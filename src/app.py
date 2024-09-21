"""App module to run the FastAPI app."""

import sqlite3
from pathlib import Path

import pandas as pd
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf
from unidecode import unidecode

import genrelabeller

#### Get Paths to be Used in the App Downstream.
BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_PATH / "artifacts" / "model" / "model_weights.pth"
CONFIG_PATH = BASE_PATH / "conf" / "inference_config.yaml"
SCALER_PATH = BASE_PATH / "artifacts" / "scaler.pkl"
DB_PATH = BASE_PATH / "artifacts" / "inference_results.db"
OPENAPI_YAML_PATH = BASE_PATH / "docs" / "openapi.yaml"

#### Mini Setup, Load the Model and Config.
model = None
config = OmegaConf.load(CONFIG_PATH)
data_preparation_params = config["datapreparation"]
data_preprocess_params = config["datapreprocessing"]
model_params = config["model"]

#### Load the OpenAPI specification, I've written it in YAML format so I'll use PyYAML
with open(OPENAPI_YAML_PATH, "r") as file:
    openapi_spec = yaml.safe_load(file)

#### Initialize the FastAPI app.
app = FastAPI(
    title=openapi_spec["info"]["title"],
    description=openapi_spec["info"]["description"],
    version=openapi_spec["info"]["version"],
    openapi_url="/openapi.json",
)

#### Add the OpenAPI schema to the app.
if not app.openapi_schema:
    app.openapi_schema = openapi_spec


@app.get("/docs", include_in_schema=False)
async def swagger_html():
    """Use Swagger UI to render the OpenAPI schema. UI is served at /docs."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json", title="Music Genre Prediction API"
    )


@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi_endpoint():
    return JSONResponse(content=openapi_spec)


@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi_yaml():
    with open(OPENAPI_YAML_PATH, "r") as file:
        return JSONResponse(content=file.read())


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Collapse the schema by default in Swagger UI, makes it nicer to look at."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Music Genre Prediction API",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    )


@app.on_event("startup")
async def load_model():
    """Load the model on startup of the FastAPI app."""
    global model
    model = genrelabeller.model.model.GenrePredictionModel(model_params)
    weights = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    print("Model loaded successfully.")


def preprocess_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data for inference.

    This function performs the same steps as the inference pipeline in sequence to
    preprocess the input data for inference and output a test PyTorch dataloader.
    """

    data_preparation = (
        genrelabeller.data_preprocessing.data_preparation.DataPreparation(
            input_data=input_data, params=data_preparation_params
        )
    )
    data_preparation.input_data["key"] = data_preparation.input_data["key"].astype(
        float
    )
    data_preparation.input_data["time_signature"] = data_preparation.input_data[
        "time_signature"
    ].astype(float)
    (
        data_preparation.drop_na()
        .remove_stopwords_title()
        .remove_stopwords_tags()
        .get_word_embeddings_title()
        .get_word_embeddings_tags()
        .one_hot_encode_time_sig()
        .one_hot_encode_key()
    )
    cleaned_data = data_preparation.data
    data_preprocessor = genrelabeller.data_preprocessing.data_preprocess.DataPreprocess(
        clean_data=cleaned_data, params=data_preprocess_params
    )
    data_preprocessor.load_scaler_path = SCALER_PATH
    (
        data_preprocessor.separate_embeddings()
        .scale_data()
        .drop_trackid_val()
        .combine_features()
        .create_datasets()
        .create_dataloaders()
    )
    test_dataloader = data_preprocessor.val_dataloader
    return test_dataloader


@app.post("/predict/", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Make predictions on the uploaded file.

    This endpoint mirrors the inference pipeline in src/inference_pipeline.py by calling
    the preprocess_data() function defined above. It essentially takes in raw data,
    preprocesses it, and outputs a test dataloader. In hindsight, I could have maybe
    just called the inference pipeline class directly, but perhaps this is more flexible
    as it allows for leaving out some transformations easily.

    In addition to performing inference, a SQLite database is initialised using sqlite3
    and the results are logged in the database as per the requirements of the task.
    """
    try:
        #### Initialize the SQLite database to store the results.
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS inference_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trackID TEXT NOT NULL,
            title TEXT NOT NULL,
            predicted_genre TEXT NOT NULL
        )
        """
        )

        #### Inference Pipeline Mirrored Here to get the Test Dataloader.
        input_df = pd.read_csv(file.file)
        infer_dataloader = preprocess_data(input_df)

        #### Perform Inference Using the Model.
        predictions, _ = model.predict(infer_dataloader)

        genre_mapping = config.get("genre_mapping")
        predictions = [genre_mapping[int(i)] for i in predictions]
        predictions_ascii = [unidecode(genre) for genre in predictions]
        formatted_predictions = [
            {
                "track_id": row["trackID"],
                "title": row["title"],
                "predicted_genre": predictions_ascii[i],
            }
            for i, row in input_df.iterrows()
        ]
        #### Log the Results in the SQLite Database
        for index, row in input_df.iterrows():
            trackID = row.get("trackID", "Unknown")
            title = row.get("title", "Unknown")
            predicted_genre = predictions_ascii[index]
            cursor.execute(
                """
                INSERT INTO inference_results (trackID, title, predicted_genre) 
                VALUES (?, ?, ?)""",
                (trackID, title, predicted_genre),
            )
        conn.commit()
        conn.close()
        return JSONResponse(content=formatted_predictions)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/genres/", tags=["Inference"])
async def get_genres():
    """
    Return a list of classified genres in the database.

    This endpoint returns a list of genres that have been classified by the model in the
    sqlite database
    """
    DB_PATH = (
        Path(__file__).resolve().parent.parent / "artifacts" / "inference_results.db"
    )
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT predicted_genre FROM inference_results")
    genres = cursor.fetchall()
    return JSONResponse(content=[genre[0] for genre in genres])


@app.get("/titles/{genre}", tags=["Inference"])
async def get_titles_by_genre(genre: str):
    DB_PATH = (
        Path(__file__).resolve().parent.parent / "artifacts" / "inference_results.db"
    )
    """
    Return a list of titles for a provided genre.
    
    This endpoint returns a list of titles that have been classified as a specific
    genre. User should match the genre with the one returned from the /genres/ endpoint.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT title FROM inference_results WHERE predicted_genre=?", (genre,)
    )
    titles = cursor.fetchall()
    return [title[0] for title in titles]
