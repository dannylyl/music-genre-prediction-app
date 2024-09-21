"""Module contains utility functions."""

import logging
import logging.config
import os
from functools import wraps
from pathlib import Path
from typing import List

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def save_object(object_name, path_name, filename):
    """Decorator to save an object after a function is run."""

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            obj = getattr(self, object_name)
            if path_name is None or getattr(self, path_name) is None:
                logger.debug(f"Skipping saving object '{object_name}' as path is None.")
                return result
            path = Path(getattr(self, path_name) / filename)
            directory = path.parent
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory {directory}")

            if isinstance(obj, dict):
                for key, data in obj.items():
                    path = getattr(self, path_name).joinpath(f"{key}_{filename}")
                    if path.suffix == ".parquet":
                        data.to_parquet(path)
                    elif path.suffix == ".csv":
                        data.to_csv(path)
                    elif path.suffix == ".png":
                        data.savefig(path)
                    else:
                        joblib.dump(data, path)
                    logger.info(f"Saved {key} - '{object_name}' to '{path}'")
            elif isinstance(obj, pd.DataFrame):
                if path.suffix == ".parquet":
                    obj.to_parquet(path)
                elif path.suffix == ".csv":
                    obj.to_csv(path)
                else:
                    joblib.dump(obj, path)
                logger.info(f"Saved DataFrame '{object_name}' to '{path}'")
            else:
                joblib.dump(obj, path)
                logger.info(f"Saved object '{object_name}' to '{path}'")
            return result

        return wrapper

    return decorator


def load_object(attr_name, path_source):
    """Decorator to load an object from a file before the function is called."""

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Determine the actual file path to use
            if isinstance(path_source, str):
                if hasattr(self, path_source):
                    file_path = getattr(self, path_source)  # Dynamic attribute path
                else:
                    file_path = path_source  # Direct path string
            else:
                raise TypeError(
                    "path_source must be a string representing a path or an attribute name."
                )

            # Load the object if the file path exists
            if file_path and Path.exists(file_path):
                if file_path.suffix == ".parquet":
                    obj = pd.read_parquet(file_path)
                else:
                    obj = joblib.load(file_path)
                setattr(self, attr_name, obj)
                logger.debug(f"Object loaded from {file_path}")
            else:
                logger.debug(f"No such file: {file_path}")
            return method(self, *args, **kwargs)

        return wrapper

    return decorator
