#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
uvicorn src.app:app --reload
