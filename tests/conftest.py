from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd
import pytest
from joblib import load
from app import app


@pytest.fixture
def model():
    model = load(Path(__file__).parent.parent.joinpath("model/logistic_regression_model.joblib"))
    return model

@pytest.fixture
def input_data():
    path_dir = Path(__file__).parent
    file_paths = ["input_data/FS1.txt", "input_data/PS2.txt"]
    data_frames = [pd.read_csv(path_dir.joinpath(file_path), sep='\t', header=None) for file_path in file_paths]
    data_frames = pd.concat(data_frames, axis=1)
    return data_frames

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

