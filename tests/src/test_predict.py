import pytest

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
from src.predict import make_predictions


def test_make_predictions(model, input_data):
    """Test de la fonction make_predictions."""
    predictions = make_predictions(model, input_data)

    assert isinstance(predictions.tolist(), list)
    assert isinstance(len(predictions.tolist()), 2)


