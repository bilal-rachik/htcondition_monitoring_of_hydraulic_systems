from pathlib import Path
import argparse

from utils import load_dataset

from joblib import load
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator


def make_predictions(model: BaseEstimator, data: pd.DataFrame) -> ndarray:
    """Use the model to make predictions on the data."""
    return model.predict(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make predictions using a pre-trained Logistic Regression model.")

    parser.add_argument('--model-input', type=str, default='model/logistic_regression_model.joblib',
                        help='Input path for the pre-trained model')
    parser.add_argument('--data-dir', type=str, default='data_subset', help='Directory containing the data files')
    parser.add_argument('--feature-files', nargs='+', default=['FS1.txt', 'PS2.txt'], help='List of feature files')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load the data for inference
    features = load_dataset(data_dir, args.feature_files)

    # Load the pre-trained model
    model = load(args.model_input)

    # Make predictions
    predictions = make_predictions(model, features)

    # Output the predictions
    for i, pred in enumerate(predictions, 1):
        print(f"Instance {i}: Predicted class - {pred}")