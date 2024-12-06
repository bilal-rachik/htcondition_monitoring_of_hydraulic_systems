from pathlib import Path
from typing import Tuple
import os
import argparse
from utils import load_dataset
from pandas import DataFrame
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib


def train_test_split(dataset: DataFrame, split_index: int) -> Tuple[DataFrame, DataFrame]:
    """Split the dataset into training and testing sets based on the given index."""
    return dataset.iloc[:split_index], dataset.iloc[split_index:]

def train_logistic_regression(x_train: DataFrame, y_train: DataFrame, random_state: int = 42) -> LogisticRegression:
    """Train a Logistic Regression model."""
    model = LogisticRegression(random_state=random_state)
    model.fit(x_train, y_train.values.ravel())
    return model

def evaluate_model(model: BaseEstimator, x_test: DataFrame, y_test: DataFrame) -> None:
    """Evaluate the model and print out metrics."""
    pred_test_y = model.predict(x_test)

    print(classification_report(y_test, pred_test_y))
    print("Accuracy Score: ", accuracy_score(y_test, pred_test_y))
    print("Precision Score: ", precision_score(y_test, pred_test_y, average='weighted'))
    print("Recall Score: ", recall_score(y_test, pred_test_y, average='weighted'))
    print("F1 Score: ", f1_score(y_test, pred_test_y, average='weighted'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a Logistic Regression model on provided dataset.")

    parser.add_argument('--data-dir', type=str, default='data_subset', help='Directory containing the data files')
    parser.add_argument('--feature-files', nargs='+', default=['FS1.txt', 'PS2.txt'], help='List of feature files')
    parser.add_argument('--target-file', type=str, default='profile.txt', help='Target file')
    parser.add_argument('--split-index', type=int, default=2000,
                        help='Index to split the dataset into train and test sets')
    parser.add_argument('--model-output', type=str, default='model/logistic_regression_model.joblib',
                        help='Output path for the trained model')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load and prepare the dataset
    features = load_dataset(data_dir, args.feature_files)
    target = load_dataset(data_dir, [args.target_file])

    # Split the dataset into train and test sets
    x_train, x_test = train_test_split(features, args.split_index)
    y_train, y_test = train_test_split(target, args.split_index)
    y_train, y_test = y_train[[1]], y_test[[1]]

    # Shuffle the training data
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    # Train the model
    model = train_logistic_regression(x_train, y_train, random_state=42)

    # Evaluate the model
    evaluate_model(model, x_test, y_test)

    # Save the model
    model_output_dir = Path(args.model_output).parent
    os.makedirs(model_output_dir, exist_ok=True)

    joblib.dump(model, args.model_output)



