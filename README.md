# htcondition_monitoring_of_hydraulic_systems

### Train and Evaluate a Logistic Regression Model

To train and evaluate a Logistic Regression model on a specified dataset, run the following command:

```bash
python src/tarin.py --data-dir path/to/your/data --feature-files file1.txt file2.txt --target-file target.txt --split-index 2000 --model-output path/to/save/model.joblib
```

**Description:**
- `--data-dir`: Specifies the directory containing the dataset files (default: `data_subset`).
- `--feature-files`: Lists the feature files required for training, separated by spaces (default: `FS1.txt PS2.txt`).
- `--target-file`: Specifies the target file containing the labels (default: `profile.txt`).
- `--split-index`: Defines the index for splitting the dataset into training and testing sets (default: `2000`).
- `--model-output`: Specifies the path to save the trained Logistic Regression model (default: `model/logistic_regression_model.joblib`).

**Example:**
If your data is stored in `data_subset`, and you want to train a model using `FS1.txt` and `PS2.txt` as features with `profile.txt` as the target file, use:

```bash
python src/train.py --data-dir data_subset --feature-files FS1.txt PS2.txt --target-file profile.txt --split-index 2000 --model-output model/logistic_regression_model.joblib
```

The script will:
1. Load the dataset (features and target).
2. Split the dataset into training and testing sets using the specified `split-index`.
3. Shuffle the training data for better model performance.
4. Train a Logistic Regression model on the training data.
5. Evaluate the model on the test data and print metrics, such as accuracy, precision, recall, F1 score, and a detailed classification report.
6. Save the trained model to the specified output path.