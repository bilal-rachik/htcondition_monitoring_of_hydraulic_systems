# htcondition_monitoring_of_hydraulic_systems


## Predictive Maintenance and Valve Condition Optimization

This project focuses on implementing a predictive maintenance system for factory machines, with a specific emphasis on analyzing and addressing the factors that lead to suboptimal valve conditions during production cycles. By leveraging machine learning, the goal is to predict whether the valve condition is optimal (100%) or not for each production cycle, enabling proactive interventions and minimizing downtime


---
### Dataset:
The data used in this project is sourced from the [Condition Monitoring of Hydraulic Systems dataset](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems). The dataset contains measurements from a hydraulic system, and each line represents a single production cycle.

The following files are used:
- **PS2:** Pressure data (bar) sampled at 100Hz.
- **FS1:** Volume flow data (l/min) sampled at 10Hz.
- **Profile:** Contains various variables, including the "valve condition," which indicates whether the valve is in an optimal state (100%).
---
### Train and Evaluate a Logistic Regression Model
```bash
python src/train.py --data-dir path/to/your/data --feature-files file1.txt file2.txt --target-file target.txt --split-index 2000 --model-output path/to/save/model.joblib
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

### Run Predictions

To make predictions using the pre-trained Logistic Regression model, execute the following command:

```bash
python src/predict.py --model-input path/to/your/model.joblib --data-dir path/to/your/data --feature-files file1.txt file2.txt
```

**Description:**
- `--model-input`: Specifies the path to the pre-trained model file (default: `model/logistic_regression_model.joblib`).
- `--data-dir`: Indicates the directory containing the dataset files to be used for predictions (default: `data_subset`).
- `--feature-files`: Lists the feature files required for prediction, separated by spaces (default: `FS1.txt PS2.txt`).

**Example:**
If you have your model at `model/logistic_regression_model.joblib` and your data in the `data_subset` directory, you can run:

```bash
python src/predict.py --model-input model/logistic_regression_model.joblib --data-dir data_subset --feature-files FS1.txt PS2.txt
```

The script will load the data, apply the pre-trained model, and print predictions for each instance in the dataset. For example:

```
Instance 1: Predicted class - 100
Instance 2: Predicted class - 90
...
```

### Launching the API
The project includes an API for making predictions using the trained model. Follow these steps to launch the API and interact with it:

#### Start the API with Two Workers
To start the API using `uvicorn` with two workers, run the following command:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```

---

### Example Requests

Once the API is running, you can send requests to it for predictions.


```python
import requests
import json

url = "http://127.0.0.1:8000/predict"

FS1 = [8.99, 0.77, 0.641, 0.006, 0, 0, 0.001, 0.003, ...]
PS2 = [125.5, 125.39, 125.4, 125.03, 124.05, 123.18, ...]
    

data = {
"features": [
    FS1+PS2,
    FS1+PS2
    ]
    
}
data_json = json.dumps(data)
response = requests.post(url, data=data_json)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)
```
---

### Build and Run the Docker Container:

To build and run your Docker container, follow these steps:

```bash
docker build -t fastapi-app .
docker run -d -p 8000:8000 fastapi-app
```