import requests
import json
from pathlib import Path
from src.utils import load_dataset

features = load_dataset(Path('data_subset'), ['FS1.txt', 'PS2.txt'])

url = 'http://localhost:8000/predict'
data = {
    "features": [
        features.iloc[0].values.tolist(),
        features.iloc[1].values.tolist(),
    ]
}

data_json = json.dumps(data)
response = requests.post(url, data=data_json, headers={'Content-Type': 'application/json'})

if response.status_code == 200:
    predictions = response.json()['predictions']
    print(f"Predictions: {predictions}")
else:
    print(f"Request failed with status code: {response.status_code} and message: {response.text}")