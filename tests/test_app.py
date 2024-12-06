
def test_predict_success(client, input_data):
    input_data = {
        "features": [
            input_data.iloc[0].values.tolist(),
            input_data.iloc[1].values.tolist(),
        ]
    }
    response = client.post("/predict", json=input_data)

    # Assert the status code is 200 (success)
    assert response.status_code == 200

    # Assert that the response contains a 'predictions' key
    assert "predictions" in response.json()

    # You can also add further assertions based on the expected predictions
    predictions = response.json()["predictions"]
    assert isinstance(predictions, list)  # Ensure predictions is a list
    assert all(isinstance(pred, int) for pred in predictions)  # Ensure each prediction is an integer


# Test case for invalid input (e.g., missing features)
def test_predict_invalid_input(client):
    input_data = {
        # Missing the 'features' key
    }

    response = client.post("/predict", json=input_data)

    # Assert the status code is 422 (Unprocessable Entity)
    assert response.status_code == 422
    # Assert that the response contains a validation error message
    assert "detail" in response.json()