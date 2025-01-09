# tests/test_linear_regression_model.py

import tensorflow as tf
import numpy as np
import pytest

# --- Test Cases ---
def test_model_accuracy():
    """
    Verify the accuracy of the TensorFlow Lite model using random dummy data.
    """
    interpreter = tf.lite.Interpreter(model_path="../home_price_model.tflite")
    interpreter.allocate_tensors()

    # Generate random test data and calculate expected prices
    np.random.seed(1)  # Use a different seed than your training data
    test_zipcodes = np.random.randint(10000, 99999, size=100)
    test_sqft = np.random.randint(500, 5000, size=100)
    # Use the same equation from your training data generation
    test_home_prices = 500 * test_sqft + 10 * test_zipcodes + 10000  

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    for zipcode, sqft in zip(test_zipcodes, test_sqft):
        input_data = np.array([zipcode, sqft], dtype=np.float32).reshape(1, 2)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0][0])

    # Calculate Mean Squared Error
    mse = np.mean((test_home_prices - predictions)**2)

    # Assert that the MSE is below a threshold (adjust as needed)
    assert mse < 10000, f"MSE too high: {mse}"

def test_input_shape():
    """
    Ensure the model handles the correct input shape.
    """
    interpreter = tf.lite.Interpreter(model_path="../home_price_model.tflite")  # Updated path
    interpreter.allocate_tensors()

    # Correct input shape
    input_data = np.array([50000, 1500], dtype=np.float32).reshape(1, 2)  
    
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data) 
    interpreter.invoke() # Should not raise any errors

    # Incorrect input shape (optional)
    with pytest.raises(ValueError): 
        incorrect_input_data = np.array([50000, 1500], dtype=np.float32) # Shape (2,)
        interpreter.set_tensor(input_details[0]['index'], incorrect_input_data)
        interpreter.invoke()

def test_model_loading():
    """
    Confirm that the .tflite model can be loaded successfully.
    """
    try:
        interpreter = tf.lite.Interpreter(model_path="../home_price_model.tflite")  # Updated path
        interpreter.allocate_tensors() 
    except Exception as e:
        pytest.fail(f"Model loading failed: {e}")