# House Price Prediction App

This Android app predicts house prices based on lot area and year built. It uses a pre-trained neural network model and median values for other features.

The model training and conversion code can be found here: [https://github.com/Prajwal10031999/House-Price-Prediction-with-ANN/tree/main]

## How it works

1.  **Input:** The user enters the lot area and year built of the house.
2.  **Prediction:** The app uses a TensorFlow Lite model (`house_price_model.tflite`) to predict the price.
3.  **Median values:** For features not provided by the user, the app uses median values calculated from the training data. This makes the model more robust and able to provide predictions with limited input.
4.  **Output:** The app displays the predicted house price.

## Why median values?

Using median values for missing features helps to:

*   **Provide predictions with limited input:** Users only need to enter two values.
*   **Make the model more robust:**  The model can handle cases where some feature information is unavailable.
*   **Reduce bias:** Median values are less sensitive to outliers compared to mean values.

## Files

*   **`house_price_model.tflite`:** The pre-trained TensorFlow Lite model.
*   **`model.py`:**  Contains the prediction logic and uses the TensorFlow Lite model.
*   **`train_and_convert.py`:** (Optional) Used to train and convert the model (for future retraining).

## Dependencies

*   Chaquopy (for Python integration in Android)
*   TensorFlow Lite (included in the app)

## Future improvements

*   Add more input features for better accuracy.
*   Implement model updates.
*   Improve UI/UX.