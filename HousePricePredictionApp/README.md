# Run a simple ML model  on Android

This repository demonstrates how to run a simple linear regression model on an Android device (for now). The project includes a pre-trained TensorFlow Lite model and a basic Android application that uses the model to make predictions.

## Project Structure

* **app:** Contains the Android application code.
* **model:** Contains the pre-trained TensorFlow Lite model.


## Getting Started

1. **Clone the repository:** `git clone https://github.com/your-username/your-repo-name.git`
2. **Open the project in Android Studio:** Open the `app` folder in Android Studio.
3. **Build and run:** Build the app and run it on your Android device or emulator.


## App Usage

1. **Input data:** Enter the input values in the provided fields in the app's UI.
2. **Run prediction:** Tap the "Predict" button to run the model and get the prediction.
3. **View output:** The predicted value will be displayed on the screen.

## Dependencies

The Android app uses the following TensorFlow Lite dependencies:

**Gradle**

```gradle
dependencies {
    // ... other dependencies ...
    implementation 'org.tensorflow:tensorflow-lite:2.12.0' 
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.3' 
}
```

## Additional Notes

* This is a basic example to illustrate how to run a machine learning model on Android.
* You can extend this project by adding more features, improving the UI, or using more complex models.
* Consider optimizing the model for mobile deployment using techniques like quantization.
* Make sure to handle potential errors during model loading and inference.

## Contributing

Feel free to contribute to this project by submitting bug reports, feature requests, or pull requests.
