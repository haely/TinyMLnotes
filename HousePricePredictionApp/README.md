# ML Model Performance Comparison

This folder contains code and results for comparing the performance of different machine learning models in their original format (e.g., Keras) and their TensorFlow Lite versions. The goal is to evaluate the potential benefits of using TensorFlow Lite in terms of execution time, memory usage, and accuracy.

## Subfolders

Each subfolder represents a different machine learning model:


## Contents of Each Model Folder

Within each model folder, you'll find:

* **`model.keras`:** The original Keras model file.
* **`model.tflite`:** The TensorFlow Lite model file.
* **`predict_price.py`:** A Python script to load both models, make predictions, and compare their performance (execution time, memory usage, and prediction results).


