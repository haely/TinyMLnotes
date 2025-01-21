import tensorflow as tf
import numpy as np

# Load the original model
original_model = tf.keras.models.load_model('reg_dnn_model_weights.h5')

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='reg_dnn_model.tflite')
interpreter.allocate_tensors()

# Input features
input_features = np.array([
    [longitude, latitude, housing_median_age, total_rooms,
     total_bedrooms, population, households, median_income,
     rooms_per_household, bedrooms_per_room, population_per_household]
])

# Make predictions with the original model
original_predictions = original_model.predict(input_features)

# Make predictions with the TensorFlow Lite model
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_features)
interpreter.invoke()
tflite_predictions = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# Compare predictions
print("Original model predictions:", original_predictions)
print("TensorFlow Lite model predictions:", tflite_predictions)

# Measure execution time
import time

# Original model timing
start_time = time.time()
for _ in range(100):
    original_model.predict(input_features)
end_time = time.time()
original_time = end_time - start_time
print("Original model execution time:", original_time, "seconds")

# TensorFlow Lite model timing
start_time = time.time()
for _ in range(100):
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_features)
    interpreter.invoke()
end_time = time.time()
tflite_time = end_time - start_time
print("TensorFlow Lite model execution time:", tflite_time, "seconds")

# Measure memory usage
import sys

# Original model memory usage
original_memory_usage = sys.getsizeof(original_model.get_weights())
print("Original model memory usage:", original_memory_usage, "bytes")

# TensorFlow Lite model memory usage
tflite_memory_usage = interpreter.get_tensor_size(interpreter.get_output_details()[0]['index'])
print("TensorFlow Lite model memory usage:", tflite_memory_usage, "bytes")
