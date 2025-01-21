import tensorflow as tf
import numpy as np

# Load the original model (assuming it's saved in .keras format)
original_model = tf.keras.models.load_model('reg_dnn_model.keras')  # Update with your model name

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='reg_dnn_model.tflite')  # Update with your model name
interpreter.allocate_tensors()

# Get user input for latitude, longitude, and total_rooms
latitude = float(input("Enter latitude: "))
longitude = float(input("Enter longitude: "))
total_rooms = float(input("Enter total_rooms: "))

# Median values for other features (replace with your actual median values)
housing_median_age = 29.0
total_bedrooms = 538.0
population = 1426.0
households = 500.0
median_income = 3.87
rooms_per_household = 4.97
bedrooms_per_room = 0.21
population_per_household = 2.83

# Input features
input_features = np.array([
    [longitude, latitude, housing_median_age, total_rooms,
     total_bedrooms, population, households, median_income,
     rooms_per_household, bedrooms_per_room, population_per_household]
])

# Convert input features to FLOAT32
input_features = input_features.astype(np.float32)

# Tile the input features to match the expected input shape
input_features = np.tile(input_features, (136, 1))

# Make predictions with the original model
original_predictions = original_model.predict(input_features)

# Make predictions with the TensorFlow Lite model
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_features)
interpreter.invoke()
tflite_predictions = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# Compare predictions (take the average of the 136 predictions)
print("Original model predictions (average):", np.mean(original_predictions))
print("TensorFlow Lite model predictions (average):", np.mean(tflite_predictions))

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

# Original model memory usage (this will include the model architecture and weights)
original_memory_usage = sys.getsizeof(original_model) + sys.getsizeof(original_model.get_weights())
print("Original model memory usage:", original_memory_usage, "bytes")

# TensorFlow Lite model memory usage
output_details = interpreter.get_output_details()[0]
output_data = interpreter.get_tensor(output_details['index'])
tflite_memory_usage = output_data.size * output_data.itemsize
print("TensorFlow Lite model memory usage:", tflite_memory_usage, "bytes")