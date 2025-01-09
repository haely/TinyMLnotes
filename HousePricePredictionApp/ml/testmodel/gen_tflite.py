# create_home_price_model.py

import tensorflow as tf
import numpy as np

# Generate random data for training 
np.random.seed(0)
m = 500  # Slope for sqft
n = 10   # Slope for zipcode
c = 10000  # Intercept

zipcodes = np.random.randint(10000, 99999, size=1000)
sqft = np.random.randint(500, 5000, size=1000)
home_prices = m * sqft + n * zipcodes + c  # Linear relationship

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2])  # 2 inputs (zipcode, sqft)
])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Train the model
model.fit(x=np.column_stack((zipcodes, sqft)), y=home_prices, epochs=10)

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('home_price_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved to home_price_model.tflite")
