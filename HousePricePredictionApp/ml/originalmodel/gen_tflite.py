import tensorflow as tf

# --- Define your model architecture ---

model = Sequential()
model.add(Dense(units=256, activation='linear', input_dim=17))  # Set input_dim
model.add(BatchNormalization())
model.add(Dense(units=128, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=32, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=8, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=1, activation='linear'))

# --- Compile the model ---

model.compile(optimizer='adam', loss='mean_squared_error') 


# --- Convert to TensorFlow Lite ---

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# --- Save the .tflite file ---

with open('house_price_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved to home_price_model.tflite")