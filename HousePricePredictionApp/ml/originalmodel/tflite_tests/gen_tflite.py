import tensorflow as tf

# Load the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=256, activation='linear', input_shape=(17,), name="dense_0"),  # Set input_shape
    tf.keras.layers.BatchNormalization(name="batch_normalization_0"),
    tf.keras.layers.Dense(units=128, activation='linear', name="dense_1"),
    tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
    tf.keras.layers.Dense(units=64, activation='linear', name="dense_2"),
    tf.keras.layers.BatchNormalization(name="batch_normalization_2"),
    tf.keras.layers.Dense(units=32, activation='linear', name="dense_3"),
    tf.keras.layers.BatchNormalization(name="batch_normalization_3"),
    tf.keras.layers.Dense(units=16, activation='linear', name="dense_4"),
    tf.keras.layers.BatchNormalization(name="batch_normalization_4"),
    tf.keras.layers.Dense(units=8, activation='linear', name="dense_5"),
    tf.keras.layers.BatchNormalization(name="batch_normalization_5"),
    tf.keras.layers.Dense(units=1, activation='linear', name="dense_6")
])

# print summary
model.summary()
# Load the weights
model.load_weights("weights.h5")


# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('house_price_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved to house_price_model.tflite")