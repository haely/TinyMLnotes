import tensorflow as tf

try:
    interpreter = tf.lite.Interpreter(model_path="home_price_model.tflite")
    interpreter.allocate_tensors()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {e}")
