import tensorflow as tf
import h5py
import numpy as np

def print_weights_from_h5(file_path):
  with h5py.File(file_path, 'r') as f:
    for layer_name in f.keys():
      layer_data = f[layer_name]
      print("Layer:", layer_name)

      if 'weight' in layer_data.keys() and 'bias' in layer_data.keys():
          print("  Weight:")
          print(np.array(layer_data['weight']))
          print("  Bias:")
          print(np.array(layer_data['bias']))
      else:  # For layers like BatchNormalization
          print("  Parameters:")
          for param_name in layer_data.keys():
              print(f"    {param_name}: {np.array(layer_data[param_name])}")

      print("------------------")

file_path = 'reg_dnn_model_weights.h5'
print_weights_from_h5(file_path)

# --- Load the pre-trained model ---
model = tf.keras.models.load_model("reg_dnn_model.keras")  # Load the entire model
print(model.summary())
