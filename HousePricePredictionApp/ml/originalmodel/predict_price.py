from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np

# --- Load the Pre-trained Model ---

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="house_price_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Median Values for Other Features ---
# (Replace these with the actual median values from your training data)
median_values = np.array([
    3.0,  # bedrooms
    2.25,  # bathrooms
    2080.0,  # sqft_living
    5000.0,  # sqft_lot
    1.0,  # floors
    0.0,  # waterfront
    0.0,  # view
    3.0,  # condition
    7.0,  # grade
    1780.0,  # sqft_above
    0.0,  # sqft_basement
    1975.0,  # yr_built
    0.0,  # yr_renovated
    47.57,  # lat
    -122.23,  # long
    1990.0,  # sqft_living15
    5100.0  # sqft_lot15
])

# --- Scaler ---

scaler = MinMaxScaler()
# Set the min and max values (replace with actual values from your data)
scaler.fit([[0, 1900],  # Example min values for lot_area and yr_built
           [215000, 2015]])  # Example max values

# --- Prediction Function ---

def predict_house_price(lot_area, year_built):
    """
    Predicts the house price using only lot_area and year_built as input,
    with median values for other features.

    Args:
        lot_area (float): The lot area of the house.
        year_built (int): The year the house was built.

    Returns:
        float: The predicted house price.
    """

    # Create input features with median values
    input_features = median_values.copy()

    # Set the input values for lot_area and year_built
    input_features[3] = lot_area  # Index 3 for 'sqft_lot'
    input_features[11] = year_built  # Index 11 for 'yr_built'

    # Preprocess the input
    input_data = scaler.transform([input_features])

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]