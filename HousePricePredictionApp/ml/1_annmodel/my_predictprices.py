import tensorflow as tf
import numpy as np

# --- Load the pre-trained model ---
model = tf.keras.models.load_model("my_model.keras")  # Load the entire model

# --- Median Values for All Features ---
# (Replace these with the actual median values from your training data)
median_values = np.array([
    3.0,   # bedrooms
    2.25,  # bathrooms
    2080.0,  # sqft_living
    5000.0,  # sqft_lot
    1.0,   # floors
    0.0,   # waterfront
    0.0,   # view
    3.0,   # condition
    7.0,   # grade
    1780.0,  # sqft_above
    0.0,   # sqft_basement
    1975.0,  # yr_built
    0.0,   # yr_renovated
    47.57,  # lat
    -122.23,  # long
    1990.0,  # sqft_living15
    5100.0   # sqft_lot15
])

# --- Prediction function ---
def predict_house_price(lot_area, year_built, median_values):
    """
    Predicts the house price using only lot_area and year_built as input,
    with median values for other features.

    Args:
        lot_area (float): The lot area of the house.
        year_built (int): The year the house was built.
        median_values (np.array): Median values for all features.

    Returns:
        float: The predicted house price.
    """

    # Create input features with median values
    input_features = median_values.copy()

    # Set the input values for lot_area and year_built
    input_features[3] = lot_area  # Index 3 for 'sqft_lot' (adjust index if needed)
    input_features[11] = year_built  # Index 11 for 'yr_built' (adjust index if needed)

    # Reshape for the model
    input_data = input_features.reshape(1, -1) 

    # Make the prediction
    prediction = model.predict(input_data)
    return prediction[0][0]  # Extract the predicted price

if __name__ == "__main__":
    lot_area = 50000  # Example lot area
    year_built = 2000  # Example year built

    predicted_price = predict_house_price(lot_area, year_built, median_values)
    print(f"Predicted house price: ${predicted_price:.2f}")