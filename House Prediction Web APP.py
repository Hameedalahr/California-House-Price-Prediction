import pickle
import streamlit as st
import numpy as np

# Load trained model
model = pickle.load(open("house_prediction_training_model.sav", 'rb'))

# Define prediction logic
def house_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float32)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    result = model.predict(input_data_reshaped)
    return result * 100000

# Streamlit UI
def main():
    st.title("🏡 California House Price Prediction")

    MedInc = st.text_input("Enter the Median Income")
    HouseAge = st.text_input("Enter House Age")
    AveRooms = st.text_input("Enter Average Rooms")
    AveBedrms = st.text_input("Enter Average BedRooms")
    Population = st.text_input("Enter Population")
    AveOccup = st.text_input("Enter Average Occupancy")
    Latitude = st.text_input("Enter Latitude of House")
    Longitude = st.text_input("Enter Longitude of House")

    if st.button("💰 Find Value of House"):
        try:
            input_values = [float(MedInc), float(HouseAge), float(AveRooms), float(AveBedrms),
                            float(Population), float(AveOccup), float(Latitude), float(Longitude)]
            price = house_prediction(input_values)
            st.success(f"Estimated House Price: ₹{price[0]:,.2f}")
        except ValueError:
            st.error("❌ Please enter valid numeric values in all fields.")

if __name__ == '__main__':
    main()
