import streamlit as st
import pickle as pk
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 70)) 

#loading the saved model
with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pk.load(model_file)

data = loaded_model

# Define X_train globally
X_train = pd.read_csv('X_train.csv')
#normalized_dataq = pd.read_csv('normalized.csv')


#scaler.fit(normalized_dataq)  # Fit the scaler with X_train


# def denormalize_prediction(prediction, scaler):
#     # Reshape the prediction to match the scaler's expected input shape
#     prediction = prediction.reshape(-1, 1)

#     # Inverse transform the prediction to get the denormalized value
#     denormalized_prediction = scaler.inverse_transform(prediction)

#     return denormalized_prediction[0][0]



def show_predict():
    st.title('CO2 emission prediction')
    
    st.write("""### We need more info""")
    
    Makes= ('ACURA', 'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW',
       'BUGATTI', 'BUICK', 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'DODGE',
       'FIAT', 'FORD', 'GENESIS', 'GMC', 'HONDA', 'HYUNDAI', 'INFINITI',
       'JAGUAR', 'JEEP', 'KIA', 'LAMBORGHINI', 'LAND ROVER', 'LEXUS',
       'LINCOLN', 'MASERATI', 'MAZDA', 'MERCEDES-BENZ', 'MINI',
       'MITSUBISHI', 'NISSAN', 'PORSCHE', 'RAM', 'ROLLS-ROYCE', 'SCION',
       'SMART', 'SRT', 'SUBARU', 'TOYOTA', 'VOLKSWAGEN', 'VOLVO')
    
    
    Vehicle_classes = ('COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'MINICOMPACT',
       'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL',
       'SUV - STANDARD', 'VAN - CARGO', 'VAN - PASSENGER',
       'PICKUP TRUCK - STANDARD', 'SPECIAL PURPOSE VEHICLE',
       'PICKUP TRUCK - SMALL', 'MINIVAN', 'STATION WAGON - MID-SIZE')
    
    Transmissions = ('AS5', 'M6', 'AV7', 'AS6', 'AM7', 'AM8', 'AS9', 'AM9', 'AS10',
       'AM6', 'A8', 'A6', 'M7', 'AV8', 'AS8', 'A7', 'AS7', 'A9', 'AV',
       'A10', 'A4', 'M5', 'A5', 'AV6', 'AV10', 'AS4', 'AM5')
    
    Fuel_types = ('D', 'E', 'N', 'X', 'Z')
    
    Cylinders_types = (4 ,  6 , 12 ,  8 , 14 , 10 ,  5 , 16 ,  3 )
    

    Make = st.selectbox("Make(Brand)", Makes)
    
    Model = st.text_input("Enter the Model of vehicel")
  
    Class = st.selectbox("Vehicle class", Vehicle_classes)
    
    Transmission = st.selectbox("Gear Transmission", Transmissions)
    
    Fuel_type = st.selectbox("Type of fuel used", Fuel_types)
    
    Cylinders_type = st.selectbox("Number of cylinders used", Cylinders_types)

    # Add a number input box
    Engine_Size = st.number_input("Enter Engine Size in Liters", min_value=0.0, max_value=10.0, step=0.1)
    
    Fuel_Consumption_City = st.number_input("Fuel Consumption City (L/100 km)", min_value=0.0, max_value=40.0, step=0.1)
    
    Fuel_Consumption_Hwy = st.number_input("Fuel Consumption Highway (L/100 km)", min_value=0.0, max_value=30.0, step=0.1)
    
    ok = st.button('calculate CO2 emission')
    
    if ok:
        # Create a dictionary with user input
        categorical_columns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']

        user_input = {
            'Make': [Make],
            'Model': [Model],
            'Vehicle Class': [Class],
            'Engine Size(L)': [Engine_Size],
            'Cylinders': [Cylinders_type],
            'Transmission': [Transmission],
            'Fuel Type': [Fuel_type],
            'Fuel Consumption City (L/100 km)': [Fuel_Consumption_City],
            'Fuel Consumption Hwy (L/100 km)': [Fuel_Consumption_Hwy],
        }

        # Create a DataFrame from the user input
        user_input_df = pd.DataFrame(user_input)

        # One-hot encode the categorical variables to match the model's columns
        user_input_encoded = pd.get_dummies(user_input_df, columns=categorical_columns)

        # Initialize the user input DataFrame with all zeros and columns from X_train
        user_input_for_prediction = pd.DataFrame(0, index=[0], columns=X_train.columns)

        # Fill in the columns that match based on user input
        for col in user_input_encoded.columns:
            if col in user_input_for_prediction.columns:
                user_input_for_prediction.loc[0, col] = user_input_encoded[col].iloc[0]

        # Calculate 'Fuel Consumption Comb (L/100 km)' based on 'Fuel Consumption City (L/100 km)' and 'Fuel Consumption Hwy (L/100 km)'
        user_input_for_prediction['Fuel Consumption Comb (L/100 km)'] = (0.55 * user_input_for_prediction['Fuel Consumption City (L/100 km)'] +
                                                                      0.45 * user_input_for_prediction['Fuel Consumption Hwy (L/100 km)'])

        user_input_for_prediction['Fuel Consumption Comb (mpg)'] = 282.481/ user_input_for_prediction['Fuel Consumption Comb (L/100 km)']

        
        # Make predictions
        predictions = loaded_model.predict(user_input_for_prediction)

        # The 'predictions' variable now contains the predicted values for the user input
        st.write(f"Predicted CO2 Emission: {predictions[0]}")

        # Optionally, you can denormalize the prediction using the scaler
        # denormalized_prediction = denormalize_prediction(predictions, scaler)

        # # Display the denormalized CO2 emission
        # st.write(f"Predicted CO2 Emission (Denormalized): {denormalized_prediction}")