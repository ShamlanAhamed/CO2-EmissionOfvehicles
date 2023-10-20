import streamlit as st
import pickle as pk
import numpy as np
import pandas as pd

#from sklearn.preprocessing import MinMaxScaler


def set_bg_hack_url():
   
    st.markdown(
         f"""
         <style>
         .stApp {{
            background: url("");
            background-size: cover
         }}

         
         /*.e1f1d6gn0 {{
                
            margin-left: 300px;
        }}*/
        
        .ezrtsby2{{
            background-color: rgba(0,0,0,0)
        }}

         
         </style>
         """,
         unsafe_allow_html=True
     )
#
set_bg_hack_url()

st.markdown("",unsafe_allow_html=True)

##scaler = MinMaxScaler(feature_range=(0, 70)) 



#loading all the saved model 
with open('svm_model.pkl', 'rb') as model_file:
   loaded_model_svm = pk.load(model_file)

with open('random_forst_model.pkl', 'rb') as model_file:
   loaded_model_rf = pk.load(model_file)

with open('leaniermodel.pkl', 'rb') as model_file:
   loaded_model_lr = pk.load(model_file)

with open('Decisiontree.pkl', 'rb') as model_file:
   loaded_model_dt = pk.load(model_file)

#data = loaded_model_svm
preferd_model =('Decision tree Regression','Random Forest regressor' , 'Support vector Machine' , 'Linear regression'  )

# Add a sidebar
with st.sidebar:
    st.markdown("<h1 style='color: green; text-align: center;  margin-top: -30px;'>CO2 Emission</h1>", unsafe_allow_html=True)
    
    # Add an image to the sidebar
    st.image("https://png.pngtree.com/png-vector/20220513/ourmid/pngtree-ecological-stop-co2-emissions-sign-on-white-background-png-image_4595665.png", use_column_width=True)
    
    # Add a radio button to select the ML model
    st.markdown("<h3 style='text-align: center; color: green; margin-bottom: -50px;'>Choose your preferred model:</h3>", unsafe_allow_html=True)
    ML_model = st.radio("", preferd_model)
    
    if ML_model == 'Support vector Machine':
        loaded_model = loaded_model_svm
    elif ML_model == 'Random Forest regressor':
        loaded_model = loaded_model_rf
    elif ML_model == 'Linear regression':
        loaded_model = loaded_model_lr
    elif ML_model == 'Decision tree Regression':
        loaded_model = loaded_model_dt


# Define X_train globally
X_train = pd.read_csv('X_train.csv')
#normalized_dataq = pd.read_csv('normalized.csv')

def topic():
    #st.markdown("<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: green; text-align: center; '>CO2 Emission Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 28px; text-align: center;'>We need more info</h2>", unsafe_allow_html=True)

topic()


#create a funtion to to do all the user interaction
def show_predict():
    

 
    
    Makes= ('ACURA', 'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW',
       'BUGATTI', 'BUICK', 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'DODGE',
       'FIAT', 'FORD', 'GENESIS', 'GMC', 'HONDA', 'HYUNDAI', 'INFINITI',
       'JAGUAR', 'JEEP', 'KIA', 'LAMBORGHINI', 'LAND ROVER', 'LEXUS',
       'LINCOLN', 'MASERATI', 'MAZDA', 'MERCEDES-BENZ', 'MINI',
       'MITSUBISHI', 'NISSAN', 'PORSCHE', 'RAM', 'ROLLS-ROYCE', 'SCION',
       'SMART', 'SRT', 'SUBARU', 'TOYOTA', 'VOLKSWAGEN', 'VOLVO')
    
    
    Transmissions = ('AS5', 'M6', 'AV7', 'AS6', 'AM7', 'AM8', 'AS9', 'AM9', 'AS10',
       'AM6', 'A8', 'A6', 'M7', 'AV8', 'AS8', 'A7', 'AS7', 'A9', 'AV',
       'A10', 'A4', 'M5', 'A5', 'AV6', 'AV10', 'AS4', 'AM5')
    
    Fuel_types = ('Super petrol', 'Diesel', 'Ethanol', 'Natural gas', 'Petrol')

    
    #Cylinders_types = (4 ,  6 , 12 ,  8 , 14 , 10 ,  5 , 16 ,  3 )ijk

    # Create a container to hold user input form and predictions
    st.write("User Input Form")
    
    NameO = st.text_input("Enter the Owner vehicel")
    
    Make = st.selectbox("Make(Brand)", Makes)

    Transmission = st.selectbox("Gear Transmission", Transmissions)
        
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)    
    Fuel_type_radio = st.radio("Type of fuel used (Radio Buttons)", Fuel_types)

    
    if Fuel_type_radio == 'Super petrol':
        Fuel_type_radio = 'Z'

    elif Fuel_type_radio == 'Diesel':
        Fuel_type_radio = 'D'

    elif Fuel_type_radio == 'Ethanol':
        Fuel_type_radio = 'E'      

    elif Fuel_type_radio == 'Natural gas':
        Fuel_type_radio = 'N'

    elif Fuel_type_radio == 'Petrol':
        Fuel_type_radio = 'X'        

        
    Cylinders_type = st.slider("Number of cylinders used", 3, 16, step=1)

        # Add a number input box
    Engine_Size = st.number_input("Enter Engine Size in Liters", min_value=0.0, max_value=10.0, step=0.1)
        
    Fuel_Consumption_City = st.number_input("Fuel Consumption City (L/100 km)", min_value=1.0, max_value=40.0, step=0.1)
        
    Fuel_Consumption_Hwy = st.number_input("Fuel Consumption Highway (L/100 km)", min_value=1.0, max_value=30.0, step=0.1)
    
    ok = st.button('Click to calculate')
    
    # Create a container to hold previous predictions and graphs
    
    previous_predictions = st.session_state.get('previous_predictions', [])
    
    if ok:
        # Create a dictionary with user input
        categorical_columns = ['Make','Transmission', 'Fuel Type']

        user_input= {
            'Make': [Make],
            'Engine Size(L)': [Engine_Size],
            'Cylinders': [Cylinders_type],
            'Transmission': [Transmission],
            'Fuel Type': [Fuel_type_radio],
            'Fuel Consumption City (L/100 km)': [Fuel_Consumption_City],
            'Fuel Consumption Hwy (L/100 km)': [Fuel_Consumption_Hwy],
            
        }

        # Create a DataFrame from the user input
        user_input_df = pd.DataFrame(user_input)

        # One-hot encode the categorical variables to match X_train columns
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

        # Calculate 'Fuel Consumption Comb (mpg)' based on 'Fuel Consumption Comb (L/100 km)'
        user_input_for_prediction['Fuel Consumption Comb (mpg)'] = 282.481/ user_input_for_prediction['Fuel Consumption Comb (L/100 km)']

        # Make predictions
        predictions = loaded_model.predict(user_input_for_prediction)

        # The 'predictions' variable now contains the predicted values for the user input
        print(predictions)

        #denormalized_userinput = scaler.inverse_transform(predictions)
        #print(denormalized_userinput)

        denormalized_prediction = (predictions*426.0)/70 + 96.0


        # The 'predictions' variable now contains the predicted values for the user input
        st.write(f"Predicted CO2 Emission: {denormalized_prediction[0]}")
        
        
        previous_predictions.append({
            'Make': Make,
            'CO2 Emission': denormalized_prediction[0],
            'Name':NameO
        })

        st.session_state.previous_predictions = previous_predictions
        
        # CO2 Emission level (you can replace this with actual data)

        # CO2 Emission level (you can replace this with actual data)
        co2_emission = denormalized_prediction[0]

        # Determine the emission category
        if co2_emission < 160:
            emission_category = "Good"
            icon = "🟢"  # Green icon
            description = "This emission level is good for the environment."
        elif 160 <= co2_emission <= 255:
            emission_category = "Medium"
            icon = "🔵"  # Blue icon
            description = "This emission level is considered medium for the environment."
        else:
            emission_category = "High"
            icon = "🔴"  # Red icon
            description = "This emission level is high and has a significant impact on the environment."

        # Display CO2 Emission and category with colored icon
        st.markdown(f"<h3>{icon} CO2 Emission: {co2_emission} (Category: {emission_category})</h3>", unsafe_allow_html=True)

        # Display the description
        st.write(description)
        
        
    if previous_predictions:
        st.write("Previous Predictions:")
        df_previous_predictions = pd.DataFrame(previous_predictions)
        st.dataframe(df_previous_predictions)

        # Create a scatter chart of previous predictions
        st.write("Previous Predictions Scatter Chart:")
        st.scatter_chart(df_previous_predictions, x='Make', y='CO2 Emission', color='Make')
        
                        
show_predict()       









