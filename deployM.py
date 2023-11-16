# Import necessary libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense

# Define the MLP model with three hidden layers using the Functional API
def create_mlp_model(input_shape):
    inputs = Input(shape=(input_shape,))
    hidden1 = Dense(64, activation='relu')(inputs)
    hidden2 = Dense(32, activation='relu')(hidden1)
    hidden3 = Dense(16, activation='relu')(hidden2)
    output = Dense(1, activation='sigmoid')(hidden3)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Load the trained machine learning model and scaler from the disk
optimized_model = joblib.load('optimized_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to preprocess user input without any encoding
def preprocess_input(tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, OnlineSecurity, TechSupport, InternetService):
    # Convert categorical values to numerical
    tenure = int(tenure)  # Convert to integer
    MonthlyCharges = float(MonthlyCharges)  # Convert to float
    TotalCharges = float(TotalCharges)  # Convert to float
    Contract = 0 if Contract == 'Month-to-month' else 1 if Contract == 'One year' else 2
    PaymentMethod = 0 if PaymentMethod == 'Electronic check' else 1 if PaymentMethod == 'Mailed check' else 2 if PaymentMethod == 'Bank transfer (automatic)' else 3
    OnlineSecurity = 1 if OnlineSecurity == 'Yes' else 0
    TechSupport = 1 if TechSupport == 'Yes' else 0
    InternetService = 0 if InternetService == 'DSL' else 1  # Add encoding for InternetService

    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'MonthlyCharges': [MonthlyCharges],
        'tenure': [tenure],
        'TotalCharges': [TotalCharges],
        'Contract': [Contract],
        'PaymentMethod': [PaymentMethod],
        'TechSupport': [TechSupport],
        'OnlineSecurity': [OnlineSecurity],
        'InternetService': [InternetService]
    })

    return user_input.values  # Return as NumPy array

# Main function for Streamlit app
def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title='Customer Churn Prediction',
        page_icon='🔄', 
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.title('Customer Churn Prediction 🔄')
    st.write(
        """
        Welcome to the Customer Churn Prediction app! 🔄
        Enter the customer details on the left, and we'll predict the likelihood of churn.
        """
    )

    # Input features in the sidebar
    st.sidebar.header('Customer Details')

    # Sidebar input fields for user input
    tenure = st.sidebar.text_input('Tenure', '24')  # Default value is set to '24', change as needed
    MonthlyCharges = st.sidebar.text_input('Monthly Charges', '50.0')  # Default value is set to '50.0', change as needed
    TotalCharges = st.sidebar.text_input('Total Charges', '2000.0')  # Default value is set to '2000.0', change as needed
    Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two years'])
    PaymentMethod = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    OnlineSecurity = st.sidebar.selectbox('Online Security', ['No', 'Yes'])
    TechSupport = st.sidebar.selectbox('Tech Support', ['No', 'Yes'])
    InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic'])

    if st.sidebar.button('Predict'):
        # Preprocess the user input and make predictions
        user_input_scaled = scaler.transform(preprocess_input(tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, OnlineSecurity, TechSupport, InternetService))

        # Debugging prints
        # st.write('Debugging Prints:')
        # st.write(f'user_input_scaled: {user_input_scaled}')

        prediction = optimized_model.predict(user_input_scaled)

        # More debugging prints
        # st.write(f'Raw Prediction: {prediction}')

        # Display the prediction result
        st.subheader('Prediction')
        predicted_likelihood = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction  # Handle scalar case

        # If predicting "Yes"
        if predicted_likelihood >= 0.5:
            confidence_factor = predicted_likelihood
            churn_decision = "Yes, customer is likely to churn"
        else:
            # If predicting "No"
            confidence_factor = 1 - predicted_likelihood
            churn_decision = "No, customer is not likely to churn"

        st.write(f'The predicted likelihood of churn (Confidence Factor) is: {confidence_factor:.2f}')
        st.write(f'Decision: {churn_decision}')
        # st.write(prediction)

# Run the Streamlit app
if __name__ == '__main__':
    main()



# python -m streamlit run deployM.py (To run the program)


