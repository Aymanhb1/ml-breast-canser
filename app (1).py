import streamlit as st
import joblib
import numpy as np

# Load the trained SVM model
model = joblib.load('svm_model.joblib')

# Load the fitted StandardScaler
scaler = joblib.load('scaler.joblib')

st.title('Breast Cancer Prediction')

st.header('Enter Patient Data:')

# Define the features based on the training data columns
# Assuming X_train was defined in a previous step and is available
# If not, you would need to load the data and define X
# For this example, I'll list the columns based on the previous X_train output

feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                 'perimeter_worst', 'area_worst', 'smoothness_worst',
                 'compactness_worst', 'concavity_worst', 'concave points_worst',
                 'symmetry_worst', 'fractal_dimension_worst']

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature.replace('_', ' ').title(), value=0.0)

# When a button is clicked, make a prediction
if st.button('Predict'):
    # Prepare the input data for prediction
    # Convert the dictionary of inputs to a numpy array in the correct order
    input_values = np.array([input_data[feature] for feature in feature_names]).reshape(1, -1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_values)

    # Make a prediction
    prediction = model.predict(input_data_scaled)

    # Display the prediction result
    if prediction[0] == 0:
        st.success('Prediction: Benign (B)')
    else:
        st.error('Prediction: Malignant (M)')
