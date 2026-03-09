import streamlit as st 
import pandas as pd 
import joblib
import numpy as np 

#load preprocess and model from mlflow
#preprocess
scaler = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

def main():
    st.title('Machine Learning Heart Attack Prediction Model Deployment')
    
    #add components for 12   features
    # Healthy defaults for features
    age = st.number_input('Input age', min_value=29, max_value=77, value=29)
    sex = st.number_input('Input sex', min_value=0, max_value=1, value=0)
    cp = st.number_input(
        'Input chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)',
        min_value=0, max_value=3, value=0
    )
    trestbps = st.slider('Input resting blood pressure (mm Hg)', min_value=94, max_value=200, value=120)
    chol = st.slider('Input serum cholesterol (mg/dl)', min_value=126, max_value=564, value=200)
    fbs = st.number_input('Input fasting blood sugar > 120 mg/dl (0 = false, 1 = true)', min_value=0, max_value=1, value=0)
    restecg = st.number_input(
        'Input resting ECG (0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy)',
        min_value=0, max_value=2, value=0
    )
    thalach = st.slider('Input max heart rate achieved', min_value=71, max_value=202, value=180)
    exang = st.number_input('Input exercise-induced angina (0 = no, 1 = yes)', min_value=0, max_value=1, value=0)
    oldpeak = st.number_input('Input ST depression induced by exercise relative to rest', min_value=0.0, max_value=6.2, value=0.0, step=0.1)
    slope = st.number_input('Input slope of peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)', min_value=0, max_value=2, value=0)
    ca = st.number_input('Input number of major vessels colored by fluoroscopy (0-4)', min_value=0, max_value=4, value=0)
    thal = st.number_input('Input thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect, 3 = other)', min_value=0, max_value=3, value=0)
    

    if st.button('Make Prediction'):
        features = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

        
def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(input_array)
    prediction = model.predict(X_scaled)
    return int(prediction[0])

if __name__ == '__main__':

    main()      
