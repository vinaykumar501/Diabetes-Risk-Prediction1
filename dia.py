import streamlit as st
import pickle
import numpy as np
import pandas as pd

transformer = pickle.load(open('XG_transformer.pkl', 'rb'))
model = pickle.load(open('XG_model.pkl', 'rb'))

st.header("Diabetes Risk Prediction")

st.write("Enter patient details to predict diabetes risk")

gender = st.selectbox("Gender", ["Male", "Female"])
physical_activity_level = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
family_history_diabetes = st.selectbox("Family History of Diabetes", ['Yes', 'No'])
age = st.number_input("Age", min_value=1)
bmi = st.number_input("BMI")
blood_pressure = st.number_input("Blood Pressure")
fasting_glucose_level = st.number_input("Fasting Glucose Level")
insulin_level = st.number_input("Insulin Level")
HbA1c_level = st.number_input("HbA1c Level")
cholesterol_level = st.number_input("Cholesterol Level")
triglycerides_level = st.number_input("Triglycerides Level")
daily_calorie_intake = st.number_input("Daily Calorie Intake")
sugar_intake_grams_per_day = st.number_input("Sugar Intake (grams/day)")
sleep_hours = st.number_input("Sleep Hours")
stress_level = st.number_input("Stress Level (1-10)")
waist_circumference_cm = st.number_input("Waist Circumference (cm)")

submit = st.button('Predict')

if submit:

    columns = ['gender','physical_activity_level','family_history_diabetes',
               'age','bmi','blood_pressure','fasting_glucose_level',
               'insulin_level','HbA1c_level','cholesterol_level',
               'triglycerides_level','daily_calorie_intake',
               'sugar_intake_grams_per_day','sleep_hours',
               'stress_level','waist_circumference_cm']

    input_data = [[gender, physical_activity_level, family_history_diabetes,
                   age, bmi, blood_pressure, fasting_glucose_level,
                   insulin_level, HbA1c_level, cholesterol_level,
                   triglycerides_level, daily_calorie_intake,
                   sugar_intake_grams_per_day, sleep_hours,
                   stress_level, waist_circumference_cm]]

    input_df = pd.DataFrame(input_data, columns=columns)

    trans_data = transformer.transform(input_df)

    risk_score = model.predict(trans_data)[0]

    st.write(f"Risk Score: {risk_score:.2f}")

    if risk_score <= 33:
        st.success("Low Risk of Diabetes")
    elif risk_score <= 66:
        st.warning("Prediabetes")
    else:
        st.error("High Risk of Diabetes")
