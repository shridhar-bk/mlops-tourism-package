
import os
import streamlit as st
import pandas as pd
import joblib

model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_model_v1.joblib"
)

model = joblib.load(model_path)

st.title("Tourism Package Prediction App")

st.write(
    "This application predicts whether a customer is likely to purchase the Wellness Tourism Package."
)

st.write(
    "Enter the customer details below."
)

Age = st.number_input("Age",18,100,30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited","Self Enquiry"]
)

CityTier = st.selectbox(
    "City Tier",
    [1,2,3]
)

DurationOfPitch = st.number_input(
    "Duration Of Pitch",
    min_value=1,
    value=10
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried","Small Business","Free Lancer","Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number Of Persons Visiting",
    min_value=1,
    value=2
)

NumberOfFollowups = st.number_input(
    "Number Of Followups",
    min_value=0,
    value=2
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic","Standard","Deluxe","Super Deluxe","King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3,4,5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single","Married","Divorced","Unmarried"]
)

NumberOfTrips = st.number_input(
    "Number Of Trips",
    min_value=0,
    value=2
)

Passport = st.selectbox(
    "Passport",
    ["Yes","No"]
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

OwnCar = st.selectbox(
    "Own Car",
    ["Yes","No"]
)

NumberOfChildrenVisiting = st.number_input(
    "Children Visiting",
    min_value=0,
    value=0
)

Designation = st.selectbox(
    "Designation",
    ["Executive","Manager","Senior Manager","AVP","VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=20000.0
)

input_data = pd.DataFrame([{

    "Age":Age,
    "CityTier":CityTier,
    "DurationOfPitch":DurationOfPitch,
    "NumberOfPersonVisiting":NumberOfPersonVisiting,
    "NumberOfFollowups":NumberOfFollowups,
    "PreferredPropertyStar":PreferredPropertyStar,
    "NumberOfTrips":NumberOfTrips,
    "Passport":1 if Passport=="Yes" else 0,
    "PitchSatisfactionScore":PitchSatisfactionScore,
    "OwnCar":1 if OwnCar=="Yes" else 0,
    "NumberOfChildrenVisiting":NumberOfChildrenVisiting,
    "MonthlyIncome":MonthlyIncome,

    "TypeofContact":TypeofContact,
    "Occupation":Occupation,
    "Gender":Gender,
    "ProductPitched":ProductPitched,
    "MaritalStatus":MaritalStatus,
    "Designation":Designation

}])

classification_threshold = 0.45

if st.button("Predict"):

    prediction_proba = model.predict_proba(input_data)[0,1]

    prediction = (
        prediction_proba >= classification_threshold
    ).astype(int)

    if prediction == 1:
        st.success(
            "The customer is likely to purchase the Wellness Tourism Package."
        )
    else:
        st.error(
            "The customer is unlikely to purchase the Wellness Tourism Package."
        )
