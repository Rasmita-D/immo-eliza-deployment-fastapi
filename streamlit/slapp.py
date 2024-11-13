import streamlit as st
import pandas as pd
import numpy as np
import joblib


def pre_process_cat(data):       
    print("inside preprocess")
    #Encoding the kitchen field
    data['equipped_kitchen']=data['kitchen_type']
    kit_encoder = joblib.load(filename='../models/kitchen_ordinal.pkl')
    data['kitchen_type']=kit_encoder.transform(data[['equipped_kitchen']])
    data=data.drop('equipped_kitchen',axis=1)

    #Encoding the state of building field
    data['state_building']=data['state_of_building']
    state_encoder = joblib.load(filename='../models/state_building_ordinal.pkl')
    data['state_of_building']=state_encoder.transform(data[['state_building']])
    data=data.drop('state_building',axis=1)
    return data

keyList = ["property_type","subproperty_type","locality","construction_year","total_area_sqm","nbr_bedrooms","kitchen_type","fl_furnished","fl_open_fire","terrace_sqm","garden_sqm","fl_swimming_pool","fl_floodzone","state_of_building","primary_energy_consumption_sqm","heating_type","fl_double_glazing","cadastral_income"]

house = {key: None for key in keyList}

st.write("Belgium Price Prediction Model")
with st.form("my_form"):
    st.write("Enter the values to get a price prediction:")
    house["property_type"]=st.radio('Property type', ['House', 'Apartment'])
    house["subproperty_type"] = st.radio('Sub Property type', ['House', 'Apartment'])
    house["locality"]=st.radio('Locality', ['Gent', 'Bruges'])
    house["construction_year"]=st.number_input('Construction Year')
    house["total_area_sqm"]=st.number_input('Total area',format="%.2f")
    house["nbr_bedrooms"]=st.number_input('Number of Bedrooms')
    house["kitchen_type"]=st.radio('Kitchen Level', ['SEMI_EQUIPPED', 'HYPER_EQUIPPED'])
    house["fl_furnished"]=0 if st.radio('Furnished?', ['yes', 'no']) =='no' else 1
    house["fl_open_fire"]=0 if st.radio('Fire Place?', ['yes', 'no']) =='no' else 1
    house["terrace_sqm"]=st.number_input('Terrace area',format="%.2f")
    house["garden_sqm"]=st.number_input('Garden area',format="%.2f")
    house["fl_swimming_pool"]=0 if st.radio('Swiiming pool?', ['yes', 'no'])=='no' else 1
    house["fl_floodzone"]=0 if st.radio('Flood Zone?', ['yes', 'no'])=='no' else 1
    house["state_of_building"]=st.radio('State of the building', ['TO_RENOVATE', 'GOOD'])
    house["primary_energy_consumption_sqm"]=st.number_input('Energy consumption',format="%.2f")
    house["heating_type"]=st.radio('Heating Type', ['GAS', 'ELECTRICITY'])
    house["fl_double_glazing"]=0 if st.radio('Double Glazing?', ['yes', 'no'])=='no' else 1
    house["cadastral_income"]=st.number_input('Cadastral Income')

    submitted = st.form_submit_button("Submit")
    if submitted:
        data=pd.DataFrame(house,index=[0])
        model=joblib.load(filename='../models/cat_boost.pkl')
        processed=pre_process_cat(data)
        p=model.predict(processed[model.feature_names_])
        st.write(f"price:{p[0]}")

