import streamlit as st
import pandas as pd
import numpy as np
import joblib


def pre_process_cat(data):       
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
    house["property_type"]=st.radio('Property type:', ['House', 'Apartment'])
    house["subproperty_type"] = st.selectbox('Sub Property type:', ['APARTMENT', 'APARTMENT_BLOCK', 'BUNGALOW', 'CASTLE', 'CHALET', 'COUNTRY_COTTAGE', 'DUPLEX', 'EXCEPTIONAL_PROPERTY', 'FARMHOUSE', 'FLAT_STUDIO', 'GROUND_FLOOR', 'HOUSE', 'KOT', 'LOFT', 'MANOR_HOUSE', 'MANSION', 'MIXED_USE_BUILDING', 'OTHER_PROPERTY', 'PENTHOUSE', 'SERVICE_FLAT', 'TOWN_HOUSE', 'TRIPLEX', 'VILLA'])
    house["locality"]=st.selectbox('Locality:', ['Aalst', 'Antwerp', 'Arlon', 'Ath', 'Bastogne', 'Brugge', 'Brussels', 'Charleroi', 'Dendermonde', 'Diksmuide', 'Dinant', 'Eeklo', 'Gent', 'Halle-Vilvoorde', 'Hasselt', 'Huy', 'Ieper', 'Kortrijk', 'Leuven', 'Liège', 'Maaseik', 'Marche-en-Famenne', 'Mechelen', 'Mons', 'Mouscron', 'Namur', 'Neufchâteau', 'Nivelles', 'Oostend', 'Oudenaarde', 'Philippeville', 'Roeselare', 'Sint-Niklaas', 'Soignies', 'Thuin', 'Tielt', 'Tongeren', 'Tournai', 'Turnhout', 'Verviers', 'Veurne', 'Virton', 'Waremme'])
    house["construction_year"]=st.number_input('Construction Year:')
    house["total_area_sqm"]=st.number_input('Total area:',format="%.2f")
    house["nbr_bedrooms"]=st.number_input('Number of Bedrooms:')
    house["kitchen_type"]=st.selectbox('Kitchen Level:', ['NOT_INSTALLED','UNINSTALLED','INSTALLED','SEMI_EQUIPPED','HYPER_EQUIPPED'])
    house["fl_furnished"]=0 if st.radio('Furnished:', ['yes', 'no']) =='no' else 1
    house["fl_open_fire"]=0 if st.radio('Fire Place:', ['yes', 'no']) =='no' else 1
    house["terrace_sqm"]=st.number_input('Terrace area:',format="%.2f")
    house["garden_sqm"]=st.number_input('Garden area:',format="%.2f")
    house["fl_swimming_pool"]=0 if st.radio('Swimming pool:', ['yes', 'no'])=='no' else 1
    house["fl_floodzone"]=0 if st.radio('Flood Zone:', ['yes', 'no'])=='no' else 1
    house["state_of_building"]=st.selectbox('State of the building:', ['TO_RENOVATE','TO_RESTORE','TO_BE_DONE_UP','GOOD','JUST_RENOVATED','AS_NEW'])
    house["primary_energy_consumption_sqm"]=st.number_input('Energy consumption:',format="%.2f")
    house["heating_type"]=st.selectbox('Heating Type:', ['GAS', 'FUELOIL', 'ELECTRIC', 'PELLET', 'WOOD', 'SOLAR', 'CARBON'])
    house["fl_double_glazing"]=0 if st.radio('Double Glazing:', ['yes', 'no'])=='no' else 1
    house["cadastral_income"]=st.number_input('Cadastral Income:')

    submitted = st.form_submit_button("Submit")
    if submitted:
        data=pd.DataFrame(house,index=[0])
        model=joblib.load(filename='../models/cat_boost.pkl')
        processed=pre_process_cat(data)
        p=model.predict(processed[model.feature_names_])
        st.write(f"price:{p[0]}")

