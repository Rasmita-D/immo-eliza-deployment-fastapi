import streamlit as st
import pandas as pd
import numpy as np
import joblib

def pre_process_predict(data):
        
        #Encoding the kitchen field
        data['equipped_kitchen']=data['kitchen_type']
        kit_encoder = joblib.load(filename='./models/kitchen_ordinal.pkl')
        data['kitchen_type']=kit_encoder.transform(data[['equipped_kitchen']])
        data=data.drop('equipped_kitchen',axis=1)

        #Encoding the state of building field
        data['state_building']=data['state_of_building']
        state_encoder = joblib.load(filename='./models/state_building_ordinal.pkl')
        data['state_of_building']=state_encoder.transform(data[['state_building']])
        data=data.drop('state_building',axis=1)
        
        #One hot encoding nominal fields-prop type, sub prop type, locality, etc
        categorical_columns=['property_type','locality','subproperty_type','heating_type']
        ohencoder = joblib.load(filename='./models/one_hot.pkl')
        one_hot_encoded = ohencoder.transform(data[categorical_columns])
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=ohencoder.get_feature_names_out(categorical_columns))

        # Concatenate the one-hot encoded dataframe with the original dataframe
        df_encoded = pd.concat([data, one_hot_df.set_axis(data.index)], axis=1)

        #Condensing features
        df_encoded['extra_features']=df_encoded['fl_swimming_pool']+df_encoded['fl_open_fire']+df_encoded['fl_double_glazing']+df_encoded['fl_furnished']
        df_encoded=df_encoded.drop(columns=['fl_swimming_pool','fl_furnished','fl_open_fire','fl_double_glazing'])

        #Scaling construction year
        df_encoded['construction_year']=2024-df_encoded['construction_year']

        # Drop the original categorical columns
        df_encoded = df_encoded.drop(categorical_columns, axis=1)
        df_encoded = df_encoded[sorted(df_encoded.columns)]
        df_encoded = df_encoded.replace(np.nan, 0)
        return df_encoded

def pre_process_cat(data):
        
        
        #Encoding the kitchen field
        data['equipped_kitchen']=data['kitchen_type']
        kit_encoder = joblib.load(filename='./models/kitchen_ordinal.pkl')
        data['kitchen_type']=kit_encoder.transform(data[['equipped_kitchen']])
        data=data.drop('equipped_kitchen',axis=1)

        #Encoding the state of building field
        data['state_building']=data['state_of_building']
        state_encoder = joblib.load(filename='./models/state_building_ordinal.pkl')
        data['state_of_building']=state_encoder.transform(data[['state_building']])
        data=data.drop('state_building',axis=1)
        
        #Condensing features
        data['extra_features']=data['fl_swimming_pool']+data['fl_open_fire']+data['fl_double_glazing']+data['fl_furnished']
        data=data.drop(columns=['fl_swimming_pool','fl_furnished','fl_open_fire','fl_double_glazing'])
        
        #Scaling construction year
        data['construction_year']=2024-data['construction_year']

        return data

keyList = ["property_type","subproperty_type","locality","construction_year","total_area_sqm","nbr_bedrooms","kitchen_type","fl_furnished","fl_open_fire","terrace_sqm","garden_sqm","fl_swimming_pool","fl_floodzone","state_of_building","primary_energy_consumption_sqm","heating_type","fl_double_glazing","cadastral_income"]

house = {key: None for key in keyList}


st.header(":house_buildings: Belgium Price Prediction Model")
st.markdown("Enter the values to get a price prediction")
with st.form("my_form"):

    house["property_type"]=st.radio('Property type:', ['Apartment', 'House']).upper()
    house["subproperty_type"] = st.selectbox('Sub Property type:', ['Apartment', 'Apartment_block', 'Bungalow', 'Castle', 'Chalet', 'Country_cottage', 'Duplex', 'Exceptional_property', 'Farmhouse', 'Flat_studio', 'Ground_floor', 'House', 'Kot', 'Loft', 'Manor_house', 'Mansion', 'Mixed_use_building', 'Penthouse', 'Service_flat', 'Town_house', 'Triplex', 'Villa', 'Other_property']).upper()
    house["locality"]=st.selectbox('Locality:', ['Aalst', 'Antwerp', 'Arlon', 'Ath', 'Bastogne', 'Brugge', 'Brussels', 'Charleroi', 'Dendermonde', 'Diksmuide', 'Dinant', 'Eeklo', 'Gent', 'Halle-Vilvoorde', 'Hasselt', 'Huy', 'Ieper', 'Kortrijk', 'Leuven', 'Liège', 'Maaseik', 'Marche-en-Famenne', 'Mechelen', 'Mons', 'Mouscron', 'Namur', 'Neufchâteau', 'Nivelles', 'Oostend', 'Oudenaarde', 'Philippeville', 'Roeselare', 'Sint-Niklaas', 'Soignies', 'Thuin', 'Tielt', 'Tongeren', 'Tournai', 'Turnhout', 'Verviers', 'Veurne', 'Virton', 'Waremme'])
    house["construction_year"]=st.number_input('Construction Year:',1000,2024,step=1)
    house["total_area_sqm"]=st.number_input('Total area:',format="%.2f")
    house["nbr_bedrooms"]=st.number_input('Number of Bedrooms:',step=1)
    house["kitchen_type"]=st.selectbox('Kitchen Level:', ['Not_installed', 'Uninstalled', 'Installed', 'Semi_equipped', 'Hyper_equipped']).upper()
    house["fl_furnished"]=0 if st.radio('Furnished:', ['yes', 'no']) =='no' else 1
    house["fl_open_fire"]=0 if st.radio('Fire Place:', ['yes', 'no']) =='no' else 1
    house["terrace_sqm"]=st.number_input('Terrace area:',format="%.2f")
    house["garden_sqm"]=st.number_input('Garden area:',format="%.2f")
    house["fl_swimming_pool"]=0 if st.radio('Swimming pool:', ['yes', 'no'])=='no' else 1
    house["fl_floodzone"]=0 if st.radio('Flood Zone:', ['yes', 'no'])=='no' else 1
    house["state_of_building"]=st.selectbox('State of the building:', ['To_renovate', 'To_restore', 'To_be_done_up', 'Good', 'Just_renovated', 'As_new']).upper()
    house["primary_energy_consumption_sqm"]=st.number_input('Energy consumption:',format="%.2f")
    house["heating_type"]=st.selectbox('Heating Type:', ['Gas', 'Fueloil', 'Electric', 'Pellet', 'Wood', 'Solar', 'Carbon']).upper()
    house["fl_double_glazing"]=0 if st.radio('Double Glazing:', ['yes', 'no'])=='no' else 1
    house["cadastral_income"]=st.number_input('Cadastral Income:')
    model_selected= st.selectbox('Select the model to use:', ["Linear Regression","Cat Boost","Random Forest"])
    submitted = st.form_submit_button("Submit")
    if submitted:

        data=pd.DataFrame(house,index=[0])

        if model_selected=="Linear Regression":
            model=joblib.load(filename='./models/linear_regression.pkl')
            processed=pre_process_predict(data)
            p=model.predict(processed)
        
        elif model_selected=="Cat Boost":
            model=joblib.load(filename='./models/cat_boost.pkl')
            processed=pre_process_cat(data)
            p=model.predict(processed[model.feature_names_])

        elif model_selected=="Random Forest":
            model=joblib.load(filename='./models/random_forest.pkl')
            processed=pre_process_predict(data)
            p=model.predict(processed)

        st.title(f":moneybag: Price: € {round(p[0],2)}")

