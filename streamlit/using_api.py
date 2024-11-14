import streamlit as st
import requests


keyList = ["property_type","subproperty_type","locality","construction_year","total_area_sqm","nbr_bedrooms","kitchen_type","fl_furnished","fl_open_fire","terrace_sqm","garden_sqm","fl_swimming_pool","fl_floodzone","state_of_building","primary_energy_consumption_sqm","heating_type","fl_double_glazing","cadastral_income"]

house = {key: None for key in keyList}

base_url = "https://immo-eliza-price-prediction-6csa.onrender.com/get_price/"
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
    house['model']= st.selectbox('Select the model to use:', ["linear","cat_boost","random_forest"])
    submitted = st.form_submit_button("Submit")

    if submitted:
        response=requests.post(base_url,json=house)
        st.title(f":moneybag: Price: € {round(response.json()['price'],2)}")

