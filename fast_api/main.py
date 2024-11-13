from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI()


class Item(BaseModel):
    property_type:str
    subproperty_type:str
    locality:str
    construction_year:int
    total_area_sqm:float
    nbr_bedrooms:int
    kitchen_type:str
    fl_furnished:int
    fl_open_fire:int
    terrace_sqm:float
    garden_sqm:float
    fl_swimming_pool:int
    fl_floodzone:int
    state_of_building:str
    primary_energy_consumption_sqm:float
    heating_type:str
    fl_double_glazing:int
    cadastral_income:float

@app.post("/get_price/")
async def read_values(item:Item):
    data=pd.DataFrame(item.model_dump(),index=[0])
    model=joblib.load(filename='../models/cat_boost.pkl')
    processed=pre_process_cat(data)
    p=model.predict(processed[model.feature_names_])
    return {"price":p[0]}
 


def pre_process_predict(data):
        
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
        
        #One hot encoding nominal fields-prop type, sub prop type, locality, etc
        categorical_columns=['property_type','locality','subproperty_type','heating_type']
        ohencoder = joblib.load(filename='../models/one_hot.pkl')
        one_hot_encoded = ohencoder.transform(data[categorical_columns])
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=ohencoder.get_feature_names_out(categorical_columns))

        # Concatenate the one-hot encoded dataframe with the original dataframe
        df_encoded = pd.concat([data, one_hot_df.set_axis(data.index)], axis=1)

        # Drop the original categorical columns
        df_encoded = df_encoded.drop(categorical_columns, axis=1)
        df_encoded = df_encoded[sorted(df_encoded.columns)]
        df_encoded = df_encoded.replace(np.nan, 0)
        return df_encoded

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
