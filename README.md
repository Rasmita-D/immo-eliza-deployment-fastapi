# immo-eliza-deployment

## :classical_building: Description
This project features 2 modes of deployments for the models built in the ML phase. The first is by api and the second is by using a web app. 

API: https://immo-eliza-price-prediction-6csa.onrender.com

Web App: https://immo-eliza-deployment-webapp.streamlit.app/

Web App(independent of the API): https://immo-eliza-price-predict-rd.streamlit.app/


##	:building_construction: Repo Structure
```
+---fast_api
|       main.py
|       
+---models
|       cat_boost.pkl
|       kitchen_ordinal.pkl
|       linear_regression.pkl
|       one_hot.pkl
|       random_forest.pkl
|       state_building_ordinal.pkl
|       
+---streamlit
|       slapp.py
|       using_api.py
|
|   .gitattributes
|   Dockerfile
|   README.md
|   requirements.txt

```
## 🛎️ How to use?

1. Open the web app link in a new tab and enter the values prompted to get a house prediction.
2. Use the API by following the specifications in the section below.

## API Specifications

**URL:**  immo-eliza-price-prediction-6csa.onrender.com

**path:**  /get_price/

**method:** POST

**Body:**

| Feature | Description| Accepted values|
| :---: | :---------- | --- |
| `model`   | Name of the model.   | random_forest, linear, cat_boost|
| `property_type`    | Type of the property.  |  HOUSE, APARTMENT|
| `subproperty_type`    | Sub-type of the property. | 'APARTMENT', 'APARTMENT_BLOCK', 'BUNGALOW', 'CASTLE', 'CHALET', 'COUNTRY_COTTAGE', 'DUPLEX', 'EXCEPTIONAL_PROPERTY', 'FARMHOUSE', 'FLAT_STUDIO', 'GROUND_FLOOR', 'HOUSE', 'KOT', 'LOFT', 'MANOR_HOUSE', 'MANSION', 'MIXED_USE_BUILDING', 'OTHER_PROPERTY', 'PENTHOUSE', 'SERVICE_FLAT', 'TOWN_HOUSE', 'TRIPLEX', 'VILLA'|
| `locality`    | Locality where the property is present. | 'Aalst', 'Antwerp', 'Arlon', 'Ath', 'Bastogne', 'Brugge', 'Brussels', 'Charleroi', 'Dendermonde', 'Diksmuide', 'Dinant', 'Eeklo', 'Gent', 'Halle-Vilvoorde', 'Hasselt', 'Huy', 'Ieper', 'Kortrijk', 'Leuven', 'Liège', 'Maaseik', 'Marche-en-Famenne', 'Mechelen', 'Mons', 'Mouscron', 'Namur', 'Neufchâteau', 'Nivelles', 'Oostend', 'Oudenaarde', 'Philippeville', 'Roeselare', 'Sint-Niklaas', 'Soignies', 'Thuin', 'Tielt', 'Tongeren', 'Tournai', 'Turnhout', 'Verviers', 'Veurne', 'Virton', 'Waremme'|
| `construction_year`    | The year in which the property was constructed. | Any integer year. |
| `total_area_sqm`    | Total area of the property in square meters. | Any float value. |
| `nbr_bedrooms`    | Numer of bedrooms.  | Any integer value. |
| `kitchen_type`    | Level of kitchen.   | 'NOT_INSTALLED','UNINSTALLED','INSTALLED','SEMI_EQUIPPED','HYPER_EQUIPPED'|
| `fl_furnished`    | Is the property is furnished or not? | 0/1 |
| `fl_open_fire`    |  Does the property have a fireplace?   | 0/1|
| `terrace_sqm`    | Terrace area of the property in sqm. | Any float value|
| `garden_sqm`    | Garden area of the property in sqm.   | Any float value|
| `fl_swimming_pool`    | Does the property have a swimming pool? | 0/1 |
| `fl_floodzone`    | Is the property in a flood zone? | 0/1 |
| `state_of_building`    | State of the building. | 'TO_RENOVATE','TO_RESTORE','TO_BE_DONE_UP','GOOD','JUST_RENOVATED','AS_NEW'|
| `primary_energy_consumption_sqm`    | Energy consumption per square meter.  | Any float value|
| `heating_type`    | Heating type used by the property | 'CARBON', 'ELECTRIC', 'FUELOIL', 'GAS', 'PELLET', 'SOLAR', 'WOOD'|
| `fl_double_glazing`    | Does the property have double glazing windows? | 0/1 | 
| `cadastral_income`    | Cadastral income of the property. | Any float value|


**NOTE: All the accepted values noted above are case sensitive**


Sample request
```
{
    "model":"cat_boost",
    "property_type":"HOUSE",
    "subproperty_type":"HOUSE",
    "locality":"Gent",
    "construction_year":2018,
    "total_area_sqm":70.4,
    "nbr_bedrooms":2,
    "kitchen_type":"SEMI_EQUIPPED",
    "fl_furnished":1,
    "fl_open_fire":0,
    "terrace_sqm":5,
    "garden_sqm":15,
    "fl_swimming_pool":1,
    "fl_floodzone":0,
    "state_of_building":"TO_RENOVATE",
    "primary_energy_consumption_sqm":100,
    "heating_type":"GAS",
    "fl_double_glazing":1,
    "cadastral_income":1200
}
```


Sample Response
```
{
    "price": 340896.68
}
```

## Enhancements
1. Add validation to the API development.
2. Allow missing values in the API.


## ⏱️ Timeline

This project took five days for completion.

## 📌 Personal Situation
This project was done as part of the AI Boocamp at BeCode.org. 

Connect with me on [LinkedIn](https://www.linkedin.com/in/rasmita-damaraju-33b577126/).


