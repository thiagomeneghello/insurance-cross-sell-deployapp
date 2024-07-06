#%% 9.1 Building class
#Class
import pickle
import pandas as pd
import numpy as np
import requests
import json
import inflection
from sklearn import preprocessing as pp

class CrossSellInsurance(object):
    def __init__(self):
        self.annual_premium_scaler = pickle.load(open('features/annual_premium_scaler.pkl', 'rb'))
        self.age_scaler = pickle.load(open('features/age_scaler.pkl', 'rb'))
        self.gender_scaler = pickle.load(open('features/gender_scaler.pkl', 'rb'))
        self.region_code_scaler = pickle.load(open('features/region_code_scaler.pkl', 'rb'))
        self.sales_channel_scaler = pickle.load(open('features/sales_channel_scaler.pkl', 'rb'))
        self.vintage_scaler = pickle.load(open('features/vintage_scaler.pkl', 'rb'))

    def data_cleaning(self, df2):
        cols_old = df2.columns
        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase, cols_old))
        df2.columns = cols_new
        return (df2)
    
    def feature_engineering(self, df3):
        df3['vehicle_age'] = ['over_2_years' if i == '> 2 Years' else
                              'between_1_2_years' if i == '1-2 Year' else
                              'below_1_year' for i in df3['vehicle_age']]
        
        df3['vehicle_damage'] = [1 if i== 'Yes' else 0 for i in df3['vehicle_damage']]
        return df3
    
    def data_preparation(self, df4):
        #'annual_premium'
        df4['annual_premium'] = self.annual_premium_scaler.transform(df4[['annual_premium']].values)
        #'age'
        df4['age'] = self.age_scaler.transform(df4[['age']].values)
        #vintage
        df4['vintage'] = self.vintage_scaler.transform(df4[['vintage']].values)
        #gender - Label Encoding
        df4['gender'] = self.gender_scaler.transform(df4['gender'])
        #'region_code' - Target Encoding / Frequency Encoding
        df4['region_code'] = self.region_code_scaler.transform(df4['region_code'])
        #'vehicle_age' - One Hot Encoding / Order Encoding / Frequency Encoding
        df4 = pd.get_dummies(df4, prefix='vehicle_age', columns=['vehicle_age'])
        #'policy_sales_channel' - Target Encoding / Frequency Encoding
        df4.loc[:, 'policy_sales_channel'] = df4['policy_sales_channel'].map(self.sales_channel_scaler)

        cols_selected = ['age','region_code','previously_insured',
                         'vehicle_damage','annual_premium',
                         'policy_sales_channel','vintage']
        return df4[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        #prediction
        pred = model.predict_proba(test_data)
        # join pred into original date
        original_data['score'] = pred[:,1].tolist()
        return original_data.to_json( orient='records', date_format = 'iso')
