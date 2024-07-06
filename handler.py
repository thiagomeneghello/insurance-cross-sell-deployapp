#%%
#imports
import pickle
import os
import requests
import json
import pandas as pd
from sklearn import linear_model as lm
from sklearn import preprocessing as pp
from flask import Flask, Response, request
from crosssellinsurance.CrossSellInsurance import CrossSellInsurance

#%%
#loading model
model = pickle.load(open('model/model_logreg.pkl', 'rb'))

#initialize API
app = Flask(__name__) 

@app.route('/predict', methods=['POST']) 
def cross_sell_predict():
    test_json = request.get_json()

    if test_json:    #identificado data
        if isinstance(test_json, dict):   #unique example
            test_raw = pd.DataFrame(test_json, index=[0])
        
        else:    #multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        #instantiate class CrossSellInsurance
        pipeline = CrossSellInsurance()
        df1 = pipeline.data_cleaning(test_raw)
        df2 = pipeline.feature_engineering(df1)
        df3 = pipeline.data_preparation(df2)
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
    
    else:
        return Response('{no data avaible}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port)
