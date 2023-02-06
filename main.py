from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import json
import joblib
import pandas as pd

app = FastAPI(cors=False)

df = pd.read_csv('test_sample.csv')

# predictions: List[Prediction] = [
# ]
# ids: List[IdList] = [
# ]

ids = df['SK_CURR_ID']

X_test = pd.read_csv('test_encoded.csv')
X_test.fillna(X_test.median(), inplace=True)
indices_test = X_test['SK_ID_CURR']
X_test = X_test.set_index('SK_ID_CURR')

@app.get("/")
def read_root():
    return {"Welcome": "to the Project"}

@app.get("/favicon.ico")
def read_favicon():
    return {"Favicon": "OK"}

@app.get("/api/v1/df")
async def df():
    return ids

@app.get("/api/v1/predict/{predictionId}")
async def fetch_prediction(predictionId: int):
    calib_fit = joblib.load('calib_pipeline.joblib')
    id = predictionId
    test = X_test
    test_row = test[test['SK_ID_CURR'] == id]
    test_pred = calib_fit.predict_proba(test_row.values.reshape(1, -1))
    df_out = pd.DataFrame(columns=['SK_ID_CURR','TARGET'])
    df_out = df_out.append({'SK_ID_CURR':id,'TARGET':test_pred[:,1][0]}, ignore_index=True)
    return json.dumps({str(df_out.at[0,'SK_ID_CURR']): df_out.at[0,'TARGET']})

@app.get("/api/v1/characteristics/{id}")
async def fetch_characteristics(id: int):
    test = X_test
    filtered_test = test.loc[test['SK_ID_CURR'] == id]
    if filtered_test.empty:
        return json.dumps({'error': f'No data found for id {id}'})
    income_to_annuity_ratio = filtered_test['INCOME_TO_ANNUITY_RATIO'].iloc[0]
    proportion_life_employed = filtered_test['PROPORTION_LIFE_EMPLOYED'].iloc[0]
    return json.dumps({'income_to_annuity_ratio': income_to_annuity_ratio, 'proportion_life_employed': proportion_life_employed})