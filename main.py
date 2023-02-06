from typing import List
from fastapi import FastAPI, HTTPException
from models import User, Prediction, IdList
import csv
from PIL import Image
from io import BytesIO
from fastapi.responses import StreamingResponse
import joblib
import pandas as pd
import json
from sklearn.neighbors import KNeighborsClassifier
from fastapi.responses import JSONResponse

app = FastAPI(cors=False)

db: List[User] = [
    User(
        id=1234, 
        gender ="F", 
        carOwner = "Y"
    ),
    User(
        id=12454, 
        gender ="M", 
        carOwner = "N"
    )
]

predictions: List[Prediction] = [
]
ids: List[IdList] = [
]

# with open('test_sample.csv', newline='') as csv_file:
#     reader = csv.reader(csv_file)
#     headers = next(reader, None)  # Get the headers.
#     id_index = headers.index("SK_ID_CURR")  # Find the index of the "id" column.
#     for row in reader:
#         id = row[id_index]  # Extract the value of the "id" column from the current row.
#         ids.append(IdList(id = id))

df = pd.read_csv('test_sample.csv')


#Dataset preparation for KNN

path = "" # "/dataset/"

X_test = pd.read_csv(path + 'test_encoded.csv')
X_test.fillna(X_test.median(), inplace=True)
indices_test = X_test['SK_ID_CURR']
X_test = X_test.set_index('SK_ID_CURR')

### API Endpoints 

@app.get("/")
def read_root():
    return {"Welcome": "to the Project"}

@app.get("/favicon.ico")
def read_favicon():
    return {"Favicon": "OK"}

@app.get("/api/v1/predictions/{predictionId}")
async def fetch_prediction(predictionId: int):
    return predictions[predictionId]

@app.get("/api/v1/df")
async def df():
    return ids

@app.get("/api/v1/image/{file_name}")
async def read_image(file_name: str):
    try:
        img = Image.open(file_name).convert("RGBA")
        img_io = BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        return StreamingResponse(img_io, media_type='image/png')
    except:
        return {"error": "could not open the image"}

@app.get("/api/v1/predict/{predictionId}")
async def fetch_prediction(predictionId: int):
    calib_fit = joblib.load('calib_pipeline.joblib')
    id = predictionId
    test = pd.read_csv('test_encoded.csv')
    #Select the row from the test data with the specified id
    test_row = test[test['SK_ID_CURR'] == id]
    # Predict default probabilities of the test data
    test_pred = calib_fit.predict_proba(test_row.values.reshape(1, -1))
    #Adding the id back
    df_out = pd.DataFrame(columns=['SK_ID_CURR','TARGET'])
    df_out = df_out.append({'SK_ID_CURR':id,'TARGET':test_pred[:,1][0]},ignore_index=True)

    return json.dumps({str(df_out.at[0,'SK_ID_CURR']):df_out.at[0,'TARGET']})

@app.get("/api/v1/characteristics/{id}")
async def fetch_characteristics(id: int):
    test = pd.read_csv('test_encoded.csv')
    filtered_test = test.loc[test['SK_ID_CURR'] == id]
    if filtered_test.empty:
        return json.dumps({'error': f'No data found for id {id}'})
    income_to_annuity_ratio = filtered_test['INCOME_TO_ANNUITY_RATIO'].iloc[0]
    proportion_life_employed = filtered_test['PROPORTION_LIFE_EMPLOYED'].iloc[0]
    print(filtered_test)
    return json.dumps({'id': id, 'INCOME_TO_ANNUITY_RATIO':income_to_annuity_ratio, 'PROPORTION_LIFE_EMPLOYED':proportion_life_employed})

#Dataset preparation for KNN
path = "" # "/dataset/"

X_train = pd.read_csv(path + 'train_encoded.csv')
X_test = pd.read_csv(path + 'test_encoded.csv')
train_old = pd.read_csv(path + 'train_sample.csv')
X_train = X_train.merge(train_old[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR', how='left')
y_train = X_train['TARGET']
X_train = X_train.dropna(subset=['TARGET'])
X_train = X_train.drop(columns=["TARGET"], axis=1)
y_train.dropna(inplace=True)
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)
indices=X_train['SK_ID_CURR']
X_test = X_test.set_index('SK_ID_CURR')
X_train = X_train.set_index('SK_ID_CURR')

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

@app.get("/api/v1/neighbors/{id}")
async def fetch_neighbors(id: int):
    if id in X_test.index.values:
        customer_features = X_test.loc[X_test.index == id].values.reshape(1, -1)
        neighbors = knn.kneighbors(customer_features, return_distance=False)
        neighbor_ids = indices.iloc[neighbors[0]].values
        return json.dumps(neighbor_ids.tolist())
    else:
        raise HTTPException(status_code=404, detail=f"Customer with ID {id} not found.")