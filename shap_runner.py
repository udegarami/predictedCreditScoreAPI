## Shap Analysis
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# load your data
data = pd.read_csv("application_train.csv")
X = data.drop("TARGET", axis=1)
y = data["TARGET"]

# Remove entries with gender = XNA
data = data[data['CODE_GENDER'] != 'XNA']
# Remove entries with income type = maternity leave
data = data[data['NAME_INCOME_TYPE'] != 'Maternity leave']
# Remove entries with unknown family status
data = data[data['NAME_FAMILY_STATUS'] != 'Unknown']
data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

# Make category codes out of alphabetical values
data['NAME_TYPE_SUITE'] = pd.Categorical(data['NAME_TYPE_SUITE'])
data['NAME_TYPE_SUITE'] = data['NAME_TYPE_SUITE'].cat.codes
data['NAME_INCOME_TYPE'] = pd.Categorical(data['NAME_INCOME_TYPE'])
data['NAME_INCOME_TYPE'] = data['NAME_INCOME_TYPE'].cat.codes
data['NAME_EDUCATION_TYPE'] = pd.Categorical(data['NAME_EDUCATION_TYPE'])
data['NAME_EDUCATION_TYPE'] = data['NAME_EDUCATION_TYPE'].cat.codes
data['NAME_FAMILY_STATUS'] = pd.Categorical(data['NAME_FAMILY_STATUS'])
data['NAME_FAMILY_STATUS'] = data['NAME_FAMILY_STATUS'].cat.codes
data['NAME_HOUSING_TYPE'] = pd.Categorical(data['NAME_HOUSING_TYPE'])
data['NAME_HOUSING_TYPE'] = data['NAME_HOUSING_TYPE'].cat.codes
data['OCCUPATION_TYPE'] = pd.Categorical(data['OCCUPATION_TYPE'])
data['OCCUPATION_TYPE'] = data['OCCUPATION_TYPE'].cat.codes
data['WEEKDAY_APPR_PROCESS_START'] = pd.Categorical(data['WEEKDAY_APPR_PROCESS_START'])
data['WEEKDAY_APPR_PROCESS_START'] = data['WEEKDAY_APPR_PROCESS_START'].cat.codes
data['ORGANIZATION_TYPE'] = pd.Categorical(data['ORGANIZATION_TYPE'])
data['ORGANIZATION_TYPE'] = data['ORGANIZATION_TYPE'].cat.codes
data['FONDKAPREMONT_MODE'] = pd.Categorical(data['FONDKAPREMONT_MODE'])
data['FONDKAPREMONT_MODE'] = data['FONDKAPREMONT_MODE'].cat.codes
data['HOUSETYPE_MODE'] = pd.Categorical(data['HOUSETYPE_MODE'])
data['HOUSETYPE_MODE'] = data['HOUSETYPE_MODE'].cat.codes
data['WALLSMATERIAL_MODE'] = pd.Categorical(data['WALLSMATERIAL_MODE'])
data['WALLSMATERIAL_MODE'] = data['WALLSMATERIAL_MODE'].cat.codes
data['FLAG_OWN_CAR'] = pd.Categorical(data['FLAG_OWN_CAR'])
data['FLAG_OWN_CAR'] = data['FLAG_OWN_CAR'].cat.codes
data['FLAG_OWN_REALTY'] = pd.Categorical(data['FLAG_OWN_REALTY'])
data['FLAG_OWN_REALTY'] = data['FLAG_OWN_REALTY'].cat.codes

data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# Label encoder
le = LabelEncoder()

# Label encode binary features in training set
for col in data: 
    if col!='Test' and col!='TARGET' and data[col].dtype==object and data[col].nunique()==2:
        if col+'_ISNULL' in data.columns: #missing values here?
            data.loc[data[col+'_ISNULL'], col] = 'NaN'
        data[col] = le.fit_transform(data[col])
        if col+'_ISNULL' in data.columns: #re-remove missing vals
            data.loc[data[col+'_ISNULL'], col] = np.nan

X = data.drop("TARGET", axis=1)
X = np.asscalar(X)
#X = data.drop("Test", axis=1)
#X = X.drop("TARGET", axis=1)
y = data["TARGET"]

# train a model
model = RandomForestClassifier()
model.fit(X, y)

# explain the model's predictions using SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X,check_additivity=False)

fig = shap.summary_plot(shap_values, X)
plt.savefig('scratch.png')