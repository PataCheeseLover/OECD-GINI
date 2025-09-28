import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import numpy as np

idd = pd.read_csv('data/IDD.csv')
health = pd.read_csv('data/health.csv')
unemployment = pd.read_csv('data/unemployment.csv')

for df in [idd, health, unemployment]:
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

idd.TIME_PERIOD = idd.TIME_PERIOD.astype(int)
health.TIME_PERIOD = health.TIME_PERIOD.astype(int)
unemployment.TIME_PERIOD = unemployment.TIME_PERIOD.astype(int)

df = pd.merge(idd ,health, on=['REF_AREA', 'TIME_PERIOD'], how='inner')
df = pd.merge(df ,unemployment, on=['REF_AREA', 'TIME_PERIOD'], how='inner')

means = df.groupby(['REF_AREA', 'TIME_PERIOD'], as_index=False)['OBS_VALUE'].mean()
df = means.merge(df.drop(columns=['OBS_VALUE']).drop_duplicates(['REF_AREA','TIME_PERIOD']),
                 on=['REF_AREA','TIME_PERIOD'],
                 how='left')

features = ['OBS_VALUE_x', 'OBS_VALUE_y', 'OBS_VALUE', 'LABOUR_FORCE_STATUS']
df = df[features]
X = df.drop(columns=['OBS_VALUE_x'])
y = df['OBS_VALUE_x']


preprocessor = ColumnTransformer(transformers = [('num', SimpleImputer(strategy='mean'), ['OBS_VALUE', 'OBS_VALUE_y']), ('cat', OneHotEncoder(handle_unknown='ignore'), ['LABOUR_FORCE_STATUS'])])

reg = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', XGBRegressor(n_estimators=1000, learning_rate=0.05))])

scores = cross_val_score(reg, X, y, cv = 5, scoring='neg_mean_squared_error')

print(f"Model RMSE: {np.sqrt(-scores.mean())}")
