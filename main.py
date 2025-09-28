import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

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
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)


preprocessor = ColumnTransformer(transformers = [('num', SimpleImputer(strategy='mean'), ['OBS_VALUE', 'OBS_VALUE_y']), ('cat', OneHotEncoder(handle_unknown='ignore'), ['LABOUR_FORCE_STATUS'])])

reg = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])

scores = -1 * cross_val_score(reg, X, y, cv = 5, scoring='r2')

print(f"Model R^2: {scores.mean()}")
