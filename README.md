This beginner project aims to predict the Gini index of countries based on factors such as primary healthcare spending and unemployment, using XGBoost regression as the model. I have used open datasets from OECD (oecd.org), which are included in the data folder.

The model is not intended for actual prediction, but to showcase data cleaning, feature engineering, and baseline prediction using XGBoost regression.

Technologies used:1.Python
                  2.pandas
                  3.numpy
                  4.scikit-learn (OneHotEncoder, SimpleImputer, ColumnTransformer, Pipeline)
                  5.xgboost

Limitations: the dataset is small (~90 rows) and only a few features were used, resulting in relatively low predictive accuracy with linear regression. Using XGBoost has improved the performance, achieving an RMSE of approximately 0.0515.