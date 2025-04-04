import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class RegressionModels:
    
    def __init__(self):
        self.df = pd.read_csv('data/clean_data.csv')
        target = 'price'

        self.X = self.df.drop(columns=[target])
        self.y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def regression_models(self):
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
        }

        results = {}

        for name, model in models.items():
            pipeline = Pipeline([('regressor', model)])
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            results[name] = {
                "RMSE": np.sqrt(mean_squared_error(self.y_test, y_pred)),
                "MAE": mean_absolute_error(self.y_test, y_pred),
                "R² Score": r2_score(self.y_test, y_pred)
            }

        return results