import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ClassificationModels:

    def __init__(self):
        
        self.df = pd.read_csv('data/clean_data.csv')
        self.X = self.df.drop(columns=['conversion'])
        self.y = self.df['conversion']

    def classification_models(self):

        from collections import Counter

        print("Original class distribution:", Counter(self.y))

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        oversample = SMOTE(random_state=42)
        undersample = RandomUnderSampler(random_state=42)

        models = {
            "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(class_weight='balanced'),
            "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
            "XGBoost": XGBClassifier(scale_pos_weight=(self.y == 0).sum() / (self.y == 1).sum(), eval_metric='logloss')
        }

        results = {}

        # Evaluate each model with SMOTE (oversampling)
        for name, model in models.items():
            pipeline = ImbPipeline([
                ('smote', oversample),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred)
            }
            results[f"{name}_SMOTE"] = metrics

        # Evaluate each model with undersampling
        for name, model in models.items():
            pipeline = ImbPipeline([
                ('undersample', undersample),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred)
            }
            results[f"{name}_UnderSampling"] = metrics

        return results
