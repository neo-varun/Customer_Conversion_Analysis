import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

df = pd.read_csv('data/clean_data.csv')
X = df.drop(columns=['conversion', 'page'])
y = df['conversion']

print("Original class distribution:", Counter(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

oversample = SMOTE(random_state=42)
undersample = RandomUnderSampler(random_state=42)

models = {
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    "XGBoost": XGBClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum(), eval_metric='logloss')
}

# Evaluate each model with SMOTE (oversampling)
for name, model in models.items():
    print(f"\n{name} with SMOTE Oversampling:")
    pipeline = ImbPipeline([
        ('smote', oversample),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

# Evaluate each model with undersampling
for name, model in models.items():
    print(f"\n{name} with Random Undersampling:")
    pipeline = ImbPipeline([
        ('undersample', undersample),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))