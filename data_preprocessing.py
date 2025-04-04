import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numerical_features = [
    'order', 'country', 'session_id', 'page1_main_category',
    'colour', 'location', 'model_photography', 'price', 'page',
    'session_length', 'unique_pages', 'bounce', 'exit_rate', 'revisit'
]
categorical_features = ['page2_clothing_model']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])

def preprocess_file(input_path, output_path):

    df = pd.read_csv(input_path)

    df = df.drop(columns=[col for col in ['year', 'month'] if col in df.columns], errors='ignore')

    targets = df[['price_2', 'conversion']] if {'price_2', 'conversion'}.issubset(df.columns) else None

    df_processed = preprocessor.fit_transform(df)

    if hasattr(df_processed, "toarray"):
        df_processed = df_processed.toarray()

    cat_feature_names = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(cat_feature_names)

    df_cleaned = pd.DataFrame(df_processed, columns=all_feature_names)

    if targets is not None:
        df_cleaned = pd.concat([df_cleaned, targets.reset_index(drop=True)], axis=1)

    df_cleaned.to_csv(output_path, index=True)

preprocess_file("data/feature_data.csv", "data/clean_data.csv")