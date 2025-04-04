import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle
import os
import numpy as np

class DataPreprocessor:
    def __init__(self):
        # Define features - removing price_2 and conversion from numerical features
        self.numerical_features = [
            'day','order', 'country', 'page1_main_category',
            'colour', 'location', 'model_photography', 'price',
            'session_length', 'unique_pages','click_sequence',
            'bounce', 'exit_rate', 'revisit'
        ]
        self.categorical_features = ['page2_clothing_model']

        # Create numerical pipeline: impute then scale
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Create categorical pipeline: impute then one-hot encode
        self.cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine into a ColumnTransformer
        self.preprocessor = ColumnTransformer([
            ('num', self.num_pipeline, self.numerical_features),
            ('cat', self.cat_pipeline, self.categorical_features)
        ])

    def preprocess_file(self, input_path, output_path):
        # Read the file
        df = pd.read_csv(input_path)

        # Drop columns 'year' and 'month' if they exist
        df = df.drop(columns=[col for col in ['year', 'month'] if col in df.columns], errors='ignore')

        # If target columns exist, separate them
        targets = df[['price_2', 'conversion']] if {'price_2', 'conversion'}.issubset(df.columns) else None

        # Fit the preprocessor and transform the data
        df_processed = self.preprocessor.fit_transform(df)

        # Convert to dense array if the result is sparse
        if hasattr(df_processed, "toarray"):
            df_processed = df_processed.toarray()

        # Retrieve feature names for the categorical columns from the OneHotEncoder
        cat_feature_names = self.preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(self.categorical_features)
        all_feature_names = self.numerical_features + list(cat_feature_names)

        # Create a DataFrame of cleaned data
        df_cleaned = pd.DataFrame(df_processed, columns=all_feature_names)

        # Concatenate targets if they exist
        if targets is not None:
            df_cleaned = pd.concat([df_cleaned, targets.reset_index(drop=True)], axis=1)

        # Save the cleaned data to CSV with index included
        df_cleaned.to_csv(output_path, index=False)

    def save_preprocessor(self, file_path="artifacts/preprocessor_clean.pkl"):
        # Create the target directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        # Save the preprocessor as a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(self.preprocessor, f)

    def preprocess_data(self, df, model_type='regression'):
        """Process a DataFrame without reading/writing files"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Handle page2_clothing_model specially - save for later
        clothing_model = df['page2_clothing_model'].iloc[0] if 'page2_clothing_model' in df.columns else ""
        
        # Drop columns 'year' and 'month' if they exist
        df = df.drop(columns=[col for col in ['year', 'month'] if col in df.columns], errors='ignore')
        
        # Transform the data (not fitting, just transforming)
        df_processed = self.preprocessor.transform(df)
        
        # Convert to dense array if the result is sparse
        if hasattr(df_processed, "toarray"):
            df_processed = df_processed.toarray()
        
        # Get the expected feature names from the preprocessor
        cat_feature_names = self.preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(self.categorical_features)
        all_feature_names = self.numerical_features + list(cat_feature_names)
        
        # Start with one-hot encoded model columns
        valid_models = ["A1", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A2", "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29", "A3", "A30", "A31", "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39", "A4", "A40", "A41", "A42", "A43", "A5", "A6", "A7", "A8", "A9", "B1", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B19", "B2", "B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27", "B28", "B29", "B3", "B30", "B31", "B32", "B33", "B34", "B4", "B5", "B6", "B7", "B8", "B9", "C1", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C2", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C3", "C30", "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C4", "C40", "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C49", "C5", "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C59", "C6", "C7", "C8", "C9", "P1", "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19", "P2", "P20", "P21", "P23", "P24", "P25", "P26", "P27", "P29", "P3", "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37", "P38", "P39", "P4", "P40", "P41", "P42", "P43", "P44", "P45", "P46", "P47", "P48", "P49", "P5", "P50", "P51", "P52", "P53", "P55", "P56", "P57", "P58", "P59", "P6", "P60", "P61", "P62", "P63", "P64", "P65", "P66", "P67", "P68", "P69", "P7", "P70", "P71", "P72", "P73", "P74", "P75", "P76", "P77", "P78", "P8", "P80", "P81", "P82", "P9"]
        
        # Create a fresh dataframe with all numerical features from processed data
        df_result = pd.DataFrame(df_processed[:, :len(self.numerical_features)], 
                                 columns=self.numerical_features)
        
        # Create clothing model columns in correct order
        for model_code in valid_models:
            col_name = f"page2_clothing_model_{model_code}"
            df_result[col_name] = 1 if clothing_model == model_code else 0
        
        # Add price_2 column
        df_result['price_2'] = df['price_2'] if 'price_2' in df.columns else 0
        
        # Handle special columns based on model type
        if model_type == 'regression':
            # For regression, we're predicting price, so remove it
            if 'price' in df_result.columns:
                df_result = df_result.drop(columns=['price'])
            # Add conversion since regression models were trained with it
            df_result['conversion'] = 0
        elif model_type == 'classification':
            # For classification, remove features that would leak the target
            if 'conversion' in df_result.columns:
                df_result = df_result.drop(columns=['conversion'])
            # We don't need to check for page column here, as it's likely not in df_result
            # The page column removal is handled in the predict_conversion function
        
        return df_result