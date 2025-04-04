import pandas as pd
import pickle
import os

class FeatureEngineering:
    def __init__(self):
        pass

    def session_metrics(self, df):
        session_metrics = df.groupby('session_id').agg(
            session_length=('page', 'count'),
            unique_pages=('page', 'nunique')
        ).reset_index()
        df = df.merge(session_metrics, on='session_id', how='left')
        return df

    def clickstream_patterns(self, df):
        click_sequences = df.groupby('session_id')['page'].apply(
            lambda x: ''.join(map(str, x))
        ).reset_index()
        click_sequences.rename(columns={'page': 'click_sequence'}, inplace=True)
        df = df.merge(click_sequences, on='session_id', how='left')
        return df

    def behavioral_metrics(self, df):
        df['bounce'] = df['session_length'].apply(lambda x: 1 if x == 1 else 0)
        exit_counts = df.groupby('page')['session_id'].nunique()
        exit_rate = (exit_counts / exit_counts.sum()).reset_index()
        exit_rate.columns = ['page', 'exit_rate']
        df = df.merge(exit_rate, on='page', how='left')
        df['revisit'] = df.duplicated(subset=['session_id', 'page'], keep=False).astype(int)
        return df

    def conversion_label(self, df):
        df['conversion'] = df['page'].apply(lambda x: 1 if x in [4, 5] else 0)
        return df

    def process(self, df):
        df = self.session_metrics(df)
        df = self.clickstream_patterns(df)
        df = self.behavioral_metrics(df)
        df = self.conversion_label(df)
        df = df.drop(columns=['page', 'session_id'])
        return df

    def process_and_save(self, input_path, output_path):
        df = pd.read_csv(input_path)
        df = self.process(df)
        df.to_csv(output_path, index=False)
    
    def save_preprocessor(self, preprocessor=None):
        # Create artifacts directory if it doesn't exist
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")

        # Create a simple preprocessor if none is provided
        if preprocessor is None:
            preprocessor = {
                'feature_engineering_steps': [
                    'session_metrics',
                    'clickstream_patterns',
                    'behavioral_metrics',
                    'conversion_label'
                ]
            }

        # Save the preprocessor for future predictions
        with open("artifacts/preprocessor_feature.pkl", "wb") as f:
            pickle.dump(preprocessor, f)