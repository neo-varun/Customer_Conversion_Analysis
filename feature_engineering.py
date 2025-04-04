import pandas as pd

def session_metrics(df):

    session_metrics=df.groupby('session_id').agg(
        session_length=('page','count'),
        unique_pages=('page','nunique')
    ).reset_index()

    df=df.merge(session_metrics,on='session_id',how='left')

    return df

def clickstream_patterns(df):

    click_sequences=df.groupby('session_id')['page'].apply(lambda x:''.join(map(str,x))).reset_index()
    click_sequences.rename(columns={'page':'click_sequence'},inplace=True)

    df=df.merge(click_sequences,on='session_id',how='left')

    return df

def behavioral_metrics(df):

    df['bounce']=df['session_length'].apply(lambda x:1 if x==1 else 0)

    exit_counts=df.groupby('page')['session_id'].nunique()
    exit_rate = (exit_counts / exit_counts.sum()).reset_index()
    exit_rate.columns=['page','exit_rate']

    df=df.merge(exit_rate,on='page',how='left')

    df['revisit']=df.duplicated(subset=['session_id','page'],keep=False).astype(int)

    return df

def conversion_label(df):

    df['conversion'] = df['page'].apply(lambda x: 1 if x in [4, 5] else 0)

    return df

def process_and_save(input_path, output_path):
    df = pd.read_csv(input_path)
    df = session_metrics(df)
    df = clickstream_patterns(df)
    df = behavioral_metrics(df)
    df = conversion_label(df)
    df.to_csv(output_path, index=True)

process_and_save('data/train.csv', 'data/feature_data.csv')