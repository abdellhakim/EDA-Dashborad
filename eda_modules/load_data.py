import pandas as pd

def load_and_preprocess(file_like):
    df = pd.read_csv(file_like)  
    df.columns = df.columns.str.strip().str.lower()
    df.drop_duplicates(inplace=True)
    df.ffill(inplace=True)
    return df
