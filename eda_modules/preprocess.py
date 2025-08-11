import pandas as pd

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    # Basic preprocessing
    df.columns = df.columns.str.strip().str.lower()
    df.drop_duplicates(inplace=True)

    # Optional: fill or drop missing values
    df.ffill(inplace=True)


    return df
