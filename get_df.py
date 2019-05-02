import pandas as pd


# Starter code for unpickling
def get_df(path):
    df = pd.read_pickle(path)
    print(df.head(20))