import pandas as pd
import numpy as np
import pickle

# detached, non-class utility methods useful for light-weight file-loading/interpreter data management.

def removePositionFromName(player):
    return player.split(",")[0]

def removePositionsFromTable(df):
    df['PLAYER'] = df['PLAYER'].apply(removePositionFromName)
    return df

def loadDict(pkl):
    DATA_DIR = '../data/'
    with open(DATA_DIR + pkl, "rb") as f:
        return pickle.load(f)

# This emits a SettingWithCopyWarning. I don't care at the moment, but I probably should. ONLY USED FOR BASELINE MODEL.
def mergeCollegeNBA(college_df, salary_df):
    result = salary_df.copy()
    result['college_ppg'] = pd.Series(np.nan, index=result.index)
    for _, row in college_df.iterrows():
        indices = salary_df[salary_df['player'].str.contains(row['PLAYER'], case=False, regex=False)].index.values
        for nba_index in indices:
            result.at[nba_index, 'college_ppg'] = row['PTS']
    return result
