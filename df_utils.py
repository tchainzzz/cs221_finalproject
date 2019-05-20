import pandas as pd
import numpy as np

def removePositionFromName(player):
    return player.split(",")[0]

def removePositionsFromTable(df):
    df['PLAYER'] = df['PLAYER'].apply(removePositionFromName)
    return df

def getSalaries():
    return pd.read_csv('./data/NBASalaryData03-17.csv', engine='python')


# This emits a SettingWithCopyWarning. I don't care at the moment, but I probably should.
def mergeCollegeNBA(college_df, salary_df):
    result = salary_df.copy()
    result['college_ppg'] = pd.Series(np.nan, index=result.index)
    for i, row in college_df.iterrows():
        indices = salary_df[salary_df['player'].str.contains(row['PLAYER'], case=False, regex=False)].index.values
        for nba_index in indices:
            result.at[nba_index, 'college_ppg'] = row['PTS']
    return result
