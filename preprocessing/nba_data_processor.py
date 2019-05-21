import pandas as pd
import df_utils
import sys
import argparse

class NBADataManager():
    def __init__(self):
        pass

    def removeNonNCAA(self, df, reference):
        for index, row in df.iterrows():
            if row['Player'] not in reference['PLAYER'].values:
                df.drop(index=index, inplace=True)
        return df

    def removeDuplicates(self, df):
        df.drop_duplicates(subset="Player", inplace=True)
        return df

    def fullClean(self, df, collegedf):
        print("Removing non-NCAA players...")
        df = self.removeNonNCAA(df, collegedf)
        print("Removing duplicates...")
        df = self.removeDuplicates(df)
        return df

if __name__ == '__main__':
    psr = argparse.ArgumentParser()
    n = NBADataManager()
    nbadf = pd.read_csv('../csv/NBASalary_take2_csv.csv')
    collegedf = pd.read_csv('../csv/out.csv')
    nbadf = n.fullClean(nbadf, collegedf)
    nbadf.to_csv('../csv/NBA_Salary_NoDup_NCAAOnly.csv')