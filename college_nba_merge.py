import argparse
import sys
import pandas as pd

COLLEGE_CSV_PATH = './csv/out.csv'
NBA_CSV_PATH = './csv/NBA_Salary_NoDup_NCAAOnly.csv'

if __name__ == '__main__':
    print("Attempting to merge...")
    psr = argparse.ArgumentParser()
    psr.add_argument("-c", "--cross-validate", help="ensure all names appear in both Dataframes", action='store_true')
    psr.add_argument("-v", "--verbose", help="Print preview DataFrames, not just the last one", action='store_true')
    args = psr.parse_args()
    college_df = pd.read_csv(COLLEGE_CSV_PATH)
    nba_df = pd.read_csv(NBA_CSV_PATH)
    if args.verbose:
        print(college_df.head(n=10))
        print(nba_df.head(n=10))
    if args.cross_validate:
        discrepancies = 0
        for player in college_df['PLAYER'].values:
            if player not in nba_df['Player'].values:
                print("Player \""+str(player)+"\" in NCAA DataFrame but not in NBA DataFrame")
                discrepancies += 1
        for player in nba_df['Player'].values:
            if player not in college_df['PLAYER'].values:
                print("Player \""+str(player)+"\" in NBA DataFrame but not in NCAA DataFrame")
                discrepancies += 1
        print("Discrepancies:", discrepancies)
        sys.exit(0)

    # now the merge step
    print("Merging the two dataframes...")
    merged_df = pd.merge(college_df, nba_df, right_on=['Player'], left_on=['PLAYER'], how='left')
    merged_df.drop(columns=['Player'], inplace=True)
    print(merged_df.head(n=10))
    merged_df.to_csv("./csv/NCAA_NBA_merged.csv")

