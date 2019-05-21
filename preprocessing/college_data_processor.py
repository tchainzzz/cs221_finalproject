## This merges all of the raw data into a dict of dataframes.

import pandas as pd
import df_utils
import pickle
import sys
import argparse
stats = ['rebounds', 'field goals', 'free throws', '3 pointers', 'blocks', 'steals', 'assists', 'points per game', 'total points']
DATA_DIR = '../data/'
DF_DICT_NAME = 'df_dict_merged.pkl'
DF_HORIZ_DICT_NAME = 'df_horiz_merged.pkl'

# Class that manages the ESPN data
class ESPNDataManager():
    def __init__(self):
        pass

    def get_df_from_path(self, path, preview=False, preview_rows=20):
        try:
            df = pd.read_pickle(path)
            if preview: print(df.head(preview_rows))
            return df
        except FileNotFoundError:
            print("File", path, "not found")
            return None

    def get_stats(self):
        return stats

    def get_df(self, year, stat, postseason, preview=False, preview_rows=20):
        stat_indices = [i for i, full_statname in enumerate(stats) if full_statname.startswith(stat.lower())]
        if not len(stat_indices): raise Exception("Stat name not found")
        path = DATA_DIR + '_'.join(['df', str(year), str(stat_indices[0]), str(int(not not postseason))])
        return self.get_df_from_path(path, preview, preview_rows)

    def get_df_with_stat(self, stat, postseason=False, clean=True):
        df_list = []
        for year in range(2002, 2020):
            if clean:
                df_list.append(df_utils.removePositionsFromTable(self.get_df(year, stat, postseason)))
            else:
                df_list.append(self.get_df(year, stat, postseason))
        return df_list

    def get_all_dfs(self, clean=True):
        stats = {}
        for stat in self.get_stats():
            stats[(stat, True)] = self.get_df_with_stat(stat, True, clean=True)
            stats[(stat, False)] = self.get_df_with_stat(stat, False, clean=True)
        return stats

    def clean_up_names(self, df_dict):
        for _, df_list in df_dict.items():
            for df in df_list:
                df = df_utils.removePositionsFromTable(df)
        return df_dict

    def getSalaries(self):
        return pd.read_csv('../csv/NBASalaryData03-17.csv', engine='python')
        
    def drop_non_NBA_all(self, df_dict, salaries, save=False):

        def drop_non_NBA(df, salaries):
            return df[df['PLAYER'].isin(salaries['player'])]

        for _, df_list in df_dict.items():
            for i, df in enumerate(df_list):
                df_list[i] = drop_non_NBA(df, salaries)
        if save: self.save(df_dict)
        return df_dict

    # equivalent to running a vertical, column-wise concat on each list in the dict.
    def dictToLongDataframe(self, df_dict, save=False):
        for stat_id, df in df_dict.items():
            df_dict[stat_id] = pd.concat(df)
        if save: self.save(df_dict)
        return df_dict
    
    def removeDuplicates(self, longdf, save=False):
        for stat_id, df in longdf.items():
            try:
                df['TOTAL_MINS'] = df['GP'].astype('float')
                if 'MPG' in df: 
                    df['TOTAL_MINS'] = df['GP'].astype('float') * df['MPG'].astype('float')
                df = df.sort_values(by=['PLAYER', 'TOTAL_MINS'], kind='mergesort').drop(columns=['TOTAL_MINS'])
                df = df.drop_duplicates(subset=["PLAYER"], keep="last")
                longdf[stat_id] = df
            except KeyError as k:
                print("WARNING: duplicate remove on DataFrame with key ", stat_id, "failed. Error message:", k)
        if save: self.save(longdf)
        return longdf

    def save(self, obj):
        f = open(DATA_DIR + DF_DICT_NAME, "wb")
        pickle.dump(obj, f)
        f.close()

    def fullVerticalMerge(self, df_dict, salaries, save=False, verbose=False):
        if verbose: print("Dropping non-NBA players...")
        temp_dict = self.drop_non_NBA_all(df_dict, salaries)
        if verbose: print("Merging lists vertically...")
        temp_dict = self.dictToLongDataframe(temp_dict)
        if verbose: print("Removing duplicates...")
        temp_dict = self.removeDuplicates(temp_dict, save)
        if verbose: print("Merging players into single row...")
        return temp_dict

    def horizMerge(self, df_dict, save=False, csv=True):
        df_list = list(df_dict.values())
        df = df_list[0].drop(columns=['RK', 'POST', 'YR'])
        for i in range(len(df_list) - 1):
            df = df.merge(df_list[i+1].drop(columns=['RK', 'POST', 'YR']), on=['PLAYER', 'TEAM'], how='outer')
        df.to_csv('../csv/out.csv')
        print("Horizontally merged dataframe stored in csv/out.csv.")
        return df

    def preview(self, df_dict):
        for _, df_list in df_dict.items():
            print(df_list.head(n=20))
            print("Dimensions",df_list.shape)
            print("\n")

if __name__ == '__main__':
    psr = argparse.ArgumentParser()
    psr.add_argument("-v", "--verbose", help="suppress previewing all DataFrames", type=int, default=1)
    args = psr.parse_args()
    if args.verbose > 0: print("Pulling all dataframes...")
    e = ESPNDataManager()
    df_dict = e.get_all_dfs(clean=True)
    if args.verbose > 0: print("Reading salaries...")
    salaries = e.getSalaries()
    df_dict = e.fullVerticalMerge(df_dict, salaries, verbose=(args.verbose > 0))

    f1 = open(DATA_DIR + DF_DICT_NAME, "wb")
    pickle.dump(df_dict, f1)
    f1.close()

    big_df = e.horizMerge(df_dict)
    f2 = open(DATA_DIR + DF_HORIZ_DICT_NAME, "wb")
    pickle.dump(big_df, f2)
    f2.close()

    if args.verbose > 1: e.preview(df_dict)
    print("Successfully stored", len(df_dict), "DataFrames.")

# df['player'] - column with player names 

# Other useful methods
# df_pts['PLAYER'] = df_pts['PLAYER'].apply(lambda x: x.split(",")[0]) - remove player position from name field
# df_dict[(stats[0], False)].head() - preview particular stat 
