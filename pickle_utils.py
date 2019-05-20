import pandas as pd
import df_utils
import pickle
stats = ['rebounds', 'field goals', 'free throws', '3 pointers' 'blocks', 'steals', 'assists', 'points per game', 'total points']
DATA_DIR = './data/'
DF_DICT_NAME = 'df_dict_NBA_only.pkl'

# Class that manages the 
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
        for key, df_list in df_dict.items():
            for df in df_list:
                df = df_utils.removePositionsFromTable(df)
        return df_dict

        
    def drop_non_NBA_all(self, df_dict, salaries, save=False):

        def drop_non_NBA(df, salaries):
            return df[df['PLAYER'].isin(salaries['player'])]

        for key, df_list in df_dict.items():
            for df in df_list:
                df = drop_non_NBA(df, salaries)
        if save: 
            f = open(DATA_DIR + DF_DICT_NAME, "wb")
            pickle.dump(df_dict, f)
            f.close()
        return df_dict

if __name__ == '__main__':
    print("Pulling all dataframes...")
    e = ESPNDataManager()
    df_dict = e.get_all_dfs(clean=True)

# df['player'] - column with player names 

# Other useful methods
# df_pts['PLAYER'] = df_pts['PLAYER'].apply(lambda x: x.split(",")[0]) - remove player position from name field