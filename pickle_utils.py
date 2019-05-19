import pandas as pd
stats = ['rebounds', 'field goals', 'free throws', '3 pointers' 'blocks', 'steals', 'assists', 'points per game', 'total points']
DATA_DIR = './data/'

# Starter code for unpickling
def get_df_from_path(path, preview=False, preview_rows=20):
    try:
        df = pd.read_pickle(path)
        if preview: print(df.head(preview_rows))
        return df
    except FileNotFoundError:
        print("File", path, "not found")
        return None

def get_stats():
    return stats

def get_df(year, stat, postseason, preview=False, preview_rows=20):
    stat_indices = [i for i, full_statname in enumerate(stats) if full_statname.startswith(stat.lower())]
    if not len(stat_indices): raise Exception("Stat name not found")
    path = DATA_DIR + '_'.join(['df', str(year), str(stat_indices[0]), str(int(not not postseason))])
    return get_df_from_path(path, preview, preview_rows)

def get_df_with_stat(stat, postseason=False):
    df_list = []
    for year in range(2002, 2020):
        df_list.append(get_df(year, stat, postseason))
    return df_list

# Other useful methods
# df_pts['PLAYER'] = df_pts['PLAYER'].apply(lambda x: x.split(",")[0]) - remove player position from name field