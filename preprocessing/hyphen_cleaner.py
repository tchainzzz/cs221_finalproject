import pandas as pd
import math

CSV_PATH = "../csv/NCAA_NBA_merged.csv"
SAVE_PATH = "../csv/final_data.csv"

class HyphenCleaner():
    def __init__(self):
        pass
    
    def clean(self, df):
        for column in df:
            if column == 'Season': continue
            for ix, row_value in df[column].iteritems():
                if "-" in str(row_value):
                    arr = str(row_value).split("-")
                    try:
                        if float(arr[1]) == 0: 
                            pct = 0
                        else:
                            pct = float(arr[0])/float(arr[1])
                        df.loc[ix, column] = pct
                    except ValueError:
                        print("Note: attempted to convert", str(row_value) + ".", "Ignore this message if this is not a statistic.")
                        continue

if __name__ == '__main__':
    h = HyphenCleaner()
    df = pd.read_csv(CSV_PATH)
    h.clean(df)
    df.to_csv(SAVE_PATH)

    