import argparse
import pandas as pd
import math

if __name__ == '__main__':
    psr = argparse.ArgumentParser()
    psr.add_argument("-n", "--number-buckets", help="Auto-labels NBA players into percentile buckets", type=int, default=10)
    psr.add_argument("-p", "--path", help="Path to dataframe", type=str, default="./csv/final_data.csv")
    psr.add_argument("--newpath", help="New path to dataframe", type=str, default="./csv/final_data.csv")
    psr.add_argument("-c", "--column-name", help="Column name that stores the percentiles", type=str, default="Percentile")
    psr.add_argument("--class-column-name", help="Class column name", type=str, default="Class")
    args = psr.parse_args()

    df = pd.read_csv(args.path)
    df[args.class_column_name] = df[args.column_name].map(lambda pct: math.floor(float(pct) * args.number_buckets))
    df.to_csv(args.path)