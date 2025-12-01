import os
import pandas as pd
from tqdm import tqdm

data_path = "../data/baseline_2025-06-26/formatted/"

dfs = []
for year in tqdm(range(2000, 2026)):
    dfs.append(pd.read_json(data_path + str(year) + ".json"))

big_df = pd.concat(dfs, ignore_index=True)

#big_df.to_json(os.path.join(data_path, "all.json"), orient="records", lines=True)
#big_df.to_parquet(os.path.join(data_path, "all.parquet.gzip"), engine="pyarrow", compression='gzip') 
big_df.to_pickle(os.path.join(data_path, "all.pkl")) 