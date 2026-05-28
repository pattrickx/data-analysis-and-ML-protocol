from glob import glob

import pandas as pd

# Exemplo: leitura com glob e concatenação
files = glob("data/*.csv")
dfs = []
for f in files:
    print("lendo", f)
    dfs.append(pd.read_csv(f))

if dfs:
    df = pd.concat(dfs, ignore_index=True)
    print("shape total:", df.shape)
else:
    df = pd.DataFrame()

# Exemplo: leitura em chunks
for i, chunk in enumerate(pd.read_csv("data/large.csv", chunksize=10000)):
    # processar cada chunk
    print("chunk", i, chunk.shape)

# Exemplo: parquet/json
# df_parquet = pd.read_parquet('data/file.parquet')
# df_json = pd.read_json('data/file.json', lines=True)
