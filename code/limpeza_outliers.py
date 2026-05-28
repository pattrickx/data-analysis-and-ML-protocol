import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.mstats import winsorize

# df = pd.read_csv('data/sample.csv')

# Remoção de duplicatas
# df = df.drop_duplicates()

# Conversão de tipos
# df['date'] = pd.to_datetime(df['date'])

# IQR
# Q1 = df['val'].quantile(0.25)
# Q3 = df['val'].quantile(0.75)
# IQR = Q3 - Q1
# mask = (df['val'] >= Q1 - 1.5 * IQR) & (df['val'] <= Q3 + 1.5 * IQR)
# df_iqr = df.loc[mask]

# Z-score
# z = np.abs(stats.zscore(df['val'].dropna()))
# df_z = df.loc[z < 3]

# Winsorize
# arr = winsorize(df['val'], limits=[0.01, 0.01])

# Imputação por modelo (conceitual)
# from sklearn.ensemble import RandomForestRegressor
# train = df[df['x'].notna()]
# pred = df[df['x'].isna()]
# model = RandomForestRegressor(n_estimators=50)
# X = train.drop(columns=['x']).select_dtypes(include=[float, int]).fillna(0)
# y = train['x']
# model.fit(X, y)
# X_pred = pred.drop(columns=['x']).select_dtypes(include=[float, int]).fillna(0)
# df.loc[df['x'].isna(), 'x'] = model.predict(X_pred)

# Validação espacial com global_land_mask (conceitual)
# from global_land_mask import globe
# is_land = globe.is_land(df['lat'].values, df['lon'].values)
# df['is_land'] = is_land
