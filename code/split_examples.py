import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Exemplo padrão
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Split por entidade
# df = pd.read_csv('data/sample.csv')
# entities = df['mmsi'].unique()
# train_entities = np.random.choice(entities, size=int(0.8*len(entities)), replace=False)
# train = df[df['mmsi'].isin(train_entities)]
# test = df[~df['mmsi'].isin(train_entities)]

# Time-based
# df = df.sort_values('date')
# cut = int(len(df)*0.8)
# train = df.iloc[:cut]
# test = df.iloc[cut:]

# Cross-validation: GroupKFold, TimeSeriesSplit
