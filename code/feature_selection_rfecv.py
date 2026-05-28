import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif

# Correlação
# corr = X.corr()
# sns.heatmap(corr)
# plt.savefig('figures/corr_heatmap.png')

# PCA
# pca = PCA(n_components=2)
# X_p = pca.fit_transform(X_scaled)

# SelectKBest
# sel = SelectKBest(f_classif, k=10)
# sel.fit(X_train, y_train)

# RFECV
# rf = RandomForestClassifier(n_estimators=50)
# rfecv = RFECV(rf, step=1, cv=5, scoring='f1')
# rfecv.fit(X_train, y_train)
# print('n features:', rfecv.n_features_)

# LASSO (exemplo conceitual)
# from sklearn.linear_model import LogisticRegression
# l1 = LogisticRegression(penalty='l1', solver='saga')
# l1.fit(X_train, y_train)
