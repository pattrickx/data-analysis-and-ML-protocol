from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, None]}

gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring="f1")
# gs.fit(X_train, y_train)
# print(gs.best_params_)

# RandomizedSearchCV exemplo
from scipy.stats import randint

param_dist = {"n_estimators": randint(50, 200), "max_depth": [3, 5, 10, None]}
rs = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="f1",
)
# rs.fit(X_train, y_train)
# print(rs.best_params_)

# Recomenda-se usar Randomized primeiro, depois Grid na região promissora; Optuna para busca Bayesiana
