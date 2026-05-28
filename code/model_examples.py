from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Modelos básicos
models = {
    "lr": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=100),
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "svm": SVC(probability=True),
    "knn": KNeighborsClassifier(),
}

# Treinar e avaliar (exemplo)
# for name, m in models.items():
#     m.fit(X_train, y_train)
#     print(name, m.score(X_test, y_test))

# Nota: para grandes datasets, ajustar hyperparâmetros e usar CV
