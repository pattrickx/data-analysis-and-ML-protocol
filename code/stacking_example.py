from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

estimators = [
    ("rf", RandomForestClassifier(n_estimators=50)),
    ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
]

voting = VotingClassifier(estimators=estimators, voting="soft")
# voting.fit(X_train, y_train)

stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
# stack.fit(X_train, y_train)
