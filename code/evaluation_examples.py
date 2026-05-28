from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:,1]
# print('acc', accuracy_score(y_test, y_pred))
# print('precision', precision_score(y_test, y_pred))
# print('recall', recall_score(y_test, y_pred))
# print('f1', f1_score(y_test, y_pred))
# print('roc_auc', roc_auc_score(y_test, y_prob))
# cm = confusion_matrix(y_test, y_pred)

# Plot ROC / PR e curva de calibração como etapas
