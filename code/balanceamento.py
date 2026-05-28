from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

# SMOTE
# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X_train, y_train)

# Under sampling
# rus = RandomUnderSampler(random_state=42)
# X_under, y_under = rus.fit_resample(X_train, y_train)

# SMOTEENN
# se = SMOTEENN(random_state=42)
# X_se, y_se = se.fit_resample(X_train, y_train)

# Class weights
# clf = RandomForestClassifier(class_weight='balanced')
