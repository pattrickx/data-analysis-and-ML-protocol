Translated from Portuguese — original: projects/protocolo_analise/protocolo_analise.md

# Data Analysis Protocol — EN-US

This document describes a step-by-step protocol for data analysis in Python (pandas, numpy, scikit-learn, imbalanced-learn, shap, matplotlib/seaborn). All code examples are placed in Markdown code blocks and the equivalent scripts were saved in `projects/protocolo_analise/code/`.

---

TABLE OF CONTENTS (main sections)

- Importing multiple files
- Identifying general data characteristics (descriptive statistics)
- Basic and advanced treatments (outliers)
- Exploratory analysis protocol (SMART methodology and tests)
- Data splitting
- Feature selection
- Data balancing
- Classic models and neural networks
- Hyperparameter search
- Stacking and Voting
- Semi-supervised learning
- Unsupervised learning
- Model evaluation
- Explainability (SHAP)
- Drift monitoring in production

---

Note: DO NOT run the code here — corresponding scripts were saved in `projects/protocolo_analise/code/`.

1) Importing multiple files

- Objective: read multiple CSV/Parquet/JSON files and/or very large files in chunks.

Example (reading with glob and concatenation):

```python
# scripts: projects/protocolo_analise/code/importacao_multiplos_arquivos.py
import pandas as pd
from glob import glob

files = glob('data/*.csv')  # pattern: adjust to your directory
dfs = []
for f in files:
    dfs.append(pd.read_csv(f))

df = pd.concat(dfs, ignore_index=True)
print(df.shape)
```

Comment: the block above is the simplest pattern — useful when each file represents the same schema (same columns) and the volume fits in memory. Keep files ordered when temporal order matters (e.g.: daily logs).

Example (reading in chunks for large files):

```python
# scripts: projects/protocolo_analise/code/importacao_multiplos_arquivos.py (continuação)
import pandas as pd
chunks = pd.read_csv('data/large.csv', chunksize=10_000)
for i, chunk in enumerate(chunks):
    # process chunk incrementally
    print(i, chunk.shape)

# save after incremental processing
```

Comment: reading in chunks enables streaming processing (transformations and incremental writes). Recommendations: 1) apply dtypes when reading to reduce memory; 2) use parquet when possible, as it preserves types and is faster; 3) for production ETL, build a pipeline that validates schema and logs rejected rows.

Example (reading multiple formats):

```python
# read parquet, json
df_parquet = pd.read_parquet('data/file.parquet')
df_json = pd.read_json('data/file.json', lines=True)
```

When to use: glob/concat for small/medium files; chunks for datasets too large to fit in RAM; parquet for performance and type fidelity.

Operational best practices:
- Validate schema (expected columns and types) immediately after reading.
- Log counts per source (to reconcile with origins).
- Avoid duplicate reads: maintain a processing index per file (processed flag).
- Associated scripts: `projects/protocolo_analise/code/importacao_multiplos_arquivos.py` (run for initial ingestion).

---

2) Identifying general data characteristics (descriptive statistics)

- Objective: get a quick view of shape, types, nulls and statistics.

Example:

```python
# scripts: projects/protocolo_analise/code/eda_smart.py (parte: descriptiva)
import pandas as pd

df = pd.read_csv('data/sample.csv')
print(df.info())
print(df.describe(include='all'))
print('nulls per column:\n', df.isna().sum())
print('dtypes:\n', df.dtypes)
```

Context and practical objective:
- First quality checkpoint: understand how many rows/columns, types (numeric/categorical/datetime), and columns with many missing values.
- Detect incorrect types (e.g.: IDs read as floats) and necessary conversions.

Recommended steps:
1. Run `df.info()` and `df.isna().sum()`.
2. Generate histograms and boxplots for numeric variables.
3. Count categories for categorical columns (top-N).
4. Identify date columns and validate timezone/format.

Interpreting outputs:
- Very discrepant counts in `describe()` indicate nulls or inconsistent reading.
- Types like `object` where `datetime` is expected indicate conversion is needed.

Risks/behaviors to watch:
- High-cardinality columns treated as categorical can explode memory with one-hot encoding.
- Numeric-looking IDs as strings may be mistakenly transformed and lose information.

Links to scripts and execution:
- `projects/protocolo_analise/code/eda_smart.py` — run for descriptive summary.

Conceptual example (commented):

```python
# Extra example of quick category analysis
# Count top-10 of a categorical column to evaluate distribution
print(df['categoria'].value_counts().head(10))
```

---

3) Basic and advanced treatments (removal of nulls, duplicates, type conversion, outliers)

- Basics:

```python
# scripts: projects/protocolo_analise/code/limpeza_outliers.py
# remove duplicates
df = df.drop_duplicates()
# remove rows with critical missing values
df = df.dropna(subset=['target'])
# type conversion
df['date'] = pd.to_datetime(df['date'])
```

Context and practical objective:
- Removing duplicates avoids counting bias; however, in events where duplicates represent legitimate events (e.g.: reprocessed logs), understand the cause before dropping.

Recommended steps:
1. Identify deduplication key(s) and validate with samples.
2. For missing `target`, assess whether imputation is acceptable; often correct action is to remove.
3. Log how many rows were removed by each operation.

Interpreting outputs:
- Compare shape before/after and keep logs of changes (audit trail).

Risks/behaviors to watch:
- Removing too many rows can bias the sample; justify and document.

- Outliers: IQR

```python
# IQR
Q1 = df['val'].quantile(0.25)
Q3 = df['val'].quantile(0.75)
IQR = Q3 - Q1
mask = (df['val'] >= Q1 - 1.5 * IQR) & (df['val'] <= Q3 + 1.5 * IQR)
df_iqr = df.loc[mask]
```

Comment: the IQR method is robust to moderate skewness and is recommended as a first approach for outlier detection. However, it depends on context — e.g., speed/positioning data may have valid extremes.

- Z-score

```python
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df['val'].dropna()))
df_z = df.loc[z < 3]
```

Comment: Z-score assumes approximately normal distribution; avoid in strongly skewed distributions.

- Winsorizing (limit extremes):

```python
from scipy.stats.mstats import winsorize
arr = winsorize(df['val'], limits=[0.01, 0.01])
```

Comment: Winsorize preserves the number of observations by changing only extreme values — useful when you do not want to discard data.

- Model-based imputation (simple example with RandomForestRegressor style):

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# conceptual example: predict column 'x' from others
train = df[df['x'].notna()]
pred = df[df['x'].isna()]
model = RandomForestRegressor(n_estimators=50)
X = train.drop(columns=['x']).select_dtypes(include=[float, int]).fillna(0)
y = train['x']
model.fit(X, y)
X_pred = pred.drop(columns=['x']).select_dtypes(include=[float, int]).fillna(0)
df.loc[df['x'].isna(), 'x'] = model.predict(X_pred)
```

Comment: model-based imputation is powerful but can introduce leakage if information derived from the target is improperly used. Always separate a set to validate imputation performance.

- Spatial validation with global_land_mask (usage example):

```python
# Install: global_land_mask
# Conceptual usage:
from global_land_mask import globe
# validate lat/lon coordinates
is_land = globe.is_land(df['lat'].values, df['lon'].values)
df['is_land'] = is_land
```

When to use each technique: IQR/Z for robust detection; winsorize to limit without losing cases; model-based imputation when missingness is not MCAR and there are predictable relationships.

Best practices:
- Log cleaning operations and save intermediate versions (e.g.: df_v1.csv).
- Automate validations (e.g.: check if mean/median changed drastically after cleaning).
- Associated scripts: `projects/protocolo_analise/code/limpeza_outliers.py`.

---

4) Exploratory Analysis Protocol (SMART Methodology + tests)

- SMART methodology applied: Example question presented in the original protocol:

"Only 3 months after change X, does the customer churn rate (target) decrease by at least 5% compared to the previous period?"

Context and practical objective:
- Convert vague questions into measurable objectives (SMART = Specific, Measurable, Achievable, Relevant, Time-bound).

Step-by-step example (SMART -> hypothesis):
1. Specific: customer churn rate (define precise metric: monthly churn rate).
2. Measurable: absolute change in percentage points (>= 5%).
3. Achievable: check if sample has statistical power to detect 5%.
4. Relevant: business impact (e.g.: revenue).
5. Time-bound: compare 3 months before and 3 months after the change.

Convert into measurable hypothesis with ID:
- HYP-001; statement: "The difference in churn rate between period A (3 months before) and period B (3 months after change X) is >= 0.05 (5 percentage points)."; suggested_test: proportion test/bootstrapping; author: Reporter; date: 2026-05-28; status: proposed.

Creation/update of `projects/protocolo_analise/hypotheses.csv`: the file should contain columns: id,statement,suggested_test,author,date,status — the associated CSV will be created/updated locally (do not run scripts now).

Hypothesis tests (examples):

T-test (compare means, assumes normality):

```python
# scripts: projects/protocolo_analise/code/eda_smart.py (parte: testes)
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

# t-test
stat, p = ttest_ind(group_a['metric'], group_b['metric'], nan_policy='omit')
print('t-test p:', p)

# Mann-Whitney (non-parametric)
stat, p = mannwhitneyu(group_a['metric'], group_b['metric'], alternative='two-sided')
print('Mann-Whitney p:', p)

# Chi2 for contingency tables
table = pd.crosstab(df['catA'], df['catB'])
chi2, p, dof, ex = chi2_contingency(table)
print('chi2 p:', p)
```

Comments and methodological choices:
- When to use t-test: independent samples, approximately homogeneous variances and near-normal distribution; otherwise use Mann-Whitney.
- For proportions (e.g.: churn in two windows), use difference of proportions test (z-test) or bootstrapping for confidence intervals.
- If there is temporal auto-dependence (correlated series), use hierarchical models or time-series methods for control.

SMART integration -> complete practical example:

1) SMART question: "After deploying version X on 2026-01-01, did the monthly churn rate decrease by at least 5 percentage points in the 3 subsequent months (2026-01-01 to 2026-03-31) compared to the 3 previous months (2025-10-01 to 2025-12-31), with 95% confidence?"

2) Hypothesis (ID HYP-001):
- H0: p_B - p_A <= 0.05 (difference less than or equal to 5%)
- H1: p_B - p_A > 0.05

3) Suggested test: difference of proportions test or bootstrap to calculate confidence interval of the difference.

4) Example code (bootstrap) and interpretation:

```python
# scripts: projects/protocolo_analise/code/eda_smart.py (adicionar função bootstrap)
import numpy as np

def bootstrap_diff_proportions(a, b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    n_a = len(a)
    n_b = len(b)
    for _ in range(n_boot):
        sample_a = rng.choice(a, size=n_a, replace=True)
        sample_b = rng.choice(b, size=n_b, replace=True)
        diffs.append(sample_b.mean() - sample_a.mean())
    diffs = np.array(diffs)
    return np.percentile(diffs, [2.5, 97.5]), diffs.mean()

# where a and b are binary arrays (1=churn, 0=no churn)
ci, mean_diff = bootstrap_diff_proportions(a, b)
print('mean_diff', mean_diff, '95% CI', ci)
```

Interpretation of p-value and confidence intervals:
- A small p-value (<0.05) indicates the observed difference is unlikely under H0; however, p-value does not measure effect size — therefore always report CI and effect size.
- If the confidence interval does not contain the null value (0) and is entirely above 0.05 (if that is the relevant threshold), then there is evidence that the change was at least 5%.

Recording hypotheses and results:
- Save each hypothesis in `projects/protocolo_analise/hypotheses.csv` with a unique ID and keep status updated (proposed, tested, accepted, rejected).
- Associated scripts: `projects/protocolo_analise/code/eda_smart.py` (contains tests and bootstrap functions).

---

5) Data splitting

- Strategies and examples:

Train/test/validation standard:

```python
# scripts: projects/protocolo_analise/code/split_examples.py
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
```

Context and practical objective:
- The split ensures honest evaluation and avoids optimization on test data.

Concrete examples for real cases:

1) Time series (TimeSeriesSplit or temporal cut):

```python
# Time-based split - generic example
# sort by date and split by temporal index
df = df.sort_values('date')
cut = int(len(df)*0.8)
train = df.iloc[:cut]
test = df.iloc[cut:]
```

Comment: never shuffle time series when creating train/test — this causes data leakage. For validation, use `TimeSeriesSplit` which preserves order.

2) Entities (GroupKFold):

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=df['entity_id']):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
```

Comment: use when examples of the same entity (e.g.: user, vessel) cannot be in train and test simultaneously.

3) Stratification:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X, y):
    # keeps class proportion
    pass
```

Comment: stratifying is crucial for highly imbalanced classes to ensure representation in each fold.

4) Combination (Group + Stratify):
- There is no ready-made implementation in sklearn that combines GroupKFold+Stratify directly. Practical alternatives:
  - group entities and create a stratified label per entity (e.g.: proportion of the class within the entity) and use a custom procedure; or
  - use hierarchical validation: first split by group and then stratify internally.

Example practical prevention of data leakage:
- DO NOT normalize (fit/transform) with a scaler calculated over the whole dataset. Always fit scaler on the training set and apply to validation/test.

Risks of data leakage with practical examples:
- Calculating target-encoded features using the whole dataset before splitting: explicit leakage. Solution: compute encoding using only the training fold (target encoding with smoothing and internal cross-validation).

Associated scripts: `projects/protocolo_analise/code/split_examples.py`.

---

6) Feature selection

- Plots: correlation and initial selection

```python
# scripts: projects/protocolo_analise/code/feature_selection_rfecv.py
import seaborn as sns
import matplotlib.pyplot as plt
corr = X.corr()
sns.heatmap(corr)
plt.savefig('figures/corr_heatmap.png')
```

Expansion and explanations:

- Correlation: useful to identify pairs of highly correlated features (>0.9) — it is recommended to remove one of the two to avoid multicollinearity in linear models.

- PCA (reduction):

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_p = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
```

When to use PCA:
- Compression (dimensionality reduction to speed up models, reduce noise) — PCA is great when the goal is compression and performance improvement, not interpretability.
- Interpretation: PCA does not easily preserve interpretability of original features. Use loadings (components) to understand which features contribute to each component, but do not expect direct representations.

How to interpret `explained_variance_ratio_`:
- Indicates the fraction of total variance explained by each component. E.g.: if the first two components explain 80% of variance, it may be reasonable to reduce to 2D for exploratory tasks.

- RFECV (Recursive Feature Elimination with Cross-Validation model-guided):

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50)
rfecv = RFECV(rf, step=1, cv=5, scoring='f1')
rfecv.fit(X_train, y_train)
print('n features:', rfecv.n_features_)
```

Interpretation: RFECV shows which features maintain model performance during recursive removal — inspect score curves by number of features (save to `figures/rfecv_score.png`).

- LASSO (sparse linear selection):

```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)
print(lasso.coef_)
```

Interpreting LASSO coefficients:
- Coefficients close to zero indicate non-informative features; tuning regularization parameter (alpha) controls sparsity.

Pitfalls and risks:
- Multicollinearity: can confuse coefficients and importances — prefer tree models (feature importances) or strong regularization to mitigate.
- Leakage: selection performed with test data will optimistic bias performance. Execute selection inside the pipeline (with `sklearn.pipeline` and cross-validation).

Associated scripts: `projects/protocolo_analise/code/feature_selection_rfecv.py`.

---

7) Data balancing

- Techniques and examples with imbalanced-learn:

```python
# scripts: projects/protocolo_analise/code/balanceamento.py
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)
```

Comments and trade-offs:
- SMOTE creates synthetic minority examples by interpolating — improves recall but can generate artificial points near the boundary and cause overfitting.
- Undersampling reduces majority class data — simple and fast but can lose important signal.
- Class weights: alternative when you don't want to change the training base — set `class_weight='balanced'` in models that support it.

Advanced combinations:
- SMOTEENN and other combiners (SMOTE + cleaning) can improve the quality of synthetics.

Best practices:
- Always apply balancing only on the training set.
- Evaluate with appropriate metrics (F1, recall, PR-AUC) on a separate validation set.

Associated scripts: `projects/protocolo_analise/code/balanceamento.py`.

---

8) Modeling and Hyperparameters

- Simple training and evaluation examples:

```python
# scripts: projects/protocolo_analise/code/model_examples.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'lr': LogisticRegression(max_iter=1000),
    'rf': RandomForestClassifier(n_estimators=100),
    'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'svm': SVC(probability=True),
    'knn': KNeighborsClassifier()
}

for name, m in models.items():
    m.fit(X_train, y_train)
    print(name, m.score(X_test, y_test))
```

Practical recommendations for search budget and configuration:
- For RandomizedSearchCV: 50-200 iterations are a good starting point for models with 4-6 important hyperparameters.
- For GridSearchCV: use only on small spaces or for local refinement after RandomizedSearch.
- Fold counts: 5 to 10 folds are usual; for GroupKFold or TimeSeriesSplit, reduce folds if there are few entities/observations.

Basic Optuna example (study) for RandomForest:

```python
# scripts: projects/protocolo_analise/code/model_search_grid.py (adicionar função optuna)
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=3, scoring='f1').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(study.best_params_)
```

Metric tips to prioritize according to objective:
- Rare events/detection (e.g.: detecting fishing/piracy): prioritize recall (or F2) to reduce false negatives.
- Binary detection with asymmetric costs: consider a cost matrix and optimize the metric that reflects real cost.

Associated scripts: `projects/protocolo_analise/code/model_search_grid.py`.

---

9) Stacking and Voting

- VotingClassifier and StackingClassifier (sklearn):

```python
# scripts: projects/protocolo_analise/code/stacking_example.py
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

estimators = [('rf', RandomForestClassifier(n_estimators=50)), ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))]
voting = VotingClassifier(estimators=estimators, voting='soft')
voting.fit(X_train, y_train)

stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack.fit(X_train, y_train)
```

When to use: stacking for performance gain by aggregating heterogeneous models; voting for simple and robust combinations.

Best practices:
- Validate stacking with an external validation to avoid meta-model overfitting.
- Save pipelines (pickle/MLflow) including preprocessing steps.

---

10) Semi-supervised learning

- LabelPropagation / LabelSpreading and pseudo-labeling:

```python
# scripts: projects/protocolo_analise/code/semi_supervised_pseudo_label.py
from sklearn.semi_supervised import LabelPropagation

lp = LabelPropagation()
# X_unlabeled: data without labels with labels as -1
lp.fit(X_combined, y_combined)
```

Comment:
- Pseudo-labeling can expand labeled data but must be done carefully: use only high-confidence predictions and validate on holdout.

---

11) Unsupervised learning

- Clustering and dimensionality reduction:

```python
# scripts: projects/protocolo_analise/code/unsupervised_kmeans.py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=5)
d = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_p = pca.fit_transform(X_scaled)
```

Tips:
- Choose k (clusters) with metrics like silhouette score or visual inspection.
- Use DBSCAN to detect noise and arbitrarily shaped clusters.

---

12) Model evaluation

- Metrics and plots:

```python
# scripts: projects/protocolo_analise/code/evaluation_examples.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print('acc', accuracy_score(y_test, y_pred))
print('roc_auc', roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)
```

ROC vs PR (when to prefer PR-AUC):
- ROC-AUC considers true positive rate vs false positive rate and is useful when classes are relatively balanced.
- Precision-Recall (PR-AUC) is preferable in strongly imbalanced problems and when focus is on positive class performance (e.g.: rare event detection).

Practical interpretation:
- High ROC-AUC with low PR-AUC can indicate the model ranks well overall but fails to recover the minority class with good precision.

Calibration:
- Evaluate calibration curve; use isotonic/Platt calibration when necessary.

Associated scripts: `projects/protocolo_analise/code/evaluation_examples.py`.

---

13) Explainability (SHAP)

- Installation: `pip install shap`
- Example with an sklearn model:

```python
# scripts: projects/protocolo_analise/code/shap_example.py
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
shap.dependence_plot(0, shap_values.values, X_test)
```

Guidance on interpreting SHAP plots:
- Summary plot: shows average importance and direction of effect (colors indicate feature value). Use to identify features with greatest global impact.
- Dependence plot: shows marginal effect of a feature and possible interactions; interpret points outside the trend as potential outliers or unmodeled interactions.

Explainability reporting best practices:
- Save figures in `projects/protocolo_analise/figures/` with descriptive names (e.g.: `shap_summary.png`, `shap_dependence_featureX.png`).
- Generate reports by segment (e.g.: different risk segments) to see if important features change by segment.
- Document limits: SHAP explains local/global contributions of the current model — it does not prove causality.

Associated scripts: `projects/protocolo_analise/code/shap_example.py`.

---

14) Drift: monitoring and operational actions

- Concepts: data drift (change in X distribution), concept drift (change in relationship between X and y).

Operational steps to monitor drift:
1. Define comparison windows (e.g.: baseline = last stable month; current = last 7 days).
2. Comparison metrics: PSI (Population Stability Index), KL divergence, Earth Mover's Distance, changes in statistics (mean/median), proportion variation.
3. Establish operational thresholds (e.g.: PSI > 0.2 moderate alert; PSI > 0.3 critical alert).
4. Monitor model performance (ROC-AUC, recall by segment) in sliding windows.
5. Emit alert and trigger investigation/retraining pipeline when thresholds are exceeded.

Example of alert pipeline (pseudocode):

```python
# scripts: projects/protocolo_analise/code/drift_monitoring_example.py (psuedo)
THRESHOLD_PSI = 0.2
THRESHOLD_PERF = 0.05  # acceptable drop in AUC

# function that calculates PSI between two distributions
psi = calculate_psi(reference['featureX'], current['featureX'])
auc_change = baseline_auc - current_auc

if psi > THRESHOLD_PSI or auc_change > THRESHOLD_PERF:
    event = {
        'type': 'drift_alert',
        'feature': 'featureX',
        'psi': psi,
        'auc_change': auc_change,
        'timestamp': now()
    }
    send_alert(event)
    # trigger investigation job: compute per-feature psi, check campaigns/inputs, start retrain
```

Practical notes:
- Automate metric collection and save history to enable diagnosis (when it started, which features changed).
- Use Evidently or similar tools for dashboards and reports with full comparisons.

Associated scripts: `projects/protocolo_analise/code/drift_monitoring_example.py`.

---

15) Operational checklist (before considering an analysis complete)

- Ingestion:
  - [ ] Validate schema and types after ingestion.
  - [ ] Count rows per source and reconcile.

- Cleaning:
  - [ ] Remove/justify duplicates.
  - [ ] Handle critical missing values (e.g.: target).
  - [ ] Log cleaning operations (audit trail).

- Hypotheses and EDA:
  - [ ] Document hypotheses with ID and suggested test in `hypotheses.csv`.
  - [ ] Generate exploratory plots (save in `projects/protocolo_analise/figures/`).

- Features:
  - [ ] Check multicollinearity and remove/combine features.
  - [ ] Save final feature list (e.g.: `projects/protocolo_analise/features.json` optional).

- Modeling:
  - [ ] Define main metric (e.g.: recall, PR-AUC, F1) aligned with objective.
  - [ ] Set appropriate validation (GroupKFold/TimeSeries/Stratify).
  - [ ] Run hyperparameter search with defined budget and save study (Optuna).

- Validation:
  - [ ] Evaluate on final holdout and generate performance and calibration reports.
  - [ ] Generate explainability (SHAP) and save figures.

- Production and monitoring:
  - [ ] Define drift thresholds and create alert pipeline.
  - [ ] Document retraining process and rollback criteria.

- Final documentation:
  - [ ] Update `relatorio_protocolo_analise.md` with results and decisions.
  - [ ] Update `projects/protocolo_analise/manifest.json` to include new artifacts (hypotheses, figures).

---

ARTIFACTS GENERATED

- Main document: `projects/protocolo_analise/protocolo_analise.md` (updated with didactic explanations)
- Scripts (in `projects/protocolo_analise/code/`):
  - importacao_multiplos_arquivos.py
  - eda_smart.py
  - limpeza_outliers.py
  - split_examples.py
  - feature_selection_rfecv.py
  - balanceamento.py
  - model_examples.py
  - model_search_grid.py
  - stacking_example.py
  - semi_supervised_pseudo_label.py
  - unsupervised_kmeans.py
  - shap_example.py
  - evaluation_examples.py
  - drift_monitoring_example.py
- Hypotheses: `projects/protocolo_analise/hypotheses.csv` (created/updated)

Next suggested steps: invoke the Dependency Manager to generate/update `requirements.txt` with the libraries actually used (pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap, seaborn, matplotlib, optuna, evidently, global_land_mask).
