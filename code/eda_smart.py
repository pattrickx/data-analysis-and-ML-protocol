import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind

# Leitura exemplo
# df = pd.read_csv('data/sample.csv')

# Resumo básico
# print(df.info())
# print(df.describe(include='all'))
# print(df.isna().sum())

# Visualizações de exemplo
# sns.boxplot(x='group', y='metric', data=df)
# plt.savefig('figures/boxplot_group.png')

# Testes de hipótese (exemplo conceitual)
# stat, p = ttest_ind(group_a['metric'], group_b['metric'], nan_policy='omit')
# stat, p = mannwhitneyu(group_a['metric'], group_b['metric'])
# table = pd.crosstab(df['catA'], df['catB'])
# chi2, p, dof, ex = chi2_contingency(table)

# Organização de hipóteses: salvar em CSV com ID
# ex: HYP-001, descrição, teste, p-value, interpretação
