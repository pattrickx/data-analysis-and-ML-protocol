# Protocolo de Análise de Dados — PT-BR

Este documento descreve um protocolo passo a passo para análise de dados em Python (pandas, numpy, scikit-learn, imbalanced-learn, shap, matplotlib/seaborn). Todos os exemplos de código estão inseridos em blocos Markdown e os scripts equivalentes foram salvos em `projects/protocolo_analise/code/`.

---

SUMÁRIO (seções principais)

- Importação de múltiplos arquivos
- Identificação de características gerais dos dados (estatística descritiva)
- Tratamentos básicos e avançados (outliers)
- Protocolo de análise exploratória (Metodologia SMART e testes)
- Split dos dados
- Seleção de features
- Balanceamento de dados
- Modelos clássicos e redes neurais
- Busca de hiperparâmetros
- Stacking e Voting
- Aprendizado semi-supervisionado
- Aprendizado não supervisionado
- Avaliação de modelos
- Explicabilidade (SHAP)
- Monitoramento de drift em produção

---

Nota: NÃO executar o código aqui — os scripts correspondentes foram salvos em `projects/protocolo_analise/code/`.

1) Importação de múltiplos arquivos

- Objetivo: ler múltiplos CSV/Parquet/JSON e/ou arquivos grandes em chunks.

Exemplo (leitura com glob e concatenação):

```python
# scripts: projects/protocolo_analise/code/importacao_multiplos_arquivos.py
import pandas as pd
from glob import glob

files = glob('data/*.csv')  # padrão: ajustar para seu diretório
dfs = []
for f in files:
    dfs.append(pd.read_csv(f))

df = pd.concat(dfs, ignore_index=True)
print(df.shape)
```

Comentário: o bloco acima é o padrão mais simples — útil quando cada arquivo representa o mesmo esquema (mesmas colunas) e o volume cabe em memória. Mantenha os arquivos ordenados quando a ordem temporal importa (ex.: logs por dia).

Exemplo (leitura em chunks para arquivos grandes):

```python
# scripts: projects/protocolo_analise/code/importacao_multiplos_arquivos.py (continuação)
import pandas as pd
chunks = pd.read_csv('data/large.csv', chunksize=10_000)
for i, chunk in enumerate(chunks):
    # processar chunk incrementalmente
    print(i, chunk.shape)

# salvar após processamento incremental
```

Comentário: ler em chunks permite processamento streaming (transformações e escrita incremental). Recomendo: 1) aplicar tipos ao ler (dtypes) para reduzir memória; 2) usar parquet quando possível, pois preserva tipos e é mais rápido; 3) para ETL em produção, criar um pipeline que valide esquema e registre linhas rejeitadas.

Exemplo (leitura de múltiplos formatos):

```python
# leitura parquet, json
df_parquet = pd.read_parquet('data/file.parquet')
df_json = pd.read_json('data/file.json', lines=True)
```

Quando usar: glob/concat para arquivos pequenos/médios; chunks para datasets muito grandes que não cabem em RAM; parquet para performance e precisão de tipos.

Boas práticas operacionais:
- Validar esquema (colunas esperadas e tipos) imediatamente após leitura.
- Registrar contagens por fonte (para reconciliar com origens).
- Evitar leitura duplicada: manter índice de processamento por arquivo (processed flag).
- Scripts associados: `projects/protocolo_analise/code/importacao_multiplos_arquivos.py` (executar para ingestão inicial).

---

2) Identificação de características gerais dos dados (estatística descritiva)

- Objetivo: obter visão rápida da forma, tipos, nulos e estatísticas.

Exemplo:

```python
# scripts: projects/protocolo_analise/code/eda_smart.py (parte: descriptiva)
import pandas as pd

df = pd.read_csv('data/sample.csv')
print(df.info())
print(df.describe(include='all'))
print('nulos por coluna:\n', df.isna().sum())
print('tipos:\n', df.dtypes)
```

Contexto e objetivo prático:
- Primeiro checkpoint de qualidade: entender quantas linhas/colunas, tipos (numérico/ categórico/ datatime), e colunas com muitos valores ausentes.
- Detectar tipos incorretos (ex.: IDs lidos como floats) e conversões necessárias.

Passos recomendados:
1. Rodar `df.info()` e `df.isna().sum()`.
2. Gerar histogramas e boxplots das variáveis numéricas.
3. Contagem de categorias para colunas categóricas (top-N).
4. Identificar colunas de data e validar timezone/formato.

Interpretação dos outputs:
- Valores muito discrepantes no count de `describe()` indicam nulos ou leitura inconsistente.
- Tipos como `object` onde se espera `datetime` indicam necessidade de conversão.

Riscos/comportamentos a observar:
- Colunas com alta cardinalidade tratadas como categóricas podem explodir memória em one-hot encoding.
- IDs como strings numéricas podem ser erroneamente transformados e perder informações.

Links para scripts e execução:
- `projects/protocolo_analise/code/eda_smart.py` — executar para resumo descritivo.

Exemplo conceitual (comentado):

```python
# Exemplo extra de análise rápida de categorias
# Contar top-10 de uma coluna categórica para avaliar distribuição
print(df['categoria'].value_counts().head(10))
```

---

3) Tratamentos básicos e avançados (remoção de nulos, duplicatas, conversão de tipos, outliers)

- Básicos:

```python
# scripts: projects/protocolo_analise/code/limpeza_outliers.py
# remoção de duplicatas
df = df.drop_duplicates()
# remoção de linhas com missing críticos
df = df.dropna(subset=['target'])
# conversão de tipos
df['date'] = pd.to_datetime(df['date'])
```

Contexto e objetivo prático:
- Remover duplicatas evita viés de contagem; porém, em eventos onde duplicatas representam eventos legítimos (ex.: logs reprocessados), é preciso entender a causa antes de remover.

Passos recomendados:
1. Identificar chave(s) de deduplicação e validar com amostras.
2. Para missing em `target`, avaliar se imputação é aceitável; muitas vezes o correto é remover.
3. Registrar quantas linhas foram removidas por cada operação.

Interpretação dos outputs:
- Comparar shape antes/depois e manter logs de mudanças (audit trail).

Riscos/comportamentos a observar:
- Remover demasiadas linhas pode enviesar a amostra; justificar e documentar.

- Outliers: IQR

```python
# IQR
Q1 = df['val'].quantile(0.25)
Q3 = df['val'].quantile(0.75)
IQR = Q3 - Q1
mask = (df['val'] >= Q1 - 1.5 * IQR) & (df['val'] <= Q3 + 1.5 * IQR)
df_iqr = df.loc[mask]
```

Comentário: o método IQR é robusto para assimetrias moderadas e é recomendado como primeira abordagem para detecção de outliers. No entanto, depende do contexto — por exemplo, dados de velocidade/posicionamento podem ter extremos válidos.

- Z-score

```python
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df['val'].dropna()))
df_z = df.loc[z < 3]
```

Comentário: Z-score assume distribuição aproximadamente normal; evite em distribuições fortemente assimétricas.

- Winsorizing (limitar extremos):

```python
from scipy.stats.mstats import winsorize
arr = winsorize(df['val'], limits=[0.01, 0.01])
```

Comentário: Winsorize preserva o número de observações alterando apenas os valores extremos — útil quando não se deseja descartar dados.

- Imputação por modelos (exemplo simples com RandomForestImputer estilo):

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# exemplo conceitual: prever coluna 'x' a partir de outras
train = df[df['x'].notna()]
pred = df[df['x'].isna()]
model = RandomForestRegressor(n_estimators=50)
X = train.drop(columns=['x']).select_dtypes(include=[float, int]).fillna(0)
y = train['x']
model.fit(X, y)
X_pred = pred.drop(columns=['x']).select_dtypes(include=[float, int]).fillna(0)
df.loc[df['x'].isna(), 'x'] = model.predict(X_pred)
```

Comentário: imputação por modelo é poderosa, mas pode introduzir vazamento se informações derivadas do target forem usadas indevidamente. Sempre separar um conjunto para validar a performance da imputação.

- Validação espacial com global_land_mask (exemplo de uso):

```python
# Instalar: global_land_mask
# Uso conceitual:
from global_land_mask import globe
# validar coordenadas lat/lon
is_land = globe.is_land(df['lat'].values, df['lon'].values)
df['is_land'] = is_land
```

Quando usar cada técnica: IQR/Z para detecção robusta; winsorize para limitar sem perder casos; imputação por modelo quando missing não for MCAR e houver relações previsíveis.

Boas práticas:
- Registrar operações de limpeza e salvar versões intermediárias (ex.: df_v1.csv).
- Automatizar validações (ex.: checar se média/mediana mudaram drasticamente após limpeza).
- Scripts associados: `projects/protocolo_analise/code/limpeza_outliers.py`.

---

4) Protocolo de Análise Exploratória (Metodologia SMART + testes)

- Metodologia SMART aplicada: Exemplo de pergunta apresentada no protocolo original:

"Apenas 3 meses após uma alteração X, a taxa de evasão de clientes (target) diminui em pelo menos 5% comparado ao período anterior?"

Contexto e objetivo prático:
- Converter perguntas vagas em objetivos mensuráveis (SMART = Specific, Measurable, Achievable, Relevant, Time-bound).

Exemplo passo-a-passo (SMART -> hipótese):
1. Specific: taxa de evasão de clientes (definir métrica precisa: churn rate mensal).
2. Measurable: mudança absoluta em pontos percentuais (>= 5%).
3. Achievable: verificar se amostra tem poder estatístico para detectar 5%.
4. Relevant: impacto no negócio (ex.: receita).
5. Time-bound: comparar 3 meses pré e 3 meses pós alteração.

Converter em hipótese mensurável com ID:
- HYP-001; enunciado: "A diferença na taxa de evasão entre período A (3 meses antes) e período B (3 meses depois da alteração X) é >= 0.05 (5 pontos percentuais)."; teste_sugerido: teste de proporções/bootstrapping; autor: Reporter; data: 2026-05-28; estado: proposed.

Criação/atualização de `projects/protocolo_analise/hypotheses.csv`: o arquivo deve conter colunas: id,enunciado,teste_sugerido,autor,data,estado — o CSV associado será criado/atualizado localmente (não executar scripts agora).

Testes de hipótese (exemplos):

T-test (comparar médias, pressupõe normalidade):

```python
# scripts: projects/protocolo_analise/code/eda_smart.py (parte: testes)
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

# t-test
stat, p = ttest_ind(group_a['metric'], group_b['metric'], nan_policy='omit')
print('t-test p:', p)

# Mann-Whitney (não-paramétrico)
stat, p = mannwhitneyu(group_a['metric'], group_b['metric'], alternative='two-sided')
print('Mann-Whitney p:', p)

# Chi2 para tabelas de contingência
table = pd.crosstab(df['catA'], df['catB'])
chi2, p, dof, ex = chi2_contingency(table)
print('chi2 p:', p)
```

Comentários e escolhas metodológicas:
- Quando usar t-test: amostras independentes, variâncias aproximadamente homogêneas e distribuição próxima à normal; caso contrário, usar Mann-Whitney.
- Para proporções (ex.: churn em duas janelas), usar teste de diferença de proporções (z-test) ou bootstrapping para intervalos de confiança.
- Se houver auto-dependência temporal (séries correlacionadas), usar modelos hierárquicos ou séries temporais para controle.

Integração SMART -> exemplo prático completo:

1) Pergunta SMART: "Após o deploy da versão X em 2026-01-01, a taxa de churn mensal diminuiu pelo menos 5 pontos percentuais nos 3 meses subsequentes (2026-01-01 a 2026-03-31) comparada aos 3 meses anteriores (2025-10-01 a 2025-12-31), com 95% de confiança?"

2) Hipótese (ID HYP-001):
- H0: p_B - p_A <= 0.05 (diferença menor ou igual a 5%)
- H1: p_B - p_A > 0.05

3) Teste sugerido: teste de diferença de proporções ou bootstrap para calcular intervalo de confiança da diferença.

4) Exemplo de código (bootstrap) e interpretação:

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

# onde a e b são arrays binárias (1=churn, 0=no churn)
ci, mean_diff = bootstrap_diff_proportions(a, b)
print('mean_diff', mean_diff, '95% CI', ci)
```

Interpretação do p-value e intervalos de confiança:
- p-value pequeno (<0.05) indica que a diferença observada é improvável sob H0; porém, p-value não mede o tamanho do efeito — por isso sempre reporte CI e effect size.
- Se o intervalo de confiança não contém o valor nulo (0) e está totalmente acima de 0.05 (se esse é o limiar relevante), então há evidência de que a mudança foi pelo menos de 5%.

Registro de hipóteses e resultados:
- Salvar cada hipótese em `projects/protocolo_analise/hypotheses.csv` com ID único e manter status atualizado (proposed, tested, accepted, rejected).
- Scripts associados: `projects/protocolo_analise/code/eda_smart.py` (contém testes e funções de bootstrap).

---

5) Split dos dados

- Estratégias e exemplos:

Train/test/validation padrão:

```python
# scripts: projects/protocolo_analise/code/split_examples.py
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
```

Contexto e objetivo prático:
- O split garante avaliação honesta e evita otimização sobre dados de teste.

Exemplos concretos para casos reais:

1) Séries temporais (TimeSeriesSplit ou corte temporal):

```python
# Time-based split - exemplo genérico
# ordenar por data e dividir por índice temporal
df = df.sort_values('date')
cut = int(len(df)*0.8)
train = df.iloc[:cut]
test = df.iloc[cut:]
```

Comentário: nunca embaralhar séries temporais ao criar treino/teste — isso causa data leakage. Para validação, usar `TimeSeriesSplit` que preserva ordem.

2) Entidades (GroupKFold):

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=df['entity_id']):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
```

Comentário: use quando exemplos da mesma entidade (ex.: usuário, embarcação) não podem estar em treino e teste simultaneamente.

3) Estratificação:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X, y):
    # mantem proporção de classes
    pass
```

Comentário: estratificar é crucial para classes muito desbalanceadas para garantir representação em cada fold.

4) Combinação (Group + Stratify):
- Não há implementação pronta em sklearn que combine GroupKFold+Stratify diretamente. Alternativas práticas:
  - agrupar entidades e criar um rótulo estratificado por entidade (ex.: proporção da classe dentro da entidade) e usar um procedimento customizado; ou
n  - usar uma validação hierárquica: primeiro dividir por grupo e depois estratificar internamente.

Exemplo prático de prevenção de data leakage:
- NÃO normalizar (fit/transform) com scaler calculado sobre todo o dataset. Sempre ajustar scaler no conjunto de treino e aplicar em validação/teste.

Riscos de data leakage com exemplos práticos:
- Calcular target-encoded features usando toda a base antes do split: vazamento explícito. Solução: calcular encoding apenas com fold de treino (target encoding com smoothing e validação cruzada interna).

Scripts associados: `projects/protocolo_analise/code/split_examples.py`.

---

6) Seleção de features

- Plots: correlação e seleção inicial

```python
# scripts: projects/protocolo_analise/code/feature_selection_rfecv.py
import seaborn as sns
import matplotlib.pyplot as plt
corr = X.corr()
sns.heatmap(corr)
plt.savefig('figures/corr_heatmap.png')
```

Expansão e explicações:

- Correlação: útil para identificar pares de features altamente correlacionadas (>0.9) — recomenda-se remover uma das duas para evitar multicolinearidade em modelos lineares.

- PCA (redução):

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_p = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
```

Quando usar PCA:
- Compressão (redução de dimensionalidade para acelerar modelos, reduzir ruído) — PCA é ótimo quando o objetivo é compressão e melhoria de performance, não interpretabilidade.
- Interpretação: PCA não preserva facilmente a interpretabilidade das features originais. Use loadings (componentes) para compreender quais features contribuem para cada componente, mas não espere representações diretas.

Como interpretar `explained_variance_ratio_`:
- Indica a fração da variância total explicada por cada componente. Ex.: se os dois primeiros componentes explicam 80% da variância, pode ser razoável reduzir para 2D para tarefas exploratórias.

- RFECV (Recusão com validação cruzada orientada a modelo):

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50)
rfecv = RFECV(rf, step=1, cv=5, scoring='f1')
rfecv.fit(X_train, y_train)
print('n features:', rfecv.n_features_)
```

Interpretação: RFECV mostra quais features mantêm performance do modelo durante remoção recursiva — observe curvas de score por número de features (salvar em `figures/rfecv_score.png`).

- LASSO (seleção linear esparsa):

```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)
print(lasso.coef_)
```

Interpretação de coeficientes LASSO:
- Coeficientes próximos de zero indicam features não informativas; ajuste do parâmetro de regularização (alpha) controla esparsidade.

Pitfalls e riscos:
- Multicolinearidade: pode confundir coeficientes e importâncias — prefira modelos de árvore (feature importances) ou regularização forte para mitigar.
- Vazamento: seleção feita com dados de teste causará otimismo na performance. Execute seleção dentro do pipeline (com `sklearn.pipeline` e cross-validation).

Scripts associados: `projects/protocolo_analise/code/feature_selection_rfecv.py`.

---

7) Balanceamento de dados

- Técnicas e exemplos com imbalanced-learn:

```python
# scripts: projects/protocolo_analise/code/balanceamento.py
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)
```

Comentários e trade-offs:
- SMOTE cria exemplos sintéticos interpolando minoritárias — melhora recall, mas pode gerar pontos artificiais na fronteira e causar overfitting.
- Undersampling reduz dados majoritários — simples e rápido, mas pode perder sinal importante.
- Class weights: alternativa quando não se quer alterar a base de treino — ajustar `class_weight='balanced'` em modelos que suportam.

Combinações avançadas:
- SMOTEENN e outros combiners (SMOTE + cleaning) podem melhorar qualidade dos sintéticos.

Boas práticas:
- Sempre aplicar balanceamento apenas no conjunto de treino.
- Avaliar com métricas apropriadas (F1, recall, PR-AUC) em conjunto de validação separada.

Scripts associados: `projects/protocolo_analise/code/balanceamento.py`.

---

8) Modelagem e Hiperparâmetros

- Exemplos simples de treino e avaliação:

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

Recomendações práticas de budget e configuração de busca:
- Para RandomizedSearchCV: 50-200 iterações são um bom ponto de partida para modelos com 4-6 hiperparâmetros importantes.
- Para GridSearchCV: usar somente em espaços pequenos ou em refinamento local depois do RandomizedSearch.
- Fold counts: 5 a 10 folds são usuais; para GroupKFold ou TimeSeriesSplit, reduzir folds se houver poucas entidades/observações.

Exemplo básico de Optuna (estudo) para RandomForest:

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

Dicas de métricas a priorizar conforme objetivo:
- Eventos raros/recuperação (ex.: detecção de pesca/pirateamento): priorizar recall (ou F2) para reduzir falsos negativos.
- Detecção binária com custos diferenciados: considerar matriz de custo e otimizar a métrica que reflita custo real.

Scripts associados: `projects/protocolo_analise/code/model_search_grid.py`.

---

9) Stacking e Voting

- VotingClassifier e StackingClassifier (sklearn):

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

Quando usar: stacking para ganho de performance agregando modelos heterogêneos; voting para combinações simples e robustas.

Boas práticas:
- Validar stacking com uma validação externa para evitar overfitting do meta-modelo.
- Salvar pipelines (pickle/MLflow) incluindo etapas de pré-processamento.

---

10) Aprendizado semi-supervisionado

- LabelPropagation / LabelSpreading e pseudo-labeling:

```python
# scripts: projects/protocolo_analise/code/semi_supervised_pseudo_label.py
from sklearn.semi_supervised import LabelPropagation

lp = LabelPropagation()
# X_unlabeled: dados sem rótulo com rótulos -1
lp.fit(X_combined, y_combined)
```

Comentário:
- Pseudo-labeling pode ampliar dados rotulados, mas deve ser feito com cuidado: use apenas previsões de alta confiança e valide em holdout.

---

11) Aprendizado não supervisionado

- Clustering e redução de dimensionalidade:

```python
# scripts: projects/protocolo_analise/code/unsupervised_kmeans.py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=5)
d = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_p = pca.fit_transform(X_scaled)
```

Dicas:
- Escolha k (clusters) com métricas como silhouette score ou inspeção visual.
- Use DBSCAN para detectar ruído e clusters de forma arbitrária.

---

12) Avaliação de modelos

- Métricas e plots:

```python
# scripts: projects/protocolo_analise/code/evaluation_examples.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print('acc', accuracy_score(y_test, y_pred))
print('roc_auc', roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)
```

ROC vs PR (quando preferir PR-AUC):
- ROC-AUC considera taxa de verdadeiros positivos vs falsos positivos e é útil quando classes estão relativamente balanceadas.
- Precision-Recall (PR-AUC) é preferível em problemas fortemente desbalanceados e quando se quer foco no desempenho da classe positiva (ex.: detecção de evento raro).

Interpretação prática:
- Uma alta ROC-AUC com baixa PR-AUC pode indicar que o modelo classifica bem em termos gerais, mas falha em recuperar a minoria com boa precisão.

Calibração:
- Avaliar curva de calibração; usar isotonic/regression calibration quando necessário.

Scripts associados: `projects/protocolo_analise/code/evaluation_examples.py`.

---

13) Explicabilidade (SHAP)

- Instalação: `pip install shap`
- Exemplo com modelo sklearn:

```python
# scripts: projects/protocolo_analise/code/shap_example.py
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
shap.dependence_plot(0, shap_values.values, X_test)
```

Orientações sobre interpretação de plots SHAP:
- Summary plot: mostra a importância média e direção do efeito (cores indicam valor da feature). Use para identificar features com maior impacto global.
- Dependence plot: mostra efeito marginal de uma feature e possíveis interações; interpretar pontos fora da tendência como possíveis outliers ou interações não-modeladas.

Boas práticas para relatórios de explicabilidade:
- Salvar figuras em `projects/protocolo_analise/figures/` com nomes descritivos (ex.: `shap_summary.png`, `shap_dependence_featureX.png`).
- Gerar relatórios por segmento (ex.: segmentos de risco distinto) para ver se features importantes mudam por segmento.
- Documentar limites: SHAP explica contribuição local/global do modelo atual — não prova causalidade.

Scripts associados: `projects/protocolo_analise/code/shap_example.py`.

---

14) Drift: monitoramento e ações operacionais

- Conceitos: data drift (mudança na distribuição de X), concept drift (mudança na relação entre X e y).

Passos operacionais para monitorar drift:
1. Definir janelas de comparação (ex.: baseline = último mês estável; current = últimos 7 dias).
2. Métricas de comparação: PSI (Population Stability Index), KL divergence, Earth Mover's Distance, mudanças em estatísticas (mean/median), variação de proporções.
3. Estabelecer thresholds operacionais (ex.: PSI > 0.2 alerta moderado; PSI > 0.3 alerta crítico).
4. Monitorar desempenho do modelo (ROC-AUC, recall por segmento) em janelas deslizantes.
5. Emitir alerta e acionar pipeline de investigação/retraining quando thresholds forem excedidos.

Exemplo de pipeline de alarme (pseudocódigo):

```python
# scripts: projects/protocolo_analise/code/drift_monitoring_example.py (psuedo)
THRESHOLD_PSI = 0.2
THRESHOLD_PERF = 0.05  # queda aceitável em AUC

# função que calcula PSI entre duas distribuições
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
    # acionar job de investigação: computar per-feature psi, checar campanhas/inputs, iniciar retrain
```

Observações práticas:
- Automatizar coleta de métricas e salvar histórico para permitir diagnóstico (quando começou, quais features estão mudando).
- Usar Evidently ou ferramentas similares para dashboards e relatórios com comparações completas.

Scripts associados: `projects/protocolo_analise/code/drift_monitoring_example.py`.

---

15) Check-list operacional (antes de considerar uma análise completa)

- Ingestão:
  - [ ] Validar esquema e tipos após ingestão.
  - [ ] Contar linhas por fonte e reconciliar.

- Limpeza:
  - [ ] Remover/justificar duplicatas.
  - [ ] Tratar missing críticos (ex.: target).
  - [ ] Registrar operações de limpeza (audit trail).

- Hipóteses e EDA:
  - [ ] Documentar hipóteses com ID e teste sugerido em `hypotheses.csv`.
  - [ ] Gerar gráficos exploratórios (salvar em `projects/protocolo_analise/figures/`).

- Features:
  - [ ] Checar multicolinearidade e remover/combinar features.
  - [ ] Salvar lista final de features (ex.: `projects/protocolo_analise/features.json` opcional).

- Modelagem:
  - [ ] Definir métrica principal (ex.: recall, PR-AUC, F1) alinhada ao objetivo.
  - [ ] Configurar validação adequada (GroupKFold/TimeSeries/Stratify).
  - [ ] Rodar busca de hiperparâmetros com budget definido e salvar estudo (Optuna).

- Validação:
  - [ ] Avaliar em holdout final e gerar relatórios de performance e calibragem.
  - [ ] Gerar explicabilidade (SHAP) e salvar figuras.

- Produção e monitoramento:
  - [ ] Definir thresholds de drift e criar pipeline de alerta.
  - [ ] Documentar processo de retraining e critérios de rollback.

- Documentação final:
  - [ ] Atualizar `relatorio_protocolo_analise.md` com resultados e decisões.
  - [ ] Atualizar `projects/protocolo_analise/manifest.json` para incluir novos artefatos (hipóteses, figuras).

---

ARTEFATOS GERADOS

- Documento principal: `projects/protocolo_analise/protocolo_analise.md` (atualizado com explicações didáticas)
- Scripts (em `projects/protocolo_analise/code/`):
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
- Hipóteses: `projects/protocolo_analise/hypotheses.csv` (criado/atualizado)

Próximos passos sugeridos: acionar o Dependency Manager para gerar/atualizar `requirements.txt` com as bibliotecas efetivamente usadas (pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap, seaborn, matplotlib, optuna, evidently, global_land_mask).
