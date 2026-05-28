# Relatório de Metadata — Protocolo de Análise

Resumo
------
Este documento sumariza os artefatos gerados para o protocolo de análise de dados (`protocolo_analise`) e fornece instruções rápidas para uso dos scripts. O documento detalhado do protocolo está em `projeto/protocolo_analise/protocolo_analise.md` (referência principal). Este relatório serve como referência técnica curta para desenvolvedores e analistas.

Sumário (TOC)
--------------
- Resumo
- Arquivos Criados
- Como executar localmente (passos sugeridos)
- Próximos passos
- Dependências
- Observações

Arquivos Criados
----------------
Abaixo estão os arquivos presentes no protocolo, com breve descrição e path relativo.

- `projects/protocolo_analise/protocolo_analise.md`
  - Documento principal do protocolo. Contém a metodologia, exemplos de código em blocos e recomendações (documento detalhado).

- Scripts (exemplos e utilitários) — todos em `projects/protocolo_analise/code/`:
  - `importacao_multiplos_arquivos.py` — leituras em lote (CSV/Parquet/JSON) e leitura em chunks para arquivos grandes.
  - `eda_smart.py` — funções para EDA, estatísticas descritivas, testes básicos (t-test, Mann-Whitney, chi2) e visualizações.
  - `limpeza_outliers.py` — procedimentos de limpeza, remoção de duplicatas, tratamento de nulos e detecção/tratamento de outliers (IQR, z-score, winsorize).
  - `split_examples.py` — exemplos de split: stratified, group-based (por entidade), time-based e validação cruzada específica.
  - `feature_selection_rfecv.py` — exemplos de seleção de features (RFECV, SelectKBest, PCA).
  - `balanceamento.py` — uso de técnicas de balanceamento (SMOTE, RandomUnderSampler, SMOTEENN) e notas sobre class_weight.
  - `model_examples.py` — treinos básicos com diversos classificadores (LogisticRegression, RandomForest, XGBoost, SVM, KNN).
  - `model_search_grid.py` — exemplos de GridSearchCV / RandomizedSearchCV (hiperparâmetros).
  - `stacking_example.py` — exemplos de Voting e Stacking com estimadores heterogêneos.
  - `semi_supervised_pseudo_label.py` — exemplos de LabelPropagation e pseudo-labeling.
  - `unsupervised_kmeans.py` — KMeans/PCA e exemplos de clustering.
  - `shap_example.py` — criação de explicabilidade com SHAP (summary, dependence plots).
  - `evaluation_examples.py` — métricas e plots de avaliação (ROC, PR, matriz de confusão, calibração).
  - `drift_monitoring_example.py` — exemplo conceitual de monitoramento de drift (ex.: Evidently), geração de relatórios de drift.

- `projects/protocolo_analise/requirements.txt`
  - Lista de dependências sugeridas para reproduzir os exemplos e scripts.

- `projects/protocolo_analise/relatorio_protocolo_analise.md` (este arquivo)
  - Relatório de metadata conciso para referência rápida.

Como executar localmente (passos sugeridos)
-------------------------------------------
Observação: NÃO executar automaticamente — siga estes passos em um ambiente controlado.

1. Preparar ambiente Python (virtualenv/venv/conda):
   - python >= 3.8 recomendado.
   - criar e ativar ambiente: `python -m venv .venv` / `source .venv/bin/activate` (ou equivalente no Windows).

2. Instalar dependências:
   - `pip install -r projects/protocolo_analise/requirements.txt`
   - Verificar se pacotes críticos (pandas, scikit-learn, imbalanced-learn, xgboost, shap, evidently) instalaram sem erro.

3. Preparar dados:
   - Colocar datasets na pasta `data/` (ou ajustar caminhos nos scripts). Os exemplos usam caminhos relativos (ex: `data/sample.csv`, `data/large.csv`).
   - Confirmar formatos (CSV, Parquet, JSON) conforme indicado no `protocolo_analise.md`.

4. Execução de scripts (exemplos):
   - EDA rápido: `python projects/protocolo_analise/code/eda_smart.py` (verificar parâmetros esperados no topo do script).
   - Importação em lote: `python projects/protocolo_analise/code/importacao_multiplos_arquivos.py`.
   - Limpeza/outliers: `python projects/protocolo_analise/code/limpeza_outliers.py`.
   - Split e seleção de features: executar `split_examples.py` e `feature_selection_rfecv.py` conforme sequência.
   - Treino/avaliação: `python projects/protocolo_analise/code/model_examples.py` e `model_search_grid.py`.
   - Explicabilidade: `python projects/protocolo_analise/code/shap_example.py` após treinar/exportar um modelo compatível.
   - Drift monitoring (gerar relatório): `python projects/protocolo_analise/code/drift_monitoring_example.py` (requer Evidently e dados de referência e corrente).

5. Armazenar outputs:
   - Salvar figuras em `projects/protocolo_analise/figures/` (diretório sugerido).
   - Salvar relatórios HTML (Evidently) em `projects/protocolo_analise/reports/`.

Próximos passos
---------------
- Validar os scripts em um ambiente controlado com um conjunto de dados mínimo (sanity check).
- Atualizar/normalizar `requirements.txt` com versões específicas após validação (acionar o Dependency Manager para travar versões).
- Executar suíte de EDA e registrar hipóteses (cada hipótese com ID) em `projects/protocolo_analise/hypotheses.csv` ou no relatório consolidado.
- Gerar figuras e relatórios (colocar em `figures/` e `reports/`) e referenciá-los no `relatorio_protocolo_analise.md` futuro, caso necessário.

Dependências
-------------
- Veja `projects/protocolo_analise/requirements.txt` para a lista atual. Recomenda-se acionar o Dependency Manager para fixar versões após testes locais.

Observações
-----------
- O documento detalhado `projects/protocolo_analise/protocolo_analise.md` contém exemplos e explicações estendidas; use-o como referência principal.
- Este relatório é metadata conciso (1–2 páginas) e destina-se a orientar o uso rápido dos scripts e o entendimento do manifest.
- O `manifest.json` foi atualizado para incluir este relatório e lista os arquivos atualmente presentes no protocolo.



