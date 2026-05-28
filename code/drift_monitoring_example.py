# Exemplo conceitual usando evidently
# pip install evidently
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# ref_df = pd.read_csv('data/reference.csv')
# current_df = pd.read_csv('data/current.csv')

# report = Report(metrics=[DataDriftPreset()])
# report.run(reference_data=ref_df, current_data=current_df)
# report.save_html('reports/drift_report.html')

# Estratégia: calcular PSI, KL, acompanhar métricas de modelo (ROC-AUC, precision) por período
