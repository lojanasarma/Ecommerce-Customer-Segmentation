import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os

os.makedirs('outputs/figures', exist_ok=True)

print("Loading data...")
df_rfm = pd.read_csv('outputs/2_rfm_data_herath.csv')
# Standardize customer id column name
if 'Customer ID' in df_rfm.columns:
    df_rfm.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)

print("Loading model predictions...")
pred_files = {
    'K-Means': 'outputs/predictions/model1_kmeans_predictions.csv',
    'MiniBatch K-Means': 'outputs/predictions/model2_minibatch_predictions.csv',
    'Gaussian Mixture': 'outputs/predictions/model3_gmm_predictions.csv',
    'Agglomerative': 'outputs/predictions/model4_agglomerative_predictions.csv',
    'DBSCAN': 'outputs/predictions/model5_dbscan_predictions.csv',
    'Birch': 'outputs/predictions/model6_birch_predictions.csv'
}

results = []

print("\\nCalculating Metrics for all 6 Models...")
for name, file_path in pred_files.items():
    if not os.path.exists(file_path):
        print(f"[{name}] - Could not find {file_path}")
        continue
        
    df_pred = pd.read_csv(file_path)
    if 'Customer ID' in df_pred.columns:
        df_pred.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
        
    # Merge carefully to align features and labels
    merged = pd.merge(df_rfm, df_pred, on='CustomerID', how='inner')
    
    # Herath might have used Scaled_R or Recency_Scaled. Let's find out.
    if 'Scaled_R' in merged.columns:
        X = merged[['Scaled_R', 'Scaled_F', 'Scaled_M']]
    elif 'Recency_Scaled' in merged.columns:
        X = merged[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
    else:
        print(f"[{name}] - Could not find scaled feature columns!")
        continue
        
    # The cluster label is the last column
    labels = merged.iloc[:, -1].values
    
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        results.append({'Model': name, 'Silhouette': sil, 'Davies_Bouldin': db, 'Calinski_Harabasz': ch})
        print(f"[{name}] - Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f}")
    else:
        print(f"[{name}] - Failed to find multiple clusters.")
        results.append({'Model': name, 'Silhouette': 0, 'Davies_Bouldin': 0, 'Calinski_Harabasz': 0})

df_results = pd.DataFrame(results)
df_results.to_csv('outputs/predictions/model_evaluation_metrics.csv', index=False)

# Plotting the winning Silhouette Scores
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = sns.barplot(x='Silhouette', y='Model', data=df_results.sort_values('Silhouette', ascending=False), palette='viridis')

plt.title('Silhouette Score Comparison (Higher is Better)', fontsize=16, fontweight='bold')
plt.xlabel('Silhouette Score', fontsize=12)
plt.ylabel('Clustering Model', fontsize=12)

# Highlight K-Means if it wins
for i, bar in enumerate(bars.patches):
    if df_results.sort_values('Silhouette', ascending=False).iloc[i]['Model'] == 'K-Means':
        bar.set_color('crimson')

plt.tight_layout()
plt.savefig('outputs/figures/model_comparison_silhouette.png', dpi=300)
print("\\nEvaluation complete! Saved metrics to CSV and generated the comparison chart.")
