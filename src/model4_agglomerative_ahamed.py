import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import os

os.makedirs('outputs/predictions', exist_ok=True)
# Load Herath's RFM data
df_rfm = pd.read_csv('outputs/2_rfm_data_herath.csv')
X = df_rfm[['Scaled_R', 'Scaled_F', 'Scaled_M']]

print("Training Model 4: Agglomerative Hierarchical Clustering...")
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
df_rfm['Cluster_Hierarchical'] = hierarchical.fit_predict(X)

# Save the predictions
df_rfm[['CustomerID', 'Cluster_Hierarchical']].to_csv('outputs/predictions/model4_agglomerative_predictions.csv', index=False)
print("Agglomerative training complete. Predictions saved.")
