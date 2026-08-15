import pandas as pd
from sklearn.cluster import DBSCAN
import os

os.makedirs('outputs/predictions', exist_ok=True)
# Load Herath's RFM data
df_rfm = pd.read_csv('outputs/2_rfm_data_herath.csv')
X = df_rfm[['Scaled_R', 'Scaled_F', 'Scaled_M']]

print("Training Model 5: DBSCAN Clustering...")
dbscan = DBSCAN(eps=0.5, min_samples=10)
df_rfm['Cluster_DBSCAN'] = dbscan.fit_predict(X)

# Save the predictions
df_rfm[['CustomerID', 'Cluster_DBSCAN']].to_csv('outputs/predictions/model5_dbscan_predictions.csv', index=False)
print("DBSCAN training complete. Predictions saved.")
