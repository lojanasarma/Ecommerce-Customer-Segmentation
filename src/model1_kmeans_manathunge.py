import pandas as pd
from sklearn.cluster import KMeans
import os

os.makedirs('outputs/predictions', exist_ok=True)
# Load Herath's RFM data
df_rfm = pd.read_csv('outputs/2_rfm_data_herath.csv')
X = df_rfm[['Scaled_R', 'Scaled_F', 'Scaled_M']]

print("Training Model 1: K-Means...")
kmeans = KMeans(n_clusters=4, random_state=42)
df_rfm['Cluster_KMeans'] = kmeans.fit_predict(X)

# Save the predictions
df_rfm[['CustomerID', 'Cluster_KMeans']].to_csv('outputs/predictions/model1_kmeans_predictions.csv', index=False)
print("K-Means training complete. Predictions saved.")
