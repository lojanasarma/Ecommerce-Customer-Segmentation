import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import os

os.makedirs('outputs/predictions', exist_ok=True)
# Load Herath's RFM data
df_rfm = pd.read_csv('outputs/2_rfm_data_herath.csv')
X = df_rfm[['Scaled_R', 'Scaled_F', 'Scaled_M']]

print("Training Model 2: MiniBatch K-Means...")
minibatch_kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=1024)
df_rfm['Cluster_MiniBatch'] = minibatch_kmeans.fit_predict(X)

# Save the predictions
df_rfm[['CustomerID', 'Cluster_MiniBatch']].to_csv('outputs/predictions/model2_minibatch_predictions.csv', index=False)
print("MiniBatch K-Means training complete. Predictions saved.")
