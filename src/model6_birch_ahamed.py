import pandas as pd
from sklearn.cluster import Birch
import os

os.makedirs('outputs/predictions', exist_ok=True)
# Load Herath's RFM data
df_rfm = pd.read_csv('outputs/2_rfm_data_herath.csv')
X = df_rfm[['Scaled_R', 'Scaled_F', 'Scaled_M']]

print("Training Model 6: Birch Clustering...")
birch = Birch(n_clusters=4, threshold=0.5)
df_rfm['Cluster_Birch'] = birch.fit_predict(X)

# Save the predictions
df_rfm[['CustomerID', 'Cluster_Birch']].to_csv('outputs/predictions/model6_birch_predictions.csv', index=False)
print("Birch training complete. Predictions saved.")
