import pandas as pd
from sklearn.mixture import GaussianMixture
import os

os.makedirs('outputs/predictions', exist_ok=True)
# Load Herath's RFM data
df_rfm = pd.read_csv('outputs/2_rfm_data_herath.csv')
X = df_rfm[['Scaled_R', 'Scaled_F', 'Scaled_M']]

print("Training Model 3: Gaussian Mixture Model (GMM)...")
gmm = GaussianMixture(n_components=4, random_state=42)
df_rfm['Cluster_GMM'] = gmm.fit_predict(X)

# Save the predictions
df_rfm[['CustomerID', 'Cluster_GMM']].to_csv('outputs/predictions/model3_gmm_predictions.csv', index=False)
print("GMM training complete. Predictions saved.")
