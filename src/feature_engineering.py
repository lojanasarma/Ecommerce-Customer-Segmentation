import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Load Nayanajith's cleaned dataset
df = pd.read_csv('../outputs/1_cleaned_data_nayanajith.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Calculate Recency, Frequency, and Monetary (RFM)
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days, # Recency
    'Invoice': 'nunique',                                     # Frequency
    'TotalCost': 'sum'                                        # Monetary
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Apply Log Transformation & Standard Scaling
rfm['Log_R'] = np.log1p(rfm['Recency'])
rfm['Log_F'] = np.log1p(rfm['Frequency'])
rfm['Log_M'] = np.log1p(rfm['Monetary'])
scaler = StandardScaler()
rfm[['Scaled_R', 'Scaled_F', 'Scaled_M']] = scaler.fit_transform(rfm[['Log_R', 'Log_F', 'Log_M']])

# Save the engineered data
os.makedirs('../outputs', exist_ok=True)
rfm.to_csv('../outputs/2_rfm_data_herath.csv', index=False)