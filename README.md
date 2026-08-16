<div align="center">
  <img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/d2ac6b64-4941-49a9-a4bf-90e45542f6e4" />

# 🛒 E-Commerce Customer Segmentation 
**An End-to-End Unsupervised Machine Learning Pipeline**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

*Transforming raw transactional data into actionable business intelligence.*

</div>

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features & Business Impact](#-key-features--business-impact)
- [Dataset](#-dataset)
- [Machine Learning Architecture](#-machine-learning-architecture)
- [Final Model Evaluation](#-final-model-evaluation)
- [Project Structure](#-project-structure)
- [Meet the Team](#-meet-the-team)
- [How to Run Locally](#-how-to-run-locally)

---

## 📖 Project Overview
In the highly competitive e-commerce sector, one-size-fits-all marketing is obsolete. This project implements an end-to-end Machine Learning solution to automatically segment customers based on their historical purchasing behavior. By leveraging **RFM (Recency, Frequency, Monetary)** analytics and **6 distinct Unsupervised Clustering Algorithms**, we empower marketing teams to identify "Champion" customers for loyalty rewards and "At-Risk" customers for targeted interventions.

## ✨ Key Features & Business Impact
* **Automated Data Cleaning:** Robust pipeline that handles missing IDs, removes return anomalies, and engineered a `TotalPrice` feature across 1M+ rows.
* **Mathematical RFM Scaling:** Advanced log-transformations (`np.log1p`) and standard scaling applied to mitigate heavy financial right-skewness.
* **Actionable Customer Tiers:** Segments customers into distinct groups: *Champions, Loyal Customers, At-Risk, and Dormant/Lost*.
* **Targeted Marketing ROI:** Allows the business to stop wasting ad spend on dormant customers and instead deliver personalized retention campaigns to highly profitable segments.

---

## 🗄️ Dataset
This project utilizes the **Online Retail II** dataset from the UCI Machine Learning Repository, containing over 1 million real-world UK transactional records spanning from 2009 to 2011.
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
* **Size:** 1,067,371 rows
* **Domain:** E-Commerce, Retail, Financial Data

---

## 🧠 Machine Learning Architecture

We engineered a mathematical RFM matrix and trained/evaluated the following 6 clustering models to find the optimal mathematical segmentation:

### 🎯 Centroid Models
1. **K-Means Clustering** *(Baseline & Business Recommended)*
2. **MiniBatch K-Means** *(Scalability Optimization for Millions of Rows)*
3. **Gaussian Mixture Models (GMM)** *(Probabilistic Soft-Clustering)*

### 🌳 Density & Hierarchy Models
4. **Agglomerative Hierarchical** *(Bottom-Up Merging via Ward's Linkage)*
5. **DBSCAN** *(Density-Based Outlier Detection)*
6. **Birch** *(Large-Scale Tree Clustering CF Trees)*

---

## 📊 Final Model Evaluation

Because customer segmentation is an unsupervised problem without "ground truth" labels, we evaluated the models mathematically using the **Silhouette Score** and the **Davies-Bouldin Index**.

<img width="3600" height="1800" alt="model_comparison_silhouette" src="https://github.com/user-attachments/assets/5a136d5a-c312-485a-8fa3-2abb681420f8" />


| 🏆 Rank | Algorithm | Silhouette Score (Higher is Better) | Davies-Bouldin Index (Lower is Better) |
| :---: | :--- | :---: | :---: |
| **#1** | **Birch** | **0.4171** | **0.8805** |
| #2 | Agglomerative | 0.3862 | 0.9214 |
| #3 | K-Means | 0.3294 | 1.0138 |
| #4 | MiniBatch K-Means| 0.3132 | 1.0400 |
| #5 | DBSCAN | 0.2368 | 1.8006 |
| #6 | Gaussian Mixture | 0.1677 | 1.6940 |

**Business Conclusion:** While **Birch** achieved the highest mathematical score, we recommend deploying **K-Means** for final business use. K-Means provides highly interpretable "centroids" (average profiles) for each cluster, allowing marketing managers to easily understand the exact RFM metrics that define a "Loyal" vs "Dormant" customer.

---

## 📁 Project Structure

```text
Ecommerce-Customer-Segmentation/
│
├── src/                     # Python scripts for data cleaning, EDA, and ML models
├── notebooks/               # Interactive Jupyter notebooks for all 6 models
├── outputs/                 
│   ├── figures/             # Visualizations (EDA charts & Silhouette comparisons)
│   ├── predictions/         # Zipped CSV prediction results for each model
│   └── 2_rfm_data_herath.csv # Standardized RFM dataset used for modeling
├── docs/                    # Final Assignment Report PDFs and Presentations
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies (scikit-learn, pandas, etc.)
```

---

## 👨‍💻 Meet the Team

This project was built collaboratively by a cross-functional team of 6 data scientists:

| Name | Role | Core Contributions |
| :--- | :--- | :--- |
| **Sarma SSL** | *ML Evaluator* | Model Evaluation (Silhouette), Report Compilation, Ethics |
| **Nayanajith K.A.T** | *Data Engineer* | Data Cleaning, Missing Values, Anomaly Removal |
| **Herath H.M.A.R.B** | *Feature Engineer*| RFM Matrix Construction, Log Transformation, Scaling |
| **Liyanage D.D.S** | *Data Analyst* | Exploratory Data Analysis (EDA) & Visualizations |
| **Manathunge A.I**| *ML Engineer* | Centroid Model Design (K-Means, MiniBatch, GMM) |
| **Ahamed M.N.A** | *ML Engineer* | Density Model Design (Agglomerative, DBSCAN, Birch) |

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lojanasarma/Ecommerce-Customer-Segmentation.git
   cd Ecommerce-Customer-Segmentation
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the pipeline:**
   * Run the scripts in the `src/` directory in sequential order, or open the visual notebooks in the `notebooks/` folder using JupyterLab!

---
<div align="center">
  <i>Built with ❤️ for IT2011 - Artificial Intelligence and Machine Learning</i>
</div>
