# E-Commerce Customer Behavior Analysis and Segmentation

**Module:** IT2011 – Artificial Intelligence and Machine Learning
**Group ID:** 2026-Y2-S1-KU-01

## 📌 Project Overview

This project addresses a key business intelligence challenge in the retail sector: identifying and understanding different customer groups based on their purchasing behavior.

Using the **Online Retail II** dataset, the project applies unsupervised machine learning techniques to transform raw transactional data into meaningful and actionable customer segments. By moving away from a “one-size-fits-all” marketing approach, businesses can use these insights to develop targeted and personalized marketing strategies that improve customer engagement, retention, and return on investment (ROI).

## 🧠 Methodology

The project follows a data-driven machine learning pipeline consisting of the following stages:

### 1. Data Preprocessing

Raw transaction data is cleaned and prepared for analysis by:

* Handling missing customer identification values
* Removing cancelled transactions
* Correcting or filtering invalid transaction records
* Preparing relevant variables for customer-level analysis

### 2. Feature Engineering Using RFM Analysis

Customer purchasing behavior is summarized using three important metrics:

* **Recency (R):** The number of days since a customer’s most recent purchase
* **Frequency (F):** The total number of purchases or transactions made by a customer
* **Monetary Value (M):** The total amount spent by a customer

These features provide a structured representation of customer behavior and are used as inputs for the clustering model.

### 3. Machine Learning – Customer Clustering

The **K-Means clustering algorithm** is applied to automatically group customers with similar purchasing patterns.

The identified customer segments may include categories such as:

* **Champions** – Highly active customers who purchase frequently and spend significant amounts
* **Loyal Customers** – Customers who make regular purchases and demonstrate consistent engagement
* **At-Risk Customers** – Previously valuable customers who have not made recent purchases
* **Bargain Hunters** – Customers who may purchase less frequently or primarily respond to promotions and discounts

The final segment names will be assigned based on the behavioral characteristics observed in each cluster.

### 4. Model Evaluation

The quality and effectiveness of the clustering model are evaluated using mathematical performance metrics, including the:

* **Silhouette Score**

The Silhouette Score is used to measure how well customers fit within their assigned clusters and how clearly the clusters are separated from one another.

## 📂 Repository Structure

```text
E-Commerce-Customer-Behavior-Analysis/
│
├── data/
│   └── Dataset files are stored locally and excluded from Git
│
├── docs/
│   ├── Project documentation
│   ├── Proposal presentation files (.pptx)
│   └── Final project report
│
├── notebooks/
│   ├── Exploratory Data Analysis (EDA)
│   └── Initial model development and testing
│
├── src/
│   ├── Data preprocessing scripts
│   └── Customer clustering pipeline
│
├── README.md
└── requirements.txt
```

> **Note:** The `notebooks/` and `src/` directories are currently under development and will be updated as the project progresses.

## 📊 Dataset

**Dataset Name:** Online Retail II

**Source:** [UCI Machine Learning Repository – Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii?utm_source=chatgpt.com)

The dataset contains transactional records from a UK-based online retail business. It includes information such as:

* Invoice number
* Product code
* Product description
* Quantity purchased
* Invoice date
* Unit price
* Customer ID
* Country

### Dataset Setup

1. Download the **Online Retail II** dataset from the UCI Machine Learning Repository.
2. Place the downloaded dataset file inside the `data/` directory.
3. Ensure that the file name matches the name expected by the preprocessing scripts or Jupyter notebooks.

> **Note:** Dataset files are excluded from the repository using `.gitignore` because of their file size. Each user must download the dataset separately before running the project.

## 🎯 Project Objectives

The main objectives of this project are to:

* Analyze customer purchasing behavior using transactional data
* Generate customer-level features using the RFM framework
* Apply unsupervised machine learning to identify meaningful customer segments
* Evaluate the quality of the generated clusters
* Develop understandable customer personas based on cluster characteristics
* Support data-driven and personalized marketing strategies

## 🛠️ Technologies and Tools

The project is developed using the following technologies:

* **Python**
* **Jupyter Notebook**
* **Pandas** – Data manipulation and preprocessing
* **NumPy** – Numerical computations
* **Scikit-learn** – K-Means clustering and model evaluation
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical data visualization

## 👥 Group Members

| Name             | Student ID | Role             |
| ---------------- | ---------- | ---------------- |
| MANATHUNGE A.I   | IT23642096 | Group Member     |
| Ahamed M.N.A     | IT24104390 | Group Member     |
| Sarma S.L        | IT25100002 | **Group Leader** |
| Nayanajith K.A.T | IT25100020 | Group Member     |
| Herath H.M.A.R.B | IT25100117 | Group Member     |
| Liyanage D.D.S   | IT25100119 | Group Member     |

## 📈 Expected Outcomes

The project is expected to produce:

* Clearly defined customer segments based on RFM behavior
* Customer personas that support business decision-making
* Visualizations that demonstrate differences between customer groups
* Insights that can be used to improve customer retention and marketing effectiveness
* A reusable machine learning pipeline for customer segmentation

## 📄 License

This project is developed for academic purposes as part of the **IT2011 – Artificial Intelligence and Machine Learning** module.
