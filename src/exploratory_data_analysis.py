"""
Exploratory Data Analysis (EDA)
E-Commerce Customer Behavior Analysis and Segmentation
Module: IT2011 - Artificial Intelligence and Machine Learning
Group ID: 2026-Y2-S1-KU-01

EDA Lead: Liyanage D.D.S (IT25100119)

Reads:
    ../outputs/1_cleaned_data_nayanajith.csv
    ../outputs/2_rfm_data_herath.csv

Produces:
    ../outputs/figures/*.png              (all charts)
    ../outputs/customer_rfm_data.csv      (final RFM file handed to Sub-Team B)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

CLEAN_PATH = "../outputs/1_cleaned_data_nayanajith.csv"
RFM_PATH = "../outputs/2_rfm_data_herath.csv"
FIG_DIR = "../outputs/figures"
FINAL_RFM_PATH = "../outputs/customer_rfm_data.csv"

os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(CLEAN_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    rfm = pd.read_csv(RFM_PATH)
    print("Cleaned transaction data:", df.shape)
    print("RFM customer data:", rfm.shape)
    return df, rfm


def plot_top_products(df):
    top10_qty = (
        df.groupby("Description")["Quantity"].sum()
        .sort_values(ascending=False).head(10)
    )
    plt.figure()
    sns.barplot(x=top10_qty.values, y=top10_qty.index, hue=top10_qty.index,
                palette="viridis", legend=False)
    plt.title("Top 10 Selling Products by Quantity Sold")
    plt.xlabel("Total Quantity Sold")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/top10_products_quantity.png", dpi=150)
    plt.close()

    top10_rev = (
        df.groupby("Description")["TotalCost"].sum()
        .sort_values(ascending=False).head(10)
    )
    plt.figure()
    sns.barplot(x=top10_rev.values, y=top10_rev.index, hue=top10_rev.index,
                palette="magma", legend=False)
    plt.title("Top 10 Selling Products by Revenue")
    plt.xlabel("Total Revenue (GBP)")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/top10_products_revenue.png", dpi=150)
    plt.close()


def plot_sales_trends(df):
    monthly = df.set_index("InvoiceDate").resample("ME").agg(
        Revenue=("TotalCost", "sum"),
        Orders=("Invoice", "nunique")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(monthly["InvoiceDate"], monthly["Revenue"], color="tab:blue", marker="o")
    ax1.set_title("Monthly Sales Revenue Trend")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Revenue (GBP)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/monthly_revenue_trend.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly["InvoiceDate"], monthly["Orders"], color="tab:green", marker="o")
    plt.title("Monthly Order Volume Trend")
    plt.xlabel("Month")
    plt.ylabel("Number of Unique Orders")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/monthly_orders_trend.png", dpi=150)
    plt.close()


def plot_country_sales(df):
    top_countries = (
        df.groupby("Country")["TotalCost"].sum()
        .sort_values(ascending=False).head(10)
    )
    plt.figure()
    sns.barplot(x=top_countries.values, y=top_countries.index, hue=top_countries.index,
                palette="crest", legend=False)
    plt.title("Top 10 Countries by Revenue")
    plt.xlabel("Total Revenue (GBP)")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/top10_countries_revenue.png", dpi=150)
    plt.close()


def plot_rfm_distributions(rfm):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(rfm["Recency"], bins=30, kde=True, color="tab:blue", ax=axes[0])
    axes[0].set_title("Recency Distribution (days)")
    sns.histplot(rfm["Frequency"], bins=30, kde=True, color="tab:orange", ax=axes[1])
    axes[1].set_title("Frequency Distribution (orders)")
    sns.histplot(rfm["Monetary"], bins=30, kde=True, color="tab:green", ax=axes[2])
    axes[2].set_title("Monetary Distribution (GBP)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rfm_raw_distributions.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(rfm["Log_R"], bins=30, kde=True, color="tab:blue", ax=axes[0])
    axes[0].set_title("Log(Recency) Distribution")
    sns.histplot(rfm["Log_F"], bins=30, kde=True, color="tab:orange", ax=axes[1])
    axes[1].set_title("Log(Frequency) Distribution")
    sns.histplot(rfm["Log_M"], bins=30, kde=True, color="tab:green", ax=axes[2])
    axes[2].set_title("Log(Monetary) Distribution")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rfm_log_distributions.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(y=rfm["Recency"], color="tab:blue", ax=axes[0])
    axes[0].set_title("Recency Spread")
    sns.boxplot(y=rfm["Frequency"], color="tab:orange", ax=axes[1])
    axes[1].set_title("Frequency Spread")
    sns.boxplot(y=rfm["Monetary"], color="tab:green", ax=axes[2])
    axes[2].set_title("Monetary Spread")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rfm_boxplots.png", dpi=150)
    plt.close()


def plot_rfm_correlation(rfm):
    corr = rfm[["Recency", "Frequency", "Monetary"]].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Between RFM Metrics")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rfm_correlation_heatmap.png", dpi=150)
    plt.close()


def main():
    df, rfm = load_data()
    plot_top_products(df)
    plot_sales_trends(df)
    plot_country_sales(df)
    plot_rfm_distributions(rfm)
    plot_rfm_correlation(rfm)

    rfm.to_csv(FINAL_RFM_PATH, index=False)
    print("Saved final RFM dataset:", FINAL_RFM_PATH, rfm.shape)
    print("All figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
