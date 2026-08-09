import pandas as pd
import os

# -----------------------------
# 1. Load Excel file
# -----------------------------

file_path = "../data/online_retail_II.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df))

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)


# -----------------------------
# 2. Remove missing Customer IDs
# -----------------------------

df = df.dropna(subset=["Customer ID"])

print("\nRows after removing missing Customer IDs:", len(df))


# -----------------------------
# 3. Remove cancelled orders
# -----------------------------

df = df[~df["Invoice"].astype(str).str.startswith("C")]

print("Rows after removing cancelled orders:", len(df))


# -----------------------------
# 4. Remove negative quantities
# -----------------------------

df = df[df["Quantity"] > 0]

print("Rows after removing negative quantities:", len(df))


# -----------------------------
# 5. Handle price outliers
# -----------------------------

Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[
    (df["Price"] >= lower_bound) &
    (df["Price"] <= upper_bound)
]

print("Rows after handling price outliers:", len(df))


# -----------------------------
# 6. Calculate Total Cost
# -----------------------------

df["TotalCost"] = df["Quantity"] * df["Price"]

print("\nTotal Cost:")
print(df[["Quantity", "Price", "TotalCost"]].head())


# -----------------------------
# 7. Dataset information
# -----------------------------

print("\nDataset information:")
df.info()


# -----------------------------
# 8. Missing values check
# -----------------------------

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------
# 9. Negative quantity check
# -----------------------------

print("\nNegative quantities:", (df["Quantity"] < 0).sum())


# -----------------------------
# 10. Cancelled invoice check
# -----------------------------

print(
    "Cancelled invoices:",
    df["Invoice"].astype(str).str.startswith("C").sum()
)


# -----------------------------
# 11. Price check
# -----------------------------

print("Minimum price:", df["Price"].min())
print("Maximum price:", df["Price"].max())


# -----------------------------
# 12. Save cleaned dataset
# -----------------------------

output_folder = "../outputs"

os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(
    output_folder,
    "1_cleaned_data_nayanajith.csv"
)

df.to_csv(output_path, index=False)

print("\nCleaned data saved successfully!")
print(output_path)