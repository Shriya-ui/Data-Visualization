#!/usr/bin/env python
# coding: utf-8

# In[12]:


# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Diamonds Prices2022 Dataset
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# Load Dataset
# ------------------------------------------

file_path = r"C:\Users\HDC0422068\Downloads\Diamonds Prices2022.csv\Diamonds Prices2022.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!\n")

# ------------------------------------------
# Display Dataset
# ------------------------------------------

print("First 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

# ------------------------------------------
# Shape
# ------------------------------------------

print("\nDataset Shape:", df.shape)

# ------------------------------------------
# Dataset Info
# ------------------------------------------

print("\nDataset Information")
df.info()

# ------------------------------------------
# Data Types
# ------------------------------------------

print("\nData Types")
print(df.dtypes)

# ------------------------------------------
# Statistical Summary
# ------------------------------------------

print("\nStatistical Summary")
print(df.describe())

# ------------------------------------------
# Missing Values
# ------------------------------------------

print("\nMissing Values")

missing = df.isnull().sum()

print(missing)

print("\nTotal Missing Values =", missing.sum())

# ------------------------------------------
# Duplicate Rows
# ------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate Rows =", duplicates)

df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)

# ------------------------------------------
# Correlation Matrix
# ------------------------------------------

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.show()

# ------------------------------------------
# Histograms
# ------------------------------------------

numeric_df.hist(figsize=(15,10))

plt.tight_layout()

plt.show()

# ------------------------------------------
# Outlier Detection
# ------------------------------------------

print("\nOutlier Count")

numeric_columns = numeric_df.columns

outlier_count = {}

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    outlier_count[col] = len(outliers)

    print(col, ":", len(outliers))

# ------------------------------------------
# Boxplots
# ------------------------------------------

plt.figure(figsize=(18,12))

for i, col in enumerate(numeric_columns):

    plt.subplot(3,3,i+1)

    sns.boxplot(y=df[col])

    plt.title(col)

plt.tight_layout()

plt.show()

# ------------------------------------------
# Remove Outliers
# ------------------------------------------

df_clean = df.copy()

for col in numeric_columns:

    Q1 = df_clean[col].quantile(0.25)

    Q3 = df_clean[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    df_clean = df_clean[
        (df_clean[col] >= lower) &
        (df_clean[col] <= upper)
    ]

print("\nOriginal Shape :", df.shape)

print("Shape After Removing Outliers :", df_clean.shape)

# ------------------------------------------
# Save Clean Dataset
# ------------------------------------------

output = r"C:\Users\HDC0422068\Downloads\Diamonds_Cleaned.csv"

df_clean.to_csv(output, index=False)

print("\nCleaned Dataset Saved Successfully!")

# ------------------------------------------
# Final Summary
# ------------------------------------------

print("\n===============================")
print("EDA SUMMARY")
print("===============================")

print("Dataset Shape:", df.shape)

print("Missing Values:", missing.sum())

print("Duplicate Rows:", duplicates)

print("\nOutlier Count")

for k, v in outlier_count.items():
    print(k, ":", v)

print("\nEDA Completed Successfully!")


# In[ ]:




