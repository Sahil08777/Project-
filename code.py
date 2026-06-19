# -*- coding: 
# TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    confusion_matrix
)

# Load Dataset

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

df = pd.read_csv(url, header=None)

# Assign Column Headers

headers = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price"
]

df.columns = headers

print("Dataset Loaded Successfully")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 10 Rows:")
display(df.head(10))

df.replace("?", np.nan, inplace=True)

print("Missing Values:")
print(df.isnull().sum())

numeric_columns = [
    'normalized-losses',
    'bore',
    'stroke',
    'horsepower',
    'peak-rpm',
    'price'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(df.dtypes)

# ==========================================================
# Missing Value Treatment
# Mean Replacement + Mode Replacement
# ==========================================================

# Mean replacement for numerical columns
mean_columns = [
    'normalized-losses',
    'bore',
    'stroke',
    'horsepower',
    'peak-rpm'
]

for col in mean_columns:
    df[col] = df[col].fillna(df[col].mean())

# Mode replacement for categorical column
df['num-of-doors'] = df['num-of-doors'].fillna(
    df['num-of-doors'].mode()[0]
)

# Verify missing values after treatment
print("Missing Values After Treatment:")
print(df.isnull().sum())

df.dropna(subset=['price'], inplace=True)

print(df.shape)

print("Mean")
print(df.mean(numeric_only=True))

print("\nMedian")
print(df.median(numeric_only=True))

print("\nStandard Deviation")
print(df.std(numeric_only=True))

print(df.dtypes)

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    print("\n================================")
    print(col)
    print(df[col].unique())

plt.figure(figsize=(8,5))

plt.hist(df['price'], bins=20)

plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.show()

plt.figure(figsize=(8,5))

plt.hist(df['horsepower'], bins=20)

plt.title('Horsepower Distribution')
plt.xlabel('Horsepower')
plt.ylabel('Frequency')

plt.show()

plt.figure(figsize=(10,6))

sns.boxplot(
    x='body-style',
    y='price',
    data=df
)

plt.title('Price vs Body Style')

plt.show()

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

df['city-L/100km'] = 235 / df['city-mpg']

display(df[['city-mpg','city-L/100km']].head())

bins = np.linspace(
    df['horsepower'].min(),
    df['horsepower'].max(),
    4
)

labels = ['Low','Medium','High']

df['horsepower-binned'] = pd.cut(
    df['horsepower'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(df['horsepower-binned'].value_counts())

df['horsepower-normalized'] = (
    (df['horsepower'] - df['horsepower'].min())
    /
    (df['horsepower'].max() - df['horsepower'].min())
)

display(
    df[['horsepower',
        'horsepower-normalized']].head()
)

X = df[['horsepower','engine-size']]

y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("LINEAR REGRESSION RESULTS")

print("R2 Score:")
print(r2_score(y_test, y_pred))

print("\nMAE:")
print(mean_absolute_error(y_test, y_pred))

print("\nMSE:")
print(mean_squared_error(y_test, y_pred))

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("RANDOM FOREST RESULTS")

print("R2 Score:")
print(r2_score(y_test, rf_pred))

print("\nMAE:")
print(mean_absolute_error(y_test, rf_pred))

print("\nMSE:")
print(mean_squared_error(y_test, rf_pred))

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

lr2 = LinearRegression()

lr2.fit(X_train2, y_train2)

pred2 = lr2.predict(X_test2)

print("70/30 SPLIT")

print("R2:")
print(r2_score(y_test2, pred2))

print("MAE:")
print(mean_absolute_error(y_test2, pred2))

print("MSE:")
print(mean_squared_error(y_test2, pred2))

df['price-category'] = pd.qcut(
    df['price'],
    3,
    labels=['Low','Medium','High']
)

print(df['price-category'].value_counts())

X_cls = df[['horsepower','engine-size']]

y_cls = df['price-category']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls,
    y_cls,
    test_size=0.20,
    random_state=42
)

log_model = LogisticRegression(
    max_iter=1000
)

log_model.fit(
    X_train_c,
    y_train_c
)

pred_cls = log_model.predict(
    X_test_c
)

print("Accuracy")

print(
    accuracy_score(
        y_test_c,
        pred_cls
    )
)

cm = confusion_matrix(
    y_test_c,
    pred_cls
)

print(cm)

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Price")

plt.show()