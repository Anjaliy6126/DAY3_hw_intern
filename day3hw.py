# ==========================================================
# Employee Retention Prediction using Logistic Regression
# Streamlit App
# ==========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Employee Retention Prediction",
    layout="wide"
)

st.title("📊 Employee Retention Prediction")
st.write(
    "This application performs Exploratory Data Analysis (EDA) "
    "and predicts employee retention using Logistic Regression."
)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

try:
    df = pd.read_csv("HR_comma_sep.csv")
except FileNotFoundError:
    st.error("❌ HR_comma_sep.csv not found. Place the file in the same folder as day3hw.py")
    st.stop()

# ----------------------------------------------------------
# Dataset Preview
# ----------------------------------------------------------

st.header("Dataset Preview")
st.dataframe(df.head())

st.write("**Dataset Shape:**", df.shape)

# ----------------------------------------------------------
# Dataset Information
# ----------------------------------------------------------

st.header("Dataset Information")

info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values
})

st.dataframe(info_df)

# ----------------------------------------------------------
# Statistical Summary
# ----------------------------------------------------------

st.header("Statistical Summary")
st.dataframe(df.describe())

# ----------------------------------------------------------
# Average Values Grouped by Employee Retention
# ----------------------------------------------------------

st.header("Average Values Grouped by Employee Retention")
st.dataframe(df.groupby("left").mean(numeric_only=True))

# ----------------------------------------------------------
# Correlation Heatmap
# ----------------------------------------------------------

st.header("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# ----------------------------------------------------------
# Salary vs Employee Retention
# ----------------------------------------------------------

st.header("Salary vs Employee Retention")

salary_chart = pd.crosstab(df["salary"], df["left"])

fig, ax = plt.subplots(figsize=(6, 5))
salary_chart.plot(kind="bar", ax=ax)

ax.set_xlabel("Salary")
ax.set_ylabel("Employees")
ax.set_title("Salary vs Employee Retention")

st.pyplot(fig)

# ----------------------------------------------------------
# Department vs Employee Retention
# ----------------------------------------------------------

st.header("Department vs Employee Retention")

# Some datasets use "sales" instead of "Department"
department_column = "Department" if "Department" in df.columns else "sales"

dept_chart = pd.crosstab(df[department_column], df["left"])

fig, ax = plt.subplots(figsize=(10, 6))
dept_chart.plot(kind="bar", ax=ax)

ax.set_xlabel("Department")
ax.set_ylabel("Employees")
ax.set_title("Department vs Employee Retention")

st.pyplot(fig)

# ----------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------

features = df[
    [
        "satisfaction_level",
        "average_montly_hours",
        "promotion_last_5years",
        "salary"
    ]
]

# ----------------------------------------------------------
# Convert Salary into Dummy Variables
# ----------------------------------------------------------

salary_dummies = pd.get_dummies(features["salary"], prefix="salary")

X = pd.concat(
    [
        features[
            [
                "satisfaction_level",
                "average_montly_hours",
                "promotion_last_5years",
            ]
        ],
        salary_dummies,
    ],
    axis=1,
)

# Remove one dummy column
if "salary_high" in X.columns:
    X = X.drop("salary_high", axis=1)

y = df["left"]

# ----------------------------------------------------------
# Train Test Split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ----------------------------------------------------------
# Predictions
# ----------------------------------------------------------

y_pred = model.predict(X_test)

# ----------------------------------------------------------
# Accuracy
# ----------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

st.header("Model Accuracy")
st.success(f"{accuracy * 100:.2f}%")

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

st.header("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

st.header("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# ----------------------------------------------------------
# Sample Predictions
# ----------------------------------------------------------

st.header("Sample Predictions")

prediction_df = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})

st.dataframe(prediction_df)
