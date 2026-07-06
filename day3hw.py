# ==========================================================
# Employee Retention Prediction using Logistic Regression
# Streamlit App
# ==========================================================
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("HR_comma_sep.csv")

# -----------------------------
# Prepare Data
# -----------------------------
X = df[['satisfaction_level',
        'average_montly_hours',
        'promotion_last_5years',
        'salary']]

salary_dummies = pd.get_dummies(X['salary'], prefix='salary')

X = pd.concat(
    [X.drop('salary', axis=1), salary_dummies],
    axis=1
)

# Drop one dummy column
if 'salary_high' in X.columns:
    X = X.drop('salary_high', axis=1)

y = df['left']

# -----------------------------
# Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Employee Retention Predictor")

st.title("Employee Retention Prediction")

satisfaction = st.number_input(
    "Satisfaction Level",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

hours = st.number_input(
    "Average Monthly Hours",
    min_value=50,
    max_value=350,
    value=200
)

promotion = st.selectbox(
    "Promotion in Last 5 Years",
    [0, 1]
)

salary = st.selectbox(
    "Salary",
    ["Low", "Medium", "High"]
)

if st.button("Predict"):

    salary_low = 0
    salary_medium = 0

    if salary == "Low":
        salary_low = 1
    elif salary == "Medium":
        salary_medium = 1

    user_data = pd.DataFrame({
        'satisfaction_level': [satisfaction],
        'average_montly_hours': [hours],
        'promotion_last_5years': [promotion],
        'salary_low': [salary_low],
        'salary_medium': [salary_medium]
    })

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.error(
            f"Employee is likely to LEAVE ({probability[1]*100:.2f}%)"
        )
    else:
        st.success(
            f"Employee is likely to STAY ({probability[0]*100:.2f}%)"
        )

st.markdown("---")
st.header("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}%")
