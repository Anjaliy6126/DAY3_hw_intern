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
# Page Config (must be first Streamlit command)
# -----------------------------
st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="🧑‍💼",
    layout="centered"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
    <style>
    /* Overall app background */
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d29 100%);
        color: #f1f5f9;
    }

    /* Make all default text light */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #f1f5f9;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0px;
    }

    /* Subtitle / description */
    .sub-title {
        text-align: center;
        font-size: 17px;
        color: #cbd5e1;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    /* Card container look for sections */
    .card {
        background-color: #1e2130;
        padding: 22px 26px;
        border-radius: 16px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.4);
        margin-bottom: 22px;
        border: 1px solid #2d3348;
    }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }

    /* Input widgets */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #262a3d !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(90deg, #4361ee, #7209b7);
        color: white;
        font-weight: 700;
        font-size: 16px;
        padding: 10px 26px;
        border-radius: 12px;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #7209b7, #4361ee);
        transform: scale(1.02);
    }

    /* Accuracy box */
    .accuracy-box {
        background: linear-gradient(90deg, #134e4a, #1e3a5f);
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        font-size: 18px;
        font-weight: 700;
        color: #f1f5f9;
        border: 1px solid #2d3348;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 13px;
        color: #94a3b8;
    }

    /* Horizontal rule */
    hr {
        border-color: #2d3348 !important;
    }
    </style>
""", unsafe_allow_html=True)

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
st.markdown('<div class="main-title">🧑‍💼 Employee Retention Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">🔍 A Machine Learning powered app that predicts whether an employee '
    'is likely to <b>stay</b> 🟢 or <b>leave</b> 🔴 the company, based on satisfaction level, '
    'working hours, promotions, and salary band.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Input Section
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📋 Employee Details</div>', unsafe_allow_html=True)

satisfaction = st.number_input(
    "😊 Satisfaction Level",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

hours = st.number_input(
    "⏰ Average Monthly Hours",
    min_value=50,
    max_value=350,
    value=200
)

promotion = st.selectbox(
    "🏆 Promotion in Last 5 Years",
    [0, 1]
)

salary = st.selectbox(
    "💰 Salary",
    ["Low", "Medium", "High"]
)

predict_clicked = st.button("🔮 Predict")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Section
# -----------------------------
if predict_clicked:
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

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.error(
            f"🚨 Employee is likely to LEAVE 👋 ({probability[1]*100:.2f}%)"
        )
    else:
        st.success(
            f"✅ Employee is likely to STAY 🎉 ({probability[0]*100:.2f}%)"
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Model Accuracy Section
# -----------------------------
st.markdown('<div class="section-header">📈 Model Accuracy</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="accuracy-box">✅ Accuracy: {accuracy:.2f}% 🎯</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit & Logistic Regression | '
    'Employee Retention Prediction App</div>',
    unsafe_allow_html=True
)
