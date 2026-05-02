import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Dashboard", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("🎓 Student Performance Dashboard")
st.markdown("Analyze student data and predict performance 📊")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("StudentsPerformance.csv")
df.columns = df.columns.str.replace(" ", "_")

df["average_score"] = (df["math_score"] + df["reading_score"] + df["writing_score"]) / 3

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔍 Filter Data")

gender = st.sidebar.selectbox("Gender", df["gender"].unique())
test_prep = st.sidebar.selectbox("Test Preparation", df["test_preparation_course"].unique())

filtered_df = df[(df["gender"] == gender) & (df["test_preparation_course"] == test_prep)]

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(filtered_df.head())

# -----------------------------
# KPI SECTION
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Avg Score", round(filtered_df["average_score"].mean(), 2))
col2.metric("Max Math Score", filtered_df["math_score"].max())
col3.metric("Min Reading Score", filtered_df["reading_score"].min())

# -----------------------------
# BAR CHART
# -----------------------------
st.subheader("📊 Average Score by Gender")

fig, ax = plt.subplots()
sns.barplot(x="gender", y="average_score", data=df, ax=ax)
st.pyplot(fig)

# -----------------------------
# PIE CHART
# -----------------------------
st.subheader("🥧 Gender Distribution")

gender_counts = df["gender"].value_counts()

fig2, ax2 = plt.subplots()
ax2.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
st.pyplot(fig2)

# -----------------------------
# BOXPLOT
# -----------------------------
st.subheader("📦 Score Distribution")

fig3, ax3 = plt.subplots()
sns.boxplot(data=df[["math_score", "reading_score", "writing_score"]], ax=ax3)
st.pyplot(fig3)

# -----------------------------
# HEATMAP
# -----------------------------
st.subheader("📌 Correlation Heatmap")

fig4, ax4 = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

# -----------------------------
# MACHINE LEARNING MODEL
# -----------------------------
X = df[["math_score", "reading_score"]]
y = df["writing_score"]

model = LinearRegression()
model.fit(X, y)

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.subheader("🎯 Predict Writing Score")

math = st.number_input("Enter Math Score", 0, 100)
reading = st.number_input("Enter Reading Score", 0, 100)

if st.button("Predict"):
    prediction = model.predict([[math, reading]])
    st.success(f"Predicted Writing Score: {round(prediction[0],2)}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.success("Dashboard Ready 🚀")
