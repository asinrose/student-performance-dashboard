
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🎓 Student Performance Dashboard")

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Rename columns
df.columns = df.columns.str.replace(" ", "_")

# Add average column
df["average_score"] = (df["math_score"] + df["reading_score"] + df["writing_score"]) / 3

# Show dataset
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# FILTER OPTION
# -----------------------------
st.sidebar.header("Filter Data")

gender = st.sidebar.selectbox("Select Gender", df["gender"].unique())

filtered_df = df[df["gender"] == gender]

# -----------------------------
# BAR CHART
# -----------------------------
st.subheader(f"Average Score for {gender}")

avg_score = filtered_df["average_score"].mean()
st.write("Average Score:", round(avg_score, 2))

fig, ax = plt.subplots()
sns.barplot(x="gender", y="average_score", data=filtered_df, ax=ax)
st.pyplot(fig)

# -----------------------------
# HEATMAP
# -----------------------------
st.subheader("📌 Correlation Heatmap")

fig2, ax2 = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# -----------------------------
# FINAL MESSAGE
# -----------------------------
st.success("Dashboard Loaded Successfully ✅")