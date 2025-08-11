import streamlit as st
import pandas as pd
from eda_modules.summary import generate_summary
from eda_modules.correlation import generate_correlation
from eda_modules.forecast import generate_forecast
from eda_modules.patterns import detect_outliers

# Gemini API
try:
    from google import genai
except ImportError:
    genai = None

st.set_page_config(page_title="EDA App", layout="wide")
st.title("🔍 Automated EDA Dashboard")

# Sidebar: API key input
st.sidebar.subheader("🔑 AI Settings")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key (optional)", type="password")

# --- Helper: Initialize Gemini client ---
def get_gemini_client():
    if not gemini_api_key or not genai:
        return None
    return genai.Client(api_key=gemini_api_key)

# --- Rule-based Insights Function ---
def generate_rule_based_insights(df):
    insights = []
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        mean_val = df[col].mean()
        max_val = df[col].max()
        min_val = df[col].min()
        if max_val > mean_val * 1.5:
            insights.append(f"🔼 **{col}** has unusually high max values compared to its mean.")
        if min_val < mean_val * 0.5:
            insights.append(f"🔽 **{col}** has unusually low min values compared to its mean.")
    return insights or ["No significant patterns detected."]

# --- Gemini LLM Insights ---
def gemini_explain(prompt):
    client = get_gemini_client()
    if not client:
        return "❌ Gemini API key not provided or google-genai not installed."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini error: {e}"

# --- Gemini-generated dataset insights ---
def generate_llm_insights(df):
    summary = df.describe(include="all").to_string()
    prompt = f"""
    You are a data analyst. The dataset summary is:
    {summary}
    Provide clear, concise insights in bullet points about patterns, trends, and anomalies.
    """
    return gemini_explain(prompt).split("\n")

# File upload
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write("### 📊 Preview of the Dataset")
        st.dataframe(df.head())

        # Tool selection
        analysis_type = st.radio(
            "📌 Select an analysis to perform:",
            ["Summary", "Correlation", "Forecast", "Outliers", "Insights"]
        )

        if analysis_type == "Summary":
            st.subheader("📋 Dataset Summary")
            summary = generate_summary(df)
            st.dataframe(summary)

        elif analysis_type == "Correlation":
            st.subheader("🔗 Correlation Matrix")
            fig = generate_correlation(df)
            st.pyplot(fig)

            # AI explanation
            if gemini_api_key:
                st.markdown("#### 🤖 AI Explanation")
                corr_matrix = df.select_dtypes(include='number').corr().to_string()
                explanation = gemini_explain(
                    f"The correlation matrix is:\n{corr_matrix}\nExplain the main relationships in plain English."
                )
                st.write(explanation)

        elif analysis_type == "Forecast":
            st.subheader("📈 Forecasting")
            selected_column = st.selectbox(
                "Select a numeric column to forecast",
                df.select_dtypes(include='number').columns
            )
            fig = generate_forecast(df, selected_column)
            st.pyplot(fig)

            # AI explanation
            if gemini_api_key:
                st.markdown("#### 🤖 AI Forecast Interpretation")
                explanation = gemini_explain(
                    f"The dataset's '{selected_column}' values have been forecasted for the next few periods. "
                    "Explain the forecast trend and whether it shows growth, decline, or stability."
                )
                st.write(explanation)

        elif analysis_type == "Outliers":
            st.subheader("🚨 Outlier Detection")
            outliers = detect_outliers(df)
            for col, values in outliers.items():
                st.write(f"**{col}**: {values if values else 'No outliers found'}")

            # AI explanation
            if gemini_api_key:
                st.markdown("#### 🤖 AI Outlier Analysis")
                explanation = gemini_explain(
                    f"The dataset has these detected outliers: {outliers}. "
                    "Explain in plain language what these outliers might mean."
                )
                st.write(explanation)

        elif analysis_type == "Insights":
            st.subheader("💡 Insights")
            mode = st.radio("Select insight mode:", ["Rule-based (Offline)", "AI-powered (Gemini)"])
            if mode == "Rule-based (Offline)":
                insights = generate_rule_based_insights(df)
            else:
                insights = generate_llm_insights(df)

            for ins in insights:
                st.write(ins)

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
else:
    st.info("Please upload a CSV file to start.")
