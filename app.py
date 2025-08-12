import streamlit as st
import pandas as pd
from typing_extensions import TypedDict
from eda_modules.summary import generate_summary
from eda_modules.correlation import generate_correlation
from eda_modules.forecast import generate_forecast
from eda_modules.patterns import detect_outliers

# LangGraph
from langgraph.graph import StateGraph, START, END

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

# --- Gemini LLM function with skeleton loading ---
def gemini_explain(prompt):
    client = get_gemini_client()
    if not client:
        return "❌ Gemini API key not provided or google-genai not installed."
    
    placeholder = st.empty()
    placeholder.info("🤖 Thinking... Please wait.")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        placeholder.empty()
        return response.text
    except Exception as e:
        placeholder.empty()
        return f"❌ Gemini error: {e}"

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

# --- Define LangGraph State ---
class EDAState(TypedDict):
    df: pd.DataFrame
    summary: str
    ai_summary: str
    correlation_plot: any
    ai_correlation: str
    forecast_plot: any
    ai_forecast: str
    outliers: any
    ai_outliers: str

# --- LangGraph Node Functions ---
def summary_node(state: EDAState) -> EDAState:
    summary = generate_summary(state["df"])
    ai_explanation = gemini_explain(f"Here is the dataset summary:\n{summary}\nExplain in plain English.")
    state["summary"] = summary
    state["ai_summary"] = ai_explanation
    return state

def correlation_node(state: EDAState) -> EDAState:
    fig = generate_correlation(state["df"])
    corr_matrix = state["df"].select_dtypes(include='number').corr().to_string()
    ai_explanation = gemini_explain(f"The correlation matrix is:\n{corr_matrix}\nExplain the main relationships.")
    state["correlation_plot"] = fig
    state["ai_correlation"] = ai_explanation
    return state

def forecast_node(state: EDAState) -> EDAState:
    col = state["df"].select_dtypes(include='number').columns[0] if not state["df"].empty else None
    if col:
        fig = generate_forecast(state["df"], col)
        ai_explanation = gemini_explain(f"Forecast for '{col}'. Explain the trend.")
    else:
        fig = None
        ai_explanation = "No numeric columns available for forecasting."
    state["forecast_plot"] = fig
    state["ai_forecast"] = ai_explanation
    return state

def outlier_node(state: EDAState) -> EDAState:
    outliers = detect_outliers(state["df"])
    ai_explanation = gemini_explain(f"Detected outliers: {outliers}. Explain what they might mean.")
    state["outliers"] = outliers
    state["ai_outliers"] = ai_explanation
    return state

# --- Build LangGraph ---
graph = StateGraph(EDAState)

graph.add_node("Summary", summary_node)
graph.add_node("Correlation", correlation_node)
graph.add_node("Forecast", forecast_node)
graph.add_node("Outliers", outlier_node)

graph.add_edge(START, "Summary")
graph.add_edge("Summary", "Correlation")
graph.add_edge("Correlation", "Forecast")
graph.add_edge("Forecast", "Outliers")
graph.add_edge("Outliers", END)

eda_app = graph.compile()

# --- Streamlit UI ---
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write("### 📊 Preview of the Dataset")
        st.dataframe(df.head())

        analysis_type = st.radio(
            "📌 Select an analysis to perform:",
            ["Summary", "Correlation", "Forecast", "Outliers", "Insights", "Full AI-Powered Analysis"]
        )

        if analysis_type == "Summary":
            st.subheader("📋 Dataset Summary")
            st.text(generate_summary(df))

        elif analysis_type == "Correlation":
            st.subheader("🔗 Correlation Matrix")
            st.pyplot(generate_correlation(df))

        elif analysis_type == "Forecast":
            st.subheader("📈 Forecasting")
            selected_column = st.selectbox(
                "Select a numeric column to forecast",
                df.select_dtypes(include='number').columns
            )
            st.pyplot(generate_forecast(df, selected_column))

        elif analysis_type == "Outliers":
            st.subheader("🚨 Outlier Detection")
            outliers = detect_outliers(df)
            st.write(outliers)

        elif analysis_type == "Insights":
            st.subheader("💡 Insights")
            mode = st.radio("Select insight mode:", ["Rule-based (Offline)", "AI-powered (Gemini)"])
            if mode == "Rule-based (Offline)":
                for ins in generate_rule_based_insights(df):
                    st.write(ins)
            else:
                st.write(gemini_explain(
                    f"Dataset stats:\n{df.describe(include='all').to_string()}\n"
                    "Give concise insights about patterns, trends, and anomalies."
                ))

        elif analysis_type == "Full AI-Powered Analysis":
            results = eda_app.invoke({"df": df})

            st.subheader("📋 Summary")
            st.text(results["summary"])
            st.markdown(f"**AI says:** {results['ai_summary']}")

            st.subheader("🔗 Correlation")
            st.pyplot(results["correlation_plot"])
            st.markdown(f"**AI says:** {results['ai_correlation']}")

            st.subheader("📈 Forecast")
            if results["forecast_plot"]:
                st.pyplot(results["forecast_plot"])
            st.markdown(f"**AI says:** {results['ai_forecast']}")

            st.subheader("🚨 Outliers")
            st.write(results["outliers"])
            st.markdown(f"**AI says:** {results['ai_outliers']}")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
else:
    st.info("Please upload a CSV file to start.")
