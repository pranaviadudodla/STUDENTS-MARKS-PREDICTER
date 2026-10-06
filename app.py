import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Optional AI assistant (prefers the newer langchain-ollama package)
try:
    from langchain_ollama import OllamaLLM as OllamaModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaModel
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Marks Predictor")
st.write(
    "Predict student marks using Machine Learning and get study advice "
    "from an AI assistant."
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
REQUIRED_COLUMNS = [
    "hours_studied",
    "attendance",
    "previous_score",
    "assignments",
    "marks"
]

FEATURES = [
    "hours_studied",
    "attendance",
    "previous_score",
    "assignments"
]


@st.cache_data
def load_data():
    return pd.read_csv("student_marks.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "student_marks.csv was not found. Please keep the CSV file "
        "in the same folder as app.py."
    )
    st.stop()

missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

if missing_columns:
    st.error(
        f"Missing columns in student_marks.csv: {', '.join(missing_columns)}"
    )
    st.stop()

# Drop rows with blanks so the model doesn't crash
df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

if len(df) < 10:
    st.error("Not enough valid rows in student_marks.csv to train a model.")
    st.stop()


# --------------------------------------------------
# Train Machine Learning model (cached)
# --------------------------------------------------
@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[FEATURES]
    y = data["marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    return rf, mean_absolute_error(y_test, preds), r2_score(y_test, preds)


model, mae, r2 = train_model(df)


# --------------------------------------------------
# Sidebar - Student input
# --------------------------------------------------
st.sidebar.header("👨‍🎓 Student Information")

hours = st.sidebar.number_input(
    "Hours Studied per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

previous_score = st.sidebar.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

assignments = st.sidebar.number_input(
    "Assignments Completed",
    min_value=0,
    max_value=100,
    value=7,
    step=1
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.header("📈 Marks Prediction")

input_data = pd.DataFrame({
    "hours_studied": [hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "assignments": [assignments]
})

if st.button("🚀 Predict Marks", use_container_width=True):
    prediction = float(np.clip(model.predict(input_data)[0], 0, 100))
    # Remember it so the AI assistant can use it after the page reruns
    st.session_state["prediction"] = prediction

    st.metric(
        label="Predicted Marks",
        value=f"{prediction:.2f} / 100"
    )

    if prediction >= 90:
        st.success("🏆 Excellent performance!")
        st.balloons()
    elif prediction >= 75:
        st.success("⭐ Very good performance!")
    elif prediction >= 50:
        st.warning("📚 Average performance. Keep improving!")
    else:
        st.error("💪 Needs improvement. Study consistently!")


# --------------------------------------------------
# Model performance
# --------------------------------------------------
st.header("🤖 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Mean Absolute Error", f"{mae:.2f}")

with col2:
    st.metric("R² Score", f"{r2:.2f}")


# --------------------------------------------------
# Visualizations
# --------------------------------------------------
st.header("📊 Student Performance Analysis")


def make_scatter(data, x, y, title):
    """Scatter plot with a trendline; falls back if statsmodels is missing."""
    try:
        return px.scatter(data, x=x, y=y, title=title, trendline="ols")
    except Exception:
        return px.scatter(data, x=x, y=y, title=title)


col1, col2 = st.columns(2)

with col1:
    fig1 = make_scatter(df, "hours_studied", "marks", "Study Hours vs Marks")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = make_scatter(df, "attendance", "marks", "Attendance vs Marks")
    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# AI Assistant using LangChain + Ollama
# --------------------------------------------------
st.header("🤖 AI Student Assistant")

question = st.text_input(
    "Ask a question about your performance:",
    placeholder="How can I improve my marks?"
)

if st.button("💬 Ask AI"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not LANGCHAIN_AVAILABLE:
        st.error(
            "LangChain is not installed. Run: pip install langchain-ollama"
        )
    else:
        try:
            llm = OllamaModel(model="llama3.2")

            predicted = st.session_state.get("prediction")
            predicted_line = (
                f"- Predicted marks: {predicted:.2f} / 100\n"
                if predicted is not None else ""
            )

            prompt = f"""
You are a helpful student performance assistant.

Student information:
- Hours studied per day: {hours}
- Attendance: {attendance}%
- Previous score: {previous_score}
- Assignments completed: {assignments}
{predicted_line}
Student question:
{question}

Give a simple, practical and encouraging answer.
"""

            with st.spinner("AI is thinking..."):
                response = llm.invoke(prompt)

            st.write(response)

        except Exception:
            st.error(
                "Could not connect to Ollama. Make sure Ollama is running "
                "and the llama3.2 model is installed."
            )
            st.code("ollama pull llama3.2")


# --------------------------------------------------
# Dataset
# --------------------------------------------------
with st.expander("📋 View Student Dataset"):
    st.dataframe(df, use_container_width=True)