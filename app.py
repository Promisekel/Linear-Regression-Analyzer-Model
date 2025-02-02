import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Streamlit app configuration
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
st.title("📊 Diabetes Data Dashboard")

# File uploader for dataset
st.sidebar.header("📤 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded dataset
    df = pd.read_csv(uploaded_file)

    # Sidebar: Data Manipulation
    st.sidebar.header("🔍 Data Manipulation")
    if st.sidebar.checkbox("Show Raw Data"):
        st.write(df)

    # Variable Selection
    st.sidebar.header("📊 Variable Selection")
    target_var = st.sidebar.selectbox("Select Target Variable", df.columns)
    feature_vars = st.sidebar.multiselect("Select Feature Variables", df.columns.difference([target_var]))

    if feature_vars and target_var:
        X = df[feature_vars]
        y = df[target_var]

        # Data Visualization
        st.header("📈 Data Visualization")
        chart_type = st.selectbox("Choose Chart Type", ["Histogram", "Boxplot", "Scatterplot", "Correlation Heatmap"])

        if chart_type == "Histogram":
            feature = st.selectbox("Select Feature", feature_vars)
            fig = px.histogram(df, x=feature, color=target_var, barmode="overlay")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Boxplot":
            feature = st.selectbox("Select Feature", feature_vars)
            fig = px.box(df, x=target_var, y=feature, color=target_var)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatterplot":
            x_var = st.selectbox("X-axis", feature_vars)
            y_var = st.selectbox("Y-axis", feature_vars)
            fig = px.scatter(df, x=x_var, y=y_var, color=target_var)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Correlation Heatmap":
            corr = df.corr()
            fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Blues')
            st.plotly_chart(fig, use_container_width=True)

        # Model Selection
        st.sidebar.header("🤖 Model Selection")
        model_type = st.sidebar.radio("Choose Model", ["Linear Regression", "Logistic Regression"])

        if st.sidebar.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                st.subheader("Linear Regression Results")
                st.write(f"R-squared: {r2_score(y_test, predictions):.2f}")
                st.write(f"MSE: {mean_squared_error(y_test, predictions):.2f}")
                
                fig = px.scatter(x=y_test, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'})
                st.plotly_chart(fig, use_container_width=True)

            elif model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=200)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                st.subheader("Logistic Regression Results")
                st.write(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, predictions))

        # Download filtered dataset
        st.sidebar.header("📥 Download Processed Data")
        st.sidebar.download_button("Download Data", df.to_csv(index=False), file_name="processed_data.csv")
