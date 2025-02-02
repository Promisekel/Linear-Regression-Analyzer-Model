import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Streamlit app configuration
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
st.title("📊 Diabetes Data Dashboard")

# File uploader for dataset
st.sidebar.header("📤 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded dataset
    df = pd.read_csv(uploaded_file)
    
    # Ensure the correct column names for consistency
    df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']

    # Sidebar filters
    st.sidebar.header("🔍 Filter Data")
    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_filter = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))
    df_filtered = df[(df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1])]

    # Create card layout for data summary, feature distribution, and correlation heatmap
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Data Summary"):
            st.write(df_filtered.describe())

    with col2:
        if st.button("📊 Feature Distributions"):
            feature = st.selectbox("Select feature", df_filtered.columns[:-1])
            fig = px.histogram(df_filtered, x=feature, color="Outcome", barmode="overlay")
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if st.button("📉 Correlation Heatmap"):
            numeric_columns = df_filtered.select_dtypes(include=['number']).columns
            corr = df_filtered[numeric_columns].corr()
            fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Blues')
            st.plotly_chart(fig, use_container_width=True)

    # Regression Section
    st.sidebar.header("📊 Regression Model")
    model_type = st.sidebar.selectbox("Select Model Type", ["Linear Regression", "Logistic Regression"])
    
    target_variable = st.sidebar.selectbox("Select Target Variable", df.columns)
    feature_selection = st.sidebar.multiselect("Select Features", options=df.columns.difference([target_variable]), default=df.columns[:4])

    if st.sidebar.button("Train Regression Model"):
        X = df[feature_selection]
        y = df[target_variable]

        if model_type == "Linear Regression":
            if y.dtypes != 'int64' and y.dtypes != 'float64':
                st.sidebar.error("Target variable must be numeric for Linear Regression.")
            else:
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                st.subheader("Full Linear Regression Model Summary")
                st.write(model.summary())

        elif model_type == "Logistic Regression":
            if y.nunique() != 2:
                st.sidebar.error("Target variable must be binary for Logistic Regression.")
            else:
                model = LogisticRegression(max_iter=200)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.subheader("Full Logistic Regression Model Summary")
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.write("Coefficients:", pd.Series(model.coef_[0], index=feature_selection))
                st.write(f"Intercept: {model.intercept_[0]:.2f}")

    # Additional visualizations
    st.sidebar.header("📌 More Insights")
    option = st.sidebar.radio("Select a visualization:", ["Boxplot", "Diabetes Count", "Age Distribution", "Pair Plot"])

    if option == "Boxplot":
        feature = st.sidebar.selectbox("Choose feature", df_filtered.columns[:-1])
        fig = px.box(df_filtered, x="Outcome", y=feature, color="Outcome")
        st.plotly_chart(fig, use_container_width=True)

    elif option == "Diabetes Count":
        fig = px.bar(df_filtered["Outcome"].value_counts(), x=df_filtered["Outcome"].unique(), y=df_filtered["Outcome"].value_counts(), color=df_filtered["Outcome"].unique())
        st.plotly_chart(fig, use_container_width=True)

    elif option == "Age Distribution":
        fig = px.histogram(df_filtered, x="Age", color="Outcome", nbins=20, barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

    elif option == "Pair Plot":
        fig = px.scatter_matrix(df_filtered, dimensions=['Glucose', 'BloodPressure', 'BMI', 'Age'], color='Outcome')
        st.plotly_chart(fig, use_container_width=True)

    # Download option
    st.sidebar.header("📥 Download Data")
    st.sidebar.download_button("Download Filtered Data", df_filtered.to_csv(index=False), "filtered_data.csv", "text/csv")
else:
    st.write("Please upload a CSV file to start.")
