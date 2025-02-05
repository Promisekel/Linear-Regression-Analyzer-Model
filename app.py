import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Streamlit app configuration
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
st.title("📊 Diabetes Data Dashboard")

# File uploader for dataset
st.sidebar.header("📤 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Session state to preserve dummy variables
if ('edited_df' not in st.session_state):
    st.session_state.edited_df = None

if uploaded_file is not None:
    # Read the uploaded dataset
    df = pd.read_csv(uploaded_file)
    if st.session_state.edited_df is None:
        st.session_state.edited_df = df
    edited_df = st.session_state.edited_df

    # Handling missing data
    st.sidebar.header("🧹 Handle Missing Data")
    missing_option = st.sidebar.selectbox("Choose Missing Data Handling Method", ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"])

    if missing_option == "Drop Rows":
        edited_df = edited_df.dropna()
    elif missing_option == "Fill with Mean":
        edited_df = edited_df.fillna(edited_df.mean(numeric_only=True))
    elif missing_option == "Fill with Median":
        edited_df = edited_df.fillna(edited_df.median(numeric_only=True))
    elif missing_option == "Fill with Mode":
        edited_df = edited_df.apply(lambda col: col.fillna(col.mode()[0]) if col.isnull().any() else col)

    # Sidebar: Data Manipulation
    st.sidebar.header("🔍 Data Manipulation")
    if st.sidebar.checkbox("Show Raw Data"):
        st.write(edited_df)

    # Data Editing
    st.subheader("🖊️ Edit Data")
    edited_df = st.data_editor(edited_df)
    st.session_state.edited_df = edited_df

    # Distinct Value Distribution Visualization
    st.subheader("📊 Distinct Value Distribution")
    column_to_visualize = st.selectbox("Select Column to Visualize", edited_df.columns)
    if edited_df[column_to_visualize].dtype in ['int64', 'float64']:
        fig = px.histogram(edited_df, x=column_to_visualize, nbins=20, title=f'Distribution of {column_to_visualize}')
    else:
        value_counts = edited_df[column_to_visualize].value_counts()
        fig = px.bar(value_counts, x=value_counts.index, y=value_counts.values, title=f'Distribution of {column_to_visualize}')
    st.plotly_chart(fig, use_container_width=True)

    # Variable Selection
    st.sidebar.header("📊 Variable Selection")
    target_var = st.sidebar.selectbox("Select Target Variable", edited_df.columns)
    feature_vars = st.sidebar.multiselect("Select Feature Variables", edited_df.columns.difference([target_var]))

    if feature_vars and target_var:
        X = edited_df[feature_vars]
        y = edited_df[target_var]

        # Convert all feature variables to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        # Drop rows with NaN values after manipulation
        combined_data = pd.concat([X, y], axis=1).dropna()
        X = combined_data[feature_vars]
        y = combined_data[target_var]

        # Check for empty data after cleaning
        if X.empty or y.empty:
            st.error("The dataset is empty after cleaning. Please adjust the data or handling options.")
        else:
            # Model Selection
            st.sidebar.header("🤖 Model Selection")
            model_type = st.sidebar.radio("Choose Model", ["Linear Regression"])

            if st.sidebar.button("Train Model"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                X_train = sm.add_constant(X_train)  # adding a constant

                # Ensure data is numeric
                X_train = X_train.apply(pd.to_numeric, errors='coerce')
                y_train = pd.to_numeric(y_train, errors='coerce')

                # Drop any remaining NaNs after conversion
                valid_idx = X_train.dropna().index.intersection(y_train.dropna().index)
                X_train = X_train.loc[valid_idx]
                y_train = y_train.loc[valid_idx]

                if model_type == "Linear Regression" and y.dtypes not in ['int64', 'float64']:
                    st.warning("Target variable must be continuous for Linear Regression.")
                else:
                    try:
                        model = sm.OLS(y_train, X_train, missing='drop').fit()
                        results = model.summary2().tables[1]

                        st.subheader(f"{model_type} Model Summary")
                        st.write(results)
                    except Exception as e:
                        st.error(f"Model training error: {e}")
