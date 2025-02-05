\import streamlit as st
import pandas as pd
import plotly.express as px
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

    # Data Type Conversion
    st.sidebar.header("🔄 Convert Data Type")
    variable_to_convert = st.sidebar.selectbox("Select Variable to Convert", edited_df.columns)
    desired_dtype = st.sidebar.selectbox("Select Desired Data Type", ["int", "float", "str", "category"])
    if st.sidebar.button("Convert Data Type"):
        try:
            if desired_dtype == "int":
                edited_df[variable_to_convert] = edited_df[variable_to_convert].astype(int)
            elif desired_dtype == "float":
                edited_df[variable_to_convert] = edited_df[variable_to_convert].astype(float)
            elif desired_dtype == "category":
                edited_df[variable_to_convert] = edited_df[variable_to_convert].astype('category')
            else:
                edited_df[variable_to_convert] = edited_df[variable_to_convert].astype(str)
            st.success(f"Successfully converted {variable_to_convert} to {desired_dtype}.")
        except Exception as e:
            st.error(f"Error converting data type: {e}")

    # Convert Categorical to Dummies
    st.sidebar.header("🗂️ Convert Categorical Variables")
    categorical_vars = st.sidebar.multiselect("Select Categorical Variables to Convert to Dummies", edited_df.select_dtypes(include=['category', 'object']).columns)
    if st.sidebar.button("Convert to Dummies"):
        try:
            edited_df = pd.get_dummies(edited_df, columns=categorical_vars, drop_first=True, dtype='float64')
            st.session_state.edited_df = edited_df  # Save to session state
            st.success(f"Successfully converted {', '.join(categorical_vars)} to dummy variables.")
        except Exception as e:
            st.error(f"Error converting to dummies: {e}")

    # Data Visualization
    st.sidebar.header("📈 Data Visualization")
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Scatter Plot", "Bar Chart", "Histogram", "Box Plot"])
    x_var = st.sidebar.selectbox("Select X-axis Variable", edited_df.columns)
    y_var = st.sidebar.selectbox("Select Y-axis Variable", edited_df.columns)

    if st.sidebar.button("Generate Plot"):
        try:
            color_palette = px.colors.qualitative.Alphabet
            if plot_type == "Scatter Plot":
                fig = px.scatter(edited_df, x=x_var, y=y_var, color_discrete_sequence=color_palette, title=f"Scatter Plot of {y_var} vs {x_var}")
            elif plot_type == "Bar Chart":
                fig = px.bar(edited_df, x=x_var, y=y_var, color_discrete_sequence=color_palette, title=f"Bar Chart of {y_var} by {x_var}")
            elif plot_type == "Histogram":
                fig = px.histogram(edited_df, x=x_var, color_discrete_sequence=color_palette, title=f"Histogram of {x_var}")
            elif plot_type == "Box Plot":
                fig = px.box(edited_df, x=x_var, y=y_var, color_discrete_sequence=color_palette, title=f"Box Plot of {y_var} by {x_var}")

            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error generating plot: {e}")
