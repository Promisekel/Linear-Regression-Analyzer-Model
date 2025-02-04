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

if uploaded_file is not None:
    # Read the uploaded dataset
    df = pd.read_csv(uploaded_file)

    # Handling missing data
    st.sidebar.header("🧹 Handle Missing Data")
    missing_option = st.sidebar.selectbox("Choose Missing Data Handling Method", ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"])

    if missing_option == "Drop Rows":
        df = df.dropna()
    elif missing_option == "Fill with Mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif missing_option == "Fill with Median":
        df = df.fillna(df.median(numeric_only=True))
    elif missing_option == "Fill with Mode":
        df = df.apply(lambda col: col.fillna(col.mode()[0]) if col.isnull().any() else col)

    # Sidebar: Data Manipulation
    st.sidebar.header("🔍 Data Manipulation")
    if st.sidebar.checkbox("Show Raw Data"):
        edited_df = st.data_editor(df)
        st.write(edited_df)
    else:
        edited_df = df

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
            edited_df = pd.get_dummies(edited_df, columns=categorical_vars, drop_first=True)
            st.success(f"Successfully converted {', '.join(categorical_vars)} to dummy variables.")
        except Exception as e:
            st.error(f"Error converting to dummies: {e}")

    # Variable Selection
    st.sidebar.header("📊 Variable Selection")
    target_var = st.sidebar.selectbox("Select Target Variable", edited_df.columns)
    feature_vars = st.sidebar.multiselect("Select Feature Variables", edited_df.columns.difference([target_var]))

    if feature_vars and target_var:
        X = edited_df[feature_vars]
        y = edited_df[target_var]

        # Drop rows with NaN values after manipulation
        X = X.dropna()
        y = y.loc[X.index]  # Align y with X after dropping NaNs

        # Check for empty data after cleaning
        if X.empty or y.empty:
            st.error("The dataset is empty after cleaning. Please adjust the data or handling options.")
        else:
            # Model Selection
            st.sidebar.header("🤖 Model Selection")
            model_type = st.sidebar.radio("Choose Model", ["Linear Regression"])

            if st.sidebar.button("Train Model"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Ensure numeric data for OLS model
                X_train = X_train.apply(pd.to_numeric, errors='coerce')
                y_train = pd.to_numeric(y_train, errors='coerce')
                X_train = sm.add_constant(X_train)  # adding a constant

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

                        st.subheader("📋 Model Parameters")
                        param_data = {
                            "Residual Standard Error": [f"{model.bse[0]:.2f} on {model.df_resid} degrees of freedom"],
                            "Multiple R-squared": [f"{model.rsquared:.4f}"],
                            "Adjusted R-squared": [f"{model.rsquared_adj:.4f}"],
                            "F-statistic": [f"{model.fvalue:.2f} on {model.df_model} and {model.df_resid} DF"],
                            "p-value": [f"{model.f_pvalue:.4e}"]
                        }
                        st.table(pd.DataFrame(param_data))

                        st.subheader("📊 Model Visualizations")
                        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

                        residuals = model.resid
                        ax[0].scatter(model.fittedvalues, residuals, color='blue', edgecolors='black')
                        ax[0].axhline(y=0, color='red', linestyle='--')
                        ax[0].set_xlabel('Fitted Values')
                        ax[0].set_ylabel('Residuals')
                        ax[0].set_title('Residuals vs Fitted Values')

                        sns.histplot(residuals, kde=True, color='purple', ax=ax[1])
                        ax[1].set_title('Histogram of Residuals')

                        st.pyplot(fig)

                        fig = sm.qqplot(residuals, line='45')
                        st.pyplot(fig)
                        st.write("- **Q-Q Plot:** Assesses if residuals follow a normal distribution.")

                        coefficients = model.params[1:]
                        feature_names = X_train.columns[1:]
                        fig = px.bar(x=feature_names, y=coefficients, labels={'x': 'Features', 'y': 'Coefficients'}, title='Feature Importance')
                        st.plotly_chart(fig)
                        st.write("- **Feature Importance:** Displays the influence of each feature on the target variable.")
                    except Exception as e:
                        st.error(f"Model training error: {e}")