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
st.set_page_config(page_title="LINEAR REGRESSION ANALYzER MODEL", layout="wide")
st.title("📊 LINEAR REGRESSION ANALYZER MODEL")

# File uploader for dataset
st.sidebar.header("📤 Upload Your Dataset")
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
    st.sidebar.header("🧹 HANDLE MISSING DATA")
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
    st.sidebar.header("🔍 DATA MANIPULATION")
    if st.sidebar.checkbox("Show Raw Data"):
        edited_df = st.data_editor(edited_df, num_rows="dynamic")  # Editable data frame
        st.session_state.edited_df = edited_df  # Save changes

    # Data Type Conversion
    st.sidebar.header("🔄 CONVERT DATA TYPE")
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
    st.sidebar.header("🗂️ CONVERT CATEGORICAL VARIABLE")
    categorical_vars = st.sidebar.multiselect("Select Categorical Variables to Convert to Dummies", edited_df.select_dtypes(include=['category', 'object']).columns)
    if st.sidebar.button("Convert to Dummies"):
        try:
            edited_df = pd.get_dummies(edited_df, columns=categorical_vars, drop_first=True, dtype='float64')
            st.session_state.edited_df = edited_df  # Save to session state
            st.success(f"Successfully converted {', '.join(categorical_vars)} to dummy variables.")
        except Exception as e:
            st.error(f"Error converting to dummies: {e}")

    # Python Coding Environment
    #st.sidebar.header("💻 Python Coding Environment")
    #code_input = st.sidebar.text_area("Write Python Code to Manipulate Data", height=200)
    #if st.sidebar.button("Run Code"):
    #    try:
            # Safe execution of user code
     #       local_vars = {'edited_df': edited_df, 'pd': pd, 'np': np}
    #        exec(code_input, {}, local_vars)
    #        edited_df = local_vars['edited_df']
    #       st.session_state.edited_df = edited_df
    #        st.success("Code executed successfully.")
    #    except Exception as e:
    #        st.error(f"Error executing code: {e}")

    # Variable Selection
    st.sidebar.header("📊 SELECT VARAIBLE FOR REGRESSION ANALYSIS")
    target_var = st.sidebar.selectbox("Select Outcome Variable", edited_df.columns)
    feature_vars = st.sidebar.multiselect("Select Explanatory Variables", edited_df.columns.difference([target_var]))

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

        # Explicit conversion to float64 and removal of non-numeric columns
        X = X.astype('float64', errors='ignore')
        y = y.astype('float64', errors='ignore')
        X = X.select_dtypes(include=[np.number])

        # Check for empty data after cleaning
        if X.empty or y.empty:
            st.error("The dataset is empty after cleaning. Please adjust the data or handling options.")
        else:
            # Model Selection
            st.sidebar.header("🤖 MODEL")
            model_type = st.sidebar.radio("Choose Model", ["Linear Regression"])

            if st.sidebar.button("Run Model"):
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

                        # Residuals vs Fitted
                        residuals = model.resid
                        fitted_values = model.fittedvalues
                        ax[0].scatter(fitted_values, residuals)
                        ax[0].axhline(0, color='red', linestyle='--')
                        ax[0].set_xlabel('Fitted values')
                        ax[0].set_ylabel('Residuals')
                        ax[0].set_title('Residuals vs Fitted')

                        # Q-Q Plot
                        sm.qqplot(residuals, line='s', ax=ax[1])
                        ax[1].set_title('Normal Q-Q')

                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error in model training: {e}")
    # Footer
st.sidebar.markdown("---")
st.sidebar.write("Author: [Promise Bansah](https://github.com/Promisekel)")