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

    # Variable Selection
    st.sidebar.header("📊 Variable Selection")
    target_var = st.sidebar.selectbox("Select Target Variable", edited_df.columns)
    feature_vars = st.sidebar.multiselect("Select Feature Variables", edited_df.columns.difference([target_var]))

    if feature_vars and target_var:
        X = edited_df[feature_vars]
        y = edited_df[target_var]

        # Data Visualization
        st.header("📈 Data Visualization")
        chart_type = st.selectbox("Choose Chart Type", ["Histogram", "Boxplot", "Scatterplot", "Correlation Heatmap"])

        if chart_type == "Histogram":
            feature = st.selectbox("Select Feature", feature_vars)
            fig = px.histogram(edited_df, x=feature, color=target_var, barmode="overlay")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Boxplot":
            feature = st.selectbox("Select Feature", feature_vars)
            fig = px.box(edited_df, x=target_var, y=feature, color=target_var)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatterplot":
            x_var = st.selectbox("X-axis", feature_vars)
            y_var = st.selectbox("Y-axis", feature_vars)
            fig = px.scatter(edited_df, x=x_var, y=y_var, color=target_var)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Correlation Heatmap":
            corr = edited_df.corr()
            fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Blues')
            st.plotly_chart(fig, use_container_width=True)

        # Model Training
        st.sidebar.header("🤖 Model Training")

        if st.sidebar.button("Train Linear Regression Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Check if target variable is suitable for Linear Regression
            if y.dtypes not in ['int64', 'float64']:
                st.warning("Target variable must be continuous for Linear Regression.")
            else:
                # Using statsmodels for detailed output
                X_train_sm = sm.add_constant(X_train)  # adding a constant
                model = sm.OLS(y_train, X_train_sm).fit()

                # Extracting the results and formatting them
                results = model.summary2().tables[1]  # Extract coefficients, p-values, etc.

                # Display model output as a table
                st.subheader("Linear Regression Model Summary")
                st.write(results)

                # Model Parameters Table
                st.subheader("📋 Model Parameters")
                param_data = {
                    "Residual Standard Error": [f"{model.bse[0]:.2f} on {model.df_resid} degrees of freedom"],
                    "Multiple R-squared": [f"{model.rsquared:.4f}"],
                    "Adjusted R-squared": [f"{model.rsquared_adj:.4f}"],
                    "F-statistic": [f"{model.fvalue:.2f} on {model.df_model} and {model.df_resid} DF"],
                    "p-value": [f"{model.f_pvalue:.4e}"]
                }
                st.table(pd.DataFrame(param_data))

                # Visualizations
                st.subheader("📊 Model Visualizations")
                fig, ax = plt.subplots(1, 2, figsize=(14, 6))

                # Residuals plot
                residuals = model.resid
                ax[0].scatter(model.fittedvalues, residuals, color='blue', edgecolors='black')
                ax[0].axhline(y=0, color='red', linestyle='--')
                ax[0].set_xlabel('Fitted Values')
                ax[0].set_ylabel('Residuals')
                ax[0].set_title('Residuals vs Fitted Values')

                # Histogram of residuals
                sns.histplot(residuals, kde=True, color='purple', ax=ax[1])
                ax[1].set_title('Histogram of Residuals')

                st.pyplot(fig)

                # Explanations
                st.markdown("**Explanation:**")
                st.write("- **Residuals vs Fitted Values:** Helps to identify non-linearity, unequal error variances, and outliers.")
                st.write("- **Histogram of Residuals:** Shows the distribution of residuals to check for normality.")

                # Q-Q plot
                fig = sm.qqplot(residuals, line='45')
                st.pyplot(fig)
                st.write("- **Q-Q Plot:** Assesses if residuals follow a normal distribution.")

                # Feature importance plot
                coefficients = model.params[1:]  # Skip constant
                feature_names = X_train.columns
                fig = px.bar(x=feature_names, y=coefficients, labels={'x': 'Features', 'y': 'Coefficients'}, title='Feature Importance')
                st.plotly_chart(fig)
                st.write("- **Feature Importance:** Displays the influence of each feature on the target variable.")

