import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
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

            # Check if target variable is suitable for the selected model
            if model_type == "Linear Regression" and y.dtypes not in ['int64', 'float64']:
                st.warning("Target variable must be continuous for Linear Regression.")
            elif model_type == "Logistic Regression" and len(y.unique()) != 2:
                st.warning("Target variable must be binary for Logistic Regression.")
            else:
                # Using statsmodels for detailed output
                X_train_sm = sm.add_constant(X_train)  # adding a constant

                if model_type == "Linear Regression":
                    model = sm.OLS(y_train, X_train_sm).fit()
                else:
                    model = sm.Logit(y_train, X_train_sm).fit()

                # Extracting the results and formatting them
                results = model.summary2().tables[1]  # Extract the second table, which contains coefficients, p-values, etc.
                results = results[['Coef.', 'Std.Err.', 't', 'P>|t|', '[0.025', '0.975]']]

                # Display model output as a table
                st.subheader(f"{model_type} Model Summary")
                st.write(results)

                # Display additional details and explanation
                st.subheader("📑 Detailed Model Output Explanation")

                # R-squared (for Linear Regression)
                if model_type == "Linear Regression":
                    st.markdown("### R-squared:")
                    st.write(f"R-squared: {model.rsquared:.4f}")
                    st.write(f"R-squared represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A value closer to 1 indicates a better fit.")

                # Log-likelihood and AIC (for Logistic Regression)
                if model_type == "Logistic Regression":
                    st.markdown("### Log-Likelihood & AIC:")
                    st.write(f"Log-Likelihood: {model.llf:.4f}")
                    st.write(f"AIC (Akaike Information Criterion): {model.aic:.4f}")
                    st.write(f"Log-Likelihood measures how well the model fits the data. A higher value indicates a better fit. AIC is used for model comparison, with lower values indicating better models.")

                # Coefficients and Significance
                st.markdown("### Coefficients and Significance:")
                st.write(f"Coefficients represent the change in the target variable for a one-unit change in the corresponding feature. If the p-value is less than 0.05, the variable is considered statistically significant.")

                # Confidence Interval
                st.markdown("### Confidence Intervals:")
                st.write(f"The 95% confidence intervals for the coefficients provide a range in which the true coefficient is likely to fall. A wider interval suggests more uncertainty in the estimate.")

                # P-values
                st.markdown("### P-values:")
                st.write(f"P-values are used to determine the statistical significance of each variable. A p-value below 0.05 generally indicates that the feature has a significant effect on the target variable.")

        # Download filtered dataset
        st.sidebar.header("📥 Download Processed Data")
        st.sidebar.download_button("Download Data", df.to_csv(index=False), file_name="processed_data.csv")
