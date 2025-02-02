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

    # Sidebar: Data Manipulation
    st.sidebar.header("🔍 Data Manipulation")
    if st.sidebar.checkbox("Show and Edit Raw Data"):
        st.subheader("📝 Editable Dataset")

        # Editable data frame
        edited_df = st.data_editor(df, num_rows="dynamic")

        # Option to download the edited dataset
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Edited CSV",
            data=csv,
            file_name='edited_data.csv',
            mime='text/csv'
        )

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
            st.markdown("**Explanation:** This histogram shows the distribution of the selected feature, helping identify skewness, outliers, and frequency patterns.")

        elif chart_type == "Boxplot":
            feature = st.selectbox("Select Feature", feature_vars)
            fig = px.box(df, x=target_var, y=feature, color=target_var)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Explanation:** The boxplot visualizes the spread and central tendency of data, highlighting outliers and comparing distributions across categories.")

        elif chart_type == "Scatterplot":
            x_var = st.selectbox("X-axis", feature_vars)
            y_var = st.selectbox("Y-axis", feature_vars)
            fig = px.scatter(df, x=x_var, y=y_var, color=target_var)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Explanation:** The scatterplot reveals relationships between two variables, helping identify trends, clusters, and potential correlations.")

        elif chart_type == "Correlation Heatmap":
            corr = df.corr()
            fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Explanation:** The correlation heatmap displays pairwise correlations between variables. Darker shades indicate stronger relationships.")

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
                results = model.summary2().tables[1]
                results = results[['Coef.', 'Std.Err.', 't', 'P>|t|', '[0.025', '0.975]']]

                # Display model output as a table
                st.subheader(f"{model_type} Model Summary")
                st.write(results)

                # Display residual statistics for Linear Regression
                if model_type == "Linear Regression":
                    st.markdown("### Residual Statistics")
                    residuals = model.resid
                    residual_stats = {
                        "Min": residuals.min(),
                        "1Q": residuals.quantile(0.25),
                        "Median": residuals.median(),
                        "3Q": residuals.quantile(0.75),
                        "Max": residuals.max()
                    }
                    st.write(pd.DataFrame(residual_stats, index=["Residuals"]))

                    # Visualizations
                    st.markdown("### Visualizations")
                    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

                    # Residuals plot
                    ax[0].scatter(model.fittedvalues, residuals, color='blue', edgecolors='black')
                    ax[0].axhline(y=0, color='red', linestyle='--')
                    ax[0].set_xlabel('Fitted Values')
                    ax[0].set_ylabel('Residuals')
                    ax[0].set_title('Residuals vs Fitted Values')
                    st.pyplot(fig)
                    st.markdown("**Explanation:** This plot checks model assumptions by showing if residuals are randomly dispersed around zero.")

                    # Histogram of residuals
                    fig, ax = plt.subplots()
                    sns.histplot(residuals, kde=True, color='purple', ax=ax)
                    ax.set_title('Histogram of Residuals')
                    st.pyplot(fig)
                    st.markdown("**Explanation:** This histogram helps verify if residuals are normally distributed, which is key for model accuracy.")

                    # Q-Q plot
                    fig = sm.qqplot(residuals, line='45')
                    st.pyplot(fig)
                    st.markdown("**Explanation:** The Q-Q plot compares the distribution of residuals to a normal distribution. Points close to the line indicate normality.")

                    # Feature importance plot
                    coefficients = model.params[1:]
                    feature_names = X_train.columns
                    fig, ax = plt.subplots()
                    sns.barplot(x=coefficients.values, y=feature_names, palette='coolwarm', ax=ax)
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)
                    st.markdown("**Explanation:** This bar plot shows the relative importance of each feature in predicting the target variable.")
