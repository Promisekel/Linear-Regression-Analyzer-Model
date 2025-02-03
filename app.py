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

    # Feature Engineering
    st.sidebar.header("⚙️ Feature Engineering")
    if st.sidebar.checkbox("Create Interaction Term"):
        interaction_feature1 = st.sidebar.selectbox("Select First Feature", edited_df.columns)
        interaction_feature2 = st.sidebar.selectbox("Select Second Feature", edited_df.columns)
        interaction_name = f"{interaction_feature1}_x_{interaction_feature2}"
        edited_df[interaction_name] = edited_df[interaction_feature1] * edited_df[interaction_feature2]
        st.write(f"Interaction term '{interaction_name}' added to the dataset.")

    # Descriptive and Summary Statistics
    st.sidebar.header("📋 Descriptive Statistics")
    selected_var = st.sidebar.selectbox("Select Variable", edited_df.columns)
    stat_type = st.sidebar.selectbox("Select Statistics Type", ["Summary Statistics", "Descriptive Statistics"])

    if stat_type == "Summary Statistics":
        st.subheader("Summary Statistics")
        st.table(edited_df[selected_var].describe().to_frame())
    elif stat_type == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        stats_data = {
            "Mean": edited_df[selected_var].mean(),
            "Median": edited_df[selected_var].median(),
            "Mode": edited_df[selected_var].mode()[0] if not edited_df[selected_var].mode().empty else "N/A",
            "Standard Deviation": edited_df[selected_var].std(),
            "Variance": edited_df[selected_var].var(),
            "Min": edited_df[selected_var].min(),
            "Max": edited_df[selected_var].max()
        }
        st.table(pd.DataFrame(stats_data.items(), columns=["Statistic", "Value"]))

    # Section for Generating Descriptive Statistics
    st.sidebar.header("📊 Generate Descriptive Statistics")
    descriptive_vars = st.sidebar.multiselect("Select Variables for Descriptive Statistics", edited_df.columns)
    if st.sidebar.button("Generate Statistics"):
        descriptive_stats = edited_df[descriptive_vars].describe().T
        descriptive_stats["Mode"] = [edited_df[var].mode()[0] if not edited_df[var].mode().empty else "N/A" for var in descriptive_vars]

        # Beautify the table
        st.subheader("Generated Descriptive Statistics")
        styled_table = descriptive_stats.style.set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#f4f4f4'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('padding', '8px'), ('border', '1px solid #ddd')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]}
        ]).format(precision=2)

        st.dataframe(styled_table, use_container_width=True)

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
                results = model.summary2().tables[1]  # Extract coefficients, p-values, etc.

                # Display model output as a table
                st.subheader(f"{model_type} Model Summary")
                st.write(results)

                # Model Parameters Table
                st.subheader("📋 Model Parameters")
                if model_type == "Linear Regression":
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
