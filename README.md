# Linear Regression Analyzer Model

## 📊 Overview

The **Linear Regression Analyzer Model** is an interactive Streamlit web application designed for comprehensive data analysis using linear regression techniques. It provides users with tools for data preprocessing, visualization, statistical modeling, and detailed regression analysis, all within an intuitive and user-friendly interface.

## 🚀 Features

- **Data Upload:** Import CSV files directly into the app for analysis.
- **Data Cleaning:** Handle missing values by dropping rows or filling them with mean, median, or mode.
- **Data Editing:** Edit raw data directly within the app.
- **Data Manipulation:** Convert data types and transform categorical variables into dummy variables.
- **Python Coding Environment:** Execute custom Python code to manipulate data dynamically.
- **Variable Selection:** Choose target and feature variables for regression analysis.
- **Model Training:** Apply linear regression models to your dataset.
- **Model Evaluation:** Display model summaries, including coefficients, R-squared values, F-statistics, and p-values.
- **Visualizations:** Generate diagnostic plots such as Residuals vs. Fitted and Q-Q plots.

## 📦 Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd linear-regression-analyzer
   ```

3. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

1. **Upload your dataset (CSV format).**
2. **Handle missing data** via the sidebar options.
3. **Edit raw data** directly in the app if needed.
4. **Manipulate data types** and convert categorical variables into dummies.
5. **Write and execute Python code** to manipulate data dynamically.
6. **Select your target and feature variables** for analysis.
7. **Train the model** and review detailed statistical summaries.
8. **Visualize the model performance** using residual and Q-Q plots.

## ⚙️ Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Plotly
- scikit-learn

Install all dependencies with:
```bash
pip install streamlit pandas numpy matplotlib seaborn statsmodels plotly scikit-learn
```

## 📈 Model Outputs

- **Model Summary:** Regression coefficients, standard errors, t-values, p-values, R-squared, Adjusted R-squared, F-statistic.
- **Diagnostic Plots:**
  - Residuals vs. Fitted plot for checking linearity.
  - Q-Q plot for assessing normality of residuals.

## 📝 Contribution

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments

- Inspired by data science best practices.
- Built with Streamlit, Statsmodels, and Python's powerful data libraries.

## 💬 Contact

For questions, suggestions, or contributions, feel free to reach out at [your-email@example.com].

