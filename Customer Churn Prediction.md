# Customer Churn Prediction

## Project Description

This project aims to predict customer churn in the telecommunications industry using the **Telco Customer Churn** dataset. The dataset contains demographic and account information of telecom customers. The objective is to build a predictive model that can identify customers likely to churn and provide insights into how to retain valuable customers.

Dataset can be found at [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Key Objectives
- **Data Exploration and Preprocessing**: Perform exploratory data analysis (EDA), clean the dataset by handling missing values, outliers, and encode categorical variables.
- **Feature Engineering**: Create new features that could improve model performance based on customer behaviors and account details.
- **Model Development**: Use machine learning models such as logistic regression, random forests, and XGBoost to predict customer churn.
- **Model Evaluation**: Evaluate the models using accuracy, precision, recall, F1-score, and ROC-AUC to ensure the robustness and reliability of predictions.
- **Insights and Recommendations**: Based on model predictions, provide actionable insights to reduce churn and retain customers.

## Tools and Technologies
- **Python Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Machine Learning Models**: Logistic Regression, Random Forest, XGBoost
- **Data Visualization**: Seaborn, Matplotlib
- **Environment**: Jupyter Notebooks

## Features
- **Data Preprocessing**: Cleaning, handling missing data, encoding categorical variables
- **Model Building**: Logistic Regression, Random Forest, XGBoost
- **Evaluation**: Cross-validation and performance metrics
- **Feature Engineering**: Creating features from the dataset
- **Prediction**: Predict churn for new observations

## Project Structure
```bash
├── data/
│   └── telco_customer_churn.csv  # Raw dataset
├── notebooks/
│   └── customer_churn_prediction.ipynb  # Jupyter notebook with project steps
├── requirements.txt  # List of dependencies
└── README.md  # Project description
```

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/customer_churn_prediction.ipynb
   ```

4. Follow the steps in the notebook to load the data, preprocess it, build models, and evaluate performance.

## Results

- **Logistic Regression**: Achieved an accuracy of **79.91%** with F1 Score of **0.58**.
- **Random Forest Classifier**: Achieved an accuracy of **77.57%** with F1 Score of **0.51**.
- **XGBoost Classifier**: Achieved an accuracy of **77.00%** with F1 Score of **0.53**.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.