# 34222025_Churning-Customers
# Customer Churning Prediction Project

## Overview
This project focuses on predicting customer churn using machine learning techniques. 
It involves data processing, exploratory data analysis (EDA), feature selection, and the training of a multi-layer perceptron (MLP) model.
The Churning Project aims to identify and analyze the factors that contribute to customer churn, or the loss of customers over time.
By understanding the reasons why customers churn, businesses can develop strategies to retain existing customers and improve customer lifetime value.


## Libraries Used
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
- Keras
- Joblib
- Missingno
- NumPy

## Getting Started
1. Install required libraries: `pip install scikeras`
2. Mount Google Drive: Follow the instructions in the notebook for mounting Google Drive if using Colab.
3. Execute the code cells in the notebook sequentially.

## Project Structure
- **Data Processing:** Loading the dataset, inspecting initial rows, handling missing values, dropping unnecessary columns, and label encoding.
- **Feature Selection:** Using a RandomForestClassifier to select important features.
- **Exploratory Data Analysis (EDA):** Analyzing demographic information, senior citizenship, partner and dependents, service-related features, contract and payment analysis, financial metrics, tenure analysis, and visualizing the correlation matrix.
- **Scatter Plots and Pie Chart:** Adding scatter plots of Monthly Charges vs. Tenure, Total Charges vs. Tenure, and a pie chart for Churn distribution.
- **Scaling and Splitting:** Scaling features and splitting the dataset into training, testing, and validation sets.
- **MLP Model Training:** Creating and training an MLP model using the Keras Functional API. Performing hyperparameter tuning using GridSearchCV.
- **Evaluation:** Evaluating the model on the validation and test sets. Displaying accuracy, AUC score, and classification reports.
- **Model Optimization and Saving:** Optimizing the model using the best hyperparameters and saving the model and scaler using Joblib.

## Results
- The optimized MLP model achieved a test accuracy of 81.16% and an AUC score of 0.8528.

## Files
- `customer_churn_prediction.ipynb`: Jupyter notebook containing the entire code.
- `optimized_model.pkl`: Saved optimized MLP model.
- `scaler.pkl`: Saved StandardScaler.

## Acknowledgments
- This project is based on the Customer Churning dataset.

## Link to video
https://drive.google.com/drive/folders/15E7mGPeQxftMj6y_z0xAz4NeTPQYC3YW?usp=sharing
