# House Price Prediction: Advanced Regression Techniques

This project is a comprehensive solution for the Kaggle competition "House Prices: Advanced Regression Techniques". The primary goal is to predict the final sale price of residential homes in Ames, Iowa, using a dataset with 79 explanatory variables.

## 🎯 Project Goal

* To perform in-depth **Exploratory Data Analysis (EDA)** to uncover patterns, outliers, and relationships within the dataset.
* To execute extensive **data preprocessing and feature engineering**, which is critical for this competition due to the large number of features and missing values.
* To train and **compare several regression models**, from simple linear models to powerful gradient boosting ensembles.
* To select the best-performing model and evaluate it using the competition's specific metric, Root Mean Squared Logarithmic Error (RMSLE).

## 💾 Dataset

The data comes directly from the Kaggle competition and is available [at this link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). It includes a wide range of features describing various aspects of residential homes.

```
├── data/                  # Raw train.csv and test.csv files
├── lasso-model/           # Saved Lasso Regression model
├── lightgbm-model/        # Saved LightGBM model
├── random-forest-model/   # Saved Random Forest model
├── xgboost-model/         # Saved XGBoost model
├── data_description.txt   # Description of each feature in the dataset
├── requirements.txt       # List of Python dependencies to install
└── visualizations.ipynb   # Jupyter Notebook with EDA, feature engineering, and modeling
```

## ⚙️ Methodology

The solution was developed through the following key stages:

1.  **Exploratory Data Analysis (EDA):** The analysis in `visualizations.ipynb` focused on understanding the target variable (`SalePrice`), identifying feature distributions, and handling data skewness and outliers.
2.  **Data Preprocessing & Feature Engineering:** This was the most critical phase. Key steps included:
    * Log-transforming the skewed `SalePrice` target variable.
    * Handling a large number of missing values using various strategies (e.g., imputation with mean, median, or "None").
    * Transforming numerical features into categorical ones (e.g., `MSSubClass`).
    * Encoding categorical features using methods like One-Hot Encoding.
3.  **Model Training:** Four different regression models were trained to find the most accurate algorithm for this problem.
4.  **Evaluation:** Models were compared based on the **Root Mean Squared Logarithmic Error (RMSLE)**, as required by the competition.

## 🤖 Models Used

The following regression algorithms were implemented and evaluated:
* Lasso Regression
* Random Forest Regressor
* XGBoost (eXtreme Gradient Boosting)
* LightGBM (Light Gradient Boosting Machine)

## 📊 Results

The performance of each model was evaluated on the validation set. Gradient boosting models demonstrated superior performance, which is typical for complex tabular datasets.

| Model              | RMSLE (Lower is Better) |
| :----------------- | :---------------------: |
| Lasso Regression   |         ~0.13           |
| Random Forest      |         ~0.14           |
| XGBoost            |         ~0.12           |
| LightGBM           |         ~0.13           |

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/jakubkos11/house-pricing-predictor.git](https://github.com/jakubkos11/house-pricing-predictor.git)
    cd house-pricing-predictor
    ```
2.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  Open and run the `visualizations.ipynb` notebook in a Jupyter environment.

## 🛠️ Technologies

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost & LightGBM
* Matplotlib & Seaborn
* Jupyter Notebook
* Optuna
