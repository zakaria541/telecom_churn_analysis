# Telecom Customer Churn Prediction

## Project Overview
Customer churn is a critical challenge in the telecom industry due to high competition and low switching costs. This project builds and evaluates multiple machine learning models to predict which customers are likely to churn, enabling telecom providers to take proactive retention measures.

## Dataset
- **Source:** Telecom Customer Churn dataset
- **Features:** Customer demographics, service details, and usage patterns (e.g. Gender, Tenure, Monthly Charges, Contract Type)
- **Target Variable:** `Churn` — whether a customer cancelled their service

## Tools & Technologies
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

## Methodology

### Data Preprocessing
- Handled missing values using mean/mode imputation
- Encoded categorical variables (Gender, Contract Type) using Label Encoding
- Applied StandardScaler for feature normalisation
- Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)

### Models Built
| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|---|---|---|---|---|
| Logistic Regression | 81.47% | 0.68 | 0.56 | 0.61 |
| Decision Tree | 79.49% | 0.61 | 0.63 | 0.62 |
| Random Forest | 79.70% | 0.66 | 0.48 | 0.56 |

### Model Evaluation
- ROC-AUC Score (Logistic Regression): **0.86**
- Hyperparameter tuning via GridSearchCV applied to Decision Tree
- Overfitting and underfitting analysis conducted across all models

## Key Findings
- Logistic Regression achieved the highest accuracy (81.47%) and AUC (0.86), making it the strongest overall model
- Decision Tree had the best recall (63%), making it more suitable when the business priority is capturing as many churners as possible
- Random Forest suffered the most from class imbalance, with a recall of only 48% for churned customers
- K-Means Clustering did not produce meaningful customer segments due to mixed data types and high dimensionality

## Suggested Improvements
- Apply SMOTE or cost-sensitive learning to better handle class imbalance
- Engineer new features (e.g. tenure-to-charge ratio, customer engagement scores)
- Experiment with XGBoost or LightGBM for improved performance on imbalanced data
- Apply PCA before clustering to reduce dimensionality

## Files
| File | Description |
|---|---|
| `telecom_churn_analysis.ipynb` | Full Jupyter Notebook with code and outputs |
| `telecom_churn.csv` | Telecom Customer Churn dataset |
| `telecom_churn_analysis.pdf` | Full project report |

## Author
**Zakaria Gou-Ali**  
MSc Business Analytics — University of Greenwich  
[LinkedIn](https://www.linkedin.com/in/zakaria-gou-ali-896175347/)
