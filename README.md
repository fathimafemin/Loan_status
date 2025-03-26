

# 🏦 Loan Eligibility Prediction Using Machine Learning

## Solution Description
The goal of this project is to automate loan eligibility prediction for **Dream Housing Finance** using machine learning. By analyzing customer data such as gender, marital status, income, loan amount, credit history, and property area, the model predicts whether a loan will be approved (`Y`) or not (`N`).

---

## Steps Taken:

### 1. Data Preprocessing:
- **Loaded the Dataset:** Imported data from `train.csv` and `test.csv`.
- **Dropped Loan ID:** Dropped the `Loan_ID` column as it is irrelevant to model training.
- **Handled Missing Values:**
    - Numerical columns (`LoanAmount`, `Loan_Amount_Term`, `ApplicantIncome`, `CoapplicantIncome`) were imputed using **median** values.
    - Categorical columns (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`) were imputed using **mode**.
- **Encoded Categorical Variables:** Converted categorical features into numerical format using `LabelEncoder`.
- **Scaled Numerical Features:** Applied `StandardScaler` to standardize numerical columns for better model performance.

---

### 2. Model Selection:
- **Splitting the Data:** 
    - Split the data into training, validation, and test sets.
    - 70% of data was used for training, while 15% each was allocated for validation and test sets.
- **Model Training:** 
    - Used `RandomForestClassifier` with hyperparameter tuning.
    - Performed **GridSearchCV** to optimize hyperparameters, focusing on:
        - `n_estimators`: Number of trees.
        - `max_depth`: Depth of trees.
        - `min_samples_split`: Minimum samples required to split a node.
        - `min_samples_leaf`: Minimum samples required to form a leaf.
- **Best Model Parameters:** 
    - After hyperparameter tuning, the best combination was:
    ```json
    {
      "n_estimators": 200,
      "max_depth": 20,
      "min_samples_split": 5,
      "min_samples_leaf": 2
    }
    ```

---

### 3. Final Model & Prediction:
- **Training Performance:**
    - Achieved high accuracy with an optimized Random Forest model.
    - Evaluated performance using:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
    - Generated a classification report and confusion matrix.
- **Prediction on Test Data:**
    - Used the best model to make predictions on `test.csv`.
    - Predictions were mapped to `Y` and `N` and saved in `sample_submission.csv`.

---

## 4. Evaluation Metrics:
- **Validation Results:**
    - Achieved high performance on the validation set, ensuring minimal overfitting.
- **Test Results:**
    - Model performance was evaluated on unseen test data.
    - Generated a classification report to assess accuracy, precision, recall, and F1-score.

---

## 5. Feature Importance Analysis:
- Analyzed feature importance to understand the contribution of different variables.
- Top contributing features included:
    - `Credit_History`
    - `LoanAmount`
    - `ApplicantIncome`
    - `Property_Area`

---

## 6. Future Improvements:
- Implement additional models such as **XGBoost** or **Gradient Boosting** for performance comparison.
- Apply more advanced feature engineering (e.g., interaction features or derived variables).
- Explore additional data augmentation techniques to balance the dataset.
- Optimize model further by testing ensemble methods and stacking models.

---
