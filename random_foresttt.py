# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# Load Dataset
data = pd.read_csv("df1_loan.csv")

# for this project, the Team Leader provided a dataset that was already preprocessed.

# Encoding converting categorical into numbers.
data.replace({
    "Loan_Status": {'N': 0, 'Y': 1},
    "Gender": {'Male': 0, 'Female': 1},
    "Education": {'Not Graduate': 0, 'Graduate': 1},
    "Married": {'No': 0, 'Yes': 1},
    "Self_Employed": {'No': 0, 'Yes': 1}
}, inplace=True)

# for fixing dependents string issue.
data['Dependents'] = data['Dependents'].replace('3+', 3)

# features and target
y = data["Loan_Status"]
X = data.drop(['Loan_Status', 'Loan_ID'], axis=1)

# One-hot encoding for categorical columns.
X = pd.get_dummies(X, columns=["Property_Area", "Dependents"], drop_first=True)

# ensure all values are numeric.
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

# Train_Test Split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision tree (for comparison).
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Decision Tree Results")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Classification Report:\n", classification_report(y_test, dt_pred))

# Random forest with GridSearch.
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

grid_rf.fit(X_train, y_train)

rf_model = grid_rf.best_estimator_
rf_pred = rf_model.predict(X_test)

print("\nBest Random Forest Parameters:", grid_rf.best_params_)
print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

# Final Comparison.
print("\nModel Comparison")
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

 