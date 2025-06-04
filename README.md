This project aimed to predict bank customer churn using the Kaggle "Playground Series S4E1" dataset. 
Data preprocessing involved removing irrelevant columns, one-hot encoding for categorical features (Geography, Gender), and standard scaling for numerical features. 
A Logistic Regression model was employed, with hyperparameter optimization performed using GridSearchCV and 5-fold cross-validation, targeting ROC AUC maximization. 
The optimized model achieved a cross-validation ROC AUC of 0.8194 and a validation set ROC AUC of 0.8159 with an accuracy of 0.7542. 
Key predictors for churn included being in Germany, age, and being an inactive member. 
The study demonstrates the effective use of logistic regression for churn prediction, providing a baseline for further model improvements.
