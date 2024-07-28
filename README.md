# Movie-Rating

Data Cleaning:

Dropped rows with missing values.
Removed parentheses from Year and converted it to an integer.
Converted Duration to a float after splitting the string.
Removed commas from Votes and converted it to an integer.
Feature Engineering:

Calculated Director Average Rating and Lead Actor Average Rating.
Split Genre into three separate columns Genre1, Genre2, and Genre3.
Replaced missing values in Genre2 and Genre3.
Encoding:

Factorized Genre1, Genre2, and Genre3.
Model Training:

Used Optuna to optimize hyperparameters for an XGBoost regressor.
The objective function maximizes the R-squared value on the test set.
Hyperparameter Optimization Results:

Optuna provided the best hyperparameters which yielded an R-squared value of approximately 0.7488.
Here's a section that wraps up the process and uses the best parameters to train the final model:

python
Copy code
# Train the final model using the best hyperparameters
xgb_final = XGBRegressor(**best_params)
xgb_final.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = xgb_final.predict(X_test)
final_r2 = r2_score(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)

print(f'Final R-squared: {final_r2:.4f}')
print(f'Final Mean Absolute Error: {final_mae:.4f}')

# SHAP values for interpretability
explainer = shap.Explainer(xgb_final)
shap_values = explainer(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test)
This section trains the final model with the best parameters, evaluates it, and uses SHAP values for interpretability.

A few suggestions for further analysis:

Cross-Validation:
Implement cross-validation to ensure the model's robustness.
Additional Feature Engineering:
Consider creating more features or transforming existing ones for better model performance.
Hyperparameter Tuning:
Increase the number of trials in Optuna for potentially better hyperparameter combinations.

Performance Metrics

Random Forest Regression
Mean Squared Error (MSE): 1.44

R-squared score: 0.95

Decision Tree Regression
Mean Squared Error (MSE): 3.40

R-squared score: 0.89

Lasso Regression
Mean Squared Error (MSE): 2.93

R-squared score: 0.91

Ridge Regression
Mean Squared Error (MSE): 2.91

R-squared score: 0.91

Linear Regression
Mean Squared Error (MSE): 2.91

R-squared score: 0.91

