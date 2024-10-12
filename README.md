# GSTN-Analytics-Byte_Us
Steps Performed:<br>
1. **Data Preprocessing**:
   > Imputation of Missing Values
   > Discretization
   > Grouping
   > Variance Thresholding
2. **Feature Selection**:
   > Feature Importance using Random Forest
   > Feature Selection using Forward Selection (11 features selected)
3. **Balancing the Imbalanced Data**
   > Comparing different methods for Upsampling and Downsampling
   > Upsampled the data using the best method - SMOTE
4. **Modelling**
   > Compared Various ML models and found XGBoost performed best
   > Tuned the Hyperparameters of XGBoost using Bayesian Optimisation to get improved and robust performance
5. **Analysis of ROC Curve**
   > Analysed the ROC curve to obtain the best threshold and improve the F1 score on test data.
6. **Model Explainability**
   > Applied Explainable AI techniques like SHAP to explain the predictions of the model.
