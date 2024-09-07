# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE

# Preprocess training data
def preprocess_data(df):
    df = df.dropna()  # Drop missing values
    
    # Define features and target
    X = df[['ProductionVolume', 'ProductionCost', 'SupplierQuality', 'DeliveryDelay',
            'DefectRate', 'QualityScore', 'MaintenanceHours', 'DowntimePercentage',
            'InventoryTurnover', 'StockoutRate', 'WorkerProductivity', 'SafetyIncidents',
            'EnergyConsumption', 'EnergyEfficiency', 'AdditiveProcessTime',
            'AdditiveMaterialCost']]
    y = df['DefectStatus']  # Target variable (1: Defect, 0: No defect)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    
    return X_scaled, y_resampled

# Preprocess test data
def preprocess_test_data(df):
    X_test = df[['ProductionVolume', 'ProductionCost', 'SupplierQuality', 'DeliveryDelay',
                 'DefectRate', 'QualityScore', 'MaintenanceHours', 'DowntimePercentage',
                 'InventoryTurnover', 'StockoutRate', 'WorkerProductivity', 'SafetyIncidents',
                 'EnergyConsumption', 'EnergyEfficiency', 'AdditiveProcessTime',
                 'AdditiveMaterialCost']]
    
    # Scale the features
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    return X_test_scaled

# Train RandomForest model with hyperparameter tuning
def train_model_with_tuning(X, y):
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    model = RandomForestClassifier(random_state=42)
    
    # RandomizedSearchCV for hyperparameter tuning
    search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, 
                                n_iter=10, cv=5, verbose=2, n_jobs=-1, random_state=42)
    
    search.fit(X, y)
    best_model = search.best_estimator_
    
    return best_model

# Generate evaluation report
def generate_report(y_true, y_pred, y_pred_prob):
    report = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Confusion Matrix': confusion_matrix(y_true, y_pred),
        'Classification Report': classification_report(y_true, y_pred, output_dict=True),
        'AUC-ROC': roc_auc_score(y_true, y_pred_prob)
    }
    return report

# Display the report
def display_report(report):
    st.write(f"Model Accuracy: {report['Accuracy']:.2f}")
    
    st.write("Confusion Matrix:")
    st.write(report['Confusion Matrix'])
    
    st.write("Classification Report:")
    classification_report_df = pd.DataFrame(report['Classification Report']).transpose()
    st.write(classification_report_df)
    
    st.write(f"AUC-ROC Score: {report['AUC-ROC']:.2f}")

# Streamlit app layout
def main():
    st.title('Manufacturing Defect Prediction')

    # Upload train CSV
    st.subheader("Upload your training data (CSV file with DefectStatus)")
    uploaded_train_file = st.file_uploader("Upload train.csv", type="csv")
    
    # Upload test CSV
    st.subheader("Upload your test data (CSV file without DefectStatus)")
    uploaded_test_file = st.file_uploader("Upload test.csv", type="csv")
    
    if uploaded_train_file and uploaded_test_file:
        # Load training data and preprocess it
        train_data = pd.read_csv(uploaded_train_file)
        X_scaled, y = preprocess_data(train_data)
        
        # Train the model
        model = train_model_with_tuning(X_scaled, y)
        
        # Load and preprocess the test data
        test_data = pd.read_csv(uploaded_test_file)
        st.write("Test Data Preview:")
        st.write(test_data.head())
        
        X_test_scaled = preprocess_test_data(test_data)
        
        # Predict on test data
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # If the test data contains a 'DefectStatus' column, evaluate the model
        if 'DefectStatus' in test_data.columns and not test_data['DefectStatus'].isnull().all():
            y_test = test_data['DefectStatus']
            
            # Generate report
            report = generate_report(y_test, y_pred, y_pred_prob)
            
            # Display the report
            st.subheader("Model Evaluation Report")
            display_report(report)
        else:
            st.warning("No DefectStatus found in the test data. Proceeding with predictions only.")
        
        # Output predictions
        test_data['Predicted_DefectStatus'] = y_pred
        st.subheader("Test Data with Predictions")
        st.write(test_data)

        # Download the output as CSV
        output_csv = test_data.to_csv(index=False)
        st.download_button(label="Download Predictions as CSV", data=output_csv, file_name='output.csv', mime='text/csv')

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/krshnndu/" target="_blank">Krishnendu B</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

if __name__ == '__main__':
    main()