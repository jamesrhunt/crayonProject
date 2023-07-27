import pandas as pd

# no hyperparameters
dataNoTuning = {
    'Classifier': ['Logistic Regression', 'KNN', 'Random Forest', 'SVC'],
    'Precision1': [0.43, 0.24, 0.71, 0.57],
    'Recall1': [0.74, 0.75, 0.31, 0.52],
    'Precision0': [0.94, 0.92, 0.88, 0.91],
    'Recall0': [0.81, 0.55, 0.98, 0.92]
}
dfOld = pd.DataFrame(dataNoTuning)


if 1:
    # new_results-july.txt hyperparameter tuning for PRECISION
    dataPrecision = {
        'Classifier': ['Logistic Regression', 'KNN', 'Random Forest', 'SVC'],
        'Precision1': [0.43, 0.25, 0.68, 0.61],
        'Recall1': [0.75, 0.71, 0.28, 0.41],
        'Precision0': [0.94, 0.91, 0.88, 0.89],
        'Recall0': [0.81, 0.59, 0.97, 0.95]
    }
    df = pd.DataFrame(dataPrecision)

if 0:
    # new_results-july.txt hyperparameter tuning for RECALL
    dataRecall = {
        'Classifier': ['Logistic Regression', 'KNN', 'Random Forest', 'SVC'],
        'Precision1': [0.42, 0.22, 0.52, 0.44],
        'Recall1': [0.78, 0.84, 0.46, 0.78],
        'Precision0': [0.95, 0.93, 0.90, 0.95],
        'Recall0': [0.79, 0.43, 0.92, 0.81]
    }
    df = pd.DataFrame(dataRecall)


# comparing dataframes
numerical_cols = df.select_dtypes(include=[int, float]).columns
print(df[numerical_cols].subtract(dfOld[numerical_cols], fill_value=0))


#calculating number of interactions
support1 = 237
support0 = 1233
df['Tot Pos Predictions'] = df['Recall1'] * support1 + (1-df['Recall0']) * support0
df['Leavers found'] = df['Tot Pos Predictions'] * df['Precision1']
df['Unec Interacts'] = df['Tot Pos Predictions'] - df['Leavers found']
print(df)



