import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Redefine categorical and feature names into true categorical and 
# true numerical feature names
def redefine_features(categoricalFeatureNames_, numericalFeatureNames_):
    #print(categoricalFeatureNames_)
    #print(numericalFeatureNames_)
    print("\nlen(categoricalFeatureNames_), len(numericalFeatureNames_) ")
    print(len(categoricalFeatureNames_), len(numericalFeatureNames_))
    # List of categorical features that are in "numericalFeatureNames_" just because
    # they were pre-encoded in the dataset
    moveToCategorical = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 
                        'JobLevel', 'JobSatisfaction', 'RelationshipSatisfaction',
                        'StockOptionLevel',  'WorkLifeBalance']
    for featureName in moveToCategorical:
        numericalFeatureNames_.remove(featureName)
        categoricalFeatureNames_.append(featureName)
    print(len(categoricalFeatureNames_), len(numericalFeatureNames_))

    return categoricalFeatureNames_, numericalFeatureNames_


####################################################
#
# BUILD PIPELINE

def build_pipeline(categoricalFeatureNames_, numericalFeatureNames_):
    # Define the numerical pipeline
    numericalPipeline_ = Pipeline([
        ('scaler', StandardScaler())  # Numerical feature scaling
    ])

    # First, we define the preprocessing steps in the preprocessor object. 
    # It consists of a ColumnTransformer with two transformers: 
    # numericalPipeline_ for scaling numerical features and OneHotEncoder() 
    # for one-hot encoding categorical features.
    # one hot encoder: drop=first: "dummy variable trap" avoidance.
    preprocessor_ = ColumnTransformer([
        ('numerical', numericalPipeline_, numericalFeatureNames_),  # Apply scaling to numerical features
        #('categorical', OneHotEncoder(drop='first'), categoricalFeatureNames_)  # One-hot encode categorical features 
        ('categorical', OneHotEncoder(), categoricalFeatureNames_)  # One-hot encode categorical features 
    ])


    # (use imb pipeline for SMOTE)
    # The preprocessor is then included as part of the pipeline 
    # along with other steps such as SMOTE oversampling and a 
    # placeholder for the classification model.
    pipeline_ = ImbPipeline([
        ('preprocessing', preprocessor_),  # Preprocessing with separate transformations
        ('sampling', SMOTE()),  # Apply SMOTE for oversampling
        ('classification', None)  # Placeholder for the classification model
    ])

    return pipeline_, preprocessor_

# check transformed data
def check_x_transformed(x_df_,preprocessor_, numericalFeatureNames_):
    x_transformed = preprocessor_.fit_transform(x_df_)
    print("Transformed Data:")
    # Get the encoded feature names 
    # Why tranformers [1][1]:
    # first [1] to access second transformer in prepocessing (one hot encoding)
    # second [1] because the information is stored as [transformer name, actual transformer]
    encoded_feature_names = preprocessor_.transformers_[1][1].get_feature_names_out()
    # Combine numerical and categorical feature names
    feature_names = numericalFeatureNames_ + encoded_feature_names.tolist()
    #print(feature_names)
    df_transformed =  pd.DataFrame(x_transformed, columns=feature_names)
    print(df_transformed.head(1))
    print(df_transformed.dtypes)

    # having a quick look at the profiling report for the transformed data
    if 0:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df_transformed, title="df_transformed")
        profile.to_file("data/reports/df_transformed.html")

########################################

# RANDOM FOREST IMPORTANCE

def random_forest_importance(classifier_, numericalFeatureNames_, pipeline_,\
                             preprocessor_, x_df_, y_df_):
    # fit random forest classifier to get access to feature importances
    pipeline_.fit(x_df_, y_df_)
    #access the feature_importances_ attribute of the classifier  
    feature_importances = classifier_.feature_importances_
    
    # Get the encoded feature names from the preprocessor step:
    encoded_feature_names = preprocessor_.transformers_[1][1].get_feature_names_out()

    # Combine numerical and categorical feature names
    feature_names = numericalFeatureNames_ + encoded_feature_names.tolist()

    importance_df_ = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df_ = importance_df_.sort_values('Importance', ascending=False)
    print("Sorted Feature Importance:")
    #print(importance_df)
    print(importance_df_.to_string(index=False)) # to string to print every line
    #print("")
    return(importance_df_)


########################################



# PLOTTING FUNCTIONS

def plot_ranfor_importance(importance_df_):
    plt.figure()
    # Plot the sorted DataFrame using a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df_['Feature'], importance_df_['Importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Sorted Feature Importance')
    plt.xticks(rotation=20, ha='right')

    # Calculate the cumulative importance
    cumulative_importance = np.cumsum(importance_df_['Importance'])

    # Plot the cumulative importance using a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(importance_df_['Feature'], cumulative_importance)
    plt.xlabel('Feature')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.xticks(rotation=20, ha='right')

    plt.show()

def plotROC(roc_data_):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    for fpr, tpr, _, model_name, auc in roc_data_:
        ax.plot(fpr, tpr, label=model_name+str(auc))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend()
    plt.show()