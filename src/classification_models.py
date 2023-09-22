"""
Operational model stuff like buildling the pipleine, 
feature importance and plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ydata_profiling import ProfileReport # For the transformed data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Redefine categorical and feature names into true categorical and
# true numerical feature names
def redefine_features(categorical_feature_names_, numerical_feature_names_):
    """ Adding features that were initially incorrectly classified as
    numerical (due to their variable type) over to the categorical list. """

    print("\nRedefining categorical and numerical features:")
    print("\n# categorical, # numerical ")
    print(
        " Before: ",
        len(categorical_feature_names_), ",", len(numerical_feature_names_)
    )
    # List of categorical features that are in "numerical_feature_names_" just because
    # they were pre-encoded in the dataset
    move_to_categorical = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                        'JobLevel', 'JobSatisfaction', 'RelationshipSatisfaction',
                        'StockOptionLevel',  'WorkLifeBalance']
    for feature_name in move_to_categorical:
        numerical_feature_names_.remove(feature_name)
        categorical_feature_names_.append(feature_name)
    print(
        " After: ",
        len(categorical_feature_names_), ",", len(numerical_feature_names_)
    )

    return categorical_feature_names_, numerical_feature_names_


####################################################
#
# BUILD PIPELINE

def build_pipeline(categorical_feature_names_, numerical_feature_names_,onehot_drop_):
    """" Building the pipeline for all classification operations.
      The pipeline includes further preprocessing of the dataframe, x_df,
      and oversampling. """
    # Define the numerical pipeline
    numerical_pipeline_ = Pipeline([
        ('scaler', StandardScaler())  # Numerical feature scaling
    ])

    # Define the preprocessing steps in the preprocessor object.
    # It consists of a ColumnTransformer with two transformers:
    # numerical_pipeline_ for scaling numerical features and OneHotEncoder()
    # for one-hot encoding categorical features.
    # one hot encoder: drop=first: "dummy variable trap" avoidance.
    preprocessor_ = ColumnTransformer([
        # Apply scaling to numerical features:
        ('numerical', numerical_pipeline_, numerical_feature_names_),
        # One-hot encode categorical features:
        ('categorical', OneHotEncoder(drop=onehot_drop_), categorical_feature_names_)
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

def check_x_transformed(x_df_,preprocessor_, numerical_feature_names_,XDF_TRANSFORMED_REPORT_): # pylint: disable=invalid-name.
    """ Have a look at the data after it has been transformed. """
    # First transform the data using the preprocessor
    x_transformed = preprocessor_.fit_transform(x_df_)
    print("Transformed Data:")
    # Get the encoded feature names
    # Why tranformers [1][1]:
    # first [1] to access second transformer in prepocessing (one hot encoding)
    # second [1] because the information is stored as [transformer name, actual transformer]
    encoded_feature_names = preprocessor_.transformers_[1][1].get_feature_names_out()
    # Combine numerical and categorical feature names
    feature_names = numerical_feature_names_ + encoded_feature_names.tolist()
    #print(feature_names)
    df_transformed =  pd.DataFrame(x_transformed, columns=feature_names)
    print(df_transformed.head(1))
    print(df_transformed.dtypes)


    # having a quick look at the profiling report for the transformed data
    if XDF_TRANSFORMED_REPORT_:
        profile = ProfileReport(df_transformed, title="df_transformed")
        profile.to_file("data/reports/df_transformed.html")

########################################

# RANDOM FOREST IMPORTANCE

def random_forest_importance(classifier_, numerical_feature_names_, pipeline_,\
                             preprocessor_, x_df_, y_df_):
    """ Use the feature importances function of random forest and return
     the result as a dataframe, importance_df_.
      
       This result can later be plotted by plot_ranfor_importance. """

    # fit random forest classifier to get access to feature importances
    pipeline_.fit(x_df_, y_df_)
    #access the feature_importances_ attribute of the classifier
    feature_importances = classifier_.feature_importances_

    # Get the encoded feature names from the preprocessor step:
    encoded_feature_names = preprocessor_.transformers_[1][1].get_feature_names_out()

    # Combine numerical and categorical feature names
    feature_names = numerical_feature_names_ + encoded_feature_names.tolist()

    importance_df_ = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df_ = importance_df_.sort_values('Importance', ascending=False)
    print("Sorted Feature Importance:")
    #print(importance_df)
    print(importance_df_.to_string(index=False)) # to string to print every line
    #print("")
    return importance_df_


########################################



# PLOTTING FUNCTIONS

def plot_ranfor_importance(importance_df_):
    """" Plot the importance_df_ dataframe given by random_forst_importance """
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

def plot_ROC(roc_data_): # pylint: disable=invalid-name
    """ Plot a Receiver Operating Characteristic (ROC) Curve based on
    data generated with the 'predict_proba' method of cross_val_predict. """
    _, roc_ax = plt.subplots()
    roc_ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    for fpr, tpr, _, model_name, auc in roc_data_:
        roc_ax.plot(fpr, tpr, label=model_name+str(auc))
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    roc_ax.legend()
    plt.show()
