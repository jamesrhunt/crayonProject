""" 

Analysis of Attrition Data 

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, recall_score, roc_curve, \
    roc_auc_score, make_scorer, precision_score, f1_score # pylint: disable=W0611
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier # for importance reporting

import src.classifiers_and_hyperparams as classifiers_and_hyperparams
import src.data_exploration as data_exploration
import src.data_manipulation as data_manipulation
import src.classification_models as classification_models


###################################
# Boolean switches
###################################
# Data Exploration:
DO_PROF_REPORTS = False # generate profile reports
DO_UNIVARIATE_PLOTS = False # make and save all univariate plots of features for yes/no attrition
DO_BIVARIATE_CATEGORICAL_PLOTS = False # do specific bivariate plots with 2xCategorical features
DO_BIVARIATE_NUM_CAT_PLOTS = False # plot and save bivariate plots which
#                                    are a combination of categorical and numeric features
DO_HEATMAP_PLOTS = False # plot heatmap plots to see correlations between features
DO_CORRELATION_MATRIX = False # plot a correlation matrix
DO_MISC_PLOTS = False # Extra space for making miscellaneous plots
###################################
#        Data Manipulation:
# Feature Engineering:
PLOT_AVERAGE_SATISFACTION = False # Plot new feature, AverageSatisfaction 
PLOT_SALARY_DEVIATION = False # Plot new feature, SalaryDeviation
# Feature Selection:
CHECK_LINEAR_CORRELATION = False # calculate and print linear correlation of
#                               numerical features and attrition


TEST_DROPS = False # try dropping some features from the data set
#
VIEW_XDF_TRANSFORMED = False # look at x_df after it has been transformed in the pipeline
XDF_TRANSFORMED_REPORT = False # generate a profile report of the transformed x_df
###################################
# Pipeline building:

# Set to 'first' when wanting to drop the first column
# during one hot encoding, otherwise = None:
ONEHOT_DROP = None # 'first'
###################################
# Running classification models:
DO_ROC_ANALYSIS = True # perform roc analysis of each classifier
###################################


# Load the CSV file
df = pd.read_csv('data/Crayon_case_employee-attrition.csv')

########################################################################

# DATA EXPORATION -
# GENERATE PROFILE REPORTS AND DROPPING FIRST FEATURES

if DO_PROF_REPORTS:
    # Split based on attrition yes/no then generate profile reports.
    df_attrition = df[df['Attrition'] == 'Yes']
    df_no_attrition = df[df['Attrition'] == 'No']
    data_exploration.create_profiling_reports(df, df_attrition, df_no_attrition)


# Dropping unecessary features after looking at reports (constant values):
df = df.drop(["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], axis=1)

########################################################################

# DATA EXPORATION -
# CATEGORICAL AND NUMERIC SPLIT

# Get categorical and numeric feature names, also checks data types
categorical_feature_names, numerical_feature_names = \
    data_exploration.split_categorical_numerical(df)

# Have a look at the numeric data ranges etc
for column in df[numerical_feature_names].columns:
    dfDescribe = df[column].describe()
    #print(dfDescribe)
    print(f"Min: {dfDescribe['min']},\t Max: {dfDescribe['max']},\t \
          Range: {dfDescribe['max']-dfDescribe['min']}, {column}")

########################################################################

# DATA EXPORATION -
# PLOTTING

# Plot and save all of the possible univariate plots
if DO_UNIVARIATE_PLOTS:
    data_exploration.univariate_plots(df, categorical_feature_names, numerical_feature_names)

# Do some bivariate plots:
if DO_BIVARIATE_CATEGORICAL_PLOTS:
    data_exploration.bivariate_cat_plots(df)
if DO_BIVARIATE_NUM_CAT_PLOTS:
    data_exploration.bivariate_num_cat_plots(df)

# heatmap plots
if DO_HEATMAP_PLOTS:
    data_exploration.heatmap_plots(df)

# Correlation matrix (this is similar to the one
# generated by ProfileReport, but done manually):
if DO_CORRELATION_MATRIX:
    data_exploration.plot_correlation_matrix(df, numerical_feature_names)

# For general testing of plot types etc:
if DO_MISC_PLOTS:
    data_exploration.misc_plots(df)


########################################################################

# DATA MANIPULATION -
# FEATURE ENGINEERING
#

# Make two new features
df['AverageSatisfaction'], df['SalaryDeviation'] = \
    data_manipulation.feature_engineering(df, PLOT_AVERAGE_SATISFACTION, PLOT_SALARY_DEVIATION)

# Update lists with categorical and numeric feature names:
categorical_feature_names, numerical_feature_names = \
    data_exploration.split_categorical_numerical(df)


########################################################################

# DATA MANIPULATION -
# FEATURE SELECTION

# Use corwith to measure the linear relationship between
# the numerical features and attrition.
# Serves as a guide to cross check with observations from data exploration.
# It is univariate and works only with numeric data.


if CHECK_LINEAR_CORRELATION:
    # Convert attrition labels to 1s and 0s:
    attrition_codes = (df["Attrition"].astype('category')).cat.codes
    print("\n\nAttrition Codes:")
    print(attrition_codes)

    # Compute correlation coefficients.
    correlation_with_attrition = abs(df[numerical_feature_names].corrwith(attrition_codes))
    print(correlation_with_attrition.sort_values(ascending=False))


# Drop features based on data exploration and correlation coefficients:
# Explanation of drops in function:
df = data_manipulation.drop_features(df)
# update the cat and num feature names lists
categorical_feature_names, numerical_feature_names = \
    data_exploration.split_categorical_numerical(df)


########################################################################

# CLASSIFICATION MODELS
#
# SPLIT INTO TARGET (y_df) AND FEATURES (x_df)

# save features into x_df
x_df = df.drop("Attrition", axis=1)

# Update categorical features list
categorical_feature_names.remove('Attrition')
#print(categorical_feature_names)

# save target into y_df
y_df = df["Attrition"]


# encode target variable into 0s and 1s rather than Yes or Nos
# (numeric labels)

label_encoder = LabelEncoder()
y_df = label_encoder.fit_transform(y_df)


# Redefine categorical and numerical feature names so they are based on
# whether they are actually categorical and numerical rather than on their
# initial data types. This is important for SMOTE and standard scaling.
categorical_feature_names, numerical_feature_names = \
    classification_models.redefine_features(categorical_feature_names, numerical_feature_names)


########################################################################

# BUILD PIPELINE FOR ENCODING, SCALING AND FEATURE SELECTION

if TEST_DROPS:
    # Test drops based on random forest feature importance
    removeThese = ["EducationField", "Education", "RelationshipSatisfaction",
                "JobSatisfaction", "WorkLifeBalance", "JobInvolvement"]
    print(x_df.columns)
    for featureName in removeThese:
        x_df = x_df.drop([featureName], axis=1)
        categorical_feature_names.remove(featureName)

# build pipeline
pipeline, preprocessor = \
    classification_models.build_pipeline(
        categorical_feature_names, numerical_feature_names, ONEHOT_DROP
        )

# Take a look at x_df after the transformation step:
if VIEW_XDF_TRANSFORMED:
    # Set the display option to show all rows
    pd.set_option('display.max_rows', None)
    # Set the display option to show all columns
    pd.set_option('display.max_columns', None)
    classification_models.check_x_transformed(
        x_df, preprocessor, numerical_feature_names, XDF_TRANSFORMED_REPORT
        )


# select classifiers and hyperparameters with an array:
# 0: logistic regression, 1: KNN, 2: randomForest, 3: SVC   -- e.g. [1,3] KNN+SVC
#classifiers, hyperparameters = getClassAndHyp(models_=[0,1,2,3])
classifiers, hyperparameters = classifiers_and_hyperparams.getClassAndHyp(models_=[2])


print("\n\n\n\nRunning now")
roc_data = []
reportData = []
for i, classifier in enumerate(classifiers):
    print(f"Classifier: {classifier.__class__.__name__}")
    pipeline.set_params(classification=classifier)  # Set the current classifier
    print(i,pipeline,hyperparameters[i])

    # Define the grid search with the classifier's hyperparameters and cross-validation
    scoring_type = precision_score # precision_score, f1_score, recall_score
    grid_search =GridSearchCV(pipeline, hyperparameters[i], scoring=make_scorer(scoring_type), cv=5)


    # Perform grid search to find the best hyperparameters
    grid_search.fit(x_df, y_df)

    # Get the best classifier with tuned hyperparameters
    best_classifier = grid_search.best_estimator_

    print("\n\n\n")

    print(best_classifier)

    # Perform cross-validation and get predicted labels using the best classifier
    y_pred = cross_val_predict(best_classifier, x_df, y_df, cv=5)
    print(y_pred)
    report = classification_report(y_df, y_pred, output_dict=True)  # Generate classification report
    #report = classification_report(y_df, y_pred)  # Generate classification report
    print(report)
    # get the statistical fluctuations of the scores
    scores = cross_val_score(best_classifier, x_df, y_df, cv=5, scoring=make_scorer(scoring_type))
    precision_std = np.std(scores)

    print("Precision scores for each fold:", scores)
    print("Mean Precision score:", np.mean(scores))
    print("Standard Deviation of Precision:", precision_std)

    print("")

    reportData.append((classifier.__class__.__name__, report, best_classifier))

    # ROC Analysis
    if DO_ROC_ANALYSIS:
        # Perform ROC curve analysis using the best classifier
        y_scores = \
            cross_val_predict(best_classifier, x_df, y_df, cv=5, method='predict_proba')[:, 1]
        fpr, tpr, thresholds = roc_curve(y_df, y_scores, pos_label=1)
        auc = roc_auc_score(y_df, y_scores)
        roc_data.append((fpr, tpr, thresholds, classifier.__class__.__name__, auc))

        # Print the best hyperparameters and corresponding score
        print(f"Best Hyperparameters: {grid_search.best_params_}")
        print(f"Best Grid Search Score: {grid_search.best_score_}")
        print(f"AUC from the roc curve CV: {auc}")

        # Perform cross-validation and get predicted labels using the best classifier
        scores = \
            cross_val_score(best_classifier, x_df, y_df, cv=5, scoring=make_scorer(roc_auc_score))
        print("AUC scores for each fold:", scores)
        print("Mean AUC score:", np.mean(scores))


    #############################################

    # if we're at the randomforestclassifier, get the feature importances:
    print("\n\n\n\n trying feature importance \n\n\n")

    if isinstance(classifier, RandomForestClassifier):
        print("Feature Importance:")
        importance_df = classification_models.random_forest_importance( # pylint: disable=invalid-name.
            classifier, numerical_feature_names, pipeline, preprocessor, x_df, y_df
            )
        classification_models.plot_ranfor_importance(importance_df)

    #############################################

classification_models.plot_ROC(roc_data)

#print(reportData)
quit()
