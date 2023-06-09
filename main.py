import pandas as pd
import matplotlib.pyplot as plt
from src.dataExploration import *
from src.dataManipulation import *
from src.classificationModels import *
from sklearn.preprocessing import StandardScaler


# Boolean switches
doProfReports = False # generate profile reports
doUnivariatePlots = False # make and save all univariate plots of features for yes/no attrition
doBivariateCategorialPlots = True # do specific bivariate plots with 2xCategorical features
doBivariateNumCatPlots = True # plot and save bivariate plots which are a combination of categorical and numeric features

# Load the CSV file
df = pd.read_csv('data/Crayon_case_employee-attrition.csv')

if doProfReports:
    # Split into Attrition or not
    dfAttrition = df[df['Attrition'] == 'Yes']
    dfNoAttrition = df[df['Attrition'] == 'No']
    createProfilingReports(df, dfAttrition, dfNoAttrition)


# Dropping unecessary features after looking at reports (constant values):
df = df.drop(["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], axis=1)

########################################################################

# DATA EXPORATION - CATEGORICAL AND NUMERIC SPLIT

# Get categorical and numeric feature names, also checks data types
categoricalFeatureNames, numericalFeatureNames = categoricAndNumeric(df)

# Have a look at the numeric data ranges etc
for column in df[numericalFeatureNames].columns:
    dfDescribe = df[column].describe()
    #print(dfDescribe)
    print("Min: {},\t Max: {},\t Range: {}, {}".format(dfDescribe['min'],dfDescribe['max'],dfDescribe['max']-dfDescribe['min'],column))

########################################################################

# DATA EXPORATION - PLOTTING

# Plot and save all of the possible univariate plots
if doUnivariatePlots:
    univariatePlots(df, categoricalFeatureNames, numericalFeatureNames)

# Do some bivariate plots:
if doBivariateCategorialPlots:
    bivariateCatPlots(df)
if doBivariateNumCatPlots:
    bivariateNumCatPlots(df)


# For general testing of plot types etc:
#miscPlots(df)

########################################################################

# DATA MANIPULATION - FEATURE ENGINEERING 
# 
plotAverageSatisfaction = False
plotSalaryDeviation = False
# Make two new features
df['AverageSatisfaction'], df['SalaryDeviation'] = featureEngineering(df, plotAverageSatisfaction, plotSalaryDeviation)

# Update arrays with cat and numeric feature names:
categoricalFeatureNames, numericalFeatureNames = categoricAndNumeric(df)

########################################################################

# CORRELATION VALUES FOR INITAL FEATURE SELECTION

attritionCodes = (df["Attrition"].astype('category')).cat.codes
print(attritionCodes)


# Use corwith to measure the linear relationship between the numerical features
# as a light guide to cross check with observations from plots
# It is univariate and works only with numeric data
correlationWithAttrition = abs(df[numericalFeatureNames].corrwith(attritionCodes))
print(correlationWithAttrition.sort_values(ascending=False))
"""
OUTPUT:
TotalWorkingYears           0.171063
JobLevel                    0.169105
YearsInCurrentRole          0.160545
MonthlyIncome               0.159840
Age                         0.159205
YearsWithCurrManager        0.156199
AverageSatisfaction         0.146819 -- New feature
StockOptionLevel            0.137145
YearsAtCompany              0.134392
JobInvolvement              0.130016
JobSatisfaction             0.103481 - Could drop if average satisfaction is better
EnvironmentSatisfaction     0.103369 - Could drop if average satisfaction is better
DistanceFromHome            0.077924
WorkLifeBalance             0.063939
TrainingTimesLastYear       0.059478
DailyRate                   0.056652
SalaryDeviation             0.054234 -- New feature
RelationshipSatisfaction    0.045872 - Could drop if average satisfaction is better
NumCompaniesWorked          0.043494
YearsSinceLastPromotion     0.033019
Education                   0.031373
MonthlyRate                 0.015170 - Dropping
PercentSalaryHike           0.013478
EmployeeNumber              0.010577 - Dropping
HourlyRate                  0.006846 - Dropping
PerformanceRating           0.002889 - Dropping

"""

########################################################################

# DATA MANIPULATION - DROPPING FEATURES AFTER DATA EXPLORATION
print(df.columns)
df = dropFeatures(df)
# update the cat and num feature names lists
categoricalFeatureNames, numericalFeatureNames = categoricAndNumeric(df)
#print(df.columns)

########################################################################

# CLASSIFICATION MODELS 
# 
# SPLIT INTO TARGET (y_df) AND FEATURES (x_df)

# save features into x_df
x_df = df.drop("Attrition", axis=1)

# Update categorical features list
categoricalFeatureNames.remove('Attrition')
#print(categoricalFeatureNames)

# save target into y_df
y_df = df["Attrition"]
 

# Redefine categorical and numerical feature names so they are based on
# whether they are actually categorical and numerical rather than on their
# initial data types. This is important for SMOTE and standard scaling.
categoricalFeatureNames, numericalFeatureNames = redefineFeatures(categoricalFeatureNames, numericalFeatureNames)


########################################################################

# BUILD PIPELINE FOR ENCODING, SCALING AND FEATURE SELECTION


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score

if 0:
    # Test drops based on random forest feature importance
    removeThese = ["EducationField", "Education", "RelationshipSatisfaction",
                "JobSatisfaction", "WorkLifeBalance", "JobInvolvement"]
    print(x_df.columns)
    for featureName in removeThese:
        x_df = x_df.drop([featureName], axis=1)
        categoricalFeatureNames.remove(featureName)

# build pipeline
pipeline, preprocessor = buildPipeline(categoricalFeatureNames, numericalFeatureNames)


# Take a look at x_df after the transformation step:
if 0:
    check_x_transformed(x_df, preprocessor, numericalFeatureNames)



# Define the list of classifiers to evaluate
classifiers = [
    LogisticRegression(max_iter=500),
    KNeighborsClassifier(n_neighbors=5),
    RandomForestClassifier(n_estimators=500),
    SVC(probability=True)
]

########################################################################

# FIT CLASSIFIERS AND OBTRAIN SCORING METRICS

# Iterate over the classifiers in the classifiers list and 
# evaluate their performance using cross-validation. For each 
# classifier, set the current classifier in the pipeline using 
# pipeline.set_params(classification=classifier).
roc_data = []
for classifier in classifiers:
    pipeline.set_params(classification=classifier)  # Set the current classifier
    scores = cross_val_score(pipeline, x_df, y_df, cv=5)  # Perform cross-validation
    print(f"Classifier: {classifier.__class__.__name__}")
    print(f"Average accuracy: {scores.mean()}")
    print("")
    # Perform cross-validation and get predicted labels (stratify by default)
    y_pred = cross_val_predict(pipeline, x_df, y_df, cv=5)      
    report = classification_report(y_df, y_pred)  # Generate classification report
    print(report)
    print("")
    # note: in the code above, calling cross_val twice, could collapse those 
    # calls into one and save all required outputs
    # it also calls cross_val_predict again for the ROC curve

    # ROC curve
    # get the predicted probabilities for each data point
    y_scores = cross_val_predict(pipeline, x_df, y_df, cv=5, method='predict_proba')[:, 1] 
    # using pos_label = yes so that roc_curve understands 
    fpr, tpr, thresholds = roc_curve(y_df, y_scores,pos_label='Yes')
    auc = roc_auc_score(y_df, y_scores)
    roc_data.append((fpr, tpr, thresholds, classifier.__class__.__name__, auc))
    
    
    # Fit the pipeline to the entire dataset (x_df and y_df) to 
    # obtain feature importance and coefficient analysis if supported by the classifier
    pipeline.fit(x_df, y_df)
    
    # if we're at the randomforestclassifier, get the feature importances:
    if isinstance(classifier, RandomForestClassifier) and hasattr(classifier, 'feature_importances_'):
        print("Feature Importance:")

        #access the feature_importances_ attribute of the classifier  
        feature_importances = classifier.feature_importances_
        
        # Get the encoded feature names from the preprocessor step:
        encoded_feature_names = pipeline['preprocessing'].transformers_[1][1].get_feature_names_out()

        # Combine numerical and categorical feature names
        feature_names = numericalFeatureNames + encoded_feature_names.tolist()

        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        print("Sorted Feature Importance:")
        #print(importance_df)
        print(importance_df.to_string(index=False)) # to string to print every line
        #print("")

    #if isinstance(classifier, LogisticRegression) and hasattr(pipeline['classification'], 'coef_'):
    #    print("Coefficient Analysis:")
    #    coefficients = pipeline['classification'].coef_
    #    feature_names = numericalFeatureNames + categoricalFeatureNames
    #    coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients[0]})
    #    coefficients_df = coefficients_df.sort_values('Coefficient', ascending=False)
    #    print(coefficients_df)
    #    print("")

# plot the importance_df from the random forest classifier
if 0:
    plotRandomForestImportance(importance_df)

# Plot ROC Curve
if 1:
    plotROC(roc_data)

quit()