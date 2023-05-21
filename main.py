import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataExploration import *

# Boolean switches
doProfReports = False # generate profile reports
doUnivariatePlots = False # make and save all univariate plots of features for yes/no attrition
doBivariateCategorialPlots = False # do specific bivariate plots with 2xCategorical features
doBivariateNumCatPlots = False # plot and save bivariate plots which are a combination of categorical and numeric features

# Load the CSV file
df = pd.read_csv('data/Crayon_case_employee-attrition.csv')

if doProfReports:
    # Split into Attrition or not
    dfAttrition = df[df['Attrition'] == 'Yes']
    dfNoAttrition = df[df['Attrition'] == 'No']
    createProfilingReports(df, dfAttrition, dfNoAttrition)


# Dropping unecessary features after looking at reports (constant values):
df = df.drop(["EmployeeCount", "Over18", "StandardHours"], axis=1)

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



