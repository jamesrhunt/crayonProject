import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataExploration import *


# Load the CSV file
df = pd.read_csv('data/Crayon_case_employee-attrition.csv')

# Split into Attrition or not
#dfAttrition = df[df['Attrition'] == 'Yes']
#dfNoAttrition = df[df['Attrition'] == 'No']

#createProfilingReports(df, dfAttrition, dfNoAttrition)


# Dropping unecessary features (constant values):
df = df.drop(["EmployeeCount", "Over18", "StandardHours"], axis=1)


categoricalFeatureNames, numericalFeatureNames = categoricAndNumeric(df)
# Have a look at the numeric data ranges etc
for column in df[numericalFeatureNames].columns:
    dfDescribe = df[column].describe()
    #print(dfDescribe)
    print("Min: {},\t Max: {},\t Range: {}, {}".format(dfDescribe['min'],dfDescribe['max'],dfDescribe['max']-dfDescribe['min'],column))


# UNIVARIATE PLOTS

# CATEGORICAL

if 0:
    for column in categoricalFeatureNames:
        plt.figure()
        sns.histplot(x = df[column], hue=df['Attrition'], multiple="dodge", 
                    stat = 'percent', shrink = 0.8, common_norm=False)
        plt.savefig("data/plots/dataExploration/categorical/"+column+"-AttritionBar.png")
        #plt.show()

# NUMERIC 

if 0:
    for column in numericalFeatureNames:
        plt.figure()
        sns.histplot(x = df[column], hue=df['Attrition'],  
                        stat = 'percent', shrink = 1.0, common_norm=False)
        plt.savefig("data/plots/dataExploration/numerical/"+column+"-AttritionHist.png")


# BIVARIATE PLOTS

# CATEGORICAL 

if 0:
    plt.figure()
    sns.histplot(x = df['Department'], hue=df['EducationField'], multiple="dodge", 
                    stat = 'percent', shrink = 0.8, common_norm=False)

    plt.figure()
    sns.histplot(x = df['EducationField'], hue=df['JobRole'], multiple="dodge", 
                    stat = 'percent', shrink = 0.8, common_norm=False)

    plt.figure()
    sns.histplot(x = df['Department'], hue=df['JobRole'], multiple="dodge", 
                    stat = 'percent', shrink = 0.8, common_norm=False)




# NUMERIC AND CATEGORICAL

plt.figure()
sns.catplot(data=df, x="OverTime", y="WorkLifeBalance", hue="Attrition", kind="box")

plt.figure()
sns.catplot(data=df, x="OverTime", y="MonthlyIncome", hue="Attrition", kind="box")

# Show the plots
plt.show()
"""
plt.figure()
plt.scatter(df['MonthlyIncome'], df['MonthlyRate'])
"""
