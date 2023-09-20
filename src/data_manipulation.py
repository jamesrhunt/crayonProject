import matplotlib.pyplot as plt
import seaborn as sns

def featureEngineering(df_, plotAvg, plotSal):
    # Generating average satisfaction new feature, and plot
    df_['AverageSatisfaction'] = df_[['EnvironmentSatisfaction', 'RelationshipSatisfaction', 'JobSatisfaction']].mean(axis=1)
    # Plot the new feature to have a look
    if plotAvg:
        plt.figure()
        sns.histplot(x = df_['AverageSatisfaction'], hue=df_['Attrition'], 
                    stat = 'percent', shrink = 1.0, common_norm=False)
        plt.show()

    # Generate the deviation for the mean salary at that particular job level
    df_['SalaryDeviation'] = df_['MonthlyIncome'] - df_.groupby('JobLevel')['MonthlyIncome'].transform('mean')
    # Plot the new feature to have a look
    if plotSal:
        plt.figure()
        sns.histplot(x = df_['SalaryDeviation'], hue=df_['Attrition'],
                    stat = 'percent', shrink = 1.0, common_norm=False)
        plt.show()

    return df_['AverageSatisfaction'], df_['SalaryDeviation']
 
def dropFeatures(df_):
    # FIRST - Dropping some features after data exploration

    # Dropping department because it's well represented by job role
    df_ = df_.drop(["Department"], axis=1)

    # Dropping employee number for now because not correlated with anything
    # Already dropped
    #df = df.drop(["EmployeeNumber"], axis=1)

    # Dropping monthly rate because low correlation and much more important monthly income 
    # is much higher correlated
    df_ = df_.drop(["MonthlyRate"], axis=1)

    # Dropping monthly rate similar to above
    df_ = df_.drop(["HourlyRate"], axis=1)

    # Dropping performance rating based on the plot (identical between attrition rates), and
    # also based on the simple corrwith
    df_ = df_.drop(["PerformanceRating"], axis=1)

    return df_
