""" Data Exploration """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import statements required for Plotly
import plotly.offline as py
import plotly.graph_objs as go
#import plotly.tools as tls
 
def create_profiling_reports(df, dfAttrition, dfNoAttrition):
    from ydata_profiling import ProfileReport
    

    # Generate report of all intial data
    profile = ProfileReport(df, title="Full Data")
    profile.to_file("data/reports/fullDataReport.html")

    dfAttritionProfile = ProfileReport(dfAttrition, title="Attrition")
    dfNoAttritionProfile = ProfileReport(dfNoAttrition, title="No Attrition")
    comparison_report = dfNoAttritionProfile.compare(dfAttritionProfile)
    comparison_report.to_file("data/reports/attritionComparisonReport.html")


def categoricAndNumeric(df):
    
    # Get the categorical and numeric column names, 
    # and have a look at data types
    categoricalFeatureNames_ = []
    numericalFeatureNames_ = []
    dtypes = []
    for columnName, value in df.items():
        dtypes.append(value.dtype)
        if value.dtype == 'object':
            categoricalFeatureNames_.append(columnName)
        else:
            numericalFeatureNames_.append(columnName)

    print("\n",len(categoricalFeatureNames_), "Categorical feature names:")
    print(categoricalFeatureNames_)
    print("\n",len(numericalFeatureNames_), "Numerical feature names:")
    print(numericalFeatureNames_)
    print("\n",len(dtypes), "features in total.")
    # show all unique data types
    print("\nUnique data types:")
    print(np.unique(dtypes))
    
    
    return categoricalFeatureNames_, numericalFeatureNames_
    
def plotCorrelationMatrix(df_, numericalFeatureNames_):
    # plot a correlation matrix   
    data = [
        go.Heatmap(
            z = df_[numericalFeatureNames_].astype(float).corr().values, # Generate Pearson correlation
            x = df_[numericalFeatureNames_].columns.values,
            y = df_[numericalFeatureNames_].columns.values,
            colorscale='Viridis',
            reversescale = False,
    #         text = True ,
            opacity = 1.0
        )
    ]
    
    layout = go.Layout(
        title='Pearson Correlation of numerical features',
        xaxis = dict(ticks='', nticks=36),
        yaxis = dict(ticks='' ),
        width = 900, height = 700,
        
    )

    # using py.iplot to generate interactive plot in browser or notebook
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='labelled-heatmap')
    

def univariatePlots(df_, categoricalFeatureNames_, numericalFeatureNames_):

    # UNIVARIATE PLOTS 

    # CATEGORICAL

    for column in categoricalFeatureNames_:
        plt.figure()
        sns.histplot(x = df_[column], hue=df_['Attrition'], multiple="dodge", 
                    stat = 'percent', shrink = 0.8, common_norm=False)
        plt.savefig("data/plots/dataExploration/categorical/"+column+"-AttritionBar.png")
        #plt.show()

    # NUMERIC 

    for column in numericalFeatureNames_:
        plt.figure()
        sns.histplot(x = df_[column], hue=df_['Attrition'],  
                        stat = 'percent', shrink = 1.0, common_norm=False)
        plt.savefig("data/plots/dataExploration/numerical/"+column+"-AttritionHist.png")


# BIVARIATE PLOTS - CATEGORICAL 
def bivariatePlots_categorical(df_, catFeat1_, catFeat2_):
    # This is just the function to make the plots
    plt.figure()
    sns.histplot(x = df_[catFeat1_], hue=df_[catFeat2_], multiple="dodge", 
                    stat = 'percent', shrink = 0.8, common_norm=False)
    plt.xticks(rotation=20, ha='right')
    plt.savefig("data/plots/dataExploration/categorical/bivariate/bivarPlot_cat"+catFeat1_+"_"+catFeat2_+".png")



# BIVARIATE PLOTS - NUMERIC AND CATEGORICAL
def bivariatePlots_numerCateg(df_, catFeat1_, numFeat_, catFeat2_):
    # This is just the function to make the plots
    plt.figure()
    sns.catplot(data=df_, x=catFeat1_, y=numFeat_, hue=catFeat2_, kind="box")
    plt.savefig("data/plots/dataExploration/categorical/bivariate/bivarPlot_Numcat"\
                +catFeat1_+"_"+numFeat_+"_"+catFeat2_+".png")
     
def bivariateCatPlots(df_):
    # Plot and save specific categorical plots

    # Make a list of categorical features to pair for plotting against each other
    bivariateCategoricalNames = [
        #['Department', 'EducationField'],
        #['EducationField', 'JobRole'],
        #['Department', 'JobRole'],
        #['BusinessTravel', 'MaritalStatus'],
        ['OverTime', 'JobRole']
        ]

    # plot the pairs of categorical features
    for item in bivariateCategoricalNames:
        bivariatePlots_categorical(df_, item[0], item[1])

    plt.show()

def bivariateNumCatPlots(df_):
    # plot and save bivariate plots which are a combination of categorical and numeric features

    bivariateNumCatNames = [
        #['EducationField', 'Education', 'Attrition'],
        #['OverTime', 'MonthlyIncome', 'Attrition'],
        #['MaritalStatus', 'Age', 'Attrition']
        #['JobLevel', 'MonthlyIncome', 'Attrition'],
        #['JobRole', 'MonthlyIncome', 'Attrition'],
        #['JobRole', 'MonthlyIncome', 'JobLevel'],
        ['JobRole', 'MonthlyIncome', 'Attrition']
        ]
    for item in bivariateNumCatNames:
        bivariatePlots_numerCateg(df_, item[0],item[1],item[2])
        plt.xticks(rotation=20, ha='right')

    plt.show()

def correlationPlots(df_):
    feature1_ = 'Age'
    feature2_ = 'DailyRate'
    plt.figure()
    cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)
    # Generate and plot
    sns.kdeplot(data=df_, x=feature1_, y=feature2_, cmap=cmap, shade=True)
    plt.legend()
    plt.show()
    plt.savefig("data/plots/dataExploration/heatmap"\
                +feature1_+"_"+feature2_+"_.png")

def miscPlots(df_):
    
    
    plt.figure()
    
    df_["YearsOver18"] = df_["Age"]-18
    sns.histplot(
        df_, x="YearsOver18", 
        bins=20, label="Years Over 18"
    )
    sns.histplot(
        df_, x="TotalWorkingYears", 
        bins=20, label="Total Working Years"
    )
    plt.legend()
    plt.show()

    

    #plt.figure()
    #sns.catplot(data=df, x="OverTime", y="WorkLifeBalance", hue="Attrition", kind="box")

    #plt.figure()
    #sns.catplot(data=df, x="OverTime", y="MonthlyIncome", hue="Attrition", kind="box")

    #plt.figure()
    #plt.scatter(df['JobLevel'],df['MonthlyIncome'])
    #plt.show()
    #total working years, age

# Show the plots
#plt.show()