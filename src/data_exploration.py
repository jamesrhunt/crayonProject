""" Data Exploration """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Import statements required for Plotly
import plotly.offline as py
import plotly.graph_objs as go


def create_profiling_reports(report_df_, df_attrition, df_no_attrition):
    """ Create the auto report from ydata_profiling. """

    # Generate report of all intial data
    profile = ProfileReport(report_df_, title="Full Data")
    profile.to_file("data/reports/full_data_report.html")

    df_attrition_profile = ProfileReport(df_attrition, title="Attrition")
    df_no_attrition_profile = ProfileReport(df_no_attrition, title="No Attrition")
    comparison_report = df_no_attrition_profile.compare(df_attrition_profile)
    comparison_report.to_file("data/reports/attrition_comparison_report.html")


def split_categorical_numerical(df_to_split):
    """ Split the data into categorical and numeric, based only
     on the data types contained within the raw data. This split
      will be improved later. The data is split by returning two
      lists containing only categorical or numerical feature names."""

    categorical_feature_names_ = []
    numerical_feature_names_ = []
    dtypes = []
    for column_name, value in df_to_split.items():
        dtypes.append(value.dtype)
        if value.dtype == 'object':
            categorical_feature_names_.append(column_name)
        else:
            numerical_feature_names_.append(column_name)

    print("\n",len(categorical_feature_names_), "Categorical feature names:")
    print(categorical_feature_names_)
    print("\n",len(numerical_feature_names_), "Numerical feature names:")
    print(numerical_feature_names_)
    print("\n",len(dtypes), "features in total.")
    # show all unique data types
    print("\nUnique data types:")
    print(np.unique(dtypes))

    return categorical_feature_names_, numerical_feature_names_

def univariate_plots(df_, categorical_feature_names_, numerical_feature_names_):
    """ Plot all features using a different method for categorical and numerical types. """

    # Categorical

    for column in categorical_feature_names_:
        plt.figure()
        sns.histplot(x = df_[column], hue=df_['Attrition'], multiple="dodge",
                    stat = 'percent', shrink = 0.8, common_norm=False)
        plt.savefig("data/plots/dataExploration/categorical/"+column+"-AttritionBar.png")
        #plt.show()

    # Numerical

    for column in numerical_feature_names_:
        plt.figure()
        sns.histplot(x = df_[column], hue=df_['Attrition'],
                        stat = 'percent', shrink = 1.0, common_norm=False)
        plt.savefig("data/plots/dataExploration/numerical/"+column+"-AttritionHist.png")



# BIVARIATE PLOTS - CATEGORICAL =============

def bivariate_plots_categorical(df_, cat_feat_1_, cat_feat_2_):
    """ Make the plot for function bivariate_cat_plots, below. """
    plt.figure()
    sns.histplot(x = df_[cat_feat_1_], hue=df_[cat_feat_2_], multiple="dodge",
                    stat = 'percent', shrink = 0.8, common_norm=False)
    plt.xticks(rotation=20, ha='right')
    plt.savefig(
        "data/plots/dataExploration/categorical/bivariate/bivarPlot_cat"
        +cat_feat_1_+"_"+cat_feat_2_+".png"
    )

def bivariate_cat_plots(df_):
    """ Plot and save specific combinations of categorical features,
     using seaborn's catplot. Calls bivar_plots_num_cat. """

    # Make a list of categorical features to pair for plotting against each other
    categorical_features_to_plot = [
        #['Department', 'EducationField'],
        #['EducationField', 'JobRole'],
        #['Department', 'JobRole'],
        #['BusinessTravel', 'MaritalStatus'],
        ['OverTime', 'JobRole']
        ]

    # plot the pairs of categorical features
    for item in categorical_features_to_plot:
        bivariate_plots_categorical(df_, item[0], item[1])

    plt.show()



# BIVARIATE PLOTS - NUMERIC AND CATEGORICAL ==========

def bivar_plots_num_cat(df_, cat_feat_1_, num_feat_, cat_feat_2_):
    """ Make the plots for function bivariate_num_cat_plots, below. """
    plt.figure()
    sns.catplot(data=df_, x=cat_feat_1_, y=num_feat_, hue=cat_feat_2_, kind="box")
    plt.savefig(
        "data/plots/dataExploration/categorical/bivariate/bivarPlot_Numcat"
        +cat_feat_1_+"_"+num_feat_+"_"+cat_feat_2_+".png"
    )


def bivariate_num_cat_plots(df_):
    """ Plot and save bivariate plots which are a combination of categorical
     and numeric features. All plots also have a third variable, Attrition 
     for side by side comparison. Calls bivar_plots_num_cat. """

    bivariate_num_cat_names = [
        #['EducationField', 'Education', 'Attrition'],
        #['OverTime', 'MonthlyIncome', 'Attrition'],
        #['MaritalStatus', 'Age', 'Attrition']
        #['JobLevel', 'MonthlyIncome', 'Attrition'],
        #['JobRole', 'MonthlyIncome', 'Attrition'],
        #['JobRole', 'MonthlyIncome', 'JobLevel'],
        ['JobRole', 'MonthlyIncome', 'Attrition']
        ]
    for item in bivariate_num_cat_names:
        bivar_plots_num_cat(df_, item[0],item[1],item[2])
        plt.xticks(rotation=20, ha='right')

    plt.show()


def heatmap_plots(df_):
    """ Plot a heatmap comparing two numerical features. """
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


def plot_correlation_matrix(df_, numerical_feature_names_):
    """" Plot a correlation matrix of all features which where
     identified as numerical based on their data type. (Opens in browser)"""
    # Generate Pearson correlation
    data = [
        go.Heatmap(
            z = df_[numerical_feature_names_].astype(float).corr().values,
            x = df_[numerical_feature_names_].columns.values,
            y = df_[numerical_feature_names_].columns.values,
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




def misc_plots(df_):
    """ A function for creating miscellaneous plots on the fly. """

    # Looking at years over 18 vs total working years
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
