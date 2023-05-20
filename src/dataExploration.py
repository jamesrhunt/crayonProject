import numpy as np
def createProfilingReports(df, dfAttrition, dfNoAttrition):
    from ydata_profiling import ProfileReport
    

    # Generate report of all intial data
    profile = ProfileReport(df, title="Full Data")
    profile.to_file("data/reports/fullDataReport.html")

    dfAttritionProfile = ProfileReport(dfAttrition, title="Attrition")
    dfNoAttritionProfile = ProfileReport(dfNoAttrition, title="No Attrition")
    comparison_report = dfNoAttritionProfile.compare(dfAttritionProfile)
    comparison_report.to_file("data/reports/attritionComparisonReport.html")


def categoricAndNumeric(df):
    # 
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

    print(categoricalFeatureNames_)
    print(numericalFeatureNames_)
    print(np.unique(dtypes))
    # show all unique data types
    print(len(categoricalFeatureNames_), len(numericalFeatureNames_), len(dtypes))
    return categoricalFeatureNames_, numericalFeatureNames_
    