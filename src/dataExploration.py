


def createProfilingReports(df, dfAttrition, dfNoAttrition):
    from ydata_profiling import ProfileReport
    

    # Generate report of all intial data
    profile = ProfileReport(df, title="Full Data")
    profile.to_file("data/reports/fullDataReport.html")

    dfAttritionProfile = ProfileReport(dfAttrition, title="Attrition")
    dfNoAttritionProfile = ProfileReport(dfNoAttrition, title="No Attrition")
    comparison_report = dfNoAttritionProfile.compare(dfAttritionProfile)
    comparison_report.to_file("data/reports/attritionComparisonReport.html")