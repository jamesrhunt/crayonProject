import pandas as pd
from src.dataExploration import *

# Load the CSV file
df = pd.read_csv('data/Crayon_case_employee-attrition.csv')

# Split into Attrition or not
dfAttrition = df[df['Attrition'] == 'Yes']
dfNoAttrition = df[df['Attrition'] == 'No']

createProfilingReports(df, dfAttrition, dfNoAttrition)



