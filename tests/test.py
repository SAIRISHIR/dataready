import pandas as pd
from dataready import DataReady

df = pd.read_csv("student_performance.csv")   # ← change this to your CSV filename
dr = DataReady(df, target_col="gender")  # ← change to your target column, or remove if none

dr.scan()                   # prints the quality report
clean_df = dr.fix()         # interactive — asks before each fix
# clean_df = dr.fix(auto_approve=True)  # applies all fixes automatically

dr.export_report("report.html")  # saves an HTML report you can open in browser