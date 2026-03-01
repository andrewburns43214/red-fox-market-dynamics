import main
import pandas as pd

print("\nRUNNING BUILD_DASHBOARD RETURN CAPTURE\n")

df = main.build_dashboard()

print("\nRETURN TYPE:", type(df))

if isinstance(df, pd.DataFrame):
    print("ROWS:", len(df))
    print("COLUMNS:", list(df.columns))
else:
    print("FUNCTION DOES NOT RETURN DATAFRAME")
