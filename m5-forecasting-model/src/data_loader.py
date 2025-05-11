import pandas as pd
from src.config import DATA_DIR

def load_sales_data():
    return pd.read_csv(DATA_DIR / "sales_train_validation.csv")

def load_calendar_data():
    return pd.read_csv(DATA_DIR / "calendar.csv")

def melt_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = df.columns[:6]
    value_cols = df.columns[6:]
    melted = pd.melt(df, id_vars=id_cols, value_vars=value_cols,
                    var_name="d", value_name="sales")
    return melted

def merge_calendar(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    return df.merge(calendar, on="d", how="left")
