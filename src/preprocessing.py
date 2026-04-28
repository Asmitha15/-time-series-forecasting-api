import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    # Remove unwanted columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Select correct columns
    df = df[["Date", "State", "Total"]]

    # Rename
    df.columns = ["Date", "State", "Sales"]

    return df


def preprocess_data(df: pd.DataFrame, freq: str = "W") -> dict:
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df = df.sort_values("Date")

    state_data = {}

    for state in df["State"].unique():
        state_df = df[df["State"] == state].copy()
        state_df.set_index("Date", inplace=True)

        # Weekly resample
        state_df = state_df.resample(freq).sum()

        # Ensure numeric
        state_df["Sales"] = pd.to_numeric(state_df["Sales"], errors="coerce")

        # Fill missing
        state_df["Sales"] = state_df["Sales"].interpolate()
        state_df["Sales"] = state_df["Sales"].ffill().bfill()

        state_data[state] = state_df

    return state_data