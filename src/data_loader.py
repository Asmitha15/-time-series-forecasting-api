import pandas as pd

def load_and_prepare_data(path):
    df = pd.read_excel(path)

    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        "date": "date",
        "state": "state",
        "total": "sales"
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    df = df.dropna(subset=["date", "sales"])
    df["state"] = df["state"].astype(str).str.strip()

    df = df.sort_values("date")
    df = df.set_index("date")

    state_data = {}

    for state, group in df.groupby("state"):
        g = group[["sales"]].copy()
        g = g.resample("W").mean()
        g["sales"] = g["sales"].interpolate()
        state_data[state] = g

    return state_data