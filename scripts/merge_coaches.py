import pandas as pd

dfs = []
for year in range(2003, 2026):
    df = pd.read_csv(f"../raw_data/coaches_data/coaches{year}.csv", header=1)
    # Rename duplicate columns
    df.rename(columns={
        "W": "Season Wins",
        "W.1": "School Wins",
        "W.2": "Career Wins",
        "L": "Season Losses",
        "L.1": "School Losses",
        "L.2": "Career Losses",
        "W-L%": "Season W-L%",
        "W-L%.1": "School W-L%",
        "W-L%.2": "Career W-L%",
        "NCAA": "School NCAA",
        "NCAA.1": "Career NCAA",
        "S16": "School S16",
        "S16.1": "Career S16",
        "FF": "School FF",
        "FF.1": "Career FF",
        "Chmp": "School Chmp",
        "Chmp.1": "Career Chmp",
    }, inplace=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["Coach"] = df["Coach"].str.replace("*", "", regex=False)
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv("../raw_data/coaches_merged.csv", index=False)
