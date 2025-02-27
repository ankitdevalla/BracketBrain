import pandas as pd

# Load dataset
file_path = "../raw_data/MRegularSeasonDetailedResults.csv"  # Update if needed
df = pd.read_csv(file_path)

# Compute Possessions for both winning and losing teams
df["WPoss"] = df["WFGA"] - df["WOR"] + df["WTO"] + (0.475 * df["WFTA"])
df["LPoss"] = df["LFGA"] - df["LOR"] + df["LTO"] + (0.475 * df["LFTA"])

# Compute Offensive and Defensive Ratings
df["WORtg"] = (df["WScore"] / df["WPoss"]) * 100
df["LORtg"] = (df["LScore"] / df["LPoss"]) * 100
df["WDRtg"] = df["LORtg"]  # Winner's DRtg is Loser's ORtg
df["LDRtg"] = df["WORtg"]  # Loser's DRtg is Winner's ORtg

# Separate winning and losing team stats
win_columns = ["WTeamID", "WScore", "WPoss", "WORtg", "WDRtg"]
lose_columns = ["LTeamID", "LScore", "LPoss", "LORtg", "LDRtg"]

# Rename columns for consistency
win_df = df[["Season"] + win_columns].copy()
lose_df = df[["Season"] + lose_columns].copy()
rename_dict = {col: col[1:] for col in win_columns}
win_df.rename(columns=rename_dict, inplace=True)
rename_dict = {col: col[1:] for col in lose_columns}
lose_df.rename(columns=rename_dict, inplace=True)

# Merge both winners & losers into a single dataset
team_stats = pd.concat([win_df, lose_df])

# **First Pass: Compute raw season-long team averages**
team_averages = team_stats.groupby(["Season", "TeamID"]).agg({
    "Score": "mean",
    "Poss": "mean",
    "ORtg": "mean",
    "DRtg": "mean"
}).reset_index()

# **Compute Initial Net Rating**
team_averages["NetRtg"] = team_averages["ORtg"] - team_averages["DRtg"]

# **First-pass opponent stats for SOS adjustment**
opponent_stats = team_averages[["Season", "TeamID", "NetRtg", "ORtg", "DRtg"]]
opponent_stats.columns = ["Season", "OpponentID", "Opp_NetRtg", "Opp_ORtg", "Opp_DRtg"]

# **Second Pass: Compute Adjusted Ratings**
# Merge games with opponent stats
df = df.merge(opponent_stats, left_on=["Season", "LTeamID"], right_on=["Season", "OpponentID"])

# Adjust Ortg based on Opponent Drtg
df["WAdjO"] = df["WORtg"] * (100 / df["Opp_DRtg"])
df["LAdjO"] = df["LORtg"] * (100 / df["Opp_DRtg"])

# Adjust Drtg based on Opponent Ortg
df["WAdjD"] = df["WDRtg"] * (100 / df["Opp_ORtg"])
df["LAdjD"] = df["LDRtg"] * (100 / df["Opp_ORtg"])

# Separate winners & losers again with new adjusted metrics
win_columns += ["WAdjO", "WAdjD"]
lose_columns += ["LAdjO", "LAdjD"]

win_df = df[["Season"] + win_columns].copy()
lose_df = df[["Season"] + lose_columns].copy()
rename_dict = {col: col[1:] for col in win_columns}
win_df.rename(columns=rename_dict, inplace=True)
rename_dict = {col: col[1:] for col in lose_columns}
lose_df.rename(columns=rename_dict, inplace=True)

# Merge back into team stats
team_stats = pd.concat([win_df, lose_df])

# **Final Pass: Compute Adjusted Season-Long Averages**
team_averages = team_stats.groupby(["Season", "TeamID"]).agg({
    "Score": "mean",
    "Poss": "mean",
    "ORtg": "mean",
    "DRtg": "mean",
    "AdjO": "mean",
    "AdjD": "mean"
}).reset_index()

# **Compute Final Net Rating with Adjusted Metrics**
team_averages["AdjNetRtg"] = team_averages["AdjO"] - team_averages["AdjD"]

# **Recompute Strength of Schedule with Adjusted Ratings**
sos = df.groupby(["Season", "WTeamID"])[["Opp_NetRtg", "Opp_ORtg", "Opp_DRtg"]].mean().reset_index()
sos.columns = ["Season", "TeamID", "SOS_NetRtg", "SOS_ORtg", "SOS_DRtg"]
team_averages = team_averages.merge(sos, on=["Season", "TeamID"], how="left")

# **Compute Expected Win% using Adjusted Ratings**
team_averages["Expected Win%"] = (team_averages["AdjNetRtg"] - team_averages["SOS_NetRtg"]).rank(pct=True)

# **Compute Actual Win%**
actual_win_pct = df.groupby(["Season", "WTeamID"]).size().reset_index(name="Wins").merge(
    df.groupby(["Season", "LTeamID"]).size().reset_index(name="Losses"), left_on=["Season", "WTeamID"],
    right_on=["Season", "LTeamID"], how="left").fillna(0)
actual_win_pct["Total Games"] = actual_win_pct["Wins"] + actual_win_pct["Losses"]
actual_win_pct["Actual Win%"] = actual_win_pct["Wins"] / actual_win_pct["Total Games"]

# Merge Actual Win% into dataset
team_averages = team_averages.merge(actual_win_pct[["Season", "WTeamID", "Actual Win%"]],
                                    left_on=["Season", "TeamID"], right_on=["Season", "WTeamID"], how="left")

# **Compute Luck Factor**
team_averages["Luck"] = team_averages["Actual Win%"] - team_averages["Expected Win%"]

# **Compute Clutch Performance (Win % in Close Games)**
close_games = df[(df["WScore"] - df["LScore"]).abs() <= 5]
clutch_wins = close_games.groupby(["Season", "WTeamID"]).size().reset_index(name="ClutchWins")
clutch_games = close_games.groupby(["Season", "WTeamID"]).size().reset_index(name="TotalClutchGames")
clutch_performance = clutch_wins.merge(clutch_games, on=["Season", "WTeamID"], how="left")
clutch_performance["ClutchWin%"] = clutch_performance["ClutchWins"] / clutch_performance["TotalClutchGames"]
team_averages = team_averages.merge(clutch_performance[["Season", "WTeamID", "ClutchWin%"]],
                                    left_on=["Season", "TeamID"], right_on=["Season", "WTeamID"], how="left").fillna(0)

# Save the dataset
output_path = "PreTournamentAdvancedStats.csv"
team_averages.to_csv(output_path, index=False)

print(f"✅ Adjusted Advanced Stats saved to {output_path}!")
