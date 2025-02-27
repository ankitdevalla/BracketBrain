import pandas as pd
import numpy as np

# ---------------------------
# 1. Load Data & Compute Basic Metrics
# ---------------------------
file_path = "../raw_data/MRegularSeasonDetailedResults.csv"  # Update if needed
df = pd.read_csv(file_path)

# Compute possessions (per game) for both winners and losers
df["WPoss"] = df["WFGA"] - df["WOR"] + df["WTO"] + (0.475 * df["WFTA"])
df["LPoss"] = df["LFGA"] - df["LOR"] + df["LTO"] + (0.475 * df["LFTA"])

# Compute Offensive & Defensive Ratings (per 100 possessions)
df["WORtg"] = (df["WScore"] / df["WPoss"]) * 100
df["LORtg"] = (df["LScore"] / df["LPoss"]) * 100
# For each game, winner’s DRtg is the loser’s ORtg and vice versa:
df["WDRtg"] = df["LORtg"]
df["LDRtg"] = df["WORtg"]

# ---------------------------
# 2. Create a Unified Game-Level Dataset
# ---------------------------
# For winners, the opponent is the losing team; for losers, the opponent is the winning team.
win_df = df[["Season", "WTeamID", "WScore", "WPoss", "WORtg", "WDRtg", "LTeamID"]].copy()
lose_df = df[["Season", "LTeamID", "LScore", "LPoss", "LORtg", "LDRtg", "WTeamID"]].copy()

# Rename columns so both dataframes share the same names:
win_df.rename(columns={
    "WTeamID": "TeamID",
    "WScore": "Score",
    "WPoss": "Poss",
    "WORtg": "ORtg",
    "WDRtg": "DRtg",
    "LTeamID": "OpponentID"
}, inplace=True)
lose_df.rename(columns={
    "LTeamID": "TeamID",
    "LScore": "Score",
    "LPoss": "Poss",
    "LORtg": "ORtg",
    "LDRtg": "DRtg",
    "WTeamID": "OpponentID"
}, inplace=True)

# Combine into one dataset (each row is one game from the perspective of one team)
games = pd.concat([win_df, lose_df], ignore_index=True)

# ---------------------------
# 3. First Pass – Compute Raw Season Averages
# ---------------------------
team_averages = games.groupby(["Season", "TeamID"]).agg({
    "Score": "mean",
    "Poss": "mean",
    "ORtg": "mean",
    "DRtg": "mean"
}).reset_index()
team_averages["NetRtg"] = team_averages["ORtg"] - team_averages["DRtg"]

# Create opponent stats from these averages; these will be merged back into each game.
opponent_stats = team_averages[["Season", "TeamID", "NetRtg", "ORtg", "DRtg"]].copy()
opponent_stats.rename(columns={
    "TeamID": "OpponentID",
    "NetRtg": "Opp_NetRtg",
    "ORtg": "Opp_ORtg",
    "DRtg": "Opp_DRtg"
}, inplace=True)

# ---------------------------
# 4. Merge Opponent Stats for Adjusted Ratings
# ---------------------------
# For each game record, merge in the opponent's season averages
games_adj = games.merge(opponent_stats, on=["Season", "OpponentID"], how="left")

# Adjusted Offensive Rating: scale ORtg by opponent defensive efficiency
games_adj["AdjO"] = games_adj["ORtg"] * (100 / games_adj["Opp_DRtg"])
# Adjusted Defensive Rating: scale DRtg by opponent offensive efficiency
games_adj["AdjD"] = games_adj["DRtg"] * (100 / games_adj["Opp_ORtg"])

# ---------------------------
# 5. Aggregate Adjusted Metrics to the Season Level
# ---------------------------
team_averages_adj = games_adj.groupby(["Season", "TeamID"]).agg({
    "Score": "mean",
    "Poss": "mean",
    "ORtg": "mean",
    "DRtg": "mean",
    "AdjO": "mean",
    "AdjD": "mean"
}).reset_index()
team_averages_adj["AdjNetRtg"] = team_averages_adj["AdjO"] - team_averages_adj["AdjD"]

# Compute strength of schedule (SOS) as the average opponent adjusted net rating
sos = games_adj.groupby(["Season", "TeamID"]).agg({
    "Opp_NetRtg": "mean",
    "Opp_ORtg": "mean",
    "Opp_DRtg": "mean"
}).reset_index()
sos.rename(columns={
    "Opp_NetRtg": "SOS_NetRtg",
    "Opp_ORtg": "SOS_ORtg",
    "Opp_DRtg": "SOS_DRtg"
}, inplace=True)
team_averages_adj = team_averages_adj.merge(sos, on=["Season", "TeamID"], how="left")

# ---------------------------
# 6. Compute Expected Win% via a Logistic Function
# ---------------------------
# A typical scaling factor is around 11; adjust as needed.
s = 11
team_averages_adj["Expected Win%"] = 1 / (1 + np.exp(-((team_averages_adj["AdjNetRtg"] - team_averages_adj["SOS_NetRtg"]) / s)))

# ---------------------------
# 7. Compute Actual Win%, Luck, and Clutch Performance
# ---------------------------
# Actual Win% based on wins/losses from the original data
win_counts = df.groupby(["Season", "WTeamID"]).size().reset_index(name="Wins")
lose_counts = df.groupby(["Season", "LTeamID"]).size().reset_index(name="Losses")
win_counts.rename(columns={"WTeamID": "TeamID"}, inplace=True)
lose_counts.rename(columns={"LTeamID": "TeamID"}, inplace=True)
actual_win_pct = pd.merge(win_counts, lose_counts, on=["Season", "TeamID"], how="outer").fillna(0)
actual_win_pct["TotalGames"] = actual_win_pct["Wins"] + actual_win_pct["Losses"]
actual_win_pct["Actual Win%"] = actual_win_pct["Wins"] / actual_win_pct["TotalGames"]

team_averages_adj = team_averages_adj.merge(actual_win_pct[["Season", "TeamID", "Actual Win%"]], on=["Season", "TeamID"], how="left")

# Luck = Actual Win% minus Expected Win%
team_averages_adj["Luck"] = team_averages_adj["Actual Win%"] - team_averages_adj["Expected Win%"]

# Clutch Performance: win percentage in games decided by 5 points or fewer
close_games = df[(df["WScore"] - df["LScore"]).abs() <= 5]
close_win_df = close_games[["Season", "WTeamID"]].copy()
close_win_df.rename(columns={"WTeamID": "TeamID"}, inplace=True)
close_win_df["ClutchWin"] = 1
close_lose_df = close_games[["Season", "LTeamID"]].copy()
close_lose_df.rename(columns={"LTeamID": "TeamID"}, inplace=True)
close_lose_df["ClutchWin"] = 0
close_games_df = pd.concat([close_win_df, close_lose_df], ignore_index=True)
clutch_stats = close_games_df.groupby(["Season", "TeamID"]).agg(
    TotalClutchGames=("ClutchWin", "count"),
    ClutchWins=("ClutchWin", "sum")
).reset_index()
clutch_stats["ClutchWin%"] = clutch_stats["ClutchWins"] / clutch_stats["TotalClutchGames"]

team_averages_adj = team_averages_adj.merge(clutch_stats[["Season", "TeamID", "ClutchWin%"]], on=["Season", "TeamID"], how="left").fillna(0)

# ---------------------------
# 8. Optional: Iterative Refinement of Opponent Adjustments
# ---------------------------
# Here we run a simple iterative loop to update opponent stats and re-calculate adjustments.
iterations = 3
for i in range(iterations):
    # Use the latest adjusted team averages to update opponent stats
    updated_opponent_stats = team_averages_adj[["Season", "TeamID", "AdjNetRtg", "AdjO", "AdjD"]].copy()
    updated_opponent_stats.rename(columns={
        "TeamID": "OpponentID",
        "AdjNetRtg": "Opp_NetRtg",
        "AdjO": "Opp_ORtg",
        "AdjD": "Opp_DRtg"
    }, inplace=True)
    
    # Drop previous opponent-related columns (if they exist) and merge new ones
    games_temp = games.drop(columns=["Opp_NetRtg", "Opp_ORtg", "Opp_DRtg", "AdjO", "AdjD"], errors="ignore")
    games_temp = games_temp.merge(updated_opponent_stats, on=["Season", "OpponentID"], how="left")
    
    # Recompute adjusted ratings for each game
    games_temp["AdjO"] = games_temp["ORtg"] * (100 / games_temp["Opp_DRtg"])
    games_temp["AdjD"] = games_temp["DRtg"] * (100 / games_temp["Opp_ORtg"])
    
    # Re-aggregate season averages with the new adjustments
    team_averages_adj = games_temp.groupby(["Season", "TeamID"]).agg({
        "Score": "mean",
        "Poss": "mean",
        "ORtg": "mean",
        "DRtg": "mean",
        "AdjO": "mean",
        "AdjD": "mean"
    }).reset_index()
    team_averages_adj["AdjNetRtg"] = team_averages_adj["AdjO"] - team_averages_adj["AdjD"]
    
    # Update SOS from these recalculated game-level adjustments
    sos = games_temp.groupby(["Season", "TeamID"]).agg({
        "Opp_NetRtg": "mean",
        "Opp_ORtg": "mean",
        "Opp_DRtg": "mean"
    }).reset_index()
    sos.rename(columns={
        "Opp_NetRtg": "SOS_NetRtg",
        "Opp_ORtg": "SOS_ORtg",
        "Opp_DRtg": "SOS_DRtg"
    }, inplace=True)
    team_averages_adj = team_averages_adj.merge(sos, on=["Season", "TeamID"], how="left")
    
    # Update Expected Win% using the logistic function
    team_averages_adj["Expected Win%"] = 1 / (1 + np.exp(-((team_averages_adj["AdjNetRtg"] - team_averages_adj["SOS_NetRtg"]) / s)))
    # (Note: Actual Win% and Clutch performance remain unchanged in the iteration)

# ---------------------------
# 9. Save the Final Adjusted Stats
# ---------------------------
output_path = "PreTournamentAdvancedStats_Adjusted.csv"
team_averages_adj.to_csv(output_path, index=False)
print(f"✅ Adjusted Advanced Stats saved to {output_path}!")
