import pandas as pd
import numpy as np

# ---------------------------
# 1. Load Data & Compute Basic Metrics
# ---------------------------
file_path = "../raw_data/MRegularSeasonDetailedResults.csv"
df = pd.read_csv(file_path)

# Load team names
teams_df = pd.read_csv("../raw_data/MTeams.csv")

# Compute possessions (per game) for both winners and losers
df["WPoss"] = df["WFGA"] - df["WOR"] + df["WTO"] + (0.475 * df["WFTA"])
df["LPoss"] = df["LFGA"] - df["LOR"] + df["LTO"] + (0.475 * df["LFTA"])

# Compute Offensive & Defensive Ratings (per 100 possessions)
df["WORtg"] = (df["WScore"] / df["WPoss"]) * 100
df["LORtg"] = (df["LScore"] / df["LPoss"]) * 100
# For each game, winner's DRtg is the loser's ORtg and vice versa:
df["WDRtg"] = df["LORtg"]
df["LDRtg"] = df["WORtg"]

# ---------------------------
# 2. Create a Unified Game-Level Dataset
# ---------------------------
# For winners, the opponent is the losing team; for losers, the opponent is the winning team.
win_df = df[["Season", "WTeamID", "WScore", "WPoss", "WORtg", "WDRtg", "LTeamID", 
             "WLoc", "NumOT", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]].copy()
lose_df = df[["Season", "LTeamID", "LScore", "LPoss", "LORtg", "LDRtg", "WTeamID", 
              "WLoc", "NumOT", "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]].copy()

# Rename columns so both dataframes share the same names:
win_df.rename(columns={
    "WTeamID": "TeamID", "WScore": "Score", "WPoss": "Poss", "WORtg": "ORtg", "WDRtg": "DRtg", "LTeamID": "OpponentID",
    "WFGM": "FGM", "WFGA": "FGA", "WFGM3": "FGM3", "WFGA3": "FGA3", "WFTM": "FTM", "WFTA": "FTA", 
    "WOR": "OR", "WDR": "DR", "WAst": "Ast", "WTO": "TO", "WStl": "Stl", "WBlk": "Blk", "WPF": "PF"
}, inplace=True)
lose_df.rename(columns={
    "LTeamID": "TeamID", "LScore": "Score", "LPoss": "Poss", "LORtg": "ORtg", "LDRtg": "DRtg", "WTeamID": "OpponentID",
    "LFGM": "FGM", "LFGA": "FGA", "LFGM3": "FGM3", "LFGA3": "FGA3", "LFTM": "FTM", "LFTA": "FTA", 
    "LOR": "OR", "LDR": "DR", "LAst": "Ast", "LTO": "TO", "LStl": "Stl", "LBlk": "Blk", "LPF": "PF"
}, inplace=True)

# Add win/loss indicator
win_df["WinGame"] = 1
lose_df["WinGame"] = 0

# Adjust WLoc for perspective of each team
win_df["HomeGame"] = win_df["WLoc"].apply(lambda x: 1 if x == "H" else 0)
win_df["AwayGame"] = win_df["WLoc"].apply(lambda x: 1 if x == "A" else 0)
win_df["NeutralGame"] = win_df["WLoc"].apply(lambda x: 1 if x == "N" else 0)

lose_df["HomeGame"] = lose_df["WLoc"].apply(lambda x: 1 if x == "A" else 0)
lose_df["AwayGame"] = lose_df["WLoc"].apply(lambda x: 1 if x == "H" else 0)
lose_df["NeutralGame"] = lose_df["WLoc"].apply(lambda x: 1 if x == "N" else 0)

# Calculate scoring margin
win_df["ScoreMargin"] = win_df["Score"] - lose_df["Score"]
lose_df["ScoreMargin"] = lose_df["Score"] - win_df["Score"]

# Combine into one dataset (each row is one game from the perspective of one team)
games = pd.concat([win_df, lose_df], ignore_index=True)

# ---------------------------
# 3. First Pass – Compute Raw Season Averages
# ---------------------------
team_averages = games.groupby(["Season", "TeamID"]).agg({
    "Score": "mean",
    "Poss": "mean",
    "ORtg": "mean",
    "DRtg": "mean",
    "WinGame": "mean",  # Win percentage
    "HomeGame": "mean",  # Percentage of home games
    "AwayGame": "mean",  # Percentage of away games
    "NeutralGame": "mean",  # Percentage of neutral games
    "ScoreMargin": "mean",  # Average scoring margin
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
    "AdjD": "mean",
    "WinGame": "mean",
    "HomeGame": "mean",
    "AwayGame": "mean",
    "NeutralGame": "mean",
    "ScoreMargin": "mean",
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
# 7. Calculate Additional Advanced Metrics
# ---------------------------

# 7.1 Clutch Performance: win percentage in games decided by 5 points or fewer
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

# 7.2 Scoring Distribution Metrics
scoring_metrics = games.groupby(["Season", "TeamID"]).apply(lambda x: pd.Series({
    # Three-point reliance
    "ThreePtRate": (x["FGM3"].sum() * 3) / x["Score"].sum(),
    # Free throw reliance
    "FTRate": x["FTM"].sum() / x["Score"].sum(),
    # Assist percentage (assists per made field goal)
    "AstRate": x["Ast"].sum() / x["FGM"].sum() if x["FGM"].sum() > 0 else 0,
    # Turnover percentage (turnovers per possession)
    "TORate": x["TO"].sum() / x["Poss"].sum(),
    # Offensive rebounding percentage
    "ORRate": x["OR"].sum() / (x["OR"].sum() + x["DR"].sum()),
    # Defensive rebounding percentage
    "DRRate": x["DR"].sum() / (x["OR"].sum() + x["DR"].sum()),
})).reset_index()

# 7.3 Consistency/Volatility Metrics
volatility_metrics = games.groupby(["Season", "TeamID"]).apply(lambda x: pd.Series({
    # Standard deviation of scoring
    "ScoreStdDev": x["Score"].std(),
    # Standard deviation of scoring margin
    "MarginStdDev": x["ScoreMargin"].std(),
    # Standard deviation of offensive rating
    "ORtgStdDev": x["ORtg"].std(),
    # Standard deviation of defensive rating
    "DRtgStdDev": x["DRtg"].std(),
})).reset_index()

# 7.4 Home/Away Performance Split
location_metrics = games.groupby(["Season", "TeamID"]).apply(lambda x: pd.Series({
    # Home win percentage
    "HomeWin%": x[x["HomeGame"] == 1]["WinGame"].mean() if len(x[x["HomeGame"] == 1]) > 0 else np.nan,
    # Away win percentage
    "AwayWin%": x[x["AwayGame"] == 1]["WinGame"].mean() if len(x[x["AwayGame"] == 1]) > 0 else np.nan,
    # Neutral court win percentage
    "NeutralWin%": x[x["NeutralGame"] == 1]["WinGame"].mean() if len(x[x["NeutralGame"] == 1]) > 0 else np.nan,
    # Home-Away ORtg differential
    "HomeAwayORtgDiff": x[x["HomeGame"] == 1]["ORtg"].mean() - x[x["AwayGame"] == 1]["ORtg"].mean() 
                        if len(x[x["HomeGame"] == 1]) > 0 and len(x[x["AwayGame"] == 1]) > 0 else np.nan,
})).reset_index()

# 7.5 Momentum Metrics (Last 10 games)
# Sort games by date within each season
df_sorted = df.sort_values(by=["Season", "DayNum"])

# Function to get last N games for each team in each season
def get_last_n_games(df, n=10):
    last_n_games = []
    
    for season in df["Season"].unique():
        season_df = df[df["Season"] == season]
        
        # Get all teams in this season
        all_teams = set(season_df["WTeamID"].unique()) | set(season_df["LTeamID"].unique())
        
        for team in all_teams:
            # Get games where team won
            team_wins = season_df[season_df["WTeamID"] == team].copy()
            team_wins["TeamID"] = team
            team_wins["Win"] = 1
            
            # Get games where team lost
            team_losses = season_df[season_df["LTeamID"] == team].copy()
            team_losses["TeamID"] = team
            team_losses["Win"] = 0
            
            # Combine and sort by day
            team_games = pd.concat([
                team_wins[["Season", "DayNum", "TeamID", "Win"]],
                team_losses[["Season", "DayNum", "TeamID", "Win"]]
            ]).sort_values("DayNum")
            
            # Get last N games
            if len(team_games) >= n:
                last_games = team_games.tail(n)
                last_n_games.append({
                    "Season": season,
                    "TeamID": team,
                    "Last10Win%": last_games["Win"].mean(),
                    "Last10Games": n
                })
            else:
                # If less than N games, use all available
                last_n_games.append({
                    "Season": season,
                    "TeamID": team,
                    "Last10Win%": team_games["Win"].mean(),
                    "Last10Games": len(team_games)
                })
    
    return pd.DataFrame(last_n_games)

momentum_metrics = get_last_n_games(df_sorted, 10)

# ---------------------------
# 8. Merge All Metrics Together
# ---------------------------
# Merge all the additional metrics with the base adjusted stats
final_stats = team_averages_adj.merge(clutch_stats[["Season", "TeamID", "ClutchWin%"]], 
                                     on=["Season", "TeamID"], how="left")
final_stats = final_stats.merge(scoring_metrics, on=["Season", "TeamID"], how="left")
final_stats = final_stats.merge(volatility_metrics, on=["Season", "TeamID"], how="left")
final_stats = final_stats.merge(location_metrics, on=["Season", "TeamID"], how="left")
final_stats = final_stats.merge(momentum_metrics, on=["Season", "TeamID"], how="left")

# Fill NaN values with appropriate defaults
final_stats["ClutchWin%"] = final_stats["ClutchWin%"].fillna(0.5)  # Default to 50% for teams with no close games
final_stats["HomeWin%"] = final_stats["HomeWin%"].fillna(final_stats["WinGame"])  # Default to overall win%
final_stats["AwayWin%"] = final_stats["AwayWin%"].fillna(final_stats["WinGame"])
final_stats["NeutralWin%"] = final_stats["NeutralWin%"].fillna(final_stats["WinGame"])
final_stats["HomeAwayORtgDiff"] = final_stats["HomeAwayORtgDiff"].fillna(0)

# ---------------------------
# 9. Optional: Iterative Refinement of Opponent Adjustments
# ---------------------------
# Here we run a simple iterative loop to update opponent stats and re-calculate adjustments.
iterations = 3
for i in range(iterations):
    # Use the latest adjusted team averages to update opponent stats
    updated_opponent_stats = final_stats[["Season", "TeamID", "AdjNetRtg", "AdjO", "AdjD"]].copy()
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
        "AdjD": "mean",
        "WinGame": "mean",
        "HomeGame": "mean",
        "AwayGame": "mean",
        "NeutralGame": "mean",
        "ScoreMargin": "mean",
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
    
    # Merge with the additional metrics again
    final_stats = team_averages_adj.merge(clutch_stats[["Season", "TeamID", "ClutchWin%"]], 
                                         on=["Season", "TeamID"], how="left")
    final_stats = final_stats.merge(scoring_metrics, on=["Season", "TeamID"], how="left")
    final_stats = final_stats.merge(volatility_metrics, on=["Season", "TeamID"], how="left")
    final_stats = final_stats.merge(location_metrics, on=["Season", "TeamID"], how="left")
    final_stats = final_stats.merge(momentum_metrics, on=["Season", "TeamID"], how="left")
    
    # Fill NaN values with appropriate defaults
    final_stats["ClutchWin%"] = final_stats["ClutchWin%"].fillna(0.5)
    final_stats["HomeWin%"] = final_stats["HomeWin%"].fillna(final_stats["WinGame"])
    final_stats["AwayWin%"] = final_stats["AwayWin%"].fillna(final_stats["WinGame"])
    final_stats["NeutralWin%"] = final_stats["NeutralWin%"].fillna(final_stats["WinGame"])
    final_stats["HomeAwayORtgDiff"] = final_stats["HomeAwayORtgDiff"].fillna(0)

# ---------------------------
# 10. Add Team Names
# ---------------------------
# Merge team names from MTeams.csv
final_stats_with_names = final_stats.merge(teams_df[["TeamID", "TeamName"]], on="TeamID", how="left")

# Reorder columns to put TeamName near the beginning
cols = final_stats_with_names.columns.tolist()
cols.remove("TeamName")
new_cols = ["Season", "TeamID", "TeamName"] + [col for col in cols if col not in ["Season", "TeamID"]]
final_stats_with_names = final_stats_with_names[new_cols]

# ---------------------------
# 11. Save the Final Enhanced Stats
# ---------------------------
output_path = "../pre_tourney_data/EnhancedTournamentStats.csv"
final_stats_with_names.to_csv(output_path, index=False)
print(f"✅ Enhanced Tournament Stats saved to {output_path}!")

# Print a summary of the new metrics added
print("\nNew metrics added:")
print("1. Clutch Performance: ClutchWin%")
print("2. Scoring Distribution: ThreePtRate, FTRate, AstRate, TORate, ORRate, DRRate")
print("3. Consistency/Volatility: ScoreStdDev, MarginStdDev, ORtgStdDev, DRtgStdDev")
print("4. Home/Away Performance: HomeWin%, AwayWin%, NeutralWin%, HomeAwayORtgDiff")
print("5. Momentum: Last10Win%") 