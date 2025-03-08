import pandas as pd

# --- Step 1: Load Game-by-Game Results ---
df = pd.read_csv("../raw_data/MRegularSeasonDetailedResults.csv")

# --- Include DayNum in the winner and loser DataFrames ---
winners = pd.DataFrame({
    'Season': df['Season'],
    'DayNum': df['DayNum'],
    'TeamID': df['WTeamID'],
    'Outcome': 'W',
    'Score': df['WScore'],
    'FGM': df['WFGM'],
    'FGA': df['WFGA'],
    'FGM3': df['WFGM3'],
    'FGA3': df['WFGA3'],
    'FTM': df['WFTM'],
    'FTA': df['WFTA'],
    'OR': df['WOR'],
    'DR': df['WDR'],
    'Ast': df['WAst'],
    'TO': df['WTO'],
    'Stl': df['WStl'],
    'Blk': df['WBlk'],
    'PF': df['WPF'],
    'OpponentID': df['LTeamID']
})

losers = pd.DataFrame({
    'Season': df['Season'],
    'DayNum': df['DayNum'],
    'TeamID': df['LTeamID'],
    'Outcome': 'L',
    'Score': df['LScore'],
    'FGM': df['LFGM'],
    'FGA': df['LFGA'],
    'FGM3': df['LFGM3'],
    'FGA3': df['LFGA3'],
    'FTM': df['LFTM'],
    'FTA': df['LFTA'],
    'OR': df['LOR'],
    'DR': df['LDR'],
    'Ast': df['LAst'],
    'TO': df['LTO'],
    'Stl': df['LStl'],
    'Blk': df['LBlk'],
    'PF': df['LPF'],
    'OpponentID': df['WTeamID']
})

# Combine winners and losers into one DataFrame.
games = pd.concat([winners, losers], ignore_index=True)

# --- Step 2: Compute Per-Game Averages for Each Team per Season ---

# 2A) Calculate possessions for each game using the formula:
# POSS = FGA - OR + TO + (0.475 * FTA)
games['Poss'] = games['FGA'] - games['OR'] + games['TO'] + (0.475 * games['FTA'])

# 2B) Aggregate standard stats and possessions by season/team.
agg_stats = games.groupby(['Season', 'TeamID']).agg(
    Games=('Outcome', 'count'),
    Wins=('Outcome', lambda x: (x == 'W').sum()),
    Avg_Score=('Score', 'mean'),
    Avg_FGM=('FGM', 'mean'),
    Avg_FGA=('FGA', 'mean'),
    Avg_FGM3=('FGM3', 'mean'),
    Avg_FGA3=('FGA3', 'mean'),
    Avg_FTM=('FTM', 'mean'),
    Avg_FTA=('FTA', 'mean'),
    Avg_OR=('OR', 'mean'),
    Avg_DR=('DR', 'mean'),
    Avg_Ast=('Ast', 'mean'),
    Avg_TO=('TO', 'mean'),
    Avg_Stl=('Stl', 'mean'),
    Avg_Blk=('Blk', 'mean'),
    Avg_PF=('PF', 'mean'),
    Avg_Poss=('Poss', 'mean')  # average possessions
).reset_index()

agg_stats['WinPct'] = agg_stats['Wins'] / agg_stats['Games']

# Create a lookup dictionary for win percentage per (Season, TeamID).
win_pct_dict = {(row['Season'], row['TeamID']): row['WinPct'] for _, row in agg_stats.iterrows()}

# --- Step 3: Compute Strength of Schedule (SoS) Metric ---
def get_opponent_win_pct(row):
    return win_pct_dict.get((row['Season'], row['OpponentID']), None)

games['Opponent_WinPct'] = games.apply(get_opponent_win_pct, axis=1)

sos = games.groupby(['Season', 'TeamID']).agg(
    Avg_Opp_WinPct=('Opponent_WinPct', 'mean')
).reset_index()

final_stats = pd.merge(agg_stats, sos, on=['Season', 'TeamID'])

# --- Step 4: Compute Last 30 Days Win Ratio ---
# Compute the season's maximum day number using the original df.
season_max_day = df.groupby('Season')['DayNum'].max().to_dict()
# Map season max day to each game.
games['SeasonMaxDay'] = games['Season'].map(season_max_day)
# Flag games played in the final 30 days of the season.
games['InLast30'] = games['DayNum'] >= (games['SeasonMaxDay'] - 30)

# Filter to only games in the last 30 days.
games_last30 = games[games['InLast30']]

# Group by Season and TeamID to compute last 30 days win ratio.
last30_stats = games_last30.groupby(['Season', 'TeamID']).agg(
    Last30_Games=('Outcome', 'count'),
    Last30_Wins=('Outcome', lambda x: (x == 'W').sum())
).reset_index()

last30_stats['Last30_WinRatio'] = last30_stats['Last30_Wins'] / last30_stats['Last30_Games']

# Merge the last 30 days win ratio into the final stats.
final_stats = pd.merge(final_stats, last30_stats[['Season', 'TeamID', 'Last30_WinRatio']], 
                        on=['Season', 'TeamID'], how='left')

# --- Step 5: Load Team Names and Merge ---
teams = pd.read_csv("../raw_data/MTeams.csv")
final_stats = pd.merge(final_stats, teams[['TeamID', 'TeamName']], on='TeamID', how='left')

# Reorder columns for clarity.
cols = ['Season', 'TeamID', 'TeamName', 'Games', 'Wins', 'WinPct', 'Avg_Score', 'Avg_FGM', 'Avg_FGA',
        'Avg_FGM3', 'Avg_FGA3', 'Avg_FTM', 'Avg_FTA', 'Avg_OR', 'Avg_DR', 'Avg_Ast',
        'Avg_TO', 'Avg_Stl', 'Avg_Blk', 'Avg_PF', 'Avg_Poss', 'Avg_Opp_WinPct', 'Last30_WinRatio']
final_stats = final_stats[cols]

# Display the columns for verification.
# print(final_stats.columns())

# Optionally, save the final results to a CSV file.
final_stats.to_csv("TeamSeasonAverages_with_SoS.csv", index=False)
