import pandas as pd

# Load the dataset
file_path = "../pre_tourney_data/EnhancedTournamentStats.csv"  # Update if needed
df = pd.read_csv(file_path)

# Filter for the 2025 season
df_2025 = df[df["Season"] == 2025]

# Sort by NetRtg in descending order and get the top 5 teams
top_5_teams = df_2025.sort_values(by="AdjNetRtg", ascending=False).head(20)

# Display the results
print("🏀 Top 5 Teams by Net Rating (NetRtg) in 2025 🏀")
print(top_5_teams[["TeamName", "AdjNetRtg"]])