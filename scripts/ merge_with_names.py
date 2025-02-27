import pandas as pd

# File paths (update if needed)
stats_path = "PreTournamentAdvancedStats_Adjusted.csv"  # Your advanced stats file
teams_path = "../raw_data/MTeams.csv"  # CSV mapping team IDs to names
output_path = "PreTournamentAdvancedStats_withNames.csv"  # Output file

# Load the CSV files
stats_df = pd.read_csv(stats_path)
teams_df = pd.read_csv(teams_path)

# Assume both files have a common column "TeamID"
# and that teams_df contains a column like "TeamName" with the team names.

# Merge the advanced stats with team names
merged_df = stats_df.merge(teams_df, on="TeamID", how="left")

# Save the merged dataframe to a new CSV file
merged_df.to_csv(output_path, index=False)
print(f"✅ Merged table with team names saved to {output_path}!")
