import pandas as pd
import os
import glob
import re
from fuzzywuzzy import process, fuzz

# Define the paths
coaches_data_path = '../raw_data/coaches_data/'
teams_file = '../raw_data/MTeams.csv'

# Load the MTeams.csv file to get TeamIDs
try:
    teams_df = pd.read_csv(teams_file)
    print(f"Loaded teams data with {len(teams_df)} teams")
    print(f"Sample teams data:")
    print(teams_df.head())
except Exception as e:
    print(f"Error loading teams file: {e}")
    teams_df = None

# Create a dictionary of team name variations to handle special cases
team_name_variations = {
    "Texas A&M-Corpus Christi": "TAM C. Christi",
    "Texas A&M Corpus Christi": "TAM C. Christi",
    "Texas A&M-CC": "TAM C. Christi",
    "UNC-Wilmington": "UNC Wilmington",
    "UNCW": "UNC Wilmington",
    "NC-Wilmington": "UNC Wilmington",
    "UNC-Asheville": "UNC Asheville",
    "UNCA": "UNC Asheville",
    "NC-Asheville": "UNC Asheville",
    "Miami (FL)": "Miami FL",
    "Miami (Ohio)": "Miami OH",
    "Miami FL": "Miami FL",
    "Miami OH": "Miami OH",
    "Miami, FL": "Miami FL",
    "Miami, Ohio": "Miami OH",
    "UCF": "Central Florida",
    "UC-Irvine": "UC Irvine",
    "UC-Davis": "UC Davis",
    "UC-Santa Barbara": "UC Santa Barbara",
    "UCSB": "UC Santa Barbara",
    "UC-Riverside": "UC Riverside",
    "Penn": "Pennsylvania",
    "Pitt": "Pittsburgh",
    "UConn": "Connecticut",
    "USC": "Southern California",
    "SMU": "Southern Methodist",
    "BYU": "Brigham Young",
    "VCU": "Virginia Commonwealth",
    "LSU": "Louisiana State",
    "Ole Miss": "Mississippi",
    "UNLV": "Nevada Las Vegas",
    "UAB": "Alabama Birmingham",
    "UTEP": "Texas El Paso",
    "UTSA": "Texas San Antonio",
    "UIC": "Illinois Chicago",
    "Saint Mary's": "Saint Mary's CA",
    "St. Mary's": "Saint Mary's CA",
    "Saint Mary's (CA)": "Saint Mary's CA",
    "St. Mary's (CA)": "Saint Mary's CA",
    "College of Charleston": "Charleston",
    "Charleston Southern": "Charleston Southern",
    "CSU": "Charleston Southern",
    "Cal": "California",
    "UMass": "Massachusetts",
    "UMKC": "Missouri KC",
    "UT Arlington": "Texas Arlington",
    "UTA": "Texas Arlington",
    "UTRGV": "Texas Rio Grande Valley",
}

# Function to match a team name to the MTeams.csv
def match_team_name(coach_team_name, teams_df):
    """Match a team name from coaches file to MTeams.csv and return the TeamID"""
    if not coach_team_name or pd.isna(coach_team_name):
        return None, None
    
    # Clean the team name
    coach_team_name = str(coach_team_name).strip()
    
    # Check if we have a known variation
    if coach_team_name in team_name_variations:
        coach_team_name = team_name_variations[coach_team_name]
    
    # Try direct match first
    direct_match = teams_df[teams_df['TeamName'] == coach_team_name]
    if not direct_match.empty:
        return direct_match.iloc[0]['TeamID'], direct_match.iloc[0]['TeamName']
    
    # Try removing common suffixes/prefixes
    cleaned_name = re.sub(r'University|College|State|Saint|St\.|\(.*\)', '', coach_team_name).strip()
    if cleaned_name:
        direct_match = teams_df[teams_df['TeamName'].str.contains(cleaned_name, case=False, regex=False)]
        if not direct_match.empty:
            return direct_match.iloc[0]['TeamID'], direct_match.iloc[0]['TeamName']
    
    # Try fuzzy matching
    team_names = teams_df['TeamName'].tolist()
    best_match = process.extractOne(coach_team_name, team_names, scorer=fuzz.token_sort_ratio, score_cutoff=75)
    
    if best_match:
        match_name = best_match[0]
        match_row = teams_df[teams_df['TeamName'] == match_name]
        if not match_row.empty:
            return match_row.iloc[0]['TeamID'], match_row.iloc[0]['TeamName']
    
    print(f"Could not match team name: '{coach_team_name}'")
    return None, None

# Get all CSV files in the coaches_data directory
coach_files = glob.glob(os.path.join(coaches_data_path, 'coaches*.csv'))

# Initialize an empty list to store dataframes
all_coaches_data = []
team_match_failures = []

# Process each coach file
for file in coach_files:
    # Extract season from filename (e.g., 'coaches2003.csv' -> 2003)
    season = int(os.path.basename(file).replace('coaches', '').replace('.csv', ''))
    print(f"\nProcessing file for season {season}: {file}")
    
    # Try different ways to read the file
    df = None
    
    # Try reading with different header options
    for header_option in [None, 0, 1]:
        try:
            temp_df = pd.read_csv(file, header=header_option)
            # Check if this looks like a valid dataframe with coach data
            potential_columns = ['Coach', 'School', 'NCAA', 'W', 'L', 'S16', 'FF', 'Chmp']
            found_columns = [col for col in potential_columns if any(col in str(c) for c in temp_df.columns)]
            
            if len(found_columns) >= 2:  # If we found at least 2 expected columns
                df = temp_df
                print(f"  Successfully read file with header={header_option}")
                print(f"  Found columns: {found_columns}")
                print(f"  All columns: {df.columns.tolist()}")
                break
        except Exception as e:
            print(f"  Error reading with header={header_option}: {e}")
    
    if df is None:
        print(f"  Could not read file {file} with any header option. Skipping.")
        continue
    
    # Add Season column
    df['Season'] = season
    
    # Function to find and process tournament achievement columns
    def find_and_process_column(prefix, suffix=None):
        """Find column with prefix and optional suffix, convert to numeric"""
        column = None
        
        # First look for career version (with suffix)
        if suffix:
            for col in df.columns:
                if prefix in str(col) and suffix in str(col):
                    column = col
                    print(f"  Found {prefix} {suffix} column: {column}")
                    break
        
        # If not found and no suffix specified, look for any column with prefix
        if column is None:
            for col in df.columns:
                if prefix in str(col) and (not suffix or suffix in str(col)):
                    column = col
                    print(f"  Using general {prefix} column: {column}")
                    break
        
        # Process the column if found
        if column:
            return pd.to_numeric(df[column], errors='coerce').fillna(0)
        else:
            print(f"  No {prefix} column found, using zeros")
            return 0
    
    # Find and process NCAA tournament appearances
    df['career NCAA'] = find_and_process_column('NCAA', '.1')
    
    # Find and process Sweet 16 appearances
    df['career S16'] = find_and_process_column('S16', '.1')
    
    # Find and process Final Four appearances
    df['career FF'] = find_and_process_column('FF', '.1')
    
    # Find and process Championship appearances
    df['career Chmp'] = find_and_process_column('Chmp', '.1')
    
    # Look for team/school name column
    school_column = None
    for col in df.columns:
        if 'School' in str(col) or 'Team' in str(col):
            school_column = col
            print(f"  Found school column: {school_column}")
            break
    
    if not school_column:
        print("  No school name column found. Skipping file.")
        continue
    
    # Look for coach name column
    coach_column = None
    for col in df.columns:
        if 'Coach' in str(col):
            coach_column = col
            print(f"  Found coach column: {coach_column}")
            break
    
    if not coach_column:
        print("  No coach name column found")
        df['coach_name'] = 'Unknown'
    else:
        df.rename(columns={coach_column: 'coach_name'}, inplace=True)
    
    # Match team names to TeamIDs
    if teams_df is not None:
        df['TeamID'] = None
        df['MatchedTeamName'] = None
        
        for idx, row in df.iterrows():
            team_name = row[school_column]
            team_id, matched_name = match_team_name(team_name, teams_df)
            df.at[idx, 'TeamID'] = team_id
            df.at[idx, 'MatchedTeamName'] = matched_name
            
            if team_id is None:
                team_match_failures.append({
                    'Season': season, 
                    'OriginalName': team_name
                })
        
        # Report matching results
        match_count = df['TeamID'].notnull().sum()
        print(f"  Matched {match_count} out of {len(df)} team names to TeamIDs")
    else:
        print("  No teams data available for matching TeamIDs")
        continue
    
    # Keep only rows with valid TeamIDs
    df = df[df['TeamID'].notnull()]
    
    if len(df) == 0:
        print("  No valid team matches found. Skipping file.")
        continue
    
    # Keep only the columns we need
    try:
        df = df[['TeamID', 'Season', 'coach_name', 'career NCAA', 'career S16', 'career FF', 'career Chmp']]
        df['TeamID'] = df['TeamID'].astype(int)
        all_coaches_data.append(df)
        print(f"  Successfully processed file with {len(df)} coaches")
    except Exception as e:
        print(f"  Error selecting columns: {e}")
        print(f"  Available columns: {df.columns.tolist()}")
        continue

# Concatenate all dataframes
if all_coaches_data:
    merged_coaches = pd.concat(all_coaches_data, ignore_index=True)
    
    # Fill any missing values with 0
    for col in ['career NCAA', 'career S16', 'career FF', 'career Chmp']:
        merged_coaches[col] = merged_coaches[col].fillna(0)
    
    # Save the merged data
    merged_coaches.to_csv('../pre_tourney_data/coaches_merged.csv', index=False)
    
    # Save the list of teams that couldn't be matched
    if team_match_failures:
        failures_df = pd.DataFrame(team_match_failures)
        failures_df.to_csv('../pre_tourney_data/unmatched_team_names.csv', index=False)
        print(f"\nSaved {len(failures_df)} unmatched team names to unmatched_team_names.csv")
    
    print(f"\nSuccessfully merged {len(all_coaches_data)} coach files.")
    print(f"Final dataset shape: {merged_coaches.shape}")
    print(f"Columns in final dataset: {merged_coaches.columns.tolist()}")
else:
    print("\nNo valid coach files found to merge.")
