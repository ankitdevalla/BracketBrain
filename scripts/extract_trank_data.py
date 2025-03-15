import pandas as pd
import glob
import os

def load_trank_data():
    """Load and combine all T-Rank data files"""
    print("Looking for T-Rank data files...")
    trank_files = glob.glob("../T-Rank/trank_team_table_data_*.csv")
    
    if not trank_files:
        print("No T-Rank files found. Please check the path.")
        return None
    
    print(f"Found {len(trank_files)} T-Rank data files.")
    all_data = []
    
    for file in trank_files:
        year = int(file.split('_')[-1].split('.')[0])  # Extract year from filename
        print(f"Processing file for year {year}: {file}")
        df = pd.read_csv(file)
        
        # Add year column if not already present
        if 'Year' not in df.columns:
            df['Year'] = year
            
        all_data.append(df)
    
    # Combine all years
    trank_df = pd.concat(all_data, ignore_index=True)
    
    # Clean team names
    trank_df['Team'] = trank_df['Team'].str.strip()
    
    return trank_df

def create_team_name_mapping(trank_df, teams_df):
    """Create a mapping between T-Rank team names and TeamIDs"""
    # Normalize team names for matching
    trank_df['NormalizedName'] = trank_df['Team'].str.lower().str.replace('[^a-z0-9]', '', regex=True)
    teams_df['NormalizedName'] = teams_df['TeamName'].str.lower().str.replace('[^a-z0-9]', '', regex=True)
    
    # Create mapping dictionary
    team_mapping = {}
    for _, team in teams_df.iterrows():
        team_mapping[team['NormalizedName']] = team['TeamID']
    
    # Add exact custom mapping as provided
    custom_team_names = {
        "Saint Mary's": 1388,
        "Saint Joseph's": 1386,
        'Western Kentucky': 1443,
        'Kent St.': 1245,
        'Southern Illinois': 1356,
        'IU Indy': 1237,
        'Western Michigan': 1444,
        'Stephen F. Austin': 1372,
        'Cal St. Fullerton': 1168,
        'Illinois Chicago': 1227,
        'Cal St. Northridge': 1169,
        'North Dakota St.': 1295,
        'Georgia Southern': 1204,
        'Green Bay': 1453,
        'American': 1110,
        'Saint Louis': 1387,
        'Milwaukee': 1454,
        'Charleston': 1158,
        'George Washington': 1203,
        "Mount St. Mary's": 1291,
        'Albany': 1107,
        'Eastern Michigan': 1185,
        'Middle Tennessee': 1292,
        'East Tennessee St.': 1190,
        'Central Michigan': 1141,
        'Little Rock': 1114,
        'Purdue Fort Wayne': 1236,
        'Southeastern Louisiana': 1368,
        'Central Connecticut': 1148,
        'LIU': 1254,
        'Florida Atlantic': 1194,
        'Boston University': 1131,
        'Northern Colorado': 1294,
        'Detroit Mercy': 1178,
        'Eastern Washington': 1186,
        'Western Carolina': 1441,
        'UMKC': 1282,
        'Tennessee Martin': 1404,
        'Western Illinois': 1442,
        'FIU': 1198,
        'UT Rio Grande Valley': 1410,
        'Louisiana Monroe': 1419,
        'South Dakota St.': 1355,
        'Northern Illinois': 1296,
        'Texas A&M Corpus Chris': 1394,
        'Northwestern St.': 1322,
        "Saint Peter's": 1389,
        'Eastern Kentucky': 1184,
        'Central Arkansas': 1146,
        'Coastal Carolina': 1157,
        'UTSA': 1427,
        'USC Upstate': 1367,
        'Fairleigh Dickinson': 1192,
        'Southeast Missouri St.': 1369,
        'Florida Gulf Coast': 1195,
        'North Carolina A&T': 1299,
        'Monmouth': 1284,
        'Mississippi Valley St.': 1290,
        'Saint Francis': 1384,
        'Kennesaw St.': 1244,
        'Charleston Southern': 1149,
        'Cal St. Bakersfield': 1167,
        'Southern': 1380,
        'South Carolina St.': 1354,
        'Eastern Illinois': 1183,
        'Winston Salem St.': 1445,
        'Arkansas Pine Bluff': 1115,
        'The Citadel': 1154,
        'Sacramento St.': 1170,
        'Prairie View A&M': 1341,
        'Loyola Marymount': 1258,
        'Texas Southern': 1411,
        'Grambling St.': 1212,
        'North Carolina Central': 1300,
        'Maryland Eastern Shore': 1271,
        'Abilene Christian': 1101,
        'Northern Kentucky': 1297,
        'Nebraska Omaha': 1303,
        'UMass Lowell': 1262,
        'Houston Christian': 1223,
        'SIU Edwardsville': 1188,
        'St. Thomas': 1472,
        'Queens': 1474,
        'Texas A&M Commerce': 1477
    }
    
    # Create a mapping from normalized names to TeamIDs
    for team_name, team_id in custom_team_names.items():
        normalized_name = team_name.lower().replace('.', '').replace("'", '').replace('-', '').replace(' ', '')
        team_mapping[normalized_name] = team_id
    
    # Also add the exact team names for direct matching
    for team in trank_df['Team'].unique():
        if team in custom_team_names:
            trank_df.loc[trank_df['Team'] == team, 'TeamID'] = custom_team_names[team]
    
    # Add TeamID to trank_df using normalized mapping
    trank_df.loc[trank_df['TeamID'].isna(), 'TeamID'] = trank_df.loc[trank_df['TeamID'].isna(), 'NormalizedName'].map(team_mapping)
    
    # Handle missing mappings
    missing_teams = trank_df[trank_df['TeamID'].isna()]['Team'].unique()
    if len(missing_teams) > 0:
        print(f"Warning: Could not map {len(missing_teams)} teams to TeamIDs")
        print("First few missing teams:", missing_teams[:10])
        
        # Save missing teams to a file for future reference
        with open("../missing_team_mappings.txt", "w") as f:
            for team in missing_teams:
                f.write(f"{team}\n")
        print(f"Saved list of missing teams to missing_team_mappings.txt")
    
    return trank_df

def main():
    # Load T-Rank data
    print("Loading T-Rank data...")
    trank_df = load_trank_data()
    
    if trank_df is None:
        return
    
    print(f"Loaded T-Rank data with {len(trank_df)} rows")
    
    # Load team mapping data
    print("\nLoading team mapping data...")
    try:
        teams_df = pd.read_csv("../raw_data/MTeams.csv")
        print(f"Loaded team mapping data with {len(teams_df)} teams")
    except FileNotFoundError:
        print("Error: Team mapping file not found at '../raw_data/MTeams.csv'")
        return
    
    # Map team names to IDs
    print("\nMapping team names to IDs...")
    trank_df = create_team_name_mapping(trank_df, teams_df)
    
    # Check if mapping was successful
    mapped_count = trank_df['TeamID'].notna().sum()
    print(f"Successfully mapped {mapped_count} out of {len(trank_df)} teams to TeamIDs")
    
    # Extract only the columns we need
    print("\nExtracting required columns...")
    
    # Check if Barthag and Exp. columns exist
    if 'Barthag' not in trank_df.columns and 'Exp.' not in trank_df.columns:
        print("Error: Required columns 'Barthag' and 'Exp.' not found in T-Rank data")
        print("Available columns:", trank_df.columns.tolist())
        return
    
    # Create the simplified dataframe
    simplified_df = pd.DataFrame()
    simplified_df['Season'] = trank_df['Year']
    simplified_df['TeamID'] = trank_df['TeamID']
    simplified_df['TeamName'] = trank_df['Team']
    simplified_df['Barthag'] = trank_df['Barthag']
    simplified_df['Exp'] = trank_df['Exp.']
    
    # Remove rows with missing TeamIDs
    simplified_df = simplified_df.dropna(subset=['TeamID'])
    simplified_df['TeamID'] = simplified_df['TeamID'].astype(int)
    
    # Create output directory if it doesn't exist
    output_dir = "../processed_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save to CSV
    output_file = os.path.join(output_dir, "trank_simplified.csv")
    simplified_df.to_csv(output_file, index=False)
    print(f"\nSaved simplified T-Rank data to {output_file}")
    print(f"Total rows: {len(simplified_df)}")
    
    # Display sample of the data
    print("\nSample of the simplified data:")
    print(simplified_df.head())

if __name__ == "__main__":
    main() 