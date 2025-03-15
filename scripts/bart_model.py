import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob

# --- Step 1: Load and Prepare Data ---

def load_trank_data():
    """Load and combine all T-Rank data files"""
    trank_files = glob.glob("../T-Rank/trank_team_table_data_*.csv")
    all_data = []
    
    for file in trank_files:
        year = int(file.split('_')[-1].split('.')[0])  # Extract year from filename
        df = pd.read_csv(file)
        
        # Add year column if not already present
        if 'Year' not in df.columns:
            df['Year'] = year
            
        all_data.append(df)
    
    # Combine all years
    trank_df = pd.concat(all_data, ignore_index=True)
    
    # Clean team names and create a mapping to TeamID
    trank_df['Team'] = trank_df['Team'].str.strip()
    
    return trank_df

def load_tournament_data():
    """Load tournament results and seeds"""
    # Load tournament game-by-game results
    tour_df = pd.read_csv("../raw_data/MNCAATourneyDetailedResults.csv")
    
    # Load tournament seeds
    seeds_df = pd.read_csv("../raw_data/MNCAATourneySeeds.csv")
    seeds_df['Seed'] = seeds_df['Seed'].str.extract(r'(\d+)').astype(int)
    
    # Load team mapping (to map team names to IDs)
    teams_df = pd.read_csv("../raw_data/MTeams.csv")
    
    return tour_df, seeds_df, teams_df

def create_team_name_mapping(trank_df, teams_df):
    """Create a mapping between T-Rank team names and TeamIDs"""
    # This is a simplified approach - in practice, you might need more sophisticated matching
    team_mapping = {}
    
    # Normalize team names for matching
    trank_df['NormalizedName'] = trank_df['Team'].str.lower().str.replace('[^a-z0-9]', '', regex=True)
    teams_df['NormalizedName'] = teams_df['TeamName'].str.lower().str.replace('[^a-z0-9]', '', regex=True)
    
    # Create mapping dictionary
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
    
    # Add TeamID to trank_df for the rest using normalized mapping
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

def create_matchups(tour_df, seeds_df, trank_df):
    """Create all possible tournament matchups for each season"""
    all_matchups = []
    seasons = tour_df['Season'].unique()
    
    for season in seasons:
        # Filter data for this season
        season_seeds = seeds_df[seeds_df['Season'] == season]
        season_trank = trank_df[trank_df['Year'] == season]
        
        # Skip if no T-Rank data for this season
        if len(season_trank) == 0:
            print(f"No T-Rank data for season {season}, skipping")
            continue
        
        # Get teams that participated in the tournament this season
        teams = season_seeds['TeamID'].unique()
        
        # Create all possible matchups
        for i, team1 in enumerate(teams):
            for team2 in teams[i+1:]:
                # Find actual result if these teams played in the tournament
                actual_game = tour_df[(tour_df['Season'] == season) & 
                                     (((tour_df['WTeamID'] == team1) & (tour_df['LTeamID'] == team2)) | 
                                      ((tour_df['WTeamID'] == team2) & (tour_df['LTeamID'] == team1)))]
                
                if len(actual_game) > 0:
                    # These teams played each other in the tournament
                    game = actual_game.iloc[0]
                    if game['WTeamID'] == team1:
                        winner = team1
                        loser = team2
                    else:
                        winner = team2
                        loser = team1
                    
                    all_matchups.append({
                        'Season': season,
                        'Team1': team1,
                        'Team2': team2,
                        'Winner': winner
                    })
    
    return pd.DataFrame(all_matchups)

# --- Step 2: Feature Engineering ---

def add_trank_features(matchups_df, trank_df, seeds_df):
    """Add T-Rank features to matchups"""
    # Add team1 T-Rank stats
    matchups_df = matchups_df.merge(
        trank_df[trank_df['TeamID'].notna()], 
        left_on=['Season', 'Team1'], 
        right_on=['Year', 'TeamID'],
        how='left'
    )
    
    # Rename columns for team1
    trank_cols = [col for col in trank_df.columns if col not in ['Year', 'TeamID', 'Team', 'NormalizedName']]
    for col in trank_cols:
        matchups_df.rename(columns={col: f"Team1_{col}"}, inplace=True)
    
    # Drop unnecessary columns
    matchups_df.drop(['Year', 'TeamID', 'Team', 'NormalizedName'], axis=1, errors='ignore', inplace=True)
    
    # Add team2 T-Rank stats
    matchups_df = matchups_df.merge(
        trank_df[trank_df['TeamID'].notna()], 
        left_on=['Season', 'Team2'], 
        right_on=['Year', 'TeamID'],
        how='left'
    )
    
    # Rename columns for team2
    for col in trank_cols:
        matchups_df.rename(columns={col: f"Team2_{col}"}, inplace=True)
    
    # Drop unnecessary columns
    matchups_df.drop(['Year', 'TeamID', 'Team', 'NormalizedName'], axis=1, errors='ignore', inplace=True)
    
    # Add seeds
    matchups_df = matchups_df.merge(
        seeds_df[['Season', 'TeamID', 'Seed']], 
        left_on=['Season', 'Team1'], 
        right_on=['Season', 'TeamID'],
        how='left'
    )
    matchups_df.rename(columns={'Seed': 'Team1_Seed'}, inplace=True)
    matchups_df.drop('TeamID', axis=1, inplace=True)
    
    matchups_df = matchups_df.merge(
        seeds_df[['Season', 'TeamID', 'Seed']], 
        left_on=['Season', 'Team2'], 
        right_on=['Season', 'TeamID'],
        how='left'
    )
    matchups_df.rename(columns={'Seed': 'Team2_Seed'}, inplace=True)
    matchups_df.drop('TeamID', axis=1, inplace=True)
    
    return matchups_df

def create_difference_features(matchups_df):
    """Create difference features between teams"""
    # List of T-Rank metrics to use for differences
    trank_metrics = [
        'Adj OE', 'Adj DE', 'Barthag', 'eFG', 'eFG D.', 'FT Rate', 'FT Rate D',
        'TOV%', 'TOV% D', 'O Reb%', 'Op OReb%', 'Raw T', '2P %', '2P % D.',
        '3P %', '3P % D.', 'Blk %', 'Blked %', 'Ast %', 'Op Ast %', '3P Rate',
        '3P Rate D', 'Adj. T', 'Avg Hgt.', 'Eff. Hgt.', 'Exp.', 'PAKE', 'PASE',
        'Talent', 'FT%', 'Op. FT%', 'PPP Off.', 'PPP Def.', 'Elite SOS'
    ]
    
    # Create difference features
    for metric in trank_metrics:
        col_name = metric.replace('.', '').replace(' ', '_').replace('%', 'Pct')
        team1_col = f"Team1_{metric}"
        team2_col = f"Team2_{metric}"
        
        if team1_col in matchups_df.columns and team2_col in matchups_df.columns:
            matchups_df[f"{col_name}_diff"] = matchups_df[team1_col] - matchups_df[team2_col]
    
    # Add seed difference
    matchups_df['Seed_diff'] = matchups_df['Team1_Seed'] - matchups_df['Team2_Seed']
    
    # Create target variable (1 if Team1 wins, 0 if Team2 wins)
    matchups_df['Target'] = (matchups_df['Team1'] == matchups_df['Winner']).astype(int)
    
    return matchups_df

def create_advanced_features(matchups_df):
    """Create additional advanced features"""
    # Offensive to Defensive Ratio
    matchups_df['Team1_Off_Def_Ratio'] = matchups_df['Team1_Adj OE'] / matchups_df['Team1_Adj DE']
    matchups_df['Team2_Off_Def_Ratio'] = matchups_df['Team2_Adj OE'] / matchups_df['Team2_Adj DE']
    matchups_df['Off_Def_Ratio_diff'] = matchups_df['Team1_Off_Def_Ratio'] - matchups_df['Team2_Off_Def_Ratio']
    
    # Shooting Efficiency Composite
    matchups_df['Team1_Shooting_Composite'] = (
        matchups_df['Team1_eFG'] + matchups_df['Team1_3P %'] * 0.5 + matchups_df['Team1_2P %'] * 0.3
    )
    matchups_df['Team2_Shooting_Composite'] = (
        matchups_df['Team2_eFG'] + matchups_df['Team2_3P %'] * 0.5 + matchups_df['Team2_2P %'] * 0.3
    )
    matchups_df['Shooting_Composite_diff'] = matchups_df['Team1_Shooting_Composite'] - matchups_df['Team2_Shooting_Composite']
    
    # Defensive Efficiency Composite
    matchups_df['Team1_Def_Composite'] = (
        matchups_df['Team1_Adj DE'] * 0.5 + matchups_df['Team1_eFG D.'] * 0.3 + matchups_df['Team1_TOV% D'] * 0.2
    )
    matchups_df['Team2_Def_Composite'] = (
        matchups_df['Team2_Adj DE'] * 0.5 + matchups_df['Team2_eFG D.'] * 0.3 + matchups_df['Team2_TOV% D'] * 0.2
    )
    matchups_df['Def_Composite_diff'] = matchups_df['Team1_Def_Composite'] - matchups_df['Team2_Def_Composite']
    
    # Experience vs Talent Ratio
    matchups_df['Team1_Exp_Talent_Ratio'] = matchups_df['Team1_Exp.'] / matchups_df['Team1_Talent'].replace(0, 0.001)
    matchups_df['Team2_Exp_Talent_Ratio'] = matchups_df['Team2_Exp.'] / matchups_df['Team2_Talent'].replace(0, 0.001)
    matchups_df['Exp_Talent_Ratio_diff'] = matchups_df['Team1_Exp_Talent_Ratio'] - matchups_df['Team2_Exp_Talent_Ratio']
    
    # Barthag Squared (emphasize large differences)
    matchups_df['Barthag_diff_squared'] = matchups_df['Barthag_diff'] ** 2 * np.sign(matchups_df['Barthag_diff'])
    
    return matchups_df

# --- Step 3: Model Training and Evaluation ---

def train_model_with_cv(matchups_df):
    """Train an XGBoost model with leave-one-season-out cross-validation"""
    # Define XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    num_boost_round = 200
    
    # Select features for the model
    diff_cols = [col for col in matchups_df.columns if col.endswith('_diff')]
    
    # Select team-specific features that might be valuable
    team_cols = [
        'Team1_Barthag', 'Team2_Barthag',
        'Team1_Adj OE', 'Team2_Adj OE',
        'Team1_Adj DE', 'Team2_Adj DE',
        'Team1_Off_Def_Ratio', 'Team2_Off_Def_Ratio',
        'Team1_Shooting_Composite', 'Team2_Shooting_Composite',
        'Team1_Def_Composite', 'Team2_Def_Composite',
        'Team1_Exp_Talent_Ratio', 'Team2_Exp_Talent_Ratio'
    ]
    
    # Combine features
    feature_cols = diff_cols + team_cols
    
    # Handle any NaN values
    matchups_df[feature_cols] = matchups_df[feature_cols].fillna(0)
    
    # Extract features and target
    X = matchups_df[feature_cols]
    y = matchups_df['Target']
    
    # Use GroupKFold to ensure seasons stay together (leave-one-season-out CV)
    seasons = matchups_df['Season'].unique()
    cv_results = []
    
    print("Starting leave-one-season-out cross-validation...")
    for test_season in seasons:
        print(f"Testing on season {test_season}")
        
        # Split data into train and test
        train_idx = matchups_df['Season'] != test_season
        test_idx = matchups_df['Season'] == test_season
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=feature_cols
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=feature_cols
        )
        
        # Train model
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)
        
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        
        # Evaluate
        preds = model.predict(dtest)
        auc = roc_auc_score(y_test, preds)
        loss = log_loss(y_test, preds)
        
        cv_results.append({
            'test_season': test_season,
            'auc': auc,
            'log_loss': loss,
            'test_size': len(X_test)
        })
    
    cv_results_df = pd.DataFrame(cv_results)
    print("\nCross-validation results by season:")
    print(cv_results_df)
    print(f"\nAverage AUC: {cv_results_df['auc'].mean():.4f}")
    print(f"Average Log Loss: {cv_results_df['log_loss'].mean():.4f}")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    
    # Scale all features
    scaler_final = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler_final.fit_transform(X),
        index=X.index,
        columns=feature_cols
    )
    
    dtrain_full = xgb.DMatrix(X_scaled, label=y)
    final_model = xgb.train(params, dtrain_full, num_boost_round=num_boost_round)
    
    # Get feature importance
    importance = final_model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 features by importance:")
    print(importance_df.head(15))
    
    return final_model, importance_df, feature_cols, scaler_final, cv_results_df

# --- Step 4: Main Function ---

def main():
    print("Loading T-Rank data...")
    trank_df = load_trank_data()
    print(f"Loaded T-Rank data with {len(trank_df)} rows")
    
    print("\nLoading tournament data...")
    tour_df, seeds_df, teams_df = load_tournament_data()
    print(f"Loaded tournament data with {len(tour_df)} games")
    
    print("\nCreating team name mapping...")
    trank_df = create_team_name_mapping(trank_df, teams_df)
    print(f"Mapped {trank_df['TeamID'].notna().sum()} teams to TeamIDs")
    
    print("\nCreating matchups...")
    matchups_df = create_matchups(tour_df, seeds_df, trank_df)
    print(f"Created {len(matchups_df)} matchups")
    
    print("\nAdding T-Rank features...")
    matchups_df = add_trank_features(matchups_df, trank_df, seeds_df)
    
    print("\nCreating difference features...")
    matchups_df = create_difference_features(matchups_df)
    
    print("\nCreating advanced features...")
    matchups_df = create_advanced_features(matchups_df)
    
    # Train model with cross-validation
    print("\n" + "="*50)
    print("Training XGBoost model with leave-one-season-out cross-validation...")
    model, importance_df, feature_cols, scaler, cv_results = train_model_with_cv(matchups_df)
    
    # Save the model and related artifacts
    models_dir = '../models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"\nCreated directory: {models_dir}")
    
    # Save model
    model_path = os.path.join(models_dir, 'bart_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nSaved model as '{model_path}'")
    
    # Save feature columns and scaler
    joblib.dump(feature_cols, os.path.join(models_dir, 'bart_model_features.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'bart_model_scaler.pkl'))
    
    # Save model info
    model_info = {
        'feature_importance': importance_df.to_dict(),
        'cv_results': cv_results.to_dict(),
        'avg_auc': cv_results['auc'].mean(),
        'avg_log_loss': cv_results['log_loss'].mean()
    }
    joblib.dump(model_info, os.path.join(models_dir, 'bart_model_info.pkl'))
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main() 