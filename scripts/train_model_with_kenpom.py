import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
import joblib
import os

# --- Step 1: Load Data ---

# Load tournament game-by-game results
tour_df = pd.read_csv("../raw_data/MNCAATourneyDetailedResults.csv")

# Load season-long averages (with SoS and Last30 win ratio)
season_stats = pd.read_csv("../pre_tourney_data/TeamSeasonAverages_with_SoS.csv")

# Load KenPom rankings
kenpom_df = pd.read_csv("../pre_tourney_data/KenPom-Rankings-Updated.csv")

# Load tournament seeds (only need seasons 2003 onward)
seeds_df = pd.read_csv("../raw_data/MNCAATourneySeeds.csv")
seeds_df = seeds_df[seeds_df['Season'] >= 2003].copy()
# Extract numeric part from the seed string (e.g., "W01" -> 1)
seeds_df['Seed'] = seeds_df['Seed'].str.extract(r'(\d+)').astype(int)

# --- Step 2: Create Training Data from Season Stats ---

# Create all possible tournament matchups for each season
all_matchups = []

# Get unique seasons from the tournament data
seasons = tour_df['Season'].unique()

for season in seasons:
    # Get teams that participated in the tournament this season
    season_seeds = seeds_df[seeds_df['Season'] == season]
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

matchups_df = pd.DataFrame(all_matchups)

# --- Step 3: Merge Season Stats, KenPom Rankings, and Seeds ---

# Add team1 stats
matchups_df = matchups_df.merge(
    season_stats, 
    left_on=['Season', 'Team1'], 
    right_on=['Season', 'TeamID'],
    how='left'
)
# Rename columns for team1
for col in season_stats.columns:
    if col not in ['Season', 'TeamID', 'TeamName']:
        matchups_df.rename(columns={col: f"Team1_{col}"}, inplace=True)
matchups_df.drop('TeamID', axis=1, inplace=True)

# Add team2 stats
matchups_df = matchups_df.merge(
    season_stats, 
    left_on=['Season', 'Team2'], 
    right_on=['Season', 'TeamID'],
    how='left'
)
# Rename columns for team2
for col in season_stats.columns:
    if col not in ['Season', 'TeamID', 'TeamName']:
        matchups_df.rename(columns={col: f"Team2_{col}"}, inplace=True)
matchups_df.drop('TeamID', axis=1, inplace=True)

# Add KenPom rankings for team1
matchups_df = matchups_df.merge(
    kenpom_df[['Season', 'TeamID', 'OrdinalRank']], 
    left_on=['Season', 'Team1'], 
    right_on=['Season', 'TeamID'],
    how='left'
)
matchups_df.rename(columns={'OrdinalRank': 'Team1_KenPom'}, inplace=True)
matchups_df.drop('TeamID', axis=1, inplace=True)

# Add KenPom rankings for team2
matchups_df = matchups_df.merge(
    kenpom_df[['Season', 'TeamID', 'OrdinalRank']], 
    left_on=['Season', 'Team2'], 
    right_on=['Season', 'TeamID'],
    how='left'
)
matchups_df.rename(columns={'OrdinalRank': 'Team2_KenPom'}, inplace=True)
matchups_df.drop('TeamID', axis=1, inplace=True)

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

# --- Step 4: Compute Feature Differences ---

# Create difference features (Team1 - Team2)
season_cols = ['WinPct', 'Avg_Score', 'Avg_FGM', 'Avg_FGA', 'Avg_FGM3', 'Avg_FGA3',
               'Avg_FTM', 'Avg_FTA', 'Avg_OR', 'Avg_DR', 'Avg_Ast', 'Avg_TO',
               'Avg_Stl', 'Avg_Blk', 'Avg_PF', 'Avg_Poss', 'Avg_Opp_WinPct', 'Last30_WinRatio']

for col in season_cols:
    matchups_df[f"{col}_diff"] = matchups_df[f"Team1_{col}"] - matchups_df[f"Team2_{col}"]

# Add seed difference
matchups_df['Seed_diff'] = matchups_df['Team1_Seed'] - matchups_df['Team2_Seed']

# Add KenPom ranking difference (note: lower rank is better, so we subtract Team1 from Team2)
matchups_df['KenPom_diff'] = matchups_df['Team2_KenPom'] - matchups_df['Team1_KenPom']

# Create target variable (1 if Team1 wins, 0 if Team2 wins)
matchups_df['Target'] = (matchups_df['Team1'] == matchups_df['Winner']).astype(int)

# --- Step 5: Create Additional Advanced Features ---

# Offensive and Defensive Efficiency (points per 100 possessions)
matchups_df['Team1_Off_Efficiency'] = matchups_df['Team1_Avg_Score'] / matchups_df['Team1_Avg_Poss'] * 100
matchups_df['Team2_Off_Efficiency'] = matchups_df['Team2_Avg_Score'] / matchups_df['Team2_Avg_Poss'] * 100
matchups_df['Off_Efficiency_diff'] = matchups_df['Team1_Off_Efficiency'] - matchups_df['Team2_Off_Efficiency']

# Shooting percentages
matchups_df['Team1_FG_Pct'] = matchups_df['Team1_Avg_FGM'] / matchups_df['Team1_Avg_FGA']
matchups_df['Team2_FG_Pct'] = matchups_df['Team2_Avg_FGM'] / matchups_df['Team2_Avg_FGA']
matchups_df['FG_Pct_diff'] = matchups_df['Team1_FG_Pct'] - matchups_df['Team2_FG_Pct']

matchups_df['Team1_3P_Pct'] = matchups_df['Team1_Avg_FGM3'] / matchups_df['Team1_Avg_FGA3']
matchups_df['Team2_3P_Pct'] = matchups_df['Team2_Avg_FGM3'] / matchups_df['Team2_Avg_FGA3']
matchups_df['3P_Pct_diff'] = matchups_df['Team1_3P_Pct'] - matchups_df['Team2_3P_Pct']

matchups_df['Team1_FT_Pct'] = matchups_df['Team1_Avg_FTM'] / matchups_df['Team1_Avg_FTA']
matchups_df['Team2_FT_Pct'] = matchups_df['Team2_Avg_FTM'] / matchups_df['Team2_Avg_FTA']
matchups_df['FT_Pct_diff'] = matchups_df['Team1_FT_Pct'] - matchups_df['Team2_FT_Pct']

# Turnover rate (turnovers per possession)
matchups_df['Team1_TO_Rate'] = matchups_df['Team1_Avg_TO'] / matchups_df['Team1_Avg_Poss']
matchups_df['Team2_TO_Rate'] = matchups_df['Team2_Avg_TO'] / matchups_df['Team2_Avg_Poss']
matchups_df['TO_Rate_diff'] = matchups_df['Team1_TO_Rate'] - matchups_df['Team2_TO_Rate']

# Rebound rate
matchups_df['Team1_Reb_Rate'] = (matchups_df['Team1_Avg_OR'] + matchups_df['Team1_Avg_DR']) / matchups_df['Team1_Avg_Poss']
matchups_df['Team2_Reb_Rate'] = (matchups_df['Team2_Avg_OR'] + matchups_df['Team2_Avg_DR']) / matchups_df['Team2_Avg_Poss']
matchups_df['Reb_Rate_diff'] = matchups_df['Team1_Reb_Rate'] - matchups_df['Team2_Reb_Rate']

# --- Step 6: Prepare Features for Training ---

# Select features for the model
basic_diff_cols = [col for col in matchups_df.columns if col.endswith('_diff')]
advanced_cols = ['Team1_Off_Efficiency', 'Team2_Off_Efficiency', 
                'Team1_FG_Pct', 'Team2_FG_Pct',
                'Team1_3P_Pct', 'Team2_3P_Pct',
                'Team1_FT_Pct', 'Team2_FT_Pct',
                'Team1_TO_Rate', 'Team2_TO_Rate',
                'Team1_Reb_Rate', 'Team2_Reb_Rate']

feature_cols = basic_diff_cols + advanced_cols

# Handle any NaN values
matchups_df[feature_cols] = matchups_df[feature_cols].fillna(0)

X = matchups_df[feature_cols]
y = matchups_df['Target']

# --- Step 7: Train the XGBoost Model with Leave-Season-Out Cross-Validation ---

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
    
    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
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

# --- Step 8: Train the Final Model on the Entire Dataset ---
print("\nTraining final model on all data...")
dtrain_full = xgb.DMatrix(X, label=y)
final_model = xgb.train(params, dtrain_full, num_boost_round=num_boost_round)

# Get feature importance
importance = final_model.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values('Importance', ascending=False)

print("\nFeature importance:")
print(importance_df.head(15))

# # Save feature columns for later use in prediction
# joblib.dump(feature_cols, '../models/model_features.pkl')
# print("Feature columns saved as '../models/model_features.pkl'")

# Check if models directory exists, if not create it
models_dir = '../models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")

# Save the final XGBoost model using joblib
model_path = os.path.join(models_dir, 'xgb_model_advanced.pkl')
joblib.dump(final_model, model_path)
print(f"Model saved as '{model_path}'")