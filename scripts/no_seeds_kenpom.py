import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
import joblib
import os

# --- Step 1: Load Data ---

# Load tournament game–by–game results.
tour_df = pd.read_csv("../raw_data/MNCAATourneyDetailedResults.csv")

# Load season–long averages (with SoS and Last30 win ratio) computed earlier.
season_stats = pd.read_csv("../pre_tourney_data/TeamSeasonAverages_with_SoS.csv")

# Load KenPom rankings
kenpom_df = pd.read_csv("../pre_tourney_data/KenPom-Rankings-Updated.csv")

# --- Step 2: Create Training Data from Season Stats ---

# Create all possible tournament matchups for each season
all_matchups = []

# Get unique seasons from the tournament data
seasons = tour_df['Season'].unique()

for season in seasons:
    # Get teams that participated in the tournament this season
    teams = tour_df[(tour_df['Season'] == season)]['WTeamID'].unique()
    teams = np.append(teams, tour_df[(tour_df['Season'] == season)]['LTeamID'].unique())
    teams = np.unique(teams)
    
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

# --- Step 3: Merge Season Stats and KenPom Rankings (No Seeds) ---

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

# --- Step 4: Compute Feature Differences ---

# Create difference features (Team1 - Team2)
season_cols = ['WinPct', 'Avg_Score', 'Avg_FGM', 'Avg_FGA', 'Avg_FGM3', 'Avg_FGA3',
               'Avg_FTM', 'Avg_FTA', 'Avg_OR', 'Avg_DR', 'Avg_Ast', 'Avg_TO',
               'Avg_Stl', 'Avg_Blk', 'Avg_PF', 'Avg_Opp_WinPct', 'Last30_WinRatio']

for col in season_cols:
    matchups_df[f"{col}_diff"] = matchups_df[f"Team1_{col}"] - matchups_df[f"Team2_{col}"]

# Add KenPom ranking difference (note: lower rank is better, so we subtract Team1 from Team2)
matchups_df['KenPom_diff'] = matchups_df['Team2_KenPom'] - matchups_df['Team1_KenPom']

# --- Step 5: Emphasize Strength of Schedule ---

# Create additional SoS features to emphasize strength of schedule
# Make sure each feature has a unique name
matchups_df['SoS_squared_diff'] = matchups_df['Avg_Opp_WinPct_diff'] ** 2 * np.sign(matchups_df['Avg_Opp_WinPct_diff'])
matchups_df['SoS_WinPct_interaction'] = matchups_df['Team1_Avg_Opp_WinPct'] * matchups_df['Team1_WinPct'] - matchups_df['Team2_Avg_Opp_WinPct'] * matchups_df['Team2_WinPct']
matchups_df['SoS_Last30_interaction'] = matchups_df['Team1_Avg_Opp_WinPct'] * matchups_df['Team1_Last30_WinRatio'] - matchups_df['Team2_Avg_Opp_WinPct'] * matchups_df['Team2_Last30_WinRatio']

# Create KenPom-SoS interaction feature
matchups_df['KenPom_SoS_interaction'] = matchups_df['KenPom_diff'] * matchups_df['Avg_Opp_WinPct_diff']

# Create target variable (1 if Team1 wins, 0 if Team2 wins)
matchups_df['Target'] = (matchups_df['Team1'] == matchups_df['Winner']).astype(int)

# --- Step 6: Prepare Features for Training ---

# Select features for the model (excluding any seed-related features)
feature_cols = [col for col in matchups_df.columns if col.endswith('_diff') and 'Seed' not in col]
# Add the new SoS interaction features and KenPom features (make sure there are no duplicates)
additional_features = ['SoS_squared_diff', 'SoS_WinPct_interaction', 'SoS_Last30_interaction', 'KenPom_SoS_interaction']
feature_cols = list(dict.fromkeys(feature_cols + additional_features))  # Remove duplicates while preserving order

print(f"Using {len(feature_cols)} features: {feature_cols}")

# Handle any NaN values (especially for KenPom rankings which might be missing for some teams/seasons)
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
    'subsample': 0.8,  # Added to reduce overfitting
    'colsample_bytree': 0.8,  # Added to reduce overfitting
    'seed': 42
}
num_boost_round = 150  # Increased slightly

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
print(importance_df.head(20))  # Show more features to see SoS and KenPom importance


# Check if models directory exists, if not create it
models_dir = '../models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")

# Save the final XGBoost model using joblib
model_path = os.path.join(models_dir, 'xgb_model_no_seeds_kenpom.pkl')
joblib.dump(final_model, model_path)
print(f"Model saved as '{model_path}'")

# Save feature list for future reference
feature_list_path = os.path.join(models_dir, 'no_seeds_kenpom_features.txt')
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"Feature list saved as '{feature_list_path}'")

# --- Step 9: Test Prediction Symmetry ---

print("\n\n======= DEMONSTRATING PREDICTION SYMMETRY ISSUE =======")

def get_win_probability(model, team1_stats, team2_stats):
    """
    Make a basic prediction without symmetry enforcement.
    This is to demonstrate the asymmetry issue.
    """
    # Create features dictionary
    features = {}
    
    # Create difference features
    for col in season_cols:
        features[f"{col}_diff"] = team1_stats[col] - team2_stats[col]
    
    # Add KenPom (lower is better, so subtract Team1 from Team2)
    features['KenPom_diff'] = team2_stats['KenPom'] - team1_stats['KenPom']
    
    # Add SoS emphasis features
    features['SoS_squared_diff'] = features['Avg_Opp_WinPct_diff'] ** 2 * np.sign(features['Avg_Opp_WinPct_diff'])
    features['SoS_WinPct_interaction'] = team1_stats['Avg_Opp_WinPct'] * team1_stats['WinPct'] - team2_stats['Avg_Opp_WinPct'] * team2_stats['WinPct']
    features['SoS_Last30_interaction'] = team1_stats['Avg_Opp_WinPct'] * team1_stats['Last30_WinRatio'] - team2_stats['Avg_Opp_WinPct'] * team2_stats['Last30_WinRatio']
    
    # Add KenPom-SoS interaction
    features['KenPom_SoS_interaction'] = features['KenPom_diff'] * features['Avg_Opp_WinPct_diff']
    
    # Create DataFrame with exact feature order
    X = pd.DataFrame([features])
    X = X[feature_cols]
    
    # Make prediction
    dmatrix = xgb.DMatrix(X)
    return model.predict(dmatrix)[0]

# Test with a sample matchup
if len(matchups_df) > 0:
    print("\n1. DEMONSTRATING THE PROBLEM:\n")
    
    # Get a sample matchup
    sample = matchups_df.iloc[0]
    
    # Extract team info
    team1_id = sample['Team1']
    team2_id = sample['Team2']
    team1_name = sample.get('Team1_TeamName', f"Team {team1_id}")
    team2_name = sample.get('Team2_TeamName', f"Team {team2_id}")
    
    # Create stat dictionaries
    team1_stats = {}
    team2_stats = {}
    
    for col in season_cols:
        team1_stats[col] = sample[f'Team1_{col}']
        team2_stats[col] = sample[f'Team2_{col}']
    
    # Add KenPom
    team1_stats['KenPom'] = sample['Team1_KenPom']
    team2_stats['KenPom'] = sample['Team2_KenPom']
    
    # Get probabilities from both perspectives
    team1_win_prob = get_win_probability(final_model, team1_stats, team2_stats)
    team2_win_prob = get_win_probability(final_model, team2_stats, team1_stats)
    
    print(f"Matchup: {team1_name} vs {team2_name}")
    print(f"Team1 win probability: {team1_win_prob:.4f}")
    print(f"Team2 win probability: {team2_win_prob:.4f}")
    print(f"Sum of probabilities: {team1_win_prob + team2_win_prob:.4f}")  # Should be 1.0 ideally
    print(f"Asymmetry error: {abs(1.0 - (team1_win_prob + team2_win_prob)):.4f}")
    
    print("\n2. SOLUTION APPROACH #1: AVERAGING TEAM1 & (1-TEAM2) PROBABILITIES\n")
    
    # The simplest approach to enforce symmetry: average the results
    symmetric_team1_win_prob = (team1_win_prob + (1 - team2_win_prob)) / 2
    symmetric_team2_win_prob = 1 - symmetric_team1_win_prob
    
    print(f"Team1 win prob: {team1_win_prob:.4f}")
    print(f"1 - Team2 win prob: {1 - team2_win_prob:.4f}")
    print(f"Symmetrized Team1 win prob: {symmetric_team1_win_prob:.4f}")
    print(f"Symmetrized Team2 win prob: {symmetric_team2_win_prob:.4f}")
    print(f"Sum of symmetrized probs: {symmetric_team1_win_prob + symmetric_team2_win_prob:.4f}")
    
    print("\n3. SOLUTION APPROACH #2: TRAIN A SYMMETRIC MODEL\n")
    
    print("This is the recommended long-term solution:")
    print("1. Redesign your feature creation to be inherently symmetric")
    print("2. Use absolute differences for numeric features, not directional differences")
    print("3. For ordered features like KenPom, use features that capture which team is better")
    print("   - e.g., 'better_team_KenPom' and 'worse_team_KenPom' instead of 'KenPom_diff'")
    print("4. Avoid squared terms and complex interactions that introduce asymmetry")
    print("5. Train a new model with this symmetric feature design")
    
    print("\nExample of symmetry-preserving feature design:")
    print("- avg_score_diff = abs(team1_avg_score - team2_avg_score)")
    print("- team1_is_better_kenpom = 1 if team1_kenpom < team2_kenpom else 0")
    print("- better_team_kenpom = min(team1_kenpom, team2_kenpom)")
    print("- worse_team_kenpom = max(team1_kenpom, team2_kenpom)")
    
    print("\nWhen using these kinds of features, swapping team1 and team2 will")
    print("produce exact mirror probabilities without needing to average.")
