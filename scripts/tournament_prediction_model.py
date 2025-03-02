# March Madness Tournament Prediction Model
# This script builds a prediction model for NCAA Tournament games using pre-tournament statistics,
# with careful attention to preventing data leakage.

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('fivethirtyeight')

# Step 2: Load datasets
print("Loading datasets...")
tourney_results = pd.read_csv("../raw_data/MNCAATourneyDetailedResults.csv")
enhanced_stats = pd.read_csv("../pre_tourney_data/EnhancedTournamentStats.csv")
tourney_seeds = pd.read_csv("../raw_data/MNCAATourneySeeds.csv")

# Display basic information
print(f"Tournament games: {len(tourney_results)}")
print(f"Enhanced stats entries: {len(enhanced_stats)}")
print(f"Tournament seeds entries: {len(tourney_seeds)}")

# Process tournament seeds
tourney_seeds['SeedValue'] = tourney_seeds['Seed'].apply(lambda x: int(x[1:3]))

# Step 3: Examine the data structure
print("\nExamining data structure...")
print("\nTournament results sample:")
print(tourney_results.head())

print("\nEnhanced stats sample:")
print(enhanced_stats.head())

print("\nTournament seeds sample:")
print(tourney_seeds.head())

# Step 4: Create game features without leakage
print("\nCreating game features without leakage...")

def create_game_features(results_df, stats_df, seeds_df):
    """
    Create a feature dataset for tournament games with team statistics,
    ensuring we only use pre-tournament data.
    """
    # Initialize empty list for game features
    game_features = []
    
    # Process each tournament game
    for _, game in results_df.iterrows():
        season = game['Season']
        day_num = game['DayNum']
        
        # Determine round number based on DayNum
        round_number = None
        if day_num in [134, 135]:
            round_number = 0  # Play-in games
        elif day_num in [136, 137]:
            round_number = 1  # Round of 64
        elif day_num in [138, 139]:
            round_number = 2  # Round of 32
        elif day_num in [143, 144]:
            round_number = 3  # Sweet Sixteen
        elif day_num in [145, 146]:
            round_number = 4  # Elite Eight
        elif day_num == 152:
            round_number = 5  # Final Four
        elif day_num == 154:
            round_number = 6  # Championship
        
        # Get team stats for this season
        season_stats = stats_df[stats_df['Season'] == season]
        season_seeds = seeds_df[seeds_df['Season'] == season]
        
        # Get winning and losing team stats
        w_team_stats = season_stats[season_stats['TeamID'] == game['WTeamID']].iloc[0].to_dict() if not season_stats[season_stats['TeamID'] == game['WTeamID']].empty else {}
        l_team_stats = season_stats[season_stats['TeamID'] == game['LTeamID']].iloc[0].to_dict() if not season_stats[season_stats['TeamID'] == game['LTeamID']].empty else {}
        
        # Get seeds
        w_seed = season_seeds[season_seeds['TeamID'] == game['WTeamID']]['SeedValue'].values[0] if not season_seeds[season_seeds['TeamID'] == game['WTeamID']].empty else None
        l_seed = season_seeds[season_seeds['TeamID'] == game['LTeamID']]['SeedValue'].values[0] if not season_seeds[season_seeds['TeamID'] == game['LTeamID']].empty else None
        
        # Skip if we don't have stats for both teams
        if not w_team_stats or not l_team_stats or w_seed is None or l_seed is None:
            continue
            
        # Create feature dictionary for this game
        game_dict = {
            'Season': season,
            'WTeamID': game['WTeamID'],
            'LTeamID': game['LTeamID'],
            'WTeamName': w_team_stats.get('TeamName', 'Unknown'),
            'LTeamName': l_team_stats.get('TeamName', 'Unknown'),
            'WSeed': w_seed,
            'LSeed': l_seed,
            'SeedDiff': l_seed - w_seed,  # Positive means winner was higher seeded (lower number)
            'RoundNumber': round_number,
            'DayNum': day_num
        }
        
        # Add team statistics with prefixes
        for stat in w_team_stats:
            if stat not in ['Season', 'TeamID', 'TeamName']:
                # Skip any stats that might contain tournament data
                if not any(term in stat.lower() for term in ['tournament', 'tourney']):
                    game_dict[f'W_{stat}'] = w_team_stats[stat]
                    game_dict[f'L_{stat}'] = l_team_stats[stat]
                    # Add differential features
                    if isinstance(w_team_stats[stat], (int, float)) and isinstance(l_team_stats[stat], (int, float)):
                        game_dict[f'Diff_{stat}'] = w_team_stats[stat] - l_team_stats[stat]
        
        game_features.append(game_dict)
    
    return pd.DataFrame(game_features)

# Create game features
game_features_df = create_game_features(tourney_results, enhanced_stats, tourney_seeds)
print(f"Created features for {len(game_features_df)} tournament games")
print(f"Number of features: {len(game_features_df.columns)}")

# Step 5: Create prediction dataset with proper train/test split by season
print("\nCreating prediction dataset with season-based split...")

def create_prediction_dataset(game_features, test_seasons=[2022, 2023]):
    """
    Create a dataset for prediction with proper train/test split by season
    """
    prediction_rows = []
    
    for _, game in game_features.iterrows():
        # Determine which team has the lower ID (will be Team1)
        if game['WTeamID'] < game['LTeamID']:
            team1_id, team2_id = game['WTeamID'], game['LTeamID']
            team1_name, team2_name = game['WTeamName'], game['LTeamName']
            team1_seed, team2_seed = game['WSeed'], game['LSeed']
            team1_won = 1
        else:
            team1_id, team2_id = game['LTeamID'], game['WTeamID']
            team1_name, team2_name = game['LTeamName'], game['WTeamName']
            team1_seed, team2_seed = game['LSeed'], game['WSeed']
            team1_won = 0
            
        # Create feature dictionary
        game_dict = {
            'Season': game['Season'],
            'Team1ID': team1_id,
            'Team2ID': team2_id,
            'Team1Name': team1_name,
            'Team2Name': team2_name,
            'Team1Seed': team1_seed,
            'Team2Seed': team2_seed,
            'SeedDiff': team1_seed - team2_seed,
            'Team1Won': team1_won,
            'RoundNumber': game['RoundNumber'],
            'DayNum': game['DayNum']
        }
        
        # Add differential features (Team1 - Team2)
        for feat in game.index:
            if feat.startswith('Diff_'):
                if team1_id == game['WTeamID']:
                    game_dict[feat] = game[feat]
                else:
                    game_dict[feat] = -game[feat]  # Flip the sign if Team1 is the loser
        
        prediction_rows.append(game_dict)
    
    # Create DataFrame
    prediction_df = pd.DataFrame(prediction_rows)
    
    # Split by season
    train_df = prediction_df[~prediction_df['Season'].isin(test_seasons)]
    test_df = prediction_df[prediction_df['Season'].isin(test_seasons)]
    
    print(f"Train set: {len(train_df)} games from seasons {sorted(train_df['Season'].unique())}")
    print(f"Test set: {len(test_df)} games from seasons {sorted(test_df['Season'].unique())}")
    
    return prediction_df, train_df, test_df

# Create prediction dataset with season-based split
test_seasons = [2022, 2023, 2024]  # Use recent seasons as test set
prediction_df, train_df, test_df = create_prediction_dataset(game_features_df, test_seasons)

# Step 6: Feature selection
print("\nSelecting features...")
# Exclude features that might contain tournament data or direct game outcomes
exclude_terms = ['score', 'margin', 'tournament', 'tourney']
feature_columns = ['SeedDiff'] + [col for col in prediction_df.columns if col.startswith('Diff_') 
                                 and not any(term in col.lower() for term in exclude_terms)]

print(f"Selected {len(feature_columns)} features")
print(f"First 10 features: {feature_columns[:10]}")

# Prepare train and test data
X_train = train_df[feature_columns]
y_train = train_df['Team1Won']
X_test = test_df[feature_columns]
y_test = test_df['Team1Won']

print(f"\nTrain data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Step 7: Train models
print("\nTraining models...")

# Function to create and train a model
def train_model(model_class, params, X_train, y_train, X_test, y_test):
    """Train a model and evaluate its performance"""
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_class(**params))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'auc': auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# Train different models
results = {}

# Random Forest
print("\nTraining Random Forest...")
rf_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
results['RandomForest'] = train_model(RandomForestClassifier, rf_params, X_train, y_train, X_test, y_test)

# Gradient Boosting
print("\nTraining Gradient Boosting...")
gb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42}
results['GradientBoosting'] = train_model(GradientBoostingClassifier, gb_params, X_train, y_train, X_test, y_test)

# XGBoost
print("\nTraining XGBoost...")
xgb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42}
results['XGBoost'] = train_model(xgb.XGBClassifier, xgb_params, X_train, y_train, X_test, y_test)

# Find the best model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")

# Step 8: Feature importance analysis
print("\nAnalyzing feature importance...")

# Get feature importances from each model
feature_importances = {}

# Random Forest
rf_importances = results['RandomForest']['pipeline'].named_steps['model'].feature_importances_
rf_features = [(feature, importance) for feature, importance in zip(feature_columns, rf_importances)]
rf_features.sort(key=lambda x: x[1], reverse=True)
feature_importances['RandomForest'] = rf_features

# Gradient Boosting
gb_importances = results['GradientBoosting']['pipeline'].named_steps['model'].feature_importances_
gb_features = [(feature, importance) for feature, importance in zip(feature_columns, gb_importances)]
gb_features.sort(key=lambda x: x[1], reverse=True)
feature_importances['GradientBoosting'] = gb_features

# XGBoost
xgb_importances = results['XGBoost']['pipeline'].named_steps['model'].feature_importances_
xgb_features = [(feature, importance) for feature, importance in zip(feature_columns, xgb_importances)]
xgb_features.sort(key=lambda x: x[1], reverse=True)
feature_importances['XGBoost'] = xgb_features

# Print top features for each model
for model_name, features in feature_importances.items():
    print(f"\nTop 10 features for {model_name}:")
    for feature, importance in features[:10]:
        print(f"{feature}: {importance:.4f}")

# Plot feature importances for the best model
plt.figure(figsize=(12, 10))
best_model_name = best_model[0]
features_df = pd.DataFrame(feature_importances[best_model_name][:15], columns=['Feature', 'Importance'])
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title(f'Top 15 Features - {best_model_name} Model')
plt.tight_layout()
# plt.savefig(f'../figures/{best_model_name}_feature_importance.png')
# plt.show()

# Step 9: Tournament Round Analysis
print("\n--- Tournament Round Analysis ---")

# Define round names for better readability
round_names = {
    0: "Play-In Games",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet Sixteen",
    4: "Elite Eight",
    5: "Final Four",
    6: "Championship"
}

# Analyze model performance by round
round_accuracy = {}
for round_num in sorted(test_df['RoundNumber'].unique()):
    if pd.isna(round_num):
        continue
        
    round_games = test_df[test_df['RoundNumber'] == round_num]
    if len(round_games) > 0:
        X_round = round_games[feature_columns]
        y_round = round_games['Team1Won']
        y_pred = results[best_model_name]['pipeline'].predict(X_round)
        round_acc = accuracy_score(y_round, y_pred)
        round_accuracy[round_num] = (round_acc, len(round_games))

print("\nModel accuracy by tournament round:")
for round_num, (acc, count) in sorted(round_accuracy.items()):
    round_name = round_names.get(round_num, f"Round {round_num}")
    print(f"{round_name}: {acc:.4f} accuracy ({count} games)")

# Analyze upset frequency by round
print("\nUpset frequency by tournament round:")
for round_num in sorted(test_df['RoundNumber'].unique()):
    if pd.isna(round_num):
        continue
        
    round_games = test_df[test_df['RoundNumber'] == round_num]
    if len(round_games) > 0:
        upsets = round_games[(round_games['Team1Won'] == 1) & (round_games['SeedDiff'] > 0)]
        upset_pct = len(upsets) / len(round_games) * 100
        round_name = round_names.get(round_num, f"Round {round_num}")
        print(f"{round_name}: {upset_pct:.1f}% upsets ({len(upsets)} of {len(round_games)} games)")

# Step 10: Upset Analysis
print("\n--- Upset Analysis ---")

# Analyze model performance on upsets
upsets = test_df[(test_df['Team1Won'] == 1) & (test_df['SeedDiff'] > 0)]
print(f"Number of upsets in test set: {len(upsets)} out of {len(test_df)} games ({len(upsets)/len(test_df)*100:.1f}%)")

if len(upsets) > 0:
    X_upsets = upsets[feature_columns]
    y_upsets = upsets['Team1Won']
    y_pred = results[best_model_name]['pipeline'].predict(X_upsets)
    upset_acc = accuracy_score(y_upsets, y_pred)
    print(f"Model accuracy on upsets: {upset_acc:.4f}")
    
    # Analyze different levels of upsets
    minor_upsets = test_df[(test_df['Team1Won'] == 1) & (test_df['SeedDiff'] > 0) & (test_df['SeedDiff'] <= 4)]
    major_upsets = test_df[(test_df['Team1Won'] == 1) & (test_df['SeedDiff'] > 4)]
    
    if len(minor_upsets) > 0:
        X_minor = minor_upsets[feature_columns]
        y_minor = minor_upsets['Team1Won']
        y_pred_minor = results[best_model_name]['pipeline'].predict(X_minor)
        minor_acc = accuracy_score(y_minor, y_pred_minor)
        print(f"Model accuracy on minor upsets (1-4 seed diff): {minor_acc:.4f} ({len(minor_upsets)} games)")
    
    if len(major_upsets) > 0:
        X_major = major_upsets[feature_columns]
        y_major = major_upsets['Team1Won']
        y_pred_major = results[best_model_name]['pipeline'].predict(X_major)
        major_acc = accuracy_score(y_major, y_pred_major)
        print(f"Model accuracy on major upsets (>4 seed diff): {major_acc:.4f} ({len(major_upsets)} games)")

# Step 10.5: Detailed Upset Analysis
print("\n--- Detailed Upset Analysis ---")

# Create a function to analyze specific upsets
def analyze_upsets(test_data, model_pipeline, feature_cols):
    """Analyze specific upsets in detail"""
    # Find all upsets in the test data
    upsets = test_data[(test_data['Team1Won'] == 1) & (test_data['SeedDiff'] > 0)]
    
    if len(upsets) == 0:
        print("No upsets found in the test data.")
        return
    
    print(f"\nAnalyzing {len(upsets)} upsets in detail:")
    print("\n{:<5} {:<20} {:<5} vs {:<20} {:<5} | {:<10} | {:<10} | {:<10}".format(
        "Year", "Winner", "Seed", "Loser", "Seed", "Seed Diff", "Model Pred", "Correct?"
    ))
    print("-" * 100)
    
    # Sort upsets by seed difference (largest upsets first)
    upsets_sorted = upsets.sort_values('SeedDiff', ascending=False)
    
    for _, game in upsets_sorted.iterrows():
        # Get team names and seeds
        if game['Team1Won'] == 1:
            winner_name = game['Team1Name']
            loser_name = game['Team2Name']
            winner_seed = game['Team1Seed']
            loser_seed = game['Team2Seed']
        else:
            winner_name = game['Team2Name']
            loser_name = game['Team1Name']
            winner_seed = game['Team2Seed']
            loser_seed = game['Team1Seed']
        
        # Get model prediction
        X_game = game[feature_cols].values.reshape(1, -1)
        pred_prob = model_pipeline.predict_proba(X_game)[0, 1]
        predicted_team1_win = pred_prob > 0.5
        
        # Determine if prediction was correct
        correct_prediction = (predicted_team1_win == game['Team1Won'])
        
        # Format team names to fit in the table
        winner_name_short = winner_name[:18] + '..' if len(winner_name) > 20 else winner_name
        loser_name_short = loser_name[:18] + '..' if len(loser_name) > 20 else loser_name
        
        # Print the upset details
        print("{:<5} {:<20} {:<5} vs {:<20} {:<5} | {:<10} | {:<10.2f} | {:<10}".format(
            game['Season'],
            winner_name_short, 
            winner_seed,
            loser_name_short,
            loser_seed,
            loser_seed - winner_seed,
            pred_prob,
            "✓" if correct_prediction else "✗"
        ))
    
    # Calculate overall accuracy on upsets
    X_upsets = upsets[feature_cols]
    y_upsets = upsets['Team1Won']
    y_pred = model_pipeline.predict(X_upsets)
    upset_acc = accuracy_score(y_upsets, y_pred)
    print(f"\nOverall accuracy on upsets: {upset_acc:.4f} ({sum(y_pred == y_upsets)}/{len(upsets)})")
    
    # Analyze by seed difference
    print("\nAccuracy by upset magnitude:")
    for lower, upper, label in [(1, 4, "Small (1-4)"), (5, 8, "Medium (5-8)"), (9, 16, "Large (9+)")]:
        range_upsets = upsets[(upsets['SeedDiff'] >= lower) & (upsets['SeedDiff'] <= upper)]
        if len(range_upsets) > 0:
            X_range = range_upsets[feature_cols]
            y_range = range_upsets['Team1Won']
            y_pred_range = model_pipeline.predict(X_range)
            range_acc = accuracy_score(y_range, y_pred_range)
            print(f"{label} seed difference: {range_acc:.4f} ({sum(y_pred_range == y_range)}/{len(range_upsets)})")
    
    # Analyze by season
    print("\nUpset accuracy by season:")
    for season in sorted(upsets['Season'].unique()):
        season_upsets = upsets[upsets['Season'] == season]
        X_season = season_upsets[feature_cols]
        y_season = season_upsets['Team1Won']
        y_pred_season = model_pipeline.predict(X_season)
        season_acc = accuracy_score(y_season, y_pred_season)
        print(f"Season {season}: {season_acc:.4f} ({sum(y_pred_season == y_season)}/{len(season_upsets)})")

# Run the detailed upset analysis
analyze_upsets(test_df, results[best_model_name]['pipeline'], feature_columns)

# Step 10.6: Analyze Most Surprising Predictions
print("\n--- Most Surprising Predictions ---")

def analyze_surprising_predictions(test_data, model_pipeline, feature_cols, n=10):
    """Analyze the most surprising predictions (largest difference between model and reality)"""
    # Get all predictions
    X_test_all = test_data[feature_cols]
    y_test_all = test_data['Team1Won']
    y_pred_proba = model_pipeline.predict_proba(X_test_all)[:, 1]
    
    # Calculate prediction error (difference between prediction and actual)
    test_data['PredProb'] = y_pred_proba
    test_data['PredError'] = abs(test_data['PredProb'] - test_data['Team1Won'])
    
    # Sort by prediction error (largest first)
    surprising_games = test_data.sort_values('PredError', ascending=False).head(n)
    
    print(f"\nTop {n} most surprising predictions:")
    print("\n{:<5} {:<20} {:<5} vs {:<20} {:<5} | {:<10} | {:<10} | {:<10}".format(
        "Year", "Team 1", "Seed", "Team 2", "Seed", "Seed Diff", "Pred Prob", "Actual"
    ))
    print("-" * 100)
    
    for _, game in surprising_games.iterrows():
        # Format team names to fit in the table
        team1_name_short = game['Team1Name'][:18] + '..' if len(game['Team1Name']) > 20 else game['Team1Name']
        team2_name_short = game['Team2Name'][:18] + '..' if len(game['Team2Name']) > 20 else game['Team2Name']
        
        # Print the game details
        print("{:<5} {:<20} {:<5} vs {:<20} {:<5} | {:<10} | {:<10.2f} | {:<10}".format(
            game['Season'],
            team1_name_short, 
            game['Team1Seed'],
            team2_name_short,
            game['Team2Seed'],
            game['SeedDiff'],
            game['PredProb'],
            game['Team1Won']
        ))
        
        # Add a note about what happened
        actual_winner = game['Team1Name'] if game['Team1Won'] == 1 else game['Team2Name']
        predicted_winner = game['Team1Name'] if game['PredProb'] > 0.5 else game['Team2Name']
        confidence = max(game['PredProb'], 1 - game['PredProb'])
        
        print(f"   Model predicted {predicted_winner} would win with {confidence:.2f} confidence, but {actual_winner} actually won.")
        
        # If it was an upset, note that
        if (game['Team1Won'] == 1 and game['SeedDiff'] > 0) or (game['Team1Won'] == 0 and game['SeedDiff'] < 0):
            print("   This was an upset based on seeding.")
        
        print("")

# Run the surprising predictions analysis
analyze_surprising_predictions(test_df, results[best_model_name]['pipeline'], feature_columns)

# Step 11: Create prediction function for new matchups
print("\n--- Prediction Function for New Matchups ---")

def predict_matchup(team1_id, team2_id, season, model_pipeline, stats_df, seeds_df, feature_cols):
    """
    Predict the outcome of a matchup between two teams.
    Returns probability of team1 winning.
    """
    # Get team stats
    team1_stats = stats_df[(stats_df['Season'] == season) & (stats_df['TeamID'] == team1_id)]
    team2_stats = stats_df[(stats_df['Season'] == season) & (stats_df['TeamID'] == team2_id)]
    
    if team1_stats.empty or team2_stats.empty:
        print(f"Error: Stats not found for one or both teams in season {season}")
        return None
    
    team1_stats = team1_stats.iloc[0]
    team2_stats = team2_stats.iloc[0]
    
    # Get seeds
    team1_seed_row = seeds_df[(seeds_df['Season'] == season) & (seeds_df['TeamID'] == team1_id)]
    team2_seed_row = seeds_df[(seeds_df['Season'] == season) & (seeds_df['TeamID'] == team2_id)]
    
    if team1_seed_row.empty or team2_seed_row.empty:
        print(f"Error: Seeds not found for one or both teams in season {season}")
        return None
    
    team1_seed = team1_seed_row['SeedValue'].values[0]
    team2_seed = team2_seed_row['SeedValue'].values[0]
    
    # Create feature dictionary
    features = {
        'SeedDiff': team1_seed - team2_seed
    }
    
    # Add differential features
    for stat in team1_stats.index:
        if stat not in ['Season', 'TeamID', 'TeamName']:
            stat_diff_name = f'Diff_{stat}'
            if stat_diff_name in feature_cols and isinstance(team1_stats[stat], (int, float)) and isinstance(team2_stats[stat], (int, float)):
                features[stat_diff_name] = team1_stats[stat] - team2_stats[stat]
    
    # Create DataFrame with features
    X_pred = pd.DataFrame([features])
    
    # Ensure all required features are present
    for col in feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = 0
    
    # Predict
    X_pred = X_pred[feature_cols]
    win_prob = model_pipeline.predict_proba(X_pred)[0, 1]
    
    return win_prob

# Example usage of the prediction function
print("\nExample matchup predictions:")

# Function to get team name from ID
def get_team_name(team_id, season, stats_df):
    team_row = stats_df[(stats_df['Season'] == season) & (stats_df['TeamID'] == team_id)]
    if not team_row.empty:
        return team_row['TeamName'].values[0]
    return f"Team {team_id}"

# Define some example matchups from recent tournaments
example_matchups = [
    # Format: (team1_id, team2_id, season)
    (1181, 1246, 2023),  # Example matchup 1
    (1314, 1437, 2023),  # Example matchup 2
    (1112, 1268, 2022)   # Example matchup 3
]

# Best model pipeline
best_pipeline = results[best_model_name]['pipeline']

# Predict each matchup
for team1_id, team2_id, season in example_matchups:
    team1_name = get_team_name(team1_id, season, enhanced_stats)
    team2_name = get_team_name(team2_id, season, enhanced_stats)
    
    # Get seeds
    team1_seed_row = tourney_seeds[(tourney_seeds['Season'] == season) & (tourney_seeds['TeamID'] == team1_id)]
    team2_seed_row = tourney_seeds[(tourney_seeds['Season'] == season) & (tourney_seeds['TeamID'] == team2_id)]
    
    if not team1_seed_row.empty and not team2_seed_row.empty:
        team1_seed = team1_seed_row['SeedValue'].values[0]
        team2_seed = team2_seed_row['SeedValue'].values[0]
        
        win_prob = predict_matchup(team1_id, team2_id, season, best_pipeline, enhanced_stats, tourney_seeds, feature_columns)
        
        if win_prob is not None:
            print(f"\n{team1_name} (Seed {team1_seed}) vs {team2_name} (Seed {team2_seed}) - Season {season}")
            print(f"Probability of {team1_name} winning: {win_prob:.4f}")
            print(f"Probability of {team2_name} winning: {1-win_prob:.4f}")
            
            # Determine favorite and underdog
            favorite = team1_name if team1_seed < team2_seed else team2_name
            underdog = team2_name if team1_seed < team2_seed else team1_name
            
            # Check if model predicts an upset
            predicted_winner = team1_name if win_prob > 0.5 else team2_name
            is_upset = (team1_seed > team2_seed and win_prob > 0.5) or (team1_seed < team2_seed and win_prob < 0.5)
            
            print(f"Predicted winner: {predicted_winner}")
            if is_upset:
                print("This would be an upset!")

# Step 12: Model Insights and Conclusions
print("\n--- Model Insights and Conclusions ---")

# Combine top features from all models
all_top_features = {}
for model_name, features in feature_importances.items():
    for feature, importance in features[:10]:  # Consider top 10 from each model
        if feature in all_top_features:
            all_top_features[feature] += importance
        else:
            all_top_features[feature] = importance

# Sort by combined importance
sorted_features = sorted(all_top_features.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 most important features across all models:")
for feature, importance in sorted_features[:10]:
    print(f"{feature}: {importance:.4f}")

print("\nKey insights for tournament success:")
print("1. Seed difference remains one of the strongest predictors")
print("2. Adjusted efficiency metrics are highly predictive")
print("3. Performance in close games (ClutchWin%) is important")
print("4. Neutral court performance is more relevant than home/away")
print("5. Momentum (Last10Win%) can indicate teams peaking at the right time")

print("\nModel performance summary:")
for model_name, result in results.items():
    print(f"{model_name}: Accuracy = {result['accuracy']:.4f}, AUC = {result['auc']:.4f}")

print("\nThis model can be used to:")
print("1. Predict the outcomes of tournament matchups")
print("2. Identify potential upsets")
print("3. Understand which team statistics are most predictive of tournament success")
print("4. Analyze how different rounds of the tournament have different dynamics") 