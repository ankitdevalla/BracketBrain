import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading
print("Loading data...")

# Load tournament results
tourney_results = pd.read_csv('../raw_data/MNCAATourneyDetailedResults.csv')
print(f"Tournament results shape: {tourney_results.shape}")

# Load tournament seeds
tourney_seeds = pd.read_csv('../raw_data/MNCAATourneySeeds.csv')
print(f"Tournament seeds shape: {tourney_seeds.shape}")

# Load KenPom rankings
kenpom = pd.read_csv('../pre_tourney_data/KenPom-Rankings-Updated.csv')
print(f"KenPom rankings shape: {kenpom.shape}")

# Load enhanced stats
enhanced_stats = pd.read_csv('../pre_tourney_data/EnhancedTournamentStats.csv')
print(f"Enhanced stats shape: {enhanced_stats.shape}")

# 2. Data Exploration
print("\nExploring data...")

# Display sample of tournament results
print("\nSample of tournament results:")
print(tourney_results.head())

# Display sample of tournament seeds
print("\nSample of tournament seeds:")
print(tourney_seeds.head())

# Display sample of KenPom rankings
print("\nSample of KenPom rankings:")
print(kenpom.head())

# Display sample of enhanced stats
print("\nSample of enhanced stats:")
print(enhanced_stats.head())

# Check for missing values
print("\nMissing values in tournament results:")
print(tourney_results.isnull().sum())

print("\nMissing values in tournament seeds:")
print(tourney_seeds.isnull().sum())

print("\nMissing values in KenPom rankings:")
print(kenpom.isnull().sum())

print("\nMissing values in enhanced stats:")
print(enhanced_stats.isnull().sum())


# 3. Feature Engineering
print("\nEngineering features...")

# Convert seed strings to numeric values, seeds initially look like 'W01' or 'Y16'
tourney_seeds['SeedValue'] = tourney_seeds['Seed'].str[1:3].astype(int)

def create_game_dataset(results, seeds, kenpom_data, enhanced_data):
    games = []
    
    for _, game in results.iterrows():
        season = game['Season']
        day = game['DayNum']
        
        # Get team IDs
        w_team = game['WTeamID']
        l_team = game['LTeamID']
        
        # Get seed information
        try:
            w_seed = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == w_team)]['SeedValue'].values[0]
            l_seed = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == l_team)]['SeedValue'].values[0]
        except:
            # Skip if seed info is missing
            continue
        
        # Get KenPom rankings (use the most recent ranking before the tournament)
        try:
            w_kenpom = kenpom_data[(kenpom_data['Season'] == season) & 
                                  (kenpom_data['TeamID'] == w_team)]['OrdinalRank'].values[0]
            l_kenpom = kenpom_data[(kenpom_data['Season'] == season) & 
                                  (kenpom_data['TeamID'] == l_team)]['OrdinalRank'].values[0]
        except:
            # Use a high default rank if KenPom data is missing
            w_kenpom = 400
            l_kenpom = 400
        
        # Get enhanced stats
        try:
            w_stats = enhanced_data[(enhanced_data['Season'] == season) & 
                                   (enhanced_data['TeamID'] == w_team)].iloc[0]
            l_stats = enhanced_data[(enhanced_data['Season'] == season) & 
                                   (enhanced_data['TeamID'] == l_team)].iloc[0]
        except:
            # Skip if enhanced stats are missing
            continue
        
        # Create feature dictionary with an upset indicator
        game_dict = {
            'Season': season,
            'DayNum': day,
            'WTeamID': w_team,
            'LTeamID': l_team,
            'WScore': game['WScore'],
            'LScore': game['LScore'],
            'ScoreDiff': game['WScore'] - game['LScore'],
            'WSeed': w_seed,
            'LSeed': l_seed,
            'SeedDiff': l_seed - w_seed,  # Higher seeds (worse teams) have higher numbers
            'WKenPom': w_kenpom,
            'LKenPom': l_kenpom,
            'KenPomDiff': l_kenpom - w_kenpom,  # Higher ranks (worse teams) have higher numbers
            'Upset': 1 if w_seed > l_seed else 0  # Upset indicator: 1 if winning team is lower seeded (higher seed value)
        }
        
        # Add enhanced stats features
        enhanced_features = [
            'AdjO', 'AdjD', 'AdjNetRtg', 'SOS_NetRtg', 'Expected Win%', 
            'ThreePtRate', 'FTRate', 'AstRate', 'TORate', 'ORRate', 'DRRate',
            'ScoreStdDev', 'MarginStdDev', 'ORtgStdDev', 'DRtgStdDev',
            'HomeWin%', 'AwayWin%', 'NeutralWin%', 'Last10Win%'
        ]
        
        for feature in enhanced_features:
            if feature in w_stats and feature in l_stats:
                game_dict[f'W_{feature}'] = w_stats[feature]
                game_dict[f'L_{feature}'] = l_stats[feature]
                game_dict[f'Diff_{feature}'] = w_stats[feature] - l_stats[feature]
        
        # Flip the sign for metrics where lower is better
        # This ensures positive values always mean "better" for all metrics
        lower_is_better = ['KenPomDiff', 'Diff_AdjD', 'Diff_TORate']
        for feature in lower_is_better:
            if feature in game_dict:
                game_dict[feature] = -game_dict[feature]
        
        games.append(game_dict)
    
    return pd.DataFrame(games)


# Create game dataset
game_df = create_game_dataset(tourney_results, tourney_seeds, kenpom, enhanced_stats)
print(f"Game dataset shape: {game_df.shape}")
pd.set_option('display.max_columns', None)
print(game_df.tail())

def create_balanced_dataset(game_df):
    # Create copies of the original dataframe
    original_df = game_df.copy()
    swapped_df = game_df.copy()
    
    # Swap key columns in swapped_df using temporary variables
    cols_to_swap = {
        'WTeamID': 'LTeamID', 'LTeamID': 'WTeamID',
        'WScore': 'LScore', 'LScore': 'WScore',
        'WSeed': 'LSeed', 'LSeed': 'WSeed',
        'WKenPom': 'LKenPom', 'LKenPom': 'WKenPom'
    }
    
    for col1, col2 in cols_to_swap.items():
        temp = swapped_df[col1].copy()
        swapped_df[col1] = swapped_df[col2]
        swapped_df[col2] = temp
    
    # Negate the derived difference columns in swapped_df
    for diff_col in ['ScoreDiff', 'SeedDiff', 'KenPomDiff']:
        if diff_col in swapped_df.columns:
            swapped_df[diff_col] = -swapped_df[diff_col]
    
    # Swap enhanced stats columns in swapped_df
    for col in game_df.columns:
        if col.startswith('W_'):
            l_col = 'L_' + col[2:]
            if l_col in game_df.columns:
                temp = swapped_df[col].copy()
                swapped_df[col] = swapped_df[l_col]
                swapped_df[l_col] = temp
        
        # For difference features, simply negate them
        if col.startswith('Diff_'):
            swapped_df[col] = -swapped_df[col]
    
    # Add target column: 1 for original (win perspective), 0 for swapped (loss perspective)
    original_df['Target'] = 1
    swapped_df['Target'] = 0
    
    # Combine the two DataFrames
    balanced_df = pd.concat([original_df, swapped_df], ignore_index=True)
    
    return balanced_df

# Create balanced dataset
balanced_df = create_balanced_dataset(game_df)
print(f"Balanced dataset shape: {balanced_df.shape}")
print(balanced_df)

# 5. Feature Selection
print("\nSelecting features...")

# Define features to use
seed_features = ['SeedDiff']
kenpom_features = ['KenPomDiff']
enhanced_diff_features = [col for col in balanced_df.columns if col.startswith('Diff_')]

# Combine all features
features = seed_features + kenpom_features + enhanced_diff_features
print(f"Selected {len(features)} features: {features}")

# Prepare X and y
X = balanced_df[features]
y = balanced_df['Target']

# 6. Train-Test Split
print("\nSplitting data...")

# Split by season to avoid data leakage
train_seasons = balanced_df['Season'].unique()[:-3]  # Use all but the last 3 seasons for training
val_seasons = [balanced_df['Season'].unique()[-3]]   # Use third-to-last season for validation
test_seasons = balanced_df['Season'].unique()[-2:]   # Use last 2 seasons for testing

X_train = X[balanced_df['Season'].isin(train_seasons)]
y_train = y[balanced_df['Season'].isin(train_seasons)]

X_val = X[balanced_df['Season'].isin(val_seasons)]
y_val = y[balanced_df['Season'].isin(val_seasons)]

X_test = X[balanced_df['Season'].isin(test_seasons)]
y_test = y[balanced_df['Season'].isin(test_seasons)]

print(X_test)


print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\nTraining models...")

# Define models to try
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create a pipeline with scaling; note that scaling is crucial for Logistic Regression
    # and not strictly needed for tree-based models, but it's okay to include for consistency.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_val_pred = pipeline.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    val_auc = roc_auc_score(y_val, y_val_pred)
    val_log_loss = log_loss(y_val, y_val_pred)
    
    # Store results
    results[name] = {
        'pipeline': pipeline,
        'val_auc': val_auc,
        'val_log_loss': val_log_loss
    }
    
    print(f"{name} - Validation AUC: {val_auc:.4f}, Log Loss: {val_log_loss:.4f}")

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['val_auc'])
best_pipeline = results[best_model_name]['pipeline']
print(f"\nBest model: {best_model_name} with AUC: {results[best_model_name]['val_auc']:.4f}")

print("\nTuning hyperparameters for the best model...")

if best_model_name == 'Logistic Regression':
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }
else:  # XGBoost
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.8, 0.9, 1.0]
    }

# Create grid search
grid_search = GridSearchCV(
    best_pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Get best model
best_model = grid_search.best_estimator_

# Evaluate on validation set
y_val_pred = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
val_log_loss = log_loss(y_val, y_val_pred)
print(f"Tuned model - Validation AUC: {val_auc:.4f}, Log Loss: {val_log_loss:.4f}")

# Note: For tree-based models, the StandardScaler may not be necessary.
# You could optionally remove it for efficiency if needed.

# 9. Final Evaluation on Test Set
print("\nEvaluating on test set...")

# Make predictions on the test set
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = best_model.predict(X_test)

# Calculate evaluation metrics
test_auc = roc_auc_score(y_test, y_test_pred_proba)
test_log_loss = log_loss(y_test, y_test_pred_proba)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test AUC: {test_auc:.4f}")
print(f"Test Log Loss: {test_log_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# 10. Feature Importance Analysis
print("\nAnalyzing feature importance...")

# Retrieve the final estimator from the pipeline using named_steps for robustness
model_estimator = best_model.named_steps['model'] if hasattr(best_model, 'named_steps') else best_model[-1]

# Get feature importance based on available attributes
if hasattr(model_estimator, 'feature_importances_'):
    importances = model_estimator.feature_importances_
elif hasattr(model_estimator, 'coef_'):
    importances = np.abs(model_estimator.coef_[0])
else:
    # As a fallback, we could compute permutation importance, or use zeros
    importances = np.zeros(len(features))
    print("Warning: Model does not provide built-in feature importance attributes.")

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})

# Sort features by importance in descending order
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Display top 15 features
print("\nTop 15 most important features:")
print(feature_importance.head(15))

# 11. Analyze model performance by seed difference
print("\nAnalyzing model performance by seed difference...")

# Create a dataframe with predictions and actual values
test_results = pd.DataFrame({
    'SeedDiff': X_test['SeedDiff'],
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Probability': y_test_pred_proba
})

# Define bins and corresponding labels for seed differences
bins = [-20, -10, -5, -1, 1, 5, 10, 20]
labels = ["[-20, -10]", "[-10, -5]", "[-5, -1]", "[-1, 1]", "[1, 5]", "[5, 10]", "[10, 20]"]

# Create a new column for binned seed differences
test_results['SeedDiff_bin'] = pd.cut(test_results['SeedDiff'], bins=bins, labels=labels)

# Group by the binned seed difference and calculate metrics
seed_diff_performance = test_results.groupby('SeedDiff_bin').agg(
    Count=('SeedDiff', 'count'),
    Actual_Win_Rate=('Actual', 'mean'),
    Predicted_Win_Rate=('Predicted', 'mean'),
    Avg_Probability=('Probability', 'mean')
).reset_index()

print(seed_diff_performance)

print("\nSample matchup predictions:")

# Filter the balanced test set for original perspective rows (Target == 1)
test_df = balanced_df[balanced_df['Season'].isin(test_seasons)]
test_original = test_df[test_df['Target'] == 1].copy()

# Compute predictions on these rows
X_test_original = test_original[features]
test_original['Predicted_Prob'] = best_model.predict_proba(X_test_original)[:, 1]
test_original['Predicted_Class'] = best_model.predict(X_test_original)

# For these rows, the actual winner is Team1 (WTeamID)
# We'll mark the prediction as correct if the predicted probability is >= 0.5
test_original['Prediction_Correct'] = test_original['Predicted_Prob'].apply(lambda x: "Yes" if x >= 0.5 else "No")

# Load team names from CSV (if available)
try:
    teams_df = pd.read_csv("../raw_data/MTeams.csv")
    teams_dict = dict(zip(teams_df['TeamID'], teams_df['TeamName']))
    print(f"Loaded {len(teams_dict)} team names")
except FileNotFoundError:
    print("MTeams.csv not found. Using TeamIDs instead.")
    teams_dict = {}

# Map team IDs to team names for display
test_original['Team1'] = test_original['WTeamID'].map(teams_dict)
test_original['Team2'] = test_original['LTeamID'].map(teams_dict)

# Since these rows represent the original perspective, the actual winner is Team1
test_original['Actual_Winner'] = test_original['Team1']

# Select and rename columns for clarity, including seeds for each team
sample_predictions = test_original[['Season', 'Team1', 'Team2', 'WSeed', 'LSeed', 
                                      'Predicted_Prob', 'Actual_Winner', 'Prediction_Correct']].copy()

# Rename seed columns for clarity
sample_predictions = sample_predictions.rename(columns={'WSeed': 'Team1_Seed', 'LSeed': 'Team2_Seed'})

# Format the predicted probability as a percentage for readability
sample_predictions['Predicted_Prob'] = sample_predictions['Predicted_Prob'].apply(lambda x: f"{x:.1%}")

# Take a random sample of 10 matchups for display
sample_predictions = sample_predictions.sample(20)

print("\nSample matchup predictions (each row shows the season, team names, seeds, the model's predicted win probability for Team1, the actual winner, and whether the prediction was correct):")
print(sample_predictions)

# upset analysis
print("\nFiltering to only show games where the lower seed wins (upsets)...")

# Using the 'test_original' DataFrame from earlier (original perspective where Target==1)
# Filter for upset games: winning team's seed (WSeed) is greater than the losing team's seed (LSeed)
upset_games = test_original[test_original['WSeed'] > test_original['LSeed']].copy()

# Map team names if not already mapped
try:
    teams_df = pd.read_csv("../raw_data/MTeams.csv")
    teams_dict = dict(zip(teams_df['TeamID'], teams_df['TeamName']))
except FileNotFoundError:
    teams_dict = {}

upset_games['Team1'] = upset_games['WTeamID'].map(teams_dict)
upset_games['Team2'] = upset_games['LTeamID'].map(teams_dict)
# In these rows, Team1 is the actual winner (by original perspective)
upset_games['Actual_Winner'] = upset_games['Team1']

# Rename seed columns for clarity
upset_games = upset_games.rename(columns={'WSeed': 'Team1_Seed', 'LSeed': 'Team2_Seed'})

# Select and format the desired columns:
upset_display = upset_games[['Season', 'Team1', 'Team2', 'Team1_Seed', 'Team2_Seed', 
                             'Predicted_Prob', 'Actual_Winner', 'Prediction_Correct']].copy()
upset_display['Predicted_Prob'] = upset_display['Predicted_Prob'].apply(lambda x: f"{x:.1%}")

# Sample 10 upset games to display (or show all if preferred)
sample_upsets = upset_display.sample(15)

print("\nSample upset game predictions:")
print(sample_upsets)

# save the preliminary model
import joblib

# Save the model to a file
joblib.dump(best_model, 'final_model_py2.pkl')



