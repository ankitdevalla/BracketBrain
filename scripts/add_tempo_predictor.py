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
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Building tournament prediction model with downweighted tempo features...")

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

# 2. Data Preparation
print("\nPreparing data...")

# Convert seed strings to numeric values (e.g., 'W01' -> 1)
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
            continue
        
        # Get KenPom rankings
        try:
            w_kenpom = kenpom_data[(kenpom_data['Season'] == season) & (kenpom_data['TeamID'] == w_team)]['OrdinalRank'].values[0]
            l_kenpom = kenpom_data[(kenpom_data['Season'] == season) & (kenpom_data['TeamID'] == l_team)]['OrdinalRank'].values[0]
        except:
            w_kenpom = 400
            l_kenpom = 400
        
        # Get enhanced stats
        try:
            w_stats = enhanced_data[(enhanced_data['Season'] == season) & (enhanced_data['TeamID'] == w_team)].iloc[0]
            l_stats = enhanced_data[(enhanced_data['Season'] == season) & (enhanced_data['TeamID'] == l_team)].iloc[0]
        except:
            continue
        
        # Create feature dictionary
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
            'KenPomDiff': l_kenpom - w_kenpom,  # Higher ordinal rank means worse team
            'Upset': 1 if w_seed > l_seed else 0
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
        
        # Add tempo features
        if 'Poss' in w_stats and 'Poss' in l_stats:
            game_dict['W_Poss'] = w_stats['Poss']
            game_dict['L_Poss'] = l_stats['Poss']
            game_dict['Diff_Poss'] = w_stats['Poss'] - l_stats['Poss']
            game_dict['AvgTempo'] = (w_stats['Poss'] + l_stats['Poss']) / 2
            game_dict['TempoDiff'] = abs(w_stats['Poss'] - l_stats['Poss'])
        
        # Flip the sign for metrics where lower is better
        lower_is_better = ['KenPomDiff', 'Diff_AdjD', 'Diff_TORate']
        for feature in lower_is_better:
            if feature in game_dict:
                game_dict[feature] = -game_dict[feature]
        
        games.append(game_dict)
    
    return pd.DataFrame(games)

game_df = create_game_dataset(tourney_results, tourney_seeds, kenpom, enhanced_stats)
print(f"Game dataset shape: {game_df.shape}")

def create_balanced_dataset(game_df):
    original_df = game_df.copy()
    swapped_df = game_df.copy()
    
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
    
    for diff_col in ['ScoreDiff', 'SeedDiff', 'KenPomDiff']:
        if diff_col in swapped_df.columns:
            swapped_df[diff_col] = -swapped_df[diff_col]
    
    for col in game_df.columns:
        if col.startswith('W_'):
            l_col = 'L_' + col[2:]
            if l_col in game_df.columns:
                temp = swapped_df[col].copy()
                swapped_df[col] = swapped_df[l_col]
                swapped_df[l_col] = temp
        if col.startswith('Diff_'):
            swapped_df[col] = -swapped_df[col]
    
    original_df['Target'] = 1
    swapped_df['Target'] = 0
    balanced_df = pd.concat([original_df, swapped_df], ignore_index=True)
    
    return balanced_df

balanced_df = create_balanced_dataset(game_df)
print(f"Balanced dataset shape: {balanced_df.shape}")

print("\nSelecting features...")

# Base features
seed_features = ['SeedDiff']
kenpom_features = ['KenPomDiff']
enhanced_diff_features = [col for col in balanced_df.columns if col.startswith('Diff_')]

# Downweight tempo features by 0.2
balanced_df['AvgTempo_scaled'] = balanced_df['AvgTempo'] * 0.2
balanced_df['TempoDiff_scaled'] = balanced_df['TempoDiff'] * 0.2
tempo_features = ['AvgTempo_scaled', 'TempoDiff_scaled']

# Combine features
features = seed_features + kenpom_features + enhanced_diff_features + tempo_features
print(f"Selected {len(features)} features: {features}")

X = balanced_df[features]
y = balanced_df['Target']

print("\nSplitting data...")
seasons = balanced_df['Season'].unique()
train_seasons = seasons[:-3]  # All but last 3 seasons for training
val_seasons = [seasons[-3]]   # Third-to-last season for validation
test_seasons = seasons[-2:]   # Last 2 seasons for testing

X_train = X[balanced_df['Season'].isin(train_seasons)]
y_train = y[balanced_df['Season'].isin(train_seasons)]
X_val = X[balanced_df['Season'].isin(val_seasons)]
y_val = y[balanced_df['Season'].isin(val_seasons)]
X_test = X[balanced_df['Season'].isin(test_seasons)]
y_test = y[balanced_df['Season'].isin(test_seasons)]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\nTraining models...")

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)
    loss = log_loss(y_val, y_val_pred)
    results[name] = {'pipeline': pipeline, 'val_auc': auc, 'val_log_loss': loss}
    print(f"{name} - Validation AUC: {auc:.4f}, Log Loss: {loss:.4f}")

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

grid_search = GridSearchCV(
    best_pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
val_log_loss = log_loss(y_val, y_val_pred)
print(f"Tuned model - Validation AUC: {val_auc:.4f}, Log Loss: {val_log_loss:.4f}")

print("\nEvaluating on test set...")
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = best_model.predict(X_test)
test_auc = roc_auc_score(y_test, y_test_pred_proba)
test_log_loss = log_loss(y_test, y_test_pred_proba)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Log Loss: {test_log_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

print("\nAnalyzing feature importance...")
model_estimator = best_model.named_steps['model']
if hasattr(model_estimator, 'feature_importances_'):
    importances = model_estimator.feature_importances_
elif hasattr(model_estimator, 'coef_'):
    importances = np.abs(model_estimator.coef_[0])
else:
    importances = np.zeros(len(features))
    print("Warning: Model does not provide built-in feature importance attributes.")
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nTop 15 most important features:")
print(feature_importance.head(15))

print("\nAnalyzing impact of tempo features...")
# Use the original balanced dataframe to retrieve unscaled tempo values
test_original_df = balanced_df[balanced_df['Season'].isin(test_seasons)]
test_results = pd.DataFrame({
    'AvgTempo': test_original_df['AvgTempo'],
    'TempoDiff': test_original_df['TempoDiff'],
    'Actual': y_test.values,
    'Predicted': y_test_pred,
    'Probability': y_test_pred_proba
})
tempo_bins = [50, 60, 65, 70, 75, 80, 90]
tempo_labels = ["Very Slow (<60)", "Slow (60-65)", "Moderate (65-70)", 
                "Fast (70-75)", "Very Fast (75-80)", "Extreme (>80)"]
diff_bins = [0, 2, 4, 6, 8, 10, 20]
diff_labels = ["Minimal (<2)", "Small (2-4)", "Moderate (4-6)", 
               "Large (6-8)", "Very Large (8-10)", "Extreme (>10)"]
test_results['Tempo_bin'] = pd.cut(test_results['AvgTempo'], bins=tempo_bins, labels=tempo_labels)
test_results['TempoDiff_bin'] = pd.cut(test_results['TempoDiff'], bins=diff_bins, labels=diff_labels)
tempo_performance = test_results.groupby('Tempo_bin').agg(
    Count=('AvgTempo', 'count'),
    Actual_Win_Rate=('Actual', 'mean'),
    Predicted_Win_Rate=('Predicted', 'mean'),
    Avg_Probability=('Probability', 'mean')
).reset_index()
print("\nPerformance by Game Tempo:")
print(tempo_performance)
tempo_diff_performance = test_results.groupby('TempoDiff_bin').agg(
    Count=('TempoDiff', 'count'),
    Actual_Win_Rate=('Actual', 'mean'),
    Predicted_Win_Rate=('Predicted', 'mean'),
    Avg_Probability=('Probability', 'mean')
).reset_index()
print("\nPerformance by Tempo Difference:")
print(tempo_diff_performance)

print("\nSaving model and feature names...")
joblib.dump(best_model, '../scripts/final_model_with_tempo2.pkl')
np.save('../scripts/feature_names_with_tempo.npy', features)
print("Model saved to '../scripts/final_model_with_tempo2.pkl'")
print("Feature names saved to '../scripts/feature_names_with_tempo.npy'")

print("\nSample matchup predictions:")
test_df = balanced_df[balanced_df['Season'].isin(test_seasons)]
test_original = test_df[test_df['Target'] == 1].copy()
X_test_original = test_original[features]
test_original['Predicted_Prob'] = best_model.predict_proba(X_test_original)[:, 1]
test_original['Predicted_Class'] = best_model.predict(X_test_original)
test_original['Prediction_Correct'] = test_original['Predicted_Prob'].apply(lambda x: "Yes" if x >= 0.5 else "No")
try:
    teams_df = pd.read_csv("../raw_data/MTeams.csv")
    teams_dict = dict(zip(teams_df['TeamID'], teams_df['TeamName']))
    print(f"Loaded {len(teams_dict)} team names")
except FileNotFoundError:
    print("MTeams.csv not found. Using TeamIDs instead.")
    teams_dict = {}
test_original['Team1'] = test_original['WTeamID'].map(teams_dict)
test_original['Team2'] = test_original['LTeamID'].map(teams_dict)
test_original['Actual_Winner'] = test_original['Team1']
sample_predictions = test_original[['Season', 'Team1', 'Team2', 'WSeed', 'LSeed', 
                                   'Predicted_Prob', 'Actual_Winner', 'Prediction_Correct',
                                   'AvgTempo', 'TempoDiff']].copy()
sample_predictions = sample_predictions.rename(columns={'WSeed': 'Team1_Seed', 'LSeed': 'Team2_Seed'})
sample_predictions['Predicted_Prob'] = sample_predictions['Predicted_Prob'].apply(lambda x: f"{x:.1%}")
sample_predictions = sample_predictions.sample(15)
print("\nSample matchup predictions (including tempo features):")
print(sample_predictions)

print("\nAnalysis complete! Model with downweighted tempo features has been trained and saved.")
