"""
Script to add tempo (possessions per 40 minutes) as a predictor for March Madness tournament predictions.
This can be incorporated into the Tournament_Prediction.ipynb notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import joblib
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

# 2. Feature Engineering
print("\nEngineering features with Tempo...")

# Convert seed strings to numeric values
tourney_seeds['SeedValue'] = tourney_seeds['Seed'].str[1:3].astype(int)

def create_game_dataset_with_tempo(results, seeds, kenpom_data, enhanced_data):
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
        
        # Get KenPom rankings
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
        
        # Add enhanced stats features including Tempo (Poss)
        enhanced_features = [
            'Poss',  # Add Tempo (possessions per 40 minutes)
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
        
        # Add tempo pace difference and average tempo
        if 'Poss' in w_stats and 'Poss' in l_stats:
            # Poss represents tempo (possessions per 40 minutes)
            game_dict['AvgTempo'] = (w_stats['Poss'] + l_stats['Poss']) / 2
            game_dict['TempoDiff'] = abs(w_stats['Poss'] - l_stats['Poss'])
        
        games.append(game_dict)
    
    return pd.DataFrame(games)

# Create game dataset with tempo
game_df = create_game_dataset_with_tempo(tourney_results, tourney_seeds, kenpom, enhanced_stats)
print(f"Game dataset shape with tempo features: {game_df.shape}")

# 3. Feature Selection
print("\nSelecting features...")

# Define features for prediction
features = [
    'SeedDiff', 'KenPomDiff', 
    'Diff_AdjO', 'Diff_AdjD', 'Diff_AdjNetRtg', 'Diff_SOS_NetRtg', 'Diff_Expected Win%',
    'Diff_ThreePtRate', 'Diff_FTRate', 'Diff_AstRate', 'Diff_TORate', 'Diff_ORRate', 'Diff_DRRate',
    'Diff_ScoreStdDev', 'Diff_MarginStdDev', 'Diff_ORtgStdDev', 'Diff_DRtgStdDev',
    'Diff_HomeWin%', 'Diff_AwayWin%', 'Diff_NeutralWin%', 'Diff_Last10Win%',
    # Add tempo features
    'Diff_Poss', 'AvgTempo', 'TempoDiff'
]

# Make sure all features exist in the dataframe
features = [f for f in features if f in game_df.columns]

X = game_df[features]
y = (game_df['WTeamID'] < game_df['LTeamID']).astype(int)  # 1 if Team1 wins, 0 if Team2 wins

print(f"Features used: {features}")
print(f"Feature matrix shape: {X.shape}")

# 4. Model Training
print("\nTraining model with tempo features...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# 5. Model Evaluation
print("\nEvaluating model...")

# Make predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
loss = log_loss(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Log Loss: {loss:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Feature Importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': pipeline.named_steps['classifier'].feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.savefig('../output/feature_importance_with_tempo.png')
print("Feature importance plot saved to '../output/feature_importance_with_tempo.png'")

# 7. Save the model
joblib.dump(pipeline, '../scripts/final_model_with_tempo.pkl')
print("Model saved to '../scripts/final_model_with_tempo.pkl'")

# Save feature names for later use
np.save('../scripts/feature_names_with_tempo.npy', features, allow_pickle=True)
print("Feature names saved to '../scripts/feature_names_with_tempo.npy'")

print("\nDone!")
