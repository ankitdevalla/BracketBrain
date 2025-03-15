import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load Data ---

def load_data():
    """Load tournament data, seeds, T-Rank data, and season averages"""
    # Load tournament game-by-game results
    tour_df = pd.read_csv("../raw_data/MNCAATourneyDetailedResults.csv")
    
    # Filter to only include data from 2008 onwards
    tour_df = tour_df[tour_df['Season'] >= 2008]
    
    # Load tournament seeds
    seeds_df = pd.read_csv("../raw_data/MNCAATourneySeeds.csv")
    seeds_df = seeds_df[seeds_df['Season'] >= 2008]
    seeds_df['Seed'] = seeds_df['Seed'].str.extract(r'(\d+)').astype(int)
    
    # Load simplified T-Rank data with Barthag and Exp
    trank_df = pd.read_csv("../processed_data/trank_simplified.csv")
    trank_df = trank_df[trank_df['Season'] >= 2008]
    
    # Load team season averages
    season_avg_df = pd.read_csv("../pre_tourney_data/TeamSeasonAverages_with_SoS.csv")
    season_avg_df = season_avg_df[season_avg_df['Season'] >= 2008]
    
    return tour_df, seeds_df, trank_df, season_avg_df

def create_matchups(tour_df, seeds_df, trank_df):
    """Create all possible tournament matchups for each season"""
    all_matchups = []
    seasons = tour_df['Season'].unique()
    
    for season in seasons:
        # Filter data for this season
        season_seeds = seeds_df[seeds_df['Season'] == season]
        season_trank = trank_df[trank_df['Season'] == season]
        
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

def add_features(matchups_df, trank_df, seeds_df, season_avg_df):
    """Add T-Rank features, season averages, and seeds to matchups"""
    # Add team1 T-Rank stats
    matchups_df = matchups_df.merge(
        trank_df, 
        left_on=['Season', 'Team1'], 
        right_on=['Season', 'TeamID'],
        how='left'
    )
    
    # Rename columns for team1
    matchups_df.rename(columns={
        'TeamName': 'Team1_Name',
        'Barthag': 'Team1_Barthag',
        'Exp': 'Team1_Exp'
    }, inplace=True)
    
    # Drop unnecessary columns
    matchups_df.drop('TeamID', axis=1, inplace=True)
    
    # Add team2 T-Rank stats
    matchups_df = matchups_df.merge(
        trank_df, 
        left_on=['Season', 'Team2'], 
        right_on=['Season', 'TeamID'],
        how='left'
    )
    
    # Rename columns for team2
    matchups_df.rename(columns={
        'TeamName': 'Team2_Name',
        'Barthag': 'Team2_Barthag',
        'Exp': 'Team2_Exp'
    }, inplace=True)
    
    # Drop unnecessary columns
    matchups_df.drop('TeamID', axis=1, inplace=True)
    
    # Add team1 season averages
    matchups_df = matchups_df.merge(
        season_avg_df,
        left_on=['Season', 'Team1'],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    
    # Rename season average columns for team1
    season_cols = ['WinPct', 'Avg_Score', 'Avg_FGM', 'Avg_FGA', 'Avg_FGM3', 'Avg_FGA3', 
                  'Avg_FTM', 'Avg_FTA', 'Avg_OR', 'Avg_DR', 'Avg_Ast', 'Avg_TO', 
                  'Avg_Stl', 'Avg_Blk', 'Avg_PF', 'Avg_Opp_WinPct', 'Last30_WinRatio']
    
    for col in season_cols:
        if col in matchups_df.columns:
            matchups_df.rename(columns={col: f'Team1_{col}'}, inplace=True)
    
    # Drop unnecessary columns
    matchups_df.drop('TeamID', axis=1, inplace=True)
    if 'TeamName' in matchups_df.columns:
        matchups_df.drop('TeamName', axis=1, inplace=True)
    
    # Add team2 season averages
    matchups_df = matchups_df.merge(
        season_avg_df,
        left_on=['Season', 'Team2'],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    
    # Rename season average columns for team2
    for col in season_cols:
        if col in matchups_df.columns:
            matchups_df.rename(columns={col: f'Team2_{col}'}, inplace=True)
    
    # Drop unnecessary columns
    matchups_df.drop('TeamID', axis=1, inplace=True)
    if 'TeamName' in matchups_df.columns:
        matchups_df.drop('TeamName', axis=1, inplace=True)
    
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
    # Create difference features for T-Rank stats
    # matchups_df['Barthag_diff'] = matchups_df['Team1_Barthag'] - matchups_df['Team2_Barthag']
    matchups_df['Exp_diff'] = matchups_df['Team1_Exp'] - matchups_df['Team2_Exp']
    
    # Create difference features for season averages
    season_cols = ['WinPct', 'Avg_Score', 'Avg_FGM', 'Avg_FGA', 'Avg_FGM3', 'Avg_FGA3', 
                  'Avg_FTM', 'Avg_FTA', 'Avg_OR', 'Avg_DR', 'Avg_Ast', 'Avg_TO', 
                  'Avg_Stl', 'Avg_Blk', 'Avg_PF', 'Avg_Opp_WinPct', 'Last30_WinRatio']
    
    for col in season_cols:
        team1_col = f'Team1_{col}'
        team2_col = f'Team2_{col}'
        
        if team1_col in matchups_df.columns and team2_col in matchups_df.columns:
            matchups_df[f'{col}_diff'] = matchups_df[team1_col] - matchups_df[team2_col]
    
    # Add seed difference
    matchups_df['Seed_diff'] = matchups_df['Team1_Seed'] - matchups_df['Team2_Seed']
    
    # Create target variable (1 if Team1 wins, 0 if Team2 wins)
    matchups_df['Target'] = (matchups_df['Team1'] == matchups_df['Winner']).astype(int)
    
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
    
    # Define the features to use (all difference features)
    feature_cols = [col for col in matchups_df.columns if col.endswith('_diff')]
    
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    # Handle any NaN values
    matchups_df[feature_cols] = matchups_df[feature_cols].fillna(0)
    
    # Extract features and target
    X = matchups_df[feature_cols]
    y = matchups_df['Target']
    
    # Use GroupKFold to ensure seasons stay together (leave-one-season-out CV)
    seasons = matchups_df['Season'].unique()
    cv_results = []
    
    print("\nStarting leave-one-season-out cross-validation...")
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
    
    print("\nFeature importance in the final model:")
    print(importance_df)
    
    # Save feature importance plot
    output_dir = "../analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300)
    
    return final_model, importance_df, feature_cols, scaler_final, cv_results_df

# --- Step 4: Main Function ---

def main():
    print("Loading data...")
    tour_df, seeds_df, trank_df, season_avg_df = load_data()
    print(f"Loaded tournament data with {len(tour_df)} games")
    print(f"Loaded T-Rank data with {len(trank_df)} rows")
    print(f"Loaded season averages data with {len(season_avg_df)} rows")
    
    print("\nCreating matchups...")
    matchups_df = create_matchups(tour_df, seeds_df, trank_df)
    print(f"Created {len(matchups_df)} matchups")
    
    print("\nAdding features...")
    matchups_df = add_features(matchups_df, trank_df, seeds_df, season_avg_df)
    
    print("\nCreating difference features...")
    matchups_df = create_difference_features(matchups_df)
    
    # Train model with cross-validation
    print("\n" + "="*50)
    print("TRAINING MODEL WITH ALL FEATURES")
    print("="*50)
    model, importance_df, feature_cols, scaler, cv_results = train_model_with_cv(matchups_df)
    
    # Save the model and related artifacts
    models_dir = '../models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"\nCreated directory: {models_dir}")
    
    # Save model
    model_path = os.path.join(models_dir, 'bart_model_simplified.pkl')
    joblib.dump(model, model_path)
    print(f"\nSaved model as '{model_path}'")
    
    # Save feature columns and scaler
    # joblib.dump(feature_cols, os.path.join(models_dir, 'bart_model_simplified_features.pkl'))
    # joblib.dump(scaler, os.path.join(models_dir, 'bart_model_simplified_scaler.pkl'))
    
    # Save model info
    model_info = {
        'feature_importance': importance_df.to_dict(),
        'cv_results': cv_results.to_dict(),
        'avg_auc': cv_results['auc'].mean(),
        'avg_log_loss': cv_results['log_loss'].mean(),
        'features_used': feature_cols
    }
    # joblib.dump(model_info, os.path.join(models_dir, 'bart_model_simplified_info.pkl'))
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main() 