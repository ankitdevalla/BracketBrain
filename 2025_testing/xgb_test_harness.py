import pandas as pd
import numpy as np
import joblib
import re
import xgboost as xgb

# ------------------------------
# Utility Functions
# ------------------------------

def create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed, include_tempo=True):
    """
    Create a DataFrame of features for the matchup.
    This must produce the same set and order of features that your model expects.
    
    Parameters:
    - team1_stats, team2_stats: Stats for each team (as a pd.Series)
    - team1_seed, team2_seed: Numeric seeds for each team
    - include_tempo: Whether to include tempo-related features
    
    Returns:
    - DataFrame with features in the order expected by the model
    """
    # Base features using the available season stats plus seed difference.
    base_features = {
        'WinPct_diff': team1_stats['WinPct'] - team2_stats['WinPct'],
        'Avg_Score_diff': team1_stats['Avg_Score'] - team2_stats['Avg_Score'],
        'Avg_FGM_diff': team1_stats['Avg_FGM'] - team2_stats['Avg_FGM'],
        'Avg_FGA_diff': team1_stats['Avg_FGA'] - team2_stats['Avg_FGA'],
        'Avg_FGM3_diff': team1_stats['Avg_FGM3'] - team2_stats['Avg_FGM3'],
        'Avg_FGA3_diff': team1_stats['Avg_FGA3'] - team2_stats['Avg_FGA3'],
        'Avg_FTM_diff': team1_stats['Avg_FTM'] - team2_stats['Avg_FTM'],
        'Avg_FTA_diff': team1_stats['Avg_FTA'] - team2_stats['Avg_FTA'],
        'Avg_OR_diff': team1_stats['Avg_OR'] - team2_stats['Avg_OR'],
        'Avg_DR_diff': team1_stats['Avg_DR'] - team2_stats['Avg_DR'],
        'Avg_Ast_diff': team1_stats['Avg_Ast'] - team2_stats['Avg_Ast'],
        'Avg_TO_diff': team1_stats['Avg_TO'] - team2_stats['Avg_TO'],
        'Avg_Stl_diff': team1_stats['Avg_Stl'] - team2_stats['Avg_Stl'],
        'Avg_Blk_diff': team1_stats['Avg_Blk'] - team2_stats['Avg_Blk'],
        'Avg_PF_diff': team1_stats['Avg_PF'] - team2_stats['Avg_PF'],
        'Avg_Opp_WinPct_diff': team1_stats['Avg_Opp_WinPct'] - team2_stats['Avg_Opp_WinPct'],
        'Last30_WinRatio_diff': team1_stats['Last30_WinRatio'] - team2_stats['Last30_WinRatio'],
        'Seed_diff': team1_seed - team2_seed
    }
    
    # Always include tempo features if include_tempo is True.
    if include_tempo:
        poss1 = team1_stats.get('Poss', 0)
        poss2 = team2_stats.get('Poss', 0)
        raw_avg = (poss1 + poss2) / 2
        raw_diff = abs(poss1 - poss2)
        
        tempo_features = {
            'Poss_diff': poss1 - poss2,
            'AvgTempo_scaled': raw_avg * 0.2,
            'TempoDiff_scaled': raw_diff * 0.2
        }
        base_features.update(tempo_features)
    
    # Define the order of base features (tempo features appended if included)
    base_feature_order = [
        'WinPct_diff',
        'Avg_Score_diff',
        'Avg_FGM_diff',
        'Avg_FGA_diff',
        'Avg_FGM3_diff',
        'Avg_FGA3_diff',
        'Avg_FTM_diff',
        'Avg_FTA_diff',
        'Avg_OR_diff',
        'Avg_DR_diff',
        'Avg_Ast_diff',
        'Avg_TO_diff',
        'Avg_Stl_diff',
        'Avg_Blk_diff',
        'Avg_PF_diff',
        'Avg_Opp_WinPct_diff',
        'Last30_WinRatio_diff',
        'Seed_diff'
    ]
    
    tempo_feature_order = ['Poss_diff', 'AvgTempo_scaled', 'TempoDiff_scaled']
    feature_order = base_feature_order + (tempo_feature_order if include_tempo else [])
    
    # Create DataFrame with only the features that exist in our feature dictionary, preserving order
    available_features = [f for f in feature_order if f in base_features]
    return pd.DataFrame([base_features])[available_features]


def predict_matchup(model, team1_stats, team2_stats, team1_seed, team2_seed, include_tempo=True, model_version=None):
    """
    Predict the outcome of a matchup between two teams using the trained model.
    
    Parameters:
    - model: The trained model to use for prediction
    - team1_stats, team2_stats: Stats for each team (as a pd.Series)
    - team1_seed, team2_seed: Numeric seeds for each team
    - include_tempo: Whether to include tempo features
    - model_version: String identifier for the model version (for logging)
    
    Returns:
    - Probability that team1 wins
    """
    # If team1's seed is worse than team2's, swap them to ensure better-seeded team is first.
    swap_needed = team1_seed > team2_seed
    if swap_needed:
        team1_stats, team2_stats = team2_stats, team1_stats
        team1_seed, team2_seed = team2_seed, team1_seed
    
    # Create features for the matchup
    X = create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed, include_tempo)
    
    # Get prediction from model using model.predict on a DMatrix
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    
    prob_first_wins = preds[0]
    return 1 - prob_first_wins if swap_needed else prob_first_wins



# ------------------------------
# Main Simulation Function
# ------------------------------

def simulate_first_round_model(bracket_path='bracket.csv',
                               team_stats_path='../scripts/TeamSeasonAverages_with_SoS.csv',
                               model_path='../models/xgb_model_basic.pkl'):
    """
    Simulates the first round of a tournament using your trained model.
    
    Parameters:
    - bracket_path: Path to CSV with bracket information (with columns: Seed, Team, Region)
    - team_stats_path: Path to basic stats csv
    - model_path: Path to the trained model
    
    Returns:
    - DataFrame with matchup details and predicted win probabilities
    """
    model_version = model_path.split('/')[-1]
    include_tempo = 'with_tempo' in model_version  # for xgb_model_basic.pkl, this should be False
    
    # Load the trained model
    model = joblib.load(model_path)
    print(f"Successfully loaded model from {model_path}")
    print(f"Using {'tempo-enhanced' if include_tempo else 'standard'} features")
    
    # Load team stats for latest season
    team_stats = pd.read_csv(team_stats_path)
    latest_season = team_stats['Season'].max()
    team_stats = team_stats[team_stats['Season'] == latest_season].copy()
    
    # (KenPom merging removed since our basic model doesn't use KenPom rankings.)
    
    # Clean team names
    team_stats['TeamName'] = team_stats['TeamName'].apply(lambda x: x.strip())

    # Load bracket file
    bracket = pd.read_csv(bracket_path)
    bracket['Seed'] = pd.to_numeric(bracket['Seed'])
    
    results = []
    for region, group in bracket.groupby('Region'):
        matchup_pairs = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
        seed_map = {row['Seed']: row for _, row in group.iterrows()}
        for seed_a, seed_b in matchup_pairs:
            if seed_a in seed_map and seed_b in seed_map:
                team_a_bracket = seed_map[seed_a]
                team_b_bracket = seed_map[seed_b]
                team_a_stats = team_stats[team_stats['TeamName'] == team_a_bracket['Team']]
                team_b_stats = team_stats[team_stats['TeamName'] == team_b_bracket['Team']]
                if team_a_stats.empty or team_b_stats.empty:
                    print(f"Warning: Stats not found for matchup {team_a_bracket['Team']} vs {team_b_bracket['Team']}")
                    continue
                team_a_stats = team_a_stats.iloc[0]
                team_b_stats = team_b_stats.iloc[0]
                
                prob = predict_matchup(
                    model, 
                    team_a_stats, 
                    team_b_stats, 
                    team_a_bracket['Seed'], 
                    team_b_bracket['Seed'],
                    include_tempo,
                    model_version
                )
                
                tempo_info = ""
                if include_tempo and 'Poss' in team_a_stats and 'Poss' in team_b_stats:
                    avg_tempo = (team_a_stats['Poss'] + team_b_stats['Poss']) / 2
                    tempo_diff = abs(team_a_stats['Poss'] - team_b_stats['Poss'])
                    tempo_info = f" | Avg Tempo: {avg_tempo:.1f}, Diff: {tempo_diff:.1f}"
                
                results.append({
                    'Region': region,
                    'Matchup': f"{team_a_bracket['Team']} (Seed {seed_a}) vs {team_b_bracket['Team']} (Seed {seed_b}){tempo_info}",
                    'TeamA': team_a_bracket['Team'],
                    'TeamB': team_b_bracket['Team'],
                    'SeedA': seed_a,
                    'SeedB': seed_b,
                    'TeamA_win_prob': prob,
                    'TeamB_win_prob': 1 - prob,
                    'Predicted_Winner': team_a_bracket['Team'] if prob > 0.5 else team_b_bracket['Team'],
                    'Upset': "YES" if (prob > 0.5 and seed_a > seed_b) or (prob < 0.5 and seed_a < seed_b) else "NO"
                })
    
    results_df = pd.DataFrame(results).sort_values(['Region', 'SeedA'])
    return results_df

# ------------------------------
# Compare Models (Test Harness)
# ------------------------------

def compare_models(bracket_path='bracket.csv',
                   team_stats_path='../scripts/TeamSeasonAverages_with_SoS.csv'):
    """
    Compare predictions from the basic XGBoost model on the same bracket.
    """
    models = [
        '../models/xgb_model_basic.pkl'
    ]
    
    all_results = {}
    for model_path in models:
        try:
            print(f"\n{'='*60}")
            print(f"SIMULATING WITH MODEL: {model_path}")
            print(f"{'='*60}")
            
            results = simulate_first_round_model(
                bracket_path=bracket_path,
                team_stats_path=team_stats_path,
                model_path=model_path
            )
            
            model_name = model_path.split('/')[-1].replace('.pkl', '')
            all_results[model_name] = results
            
            print("\nPredicted First Round Results:")
            for _, row in results.iterrows():
                winner = "UPSET! " if row['Upset'] == "YES" else ""
                print(f"{row['Region']}: {row['TeamA']} ({row['SeedA']}) vs {row['TeamB']} ({row['SeedB']}) → {winner}{row['Predicted_Winner']} wins ({row['TeamA_win_prob']*100:.1f}%)")
            
            upset_count = results[results['Upset'] == "YES"].shape[0]
            print(f"\nTotal predicted upsets: {upset_count}")
            
        except Exception as e:
            print(f"Error with model {model_path}: {str(e)}")
    return all_results

# ------------------------------
# Example Usage
# ------------------------------

if __name__ == "__main__":
    compare_models()
