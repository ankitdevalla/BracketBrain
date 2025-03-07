import pandas as pd
import numpy as np
import joblib
import re

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
    # Base features that all models use
    base_features = {
        'SeedDiff': team1_seed - team2_seed,
        'KenPomDiff': team1_stats['KenPom'] - team2_stats['KenPom'],
        'Diff_AdjO': team1_stats['AdjO'] - team2_stats['AdjO'],
        'Diff_AdjD': team1_stats['AdjD'] - team2_stats['AdjD'],
        'Diff_AdjNetRtg': team1_stats['AdjNetRtg'] - team2_stats['AdjNetRtg'],
        'Diff_SOS_NetRtg': team1_stats['SOS_NetRtg'] - team2_stats['SOS_NetRtg'],
        'Diff_Expected Win%': team1_stats['Expected Win%'] - team2_stats['Expected Win%'],
        'Diff_ThreePtRate': team1_stats['ThreePtRate'] - team2_stats['ThreePtRate'],
        'Diff_FTRate': team1_stats['FTRate'] - team2_stats['FTRate'],
        'Diff_AstRate': team1_stats['AstRate'] - team2_stats['AstRate'],
        'Diff_TORate': team1_stats['TORate'] - team2_stats['TORate'],
        'Diff_ORRate': team1_stats['ORRate'] - team2_stats['ORRate'],
        'Diff_DRRate': team1_stats['DRRate'] - team2_stats['DRRate'],
        'Diff_ScoreStdDev': team1_stats['ScoreStdDev'] - team2_stats['ScoreStdDev'],
        'Diff_MarginStdDev': team1_stats['MarginStdDev'] - team2_stats['MarginStdDev'],
        'Diff_ORtgStdDev': team1_stats['ORtgStdDev'] - team2_stats['ORtgStdDev'],
        'Diff_DRtgStdDev': team1_stats['DRtgStdDev'] - team2_stats['DRtgStdDev'],
        'Diff_HomeWin%': team1_stats['HomeWin%'] - team2_stats['HomeWin%'],
        'Diff_AwayWin%': team1_stats['AwayWin%'] - team2_stats['AwayWin%'],
        'Diff_NeutralWin%': team1_stats['NeutralWin%'] - team2_stats['NeutralWin%'],
        'Diff_Last10Win%': team1_stats['Last10Win%'] - team2_stats['Last10Win%']
    }
    
    # Always include tempo features if include_tempo is True.
    if include_tempo:
        # Use .get with default 0 in case 'Poss' is missing.
        poss1 = team1_stats.get('Poss', 0)
        poss2 = team2_stats.get('Poss', 0)
        raw_avg = (poss1 + poss2) / 2
        raw_diff = abs(poss1 - poss2)
        
        tempo_features = {
            'Diff_Poss': poss1 - poss2,
            'AvgTempo_scaled': raw_avg * 0.2,
            'TempoDiff_scaled': raw_diff * 0.2
        }
        base_features.update(tempo_features)
    
    # Define feature order based on whether tempo is included
    base_feature_order = [
        'SeedDiff', 'KenPomDiff', 'Diff_AdjO', 'Diff_AdjD', 'Diff_AdjNetRtg',
        'Diff_SOS_NetRtg', 'Diff_Expected Win%', 'Diff_ThreePtRate', 'Diff_FTRate',
        'Diff_AstRate', 'Diff_TORate', 'Diff_ORRate', 'Diff_DRRate',
        'Diff_ScoreStdDev', 'Diff_MarginStdDev', 'Diff_ORtgStdDev',
        'Diff_DRtgStdDev', 'Diff_HomeWin%', 'Diff_AwayWin%', 'Diff_NeutralWin%',
        'Diff_Last10Win%'
    ]
    
    tempo_feature_order = ['Diff_Poss', 'AvgTempo_scaled', 'TempoDiff_scaled']
    
    feature_order = base_feature_order + (tempo_feature_order if include_tempo else [])
    
    # Create DataFrame with only the features that exist in our feature dictionary
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
    
    # Get prediction from model
    if model_version and ('py2' in model_version or 'with_tempo2' in model_version):
        prob_first_wins = model.predict_proba(X)[0][1]
    else:
        prob_first_wins = 1 - model.predict_proba(X)[0][1]
    
    # If teams were swapped, flip the probability so it corresponds to the originally selected team1
    return 1 - prob_first_wins if swap_needed else prob_first_wins

# ------------------------------
# Main Simulation Function
# ------------------------------

def simulate_first_round_model(bracket_path='bracket.csv',
                               team_stats_path='../pre_tourney_data/EnhancedTournamentStats.csv',
                               kenpom_path='../pre_tourney_data/KenPom-Rankings-Updated.csv',
                               model_path='../models/final_model_with_tempo2.pkl'):
    """
    Simulates the first round of a tournament using your trained model.
    
    Parameters:
    - bracket_path: Path to CSV with bracket information (with columns: Seed, Team, Region)
    - team_stats_path: Path to enhanced team stats CSV
    - kenpom_path: Path to KenPom rankings CSV
    - model_path: Path to the trained model
    
    Returns:
    - DataFrame with matchup details and predicted win probabilities
    """
    model_version = model_path.split('/')[-1]
    include_tempo = 'with_tempo' in model_path
    
    # Load the trained model
    model = joblib.load(model_path)
    print(f"Successfully loaded model from {model_path}")
    print(f"Using {'tempo-enhanced' if include_tempo else 'standard'} features")
    
    # Load team stats for latest season
    team_stats = pd.read_csv(team_stats_path)
    latest_season = team_stats['Season'].max()
    team_stats = team_stats[team_stats['Season'] == latest_season].copy()
    
    # Load latest KenPom rankings and merge
    kenpom = pd.read_csv(kenpom_path)
    latest_season_kp = kenpom['Season'].max()
    kenpom = kenpom[kenpom['Season'] == latest_season_kp].copy()
    kenpom = kenpom.rename(columns={'OrdinalRank': 'KenPom'})
    team_stats = team_stats.merge(kenpom[['TeamID', 'KenPom']], on='TeamID', how='left')
    
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
# Compare Multiple Models
# ------------------------------

def compare_models(bracket_path='bracket.csv',
                   team_stats_path='../pre_tourney_data/EnhancedTournamentStats.csv',
                   kenpom_path='../pre_tourney_data/KenPom-Rankings-Updated.csv'):
    """
    Compare predictions from multiple models on the same bracket.
    """
    models = [
        '../models/final_model.pkl',
        '../models/final_model_py.pkl',
        '../models/final_model_py2.pkl',
        '../models/final_model_with_tempo2.pkl',
        '../models/modelv3.pkl'
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
                kenpom_path=kenpom_path,
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
