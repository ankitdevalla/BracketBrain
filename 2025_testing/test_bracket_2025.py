import pandas as pd
import numpy as np
import joblib
import re

# ------------------------------
# Utility Functions
# ------------------------------

def create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed):
    """
    Create a DataFrame of features for the matchup.
    This must produce the same set and order of features that your model expects.
    Here we assume the model was trained on differences of:
        'SeedDiff', 'KenPomDiff', 'Diff_AdjO', 'Diff_AdjD', 'Diff_AdjNetRtg',
        'Diff_SOS_NetRtg', 'Diff_Expected Win%', 'Diff_ThreePtRate', 'Diff_FTRate',
        'Diff_AstRate', 'Diff_TORate', 'Diff_ORRate', 'Diff_DRRate',
        'Diff_ScoreStdDev', 'Diff_MarginStdDev', 'Diff_ORtgStdDev',
        'Diff_DRtgStdDev', 'Diff_HomeWin%', 'Diff_AwayWin%', 'Diff_NeutralWin%',
        'Diff_Last10Win%'
    The features are computed as (team1 value minus team2 value).
    """
    features = {
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
    # Ensure the features are in the order expected by your model:
    feature_order = [
        'SeedDiff', 'KenPomDiff', 'Diff_AdjO', 'Diff_AdjD', 'Diff_AdjNetRtg',
        'Diff_SOS_NetRtg', 'Diff_Expected Win%', 'Diff_ThreePtRate', 'Diff_FTRate',
        'Diff_AstRate', 'Diff_TORate', 'Diff_ORRate', 'Diff_DRRate',
        'Diff_ScoreStdDev', 'Diff_MarginStdDev', 'Diff_ORtgStdDev',
        'Diff_DRtgStdDev', 'Diff_HomeWin%', 'Diff_AwayWin%', 'Diff_NeutralWin%',
        'Diff_Last10Win%'
    ]
    return pd.DataFrame([features], columns=feature_order)

def predict_matchup(model, team1_stats, team2_stats, team1_seed, team2_seed):
    """
    Predict the outcome of a matchup between two teams using the trained model.
    The model is trained to predict the probability that the first team wins.
    To ensure consistency, if team1's seed is worse (i.e. numerically higher) than team2's,
    we swap the order, and then flip the resulting probability.
    """
    swap_needed = team1_seed > team2_seed
    if swap_needed:
        team1_stats, team2_stats = team2_stats, team1_stats
        team1_seed, team2_seed = team2_seed, team1_seed
    X = create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed)
    prob_first_wins = model.predict_proba(X)[0][1]
    return 1 - prob_first_wins if swap_needed else prob_first_wins

# ------------------------------
# Main Simulation Function
# ------------------------------

def simulate_first_round_model(bracket_path='bracket.csv',
                               team_stats_path='../pre_tourney_data/EnhancedTournamentStats.csv',
                               kenpom_path='../pre_tourney_data/KenPom-Rankings-Updated.csv',
                               model_path='../scripts/final_model_py2.pkl'):
    """
    Simulates the first round of a tournament using your trained model.
    
    1. Loads the model.
    2. Loads team stats (Enhanced Tournament Stats) and KenPom rankings (latest season) and merges them.
    3. Loads the bracket CSV (assumed to have columns: Seed, Team, Region).
    4. For each region, pairs teams by standard seeding (1 vs 16, 2 vs 15, etc.) and predicts win probabilities.
    
    Returns a DataFrame with the matchup details and the predicted win probabilities.
    """
    # Load the trained model
    model = joblib.load(model_path)
    print(f"Successfully loaded model from {model_path}")
    
    # Load team stats and filter for the latest season
    team_stats = pd.read_csv(team_stats_path)
    latest_season = team_stats['Season'].max()
    team_stats = team_stats[team_stats['Season'] == latest_season].copy()
    
    # Load latest KenPom rankings and merge (rename OrdinalRank to KenPom)
    kenpom = pd.read_csv(kenpom_path)
    latest_season_kp = kenpom['Season'].max()
    kenpom = kenpom[kenpom['Season'] == latest_season_kp].copy()
    kenpom = kenpom.rename(columns={'OrdinalRank': 'KenPom'})
    
    team_stats = team_stats.merge(kenpom[['TeamID', 'KenPom']], on='TeamID', how='left')
    
    # Clean team names in team_stats (if needed)
    team_stats['TeamName'] = team_stats['TeamName'].apply(lambda x: x.strip())
    
    # Load the bracket file (assumed to have columns: Seed, Team, Region)
    bracket = pd.read_csv(bracket_path)
    bracket['Seed'] = pd.to_numeric(bracket['Seed'])
    
    results = []
    # Process each region separately
    for region, group in bracket.groupby('Region'):
        # For each matchup pair: (1 vs 16), (2 vs 15), ..., (8 vs 9)
        matchup_pairs = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
        # Create a mapping from seed to team info for this region
        seed_map = {row['Seed']: row for _, row in group.iterrows()}
        for seed_a, seed_b in matchup_pairs:
            if seed_a in seed_map and seed_b in seed_map:
                team_a_bracket = seed_map[seed_a]
                team_b_bracket = seed_map[seed_b]
                # Find the corresponding team stats by matching team names
                team_a_stats = team_stats[team_stats['TeamName'] == team_a_bracket['Team']]
                team_b_stats = team_stats[team_stats['TeamName'] == team_b_bracket['Team']]
                if team_a_stats.empty or team_b_stats.empty:
                    print(f"Warning: Stats not found for matchup {team_a_bracket['Team']} vs {team_b_bracket['Team']}")
                    continue
                # Use the first matching record (assuming unique team names)
                team_a_stats = team_a_stats.iloc[0]
                team_b_stats = team_b_stats.iloc[0]
                # Predict win probability for team A using our model
                prob = predict_matchup(model, team_a_stats, team_b_stats, team_a_bracket['Seed'], team_b_bracket['Seed'])
                results.append({
                    'Region': region,
                    'Matchup': f"{team_a_bracket['Team']} (Seed {seed_a}) vs {team_b_bracket['Team']} (Seed {seed_b})",
                    'TeamA_win_prob': prob
                })
    
    return pd.DataFrame(results)

# ------------------------------
# Example Usage
# ------------------------------

if __name__ == "__main__":
    results_df = simulate_first_round_model()
    print(results_df)
