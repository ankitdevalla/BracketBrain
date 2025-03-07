import pandas as pd
import numpy as np
import joblib
import argparse

def load_model(model_path='scripts/modelv3.pkl'):
    """Load the trained model."""
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def load_team_stats(stats_path='pre_tourney_data/EnhancedTournamentStats.csv'):
    """Load the team statistics."""
    try:
        stats = pd.read_csv(stats_path)
        # Get the most recent season data
        latest_season = stats['Season'].max()
        current_stats = stats[stats['Season'] == latest_season]
        print(f"Loaded stats for season {latest_season} with {len(current_stats)} teams")
        return current_stats
    except Exception as e:
        print(f"Error loading team stats: {str(e)}")
        return None

def load_kenpom_rankings(kenpom_path='pre_tourney_data/KenPom-Rankings-Updated.csv'):
    """Load KenPom rankings."""
    try:
        kenpom = pd.read_csv(kenpom_path)
        # Get the most recent season data
        latest_season = kenpom['Season'].max()
        current_kenpom = kenpom[kenpom['Season'] == latest_season]
        current_kenpom = current_kenpom.rename(columns={'OrdinalRank': 'KenPom'})
        print(f"Loaded KenPom rankings for season {latest_season}")
        return current_kenpom
    except Exception as e:
        print(f"Error loading KenPom rankings: {str(e)}")
        return None

def create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed):
    """Create features for prediction."""
    # Define the feature order as expected by the model
    feature_order = [
        'SeedDiff', 'KenPomDiff', 'Diff_AdjO', 'Diff_AdjD', 'Diff_AdjNetRtg',
        'Diff_SOS_NetRtg', 'Diff_Expected Win%', 'Diff_ThreePtRate', 'Diff_FTRate',
        'Diff_AstRate', 'Diff_TORate', 'Diff_ORRate', 'Diff_DRRate',
        'Diff_ScoreStdDev', 'Diff_MarginStdDev', 'Diff_ORtgStdDev',
        'Diff_DRtgStdDev', 'Diff_HomeWin%', 'Diff_AwayWin%', 'Diff_NeutralWin%',
        'Diff_Last10Win%'
    ]
    
    # Calculate all differences consistently as (team1 - team2)
    diffs = {
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
        'Diff_Last10Win%': team1_stats['Last10Win%'] - team2_stats['Last10Win%'],
        # Add tempo features
        # 'Diff_Poss': team1_stats['Poss'] - team2_stats['Poss'],
        # 'AvgTempo': (team1_stats['Poss'] + team2_stats['Poss']) / 2,
        # 'TempoDiff': abs(team1_stats['Poss'] - team2_stats['Poss'])
    }
    
    # Create DataFrame with features in the correct order
    # Only include features that exist in the feature_order list
    available_features = [f for f in feature_order if f in diffs]
    return pd.DataFrame([diffs])[available_features]

def predict_matchup(model, team1_stats, team2_stats, team1_seed, team2_seed):
    """
    Predict the outcome of a matchup between two teams.
    
    We want the model to always work with the better seeded team first. So if team1 is worse seeded 
    (has a higher seed number) than team2, we swap their order. The model is trained to output the 
    probability that the first team wins.
    
    If a swap occurs, we flip the prediction so that the returned probability corresponds to the 
    original team1.
    """
    # Determine if we need to swap teams (better seed is the lower number)
    swap_needed = team1_seed > team2_seed
    if swap_needed:
        team1_stats, team2_stats = team2_stats, team1_stats
        team1_seed, team2_seed = team2_seed, team1_seed

    # Create features with the consistent ordering (better seeded team first)
    X = create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed)
    
    # Model outputs the probability that the first team wins
    prob_first_wins = model.predict_proba(X)[0][1]
    
    # If we swapped teams, flip the probability so that it corresponds to the originally provided team1
    if swap_needed:
        return 1 - prob_first_wins
    else:
        return prob_first_wins


def main():
    parser = argparse.ArgumentParser(description='Predict NCAA basketball matchups')
    parser.add_argument('--team1', type=str, required=True, help='Name of first team')
    parser.add_argument('--seed1', type=int, required=True, help='Seed of first team (1-16)')
    parser.add_argument('--team2', type=str, required=True, help='Name of second team')
    parser.add_argument('--seed2', type=int, required=True, help='Seed of second team (1-16)')
    parser.add_argument('--model', type=str, default='../scripts/final_model_py2.pkl', 
                        help='Path to model file')
    parser.add_argument('--stats', type=str, default='../pre_tourney_data/EnhancedTournamentStats.csv',
                        help='Path to team stats file')
    parser.add_argument('--kenpom', type=str, default='../pre_tourney_data/KenPom-Rankings-Updated.csv',
                        help='Path to KenPom rankings file')
    
    args = parser.parse_args()
    
    # Load model and data
    model = load_model(args.model)
    if model is None:
        return
    
    team_stats = load_team_stats(args.stats)
    if team_stats is None:
        return
    
    kenpom = load_kenpom_rankings(args.kenpom)
    if kenpom is not None:
        # Merge KenPom rankings with team stats
        team_stats = team_stats.merge(
            kenpom[['TeamID', 'KenPom']],
            on='TeamID',
            how='left'
        )
    
    # Find the teams in the dataset
    team1_data = team_stats[team_stats['TeamName'] == args.team1]
    team2_data = team_stats[team_stats['TeamName'] == args.team2]
    
    if len(team1_data) == 0:
        print(f"Team '{args.team1}' not found in the dataset.")
        print("Available teams:")
        print(team_stats['TeamName'].sort_values().to_string(index=False))
        return
    
    if len(team2_data) == 0:
        print(f"Team '{args.team2}' not found in the dataset.")
        print("Available teams:")
        print(team_stats['TeamName'].sort_values().to_string(index=False))
        return
    
    # Get team stats
    team1_stats = team1_data.iloc[0]
    team2_stats = team2_data.iloc[0]
    
    # Make predictions in both directions
    prob_1_beats_2 = predict_matchup(model, team1_stats, team2_stats, args.seed1, args.seed2)
    prob_2_beats_1 = predict_matchup(model, team2_stats, team1_stats, args.seed2, args.seed1)
    
    # Print results
    print("\n" + "="*60)
    print(f"MATCHUP: {args.team1} (#{args.seed1}) vs {args.team2} (#{args.seed2})")
    print("="*60)
    
    print(f"\n{args.team1} (#{args.seed1}) win probability: {prob_1_beats_2:.4f} ({prob_1_beats_2*100:.1f}%)")
    print(f"{args.team2} (#{args.seed2}) win probability: {1-prob_1_beats_2:.4f} ({(1-prob_1_beats_2)*100:.1f}%)")
    
    print("\nReverse matchup check (should be complementary):")
    print(f"{args.team2} (#{args.seed2}) win probability: {prob_2_beats_1:.4f} ({prob_2_beats_1*100:.1f}%)")
    print(f"{args.team1} (#{args.seed1}) win probability: {1-prob_2_beats_1:.4f} ({(1-prob_2_beats_1)*100:.1f}%)")
    
    # Check for consistency
    diff = abs(prob_1_beats_2 - (1 - prob_2_beats_1))
    if diff < 0.0001:
        print("\n✅ CONSISTENT: Predictions are properly complementary")
    else:
        print(f"\n❌ INCONSISTENT: Predictions differ by {diff:.6f}")
    
    # Print key stats comparison
    print("\nKEY STATS COMPARISON:")
    print("-"*60)
    stats_to_compare = [
        ('KenPom', 'KenPom Ranking', True),  # True means lower is better
        ('AdjO', 'Adjusted Offensive Rating', False),
        ('AdjD', 'Adjusted Defensive Rating', True),
        ('AdjNetRtg', 'Adjusted Net Rating', False),
        ('Expected Win%', 'Expected Win %', False),
        ('SOS_NetRtg', 'Strength of Schedule', False)
        # ('Poss', 'Tempo (Possessions/40 min)', None)  # None means neither better nor worse
    ]
    
    for stat_col, stat_name, lower_is_better in stats_to_compare:
        team1_val = team1_stats[stat_col]
        team2_val = team2_stats[stat_col]
        
        if lower_is_better is True:
            better_team = args.team1 if team1_val < team2_val else args.team2
            advantage = "✓" if team1_val < team2_val else " "
            advantage2 = " " if team1_val < team2_val else "✓"
        elif lower_is_better is False:
            better_team = args.team1 if team1_val > team2_val else args.team2
            advantage = "✓" if team1_val > team2_val else " "
            advantage2 = " " if team1_val > team2_val else "✓"
        else:
            better_team = None
            advantage = ""
            advantage2 = ""
        
        print(f"{stat_name:25} {args.team1:20} {team1_val:7.2f} {advantage}   {args.team2:20} {team2_val:7.2f} {advantage2}")
    
    print("-"*60)
    
if __name__ == "__main__":
    main() 