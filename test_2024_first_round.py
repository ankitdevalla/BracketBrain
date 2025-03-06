"""
Script to test the model on the first round of the 2024 NCAA tournament.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

def main():
    print("Testing model on 2024 NCAA Tournament First Round...")
    
    # Load the model
    model = joblib.load('scripts/final_model_with_tempo.pkl')
    
    # Load feature names
    feature_names = np.load('scripts/feature_names_with_tempo.npy', allow_pickle=True)
    
    # Load tournament results
    tourney_results = pd.read_csv('raw_data/MNCAATourneyDetailedResults.csv')
    
    # Load tournament seeds
    tourney_seeds = pd.read_csv('raw_data/MNCAATourneySeeds.csv')
    
    # Convert seed strings to numeric values
    tourney_seeds['SeedValue'] = tourney_seeds['Seed'].str[1:3].astype(int)
    
    # Load KenPom rankings
    kenpom = pd.read_csv('pre_tourney_data/KenPom-Rankings-Updated.csv')
    
    # Load enhanced stats
    enhanced_stats = pd.read_csv('pre_tourney_data/EnhancedTournamentStats.csv')
    
    # Filter for 2024 tournament data
    results_2024 = tourney_results[tourney_results['Season'] == 2024]
    seeds_2024 = tourney_seeds[tourney_seeds['Season'] == 2024]
    kenpom_2024 = kenpom[kenpom['Season'] == 2024]
    
    # Get the first round games (DayNum values for first round)
    # First round games typically have the lowest DayNum values in the tournament
    first_round_day = results_2024['DayNum'] < 136
    first_round_games = results_2024[results_2024['DayNum'] == first_round_day]
    
    print(f"Found {len(first_round_games)} first round games from 2024 tournament")
    
    # Create a list to store prediction results
    predictions = []
    
    # Process each first round game
    for _, game in first_round_games.iterrows():
        # Get team IDs
        team1_id = game['WTeamID']  # Winner
        team2_id = game['LTeamID']  # Loser
        
        # Get team names
        team1_name = get_team_name(team1_id)
        team2_name = get_team_name(team2_id)
        
        # Get seeds
        team1_seed = seeds_2024[seeds_2024['TeamID'] == team1_id]['SeedValue'].values[0]
        team2_seed = seeds_2024[seeds_2024['TeamID'] == team2_id]['SeedValue'].values[0]
        
        # Get stats for the season
        team1_stats = get_team_stats(team1_id, 2024, enhanced_stats, kenpom_2024)
        team2_stats = get_team_stats(team2_id, 2024, enhanced_stats, kenpom_2024)
        
        if team1_stats is None or team2_stats is None:
            print(f"Missing stats for game between {team1_name} and {team2_name}, skipping")
            continue
        
        # Create features for both possible orderings
        X_t1_t2 = create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed, feature_names)
        X_t2_t1 = create_matchup_features(team2_stats, team1_stats, team2_seed, team1_seed, feature_names)
        
        # Make predictions
        prob_t1_wins = model.predict_proba(X_t1_t2)[0][1]
        prob_t2_wins = model.predict_proba(X_t2_t1)[0][1]
        
        # Store prediction results
        predictions.append({
            'Team1': team1_name,
            'Team2': team2_name,
            'Team1_Seed': team1_seed,
            'Team2_Seed': team2_seed,
            'Team1_Prob': prob_t1_wins,
            'Team2_Prob': prob_t2_wins,
            'Predicted_Winner': team1_name if prob_t1_wins > 0.5 else team2_name,
            'Actual_Winner': team1_name,  # Since team1 is the actual winner in our data
            'Correct_Prediction': prob_t1_wins > 0.5,  # True if model correctly predicted team1 to win
            'Upset': team1_seed > team2_seed  # True if the higher seed (worse team) won
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(predictions)
    
    # Calculate overall accuracy
    accuracy = results_df['Correct_Prediction'].mean()
    print(f"\nModel Accuracy on 2024 First Round: {accuracy:.4f}")
    
    # Calculate accuracy on upsets vs. non-upsets
    upset_accuracy = results_df[results_df['Upset']]['Correct_Prediction'].mean() if len(results_df[results_df['Upset']]) > 0 else 0
    non_upset_accuracy = results_df[~results_df['Upset']]['Correct_Prediction'].mean() if len(results_df[~results_df['Upset']]) > 0 else 0
    
    print(f"Accuracy on Upsets: {upset_accuracy:.4f} ({len(results_df[results_df['Upset']])} games)")
    print(f"Accuracy on Non-Upsets: {non_upset_accuracy:.4f} ({len(results_df[~results_df['Upset']])} games)")
    
    # Print detailed results
    print("\nDetailed Prediction Results:")
    print(results_df[['Team1', 'Team1_Seed', 'Team2', 'Team2_Seed', 'Team1_Prob', 'Predicted_Winner', 'Actual_Winner', 'Correct_Prediction', 'Upset']])
    
    # Save results to CSV
    results_df.to_csv('output/2024_first_round_predictions.csv', index=False)
    print("\nResults saved to 'output/2024_first_round_predictions.csv'")
    
    # Create a visualization of the predictions
    plt.figure(figsize=(12, 8))
    
    # Sort by seed matchup
    results_df['Matchup'] = results_df.apply(lambda x: f"{x['Team1_Seed']} {x['Team1']} vs {x['Team2_Seed']} {x['Team2']}", axis=1)
    results_df = results_df.sort_values(by=['Team1_Seed', 'Team2_Seed'])
    
    # Create the plot
    ax = sns.barplot(x='Team1_Prob', y='Matchup', data=results_df, 
                    palette=['green' if x else 'red' for x in results_df['Correct_Prediction']])
    
    # Add a vertical line at 0.5 probability
    plt.axvline(x=0.5, color='black', linestyle='--')
    
    # Add labels
    plt.xlabel('Probability of Team 1 Winning')
    plt.ylabel('Matchup')
    plt.title('2024 NCAA Tournament First Round Predictions')
    
    # Add text annotations
    for i, row in enumerate(results_df.itertuples()):
        correct = "✓" if row.Correct_Prediction else "✗"
        plt.text(0.05, i, correct, fontsize=14, 
                color='white' if row.Team1_Prob < 0.2 else 'black')
    
    plt.tight_layout()
    plt.savefig('output/2024_first_round_predictions.png')
    print("Visualization saved to 'output/2024_first_round_predictions.png'")

def get_team_name(team_id):
    """Get team name from team ID."""
    teams_df = pd.read_csv("raw_data/MTeams.csv")
    team_name = teams_df[teams_df['TeamID'] == team_id]['TeamName'].values[0]
    return team_name

def get_team_stats(team_id, season, enhanced_stats, kenpom):
    """Get team stats for a specific season."""
    # Get enhanced stats
    team_enhanced = enhanced_stats[(enhanced_stats['Season'] == season) & 
                                 (enhanced_stats['TeamID'] == team_id)]
    
    if len(team_enhanced) == 0:
        return None
    
    team_stats = team_enhanced.iloc[0].copy()
    
    # Add KenPom ranking
    team_kenpom = kenpom[(kenpom['Season'] == season) & 
                        (kenpom['TeamID'] == team_id)]
    
    if len(team_kenpom) > 0:
        team_stats['KenPom'] = team_kenpom['OrdinalRank'].values[0]
    else:
        team_stats['KenPom'] = 400  # Default high value if not found
    
    return team_stats

def create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed, feature_names):
    """Create features for prediction."""
    # Calculate differences
    diffs = {
        'SeedDiff': team1_seed - team2_seed,
        'KenPomDiff': team2_stats['KenPom'] - team1_stats['KenPom'],  # Reversed for KenPom since lower is better
        'Diff_AdjO': team1_stats['AdjO'] - team2_stats['AdjO'],
        'Diff_AdjD': team2_stats['AdjD'] - team1_stats['AdjD'],  # Reversed for defensive rating (lower is better)
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
        'Diff_Poss': team1_stats['Poss'] - team2_stats['Poss'],
        'AvgTempo': (team1_stats['Poss'] + team2_stats['Poss']) / 2,
        'TempoDiff': abs(team1_stats['Poss'] - team2_stats['Poss'])
    }
    
    # Create DataFrame with features in the correct order
    # Only include features that exist in the feature_names list
    available_features = [f for f in feature_names if f in diffs]
    return pd.DataFrame([diffs])[available_features]

if __name__ == "__main__":
    main()
