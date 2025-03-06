"""
Script to analyze why the model predicts one team over another.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Analyzing Texas vs Florida prediction...")
    
    # Load the model
    model = joblib.load('scripts/final_model_with_tempo.pkl')
    
    # Load feature names
    feature_names = np.load('scripts/feature_names_with_tempo.npy', allow_pickle=True)
    
    # Load enhanced stats
    enhanced_stats = pd.read_csv('pre_tourney_data/EnhancedTournamentStats.csv')
    latest_season = enhanced_stats['Season'].max()
    current_stats = enhanced_stats[enhanced_stats['Season'] == latest_season]
    
    # Load KenPom rankings
    kenpom_rankings = pd.read_csv('pre_tourney_data/KenPom-Rankings-Updated.csv')
    kenpom_rankings = kenpom_rankings[kenpom_rankings['Season'] == 2025]
    kenpom_rankings = kenpom_rankings.rename(columns={'OrdinalRank': 'KenPom'})
    
    # Merge KenPom rankings with current stats
    current_stats = current_stats.merge(
        kenpom_rankings[['TeamID', 'KenPom']],
        on='TeamID',
        how='left'
    )
    
    # Get Texas and Florida stats
    texas_stats = current_stats[current_stats['TeamName'] == 'Texas'].iloc[0]
    florida_stats = current_stats[current_stats['TeamName'] == 'Florida'].iloc[0]
    
    # Assume Texas is seed 6 and Florida is seed 3 (for example)
    texas_seed = 6
    florida_seed = 3
    
    # Create features for prediction
    features = create_matchup_features(texas_stats, florida_stats, texas_seed, florida_seed, feature_names)
    
    # Make prediction
    win_probability = model.predict_proba(features)[0][1]
    
    print(f"Prediction: Texas has a {win_probability*100:.1f}% chance to win against Florida")
    
    # Get feature importances
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            # Create DataFrame with feature names, values, and importances
            feature_analysis = pd.DataFrame({
                'Feature': feature_names,
                'Value': features.values[0],
                'Importance': importances,
                'Weighted_Impact': features.values[0] * importances
            })
            
            # Sort by absolute weighted impact
            feature_analysis['Abs_Weighted_Impact'] = feature_analysis['Weighted_Impact'].abs()
            feature_analysis = feature_analysis.sort_values('Abs_Weighted_Impact', ascending=False)
            
            print("\nTop Features Influencing Prediction:")
            print(feature_analysis[['Feature', 'Value', 'Importance', 'Weighted_Impact']].head(10))
            
            # Plot feature impacts
            plt.figure(figsize=(12, 8))
            colors = ['red' if x < 0 else 'green' for x in feature_analysis['Weighted_Impact'].head(10)]
            sns.barplot(x='Weighted_Impact', y='Feature', data=feature_analysis.head(10), palette=colors)
            plt.title('Top 10 Features Influencing Texas vs Florida Prediction')
            plt.tight_layout()
            plt.savefig('output/texas_florida_analysis.png')
            print("Analysis plot saved to 'output/texas_florida_analysis.png'")

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
