import pandas as pd
import numpy as np
import joblib
import re
import xgboost as xgb
import os

# ------------------------------
# Utility Functions
# ------------------------------

def create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed, 
                           team1_kenpom=None, team2_kenpom=None, 
                           team1_barthag=None, team2_barthag=None,
                           team1_exp=None, team2_exp=None,
                           model_type='basic'):
    """
    Create a DataFrame of features for the matchup with exact feature order.
    """
    # Calculate all the features first
    # Base features
    features = {}
    features['WinPct_diff'] = team1_stats['WinPct'] - team2_stats['WinPct']
    features['Avg_Score_diff'] = team1_stats['Avg_Score'] - team2_stats['Avg_Score']
    features['Avg_FGM_diff'] = team1_stats['Avg_FGM'] - team2_stats['Avg_FGM']
    features['Avg_FGA_diff'] = team1_stats['Avg_FGA'] - team2_stats['Avg_FGA']
    features['Avg_FGM3_diff'] = team1_stats['Avg_FGM3'] - team2_stats['Avg_FGM3']
    features['Avg_FGA3_diff'] = team1_stats['Avg_FGA3'] - team2_stats['Avg_FGA3']
    features['Avg_FTM_diff'] = team1_stats['Avg_FTM'] - team2_stats['Avg_FTM']
    features['Avg_FTA_diff'] = team1_stats['Avg_FTA'] - team2_stats['Avg_FTA']
    features['Avg_OR_diff'] = team1_stats['Avg_OR'] - team2_stats['Avg_OR']
    features['Avg_DR_diff'] = team1_stats['Avg_DR'] - team2_stats['Avg_DR']
    features['Avg_Ast_diff'] = team1_stats['Avg_Ast'] - team2_stats['Avg_Ast']
    features['Avg_TO_diff'] = team1_stats['Avg_TO'] - team2_stats['Avg_TO']
    features['Avg_Stl_diff'] = team1_stats['Avg_Stl'] - team2_stats['Avg_Stl']
    features['Avg_Blk_diff'] = team1_stats['Avg_Blk'] - team2_stats['Avg_Blk']
    features['Avg_PF_diff'] = team1_stats['Avg_PF'] - team2_stats['Avg_PF']
    
    # If using advanced model, add those features
    if model_type == 'advanced':
        # Get possession data
        poss1 = team1_stats.get('Avg_Poss', 0)
        poss2 = team2_stats.get('Avg_Poss', 0)
        features['Avg_Poss_diff'] = poss1 - poss2
    
    # Continue with base features
    features['Avg_Opp_WinPct_diff'] = team1_stats['Avg_Opp_WinPct'] - team2_stats['Avg_Opp_WinPct']
    features['Last30_WinRatio_diff'] = team1_stats['Last30_WinRatio'] - team2_stats['Last30_WinRatio']
    features['Seed_diff'] = team1_seed - team2_seed
    
    # If using advanced model, add remaining advanced features
    if model_type == 'advanced':
        # Calculate shooting percentages
        fg_pct1 = team1_stats['Avg_FGM'] / team1_stats['Avg_FGA'] if team1_stats['Avg_FGA'] > 0 else 0
        fg_pct2 = team2_stats['Avg_FGM'] / team2_stats['Avg_FGA'] if team2_stats['Avg_FGA'] > 0 else 0
        
        fg3_pct1 = team1_stats['Avg_FGM3'] / team1_stats['Avg_FGA3'] if team1_stats['Avg_FGA3'] > 0 else 0
        fg3_pct2 = team2_stats['Avg_FGM3'] / team2_stats['Avg_FGA3'] if team2_stats['Avg_FGA3'] > 0 else 0
        
        ft_pct1 = team1_stats['Avg_FTM'] / team1_stats['Avg_FTA'] if team1_stats['Avg_FTA'] > 0 else 0
        ft_pct2 = team2_stats['Avg_FTM'] / team2_stats['Avg_FTA'] if team2_stats['Avg_FTA'] > 0 else 0
        
        # Calculate offensive efficiency
        off_eff1 = (team1_stats['Avg_Score'] / poss1 * 100) if poss1 > 0 else 0
        off_eff2 = (team2_stats['Avg_Score'] / poss2 * 100) if poss2 > 0 else 0
        
        # Calculate turnover rate
        to_rate1 = (team1_stats['Avg_TO'] / poss1 * 100) if poss1 > 0 else 0
        to_rate2 = (team2_stats['Avg_TO'] / poss2 * 100) if poss2 > 0 else 0
        
        # Calculate rebound rate
        total_reb1 = team1_stats['Avg_OR'] + team1_stats['Avg_DR']
        total_reb2 = team2_stats['Avg_OR'] + team2_stats['Avg_DR']
        missed_shots1 = team1_stats['Avg_FGA'] - team1_stats['Avg_FGM']
        missed_shots2 = team2_stats['Avg_FGA'] - team2_stats['Avg_FGM']
        reb_opps1 = missed_shots1 + missed_shots2
        reb_opps2 = missed_shots1 + missed_shots2
        reb_rate1 = (total_reb1 / reb_opps1) if reb_opps1 > 0 else 0
        reb_rate2 = (total_reb2 / reb_opps2) if reb_opps2 > 0 else 0
        
        # Get KenPom rankings
        kenpom_diff = 0
        if team1_kenpom is not None and team2_kenpom is not None:
            kenpom_diff = team2_kenpom - team1_kenpom  # Lower rank is better
        
        # Add advanced features in the EXACT order from the error message
        features['KenPom_diff'] = kenpom_diff
        features['Off_Efficiency_diff'] = off_eff1 - off_eff2
        features['FG_Pct_diff'] = fg_pct1 - fg_pct2
        features['3P_Pct_diff'] = fg3_pct1 - fg3_pct2
        features['FT_Pct_diff'] = ft_pct1 - ft_pct2
        features['TO_Rate_diff'] = to_rate1 - to_rate2
        features['Reb_Rate_diff'] = reb_rate1 - reb_rate2
        features['Team1_Off_Efficiency'] = off_eff1
        features['Team2_Off_Efficiency'] = off_eff2
        features['Team1_FG_Pct'] = fg_pct1
        features['Team2_FG_Pct'] = fg_pct2
        features['Team1_3P_Pct'] = fg3_pct1
        features['Team2_3P_Pct'] = fg3_pct2
        features['Team1_FT_Pct'] = ft_pct1
        features['Team2_FT_Pct'] = ft_pct2
        features['Team1_TO_Rate'] = to_rate1
        features['Team2_TO_Rate'] = to_rate2
        features['Team1_Reb_Rate'] = reb_rate1
        features['Team2_Reb_Rate'] = reb_rate2
    
    # If using bart_simplified model, add Barthag and Exp features
    elif model_type == 'bart_simplified':
        # Add Barthag and Exp features
        # features['Barthag_diff'] = team1_barthag - team2_barthag
        features['Exp_diff'] = team1_exp - team2_exp
        
        # Add individual team features
        features['Team1_Barthag'] = team1_barthag
        features['Team2_Barthag'] = team2_barthag
        features['Team1_Exp'] = team1_exp
        features['Team2_Exp'] = team2_exp
        features['Team1_Seed'] = team1_seed
        features['Team2_Seed'] = team2_seed
    
    # Create DataFrame with features in the exact order expected by the model
    if model_type == 'basic':
        # Order for basic model
        feature_order = [
            'WinPct_diff', 'Avg_Score_diff', 'Avg_FGM_diff', 'Avg_FGA_diff', 
            'Avg_FGM3_diff', 'Avg_FGA3_diff', 'Avg_FTM_diff', 'Avg_FTA_diff', 
            'Avg_OR_diff', 'Avg_DR_diff', 'Avg_Ast_diff', 'Avg_TO_diff', 
            'Avg_Stl_diff', 'Avg_Blk_diff', 'Avg_PF_diff', 'Avg_Opp_WinPct_diff', 
            'Last30_WinRatio_diff', 'Seed_diff'
        ]
    elif model_type == 'advanced':
        # Order for advanced model - EXACTLY as shown in the first list in the error message
        feature_order = [
            'WinPct_diff', 'Avg_Score_diff', 'Avg_FGM_diff', 'Avg_FGA_diff', 
            'Avg_FGM3_diff', 'Avg_FGA3_diff', 'Avg_FTM_diff', 'Avg_FTA_diff', 
            'Avg_OR_diff', 'Avg_DR_diff', 'Avg_Ast_diff', 'Avg_TO_diff', 
            'Avg_Stl_diff', 'Avg_Blk_diff', 'Avg_PF_diff', 'Avg_Poss_diff', 
            'Avg_Opp_WinPct_diff', 'Last30_WinRatio_diff', 'Seed_diff', 'KenPom_diff', 
            'Off_Efficiency_diff', 'FG_Pct_diff', '3P_Pct_diff', 'FT_Pct_diff', 
            'TO_Rate_diff', 'Reb_Rate_diff', 'Team1_Off_Efficiency', 'Team2_Off_Efficiency', 
            'Team1_FG_Pct', 'Team2_FG_Pct', 'Team1_3P_Pct', 'Team2_3P_Pct', 
            'Team1_FT_Pct', 'Team2_FT_Pct', 'Team1_TO_Rate', 'Team2_TO_Rate', 
            'Team1_Reb_Rate', 'Team2_Reb_Rate'
        ]
    elif model_type == 'bart_simplified':
        # Order for bart_simplified model - EXACTLY as specified in the user query
        feature_order = [
            'Exp_diff', 'WinPct_diff', 'Avg_Score_diff', 'Avg_FGM_diff', 
            'Avg_FGA_diff', 'Avg_FGM3_diff', 'Avg_FGA3_diff', 'Avg_FTM_diff', 'Avg_FTA_diff', 
            'Avg_OR_diff', 'Avg_DR_diff', 'Avg_Ast_diff', 'Avg_TO_diff', 'Avg_Stl_diff', 
            'Avg_Blk_diff', 'Avg_PF_diff', 'Avg_Opp_WinPct_diff', 'Last30_WinRatio_diff', 'Seed_diff'
        ]
    
    # Create DataFrame with ordered features
    ordered_features = {feature: features.get(feature, 0) for feature in feature_order}
    return pd.DataFrame([ordered_features])


def predict_matchup(model, team1_stats, team2_stats, team1_seed, team2_seed, 
                   team1_kenpom=None, team2_kenpom=None, 
                   team1_barthag=None, team2_barthag=None,
                   team1_exp=None, team2_exp=None,
                   model_type='basic'):
    """
    Predict the outcome of a matchup between two teams using the trained model.
    
    Parameters:
    - model: The trained model to use for prediction
    - team1_stats, team2_stats: Stats for each team (as a pd.Series)
    - team1_seed, team2_seed: Numeric seeds for each team
    - team1_kenpom, team2_kenpom: KenPom rankings (optional)
    - team1_barthag, team2_barthag: Barthag ratings (optional)
    - team1_exp, team2_exp: Experience ratings (optional)
    - model_type: 'basic', 'advanced', or 'bart_simplified' to determine which features to include
    
    Returns:
    - Probability that team1 wins
    """
    # Create features for the matchup
    features = create_matchup_features(
        team1_stats, team2_stats, team1_seed, team2_seed,
        team1_kenpom, team2_kenpom, 
        team1_barthag, team2_barthag,
        team1_exp, team2_exp,
        model_type
    )
    
    # Convert to DMatrix for XGBoost
    dmatrix = xgb.DMatrix(features)
    
    # Make prediction
    prob = model.predict(dmatrix)[0]
    
    return prob


# ------------------------------
# Main Simulation Function
# ------------------------------

def simulate_first_round_model(bracket_path='bracket.csv',
                               team_stats_path='../pre_tourney_data/TeamSeasonAverages_with_SoS.csv',
                               kenpom_path='../pre_tourney_data/KenPom-Rankings-Updated.csv',
                               trank_path='../processed_data/trank_simplified.csv',
                               model_path='../models/xgb_model_basic.pkl'):
    """
    Simulate the first round of the tournament using the specified model.
    """
    # Determine model type from filename
    model_type = 'basic'
    if 'advanced' in model_path:
        model_type = 'advanced'
    elif 'bart_model_simplified' in model_path:
        model_type = 'bart_simplified'
    
    print(f"Using model type: {model_type}")
    
    # Load the model
    model = joblib.load(model_path)  
      
    # Load team stats for latest season
    team_stats = pd.read_csv(team_stats_path)
    latest_season = team_stats['Season'].max()
    team_stats = team_stats[team_stats['Season'] == latest_season].copy()
    
    # Load KenPom rankings if using advanced model
    kenpom_rankings = None
    if model_type == 'advanced' and os.path.exists(kenpom_path):
        kenpom_df = pd.read_csv(kenpom_path)
        kenpom_latest = kenpom_df[kenpom_df['Season'] == latest_season]
        kenpom_rankings = dict(zip(kenpom_latest['TeamID'], kenpom_latest['OrdinalRank']))
        print(f"Loaded KenPom rankings for {len(kenpom_rankings)} teams")
    
    # Load T-Rank data if using bart_simplified model
    trank_data = None
    if model_type == 'bart_simplified' and os.path.exists(trank_path):
        trank_df = pd.read_csv(trank_path)
        trank_latest = trank_df[trank_df['Season'] == latest_season]
        trank_data = {
            'barthag': dict(zip(trank_latest['TeamID'], trank_latest['Barthag'])),
            'exp': dict(zip(trank_latest['TeamID'], trank_latest['Exp']))
        }
        print(f"Loaded T-Rank data for {len(trank_data['barthag'])} teams")
    
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
                
                # Get KenPom rankings if available
                team_a_kenpom = kenpom_rankings.get(team_a_stats['TeamID']) if kenpom_rankings else None
                team_b_kenpom = kenpom_rankings.get(team_b_stats['TeamID']) if kenpom_rankings else None
                
                # Get T-Rank data if available
                team_a_barthag = trank_data['barthag'].get(team_a_stats['TeamID']) if trank_data else None
                team_b_barthag = trank_data['barthag'].get(team_b_stats['TeamID']) if trank_data else None
                team_a_exp = trank_data['exp'].get(team_a_stats['TeamID']) if trank_data else None
                team_b_exp = trank_data['exp'].get(team_b_stats['TeamID']) if trank_data else None
                
                prob = predict_matchup(
                    model, 
                    team_a_stats, 
                    team_b_stats, 
                    team_a_bracket['Seed'], 
                    team_b_bracket['Seed'],
                    team_a_kenpom,
                    team_b_kenpom,
                    team_a_barthag,
                    team_b_barthag,
                    team_a_exp,
                    team_b_exp,
                    model_type
                )
                
                # Add additional info for display
                extra_info = ""
                if model_type == 'advanced':
                    poss_a = team_a_stats.get('Avg_Poss', 0)
                    poss_b = team_b_stats.get('Avg_Poss', 0)
                    avg_tempo = (poss_a + poss_b) / 2
                    tempo_diff = abs(poss_a - poss_b)
                    
                    kenpom_info = ""
                    if team_a_kenpom and team_b_kenpom:
                        kenpom_info = f", KenPom: {team_a_kenpom} vs {team_b_kenpom}"
                    
                    extra_info = f" | Avg Tempo: {avg_tempo:.1f}, Diff: {tempo_diff:.1f}{kenpom_info}"
                elif model_type == 'bart_simplified':
                    barthag_info = ""
                    exp_info = ""
                    if team_a_barthag and team_b_barthag:
                        barthag_info = f", Barthag: {team_a_barthag:.3f} vs {team_b_barthag:.3f}"
                    if team_a_exp and team_b_exp:
                        exp_info = f", Exp: {team_a_exp:.1f} vs {team_b_exp:.1f}"
                    
                    extra_info = f"{barthag_info}{exp_info}"
                
                results.append({
                    'Region': region,
                    'Matchup': f"{team_a_bracket['Team']} (Seed {seed_a}) vs {team_b_bracket['Team']} (Seed {seed_b}){extra_info}",
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
                   team_stats_path='../pre_tourney_data/TeamSeasonAverages_with_SoS.csv',
                   kenpom_path='../pre_tourney_data/KenPom-Rankings-Updated.csv',
                   trank_path='../processed_data/trank_simplified.csv'):
    """
    Compare predictions from all available models on the same bracket.
    """
    models = [
        '../models/xgb_model_basic.pkl',
        '../models/xgb_model_advanced.pkl',
        '../models/bart_model_simplified.pkl'
    ]
    
    all_results = {}
    for model_path in models:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        try:
            print(f"\n{'='*60}")
            print(f"SIMULATING WITH MODEL: {model_path}")
            print(f"{'='*60}")
            
            results = simulate_first_round_model(
                bracket_path=bracket_path,
                team_stats_path=team_stats_path,
                kenpom_path=kenpom_path,
                trank_path=trank_path,
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
    
    # Compare model predictions if we have multiple models
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("COMPARING MODEL PREDICTIONS")
        print("="*60)
        
        # Compare each pair of models
        model_names = list(all_results.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                if model1 in all_results and model2 in all_results:
                    print(f"\nComparing {model1} vs {model2}:")
                    
                    # Merge results on matchup
                    comparison = pd.merge(
                        all_results[model1][['Region', 'TeamA', 'TeamB', 'SeedA', 'SeedB', 'TeamA_win_prob', 'Predicted_Winner']], 
                        all_results[model2][['TeamA', 'TeamB', 'TeamA_win_prob', 'Predicted_Winner']], 
                        on=['TeamA', 'TeamB'],
                        suffixes=(f'_{model1}', f'_{model2}')
                    )
                    
                    # Find disagreements
                    comparison['prob_diff'] = comparison[f'TeamA_win_prob_{model2}'] - comparison[f'TeamA_win_prob_{model1}']
                    comparison['disagree'] = comparison[f'Predicted_Winner_{model1}'] != comparison[f'Predicted_Winner_{model2}']
                    
                    print(f"Number of prediction disagreements: {comparison['disagree'].sum()} out of {len(comparison)}")
                    
                    if comparison['disagree'].sum() > 0:
                        print("\nMatchups with different predictions:")
                        for _, row in comparison[comparison['disagree']].iterrows():
                            print(f"{row['Region']}: {row['TeamA']} ({row['SeedA']}) vs {row['TeamB']} ({row['SeedB']})")
                            print(f"  {model1}: {row[f'Predicted_Winner_{model1}']} wins ({row[f'TeamA_win_prob_{model1}']*100:.1f}%)")
                            print(f"  {model2}: {row[f'Predicted_Winner_{model2}']} wins ({row[f'TeamA_win_prob_{model2}']*100:.1f}%)")
                            print(f"  Probability difference: {abs(row['prob_diff'])*100:.1f}%")
                            print()
                    
                    # Show largest probability differences
                    print("\nMatchups with largest probability differences:")
                    for _, row in comparison.sort_values('prob_diff', key=abs, ascending=False).head(5).iterrows():
                        print(f"{row['Region']}: {row['TeamA']} ({row['SeedA']}) vs {row['TeamB']} ({row['SeedB']})")
                        print(f"  {model1}: {row[f'TeamA_win_prob_{model1}']*100:.1f}% for {row['TeamA']}")
                        print(f"  {model2}: {row[f'TeamA_win_prob_{model2}']*100:.1f}% for {row['TeamA']}")
                        print(f"  Difference: {row['prob_diff']*100:.1f}%")
                        print()
    
    return all_results

# ------------------------------
# Example Usage
# ------------------------------

if __name__ == "__main__":
    compare_models()
