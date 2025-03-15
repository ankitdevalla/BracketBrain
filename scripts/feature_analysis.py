import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the same data as in bart_model_simplified.py
def load_data():
    """Load tournament data, seeds, and simplified T-Rank data"""
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
    
    return tour_df, seeds_df, trank_df

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

def add_features(matchups_df, trank_df, seeds_df):
    """Add T-Rank features and seeds to matchups"""
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
    # Create difference features
    matchups_df['Barthag_diff'] = matchups_df['Team1_Barthag'] - matchups_df['Team2_Barthag']
    matchups_df['Exp_diff'] = matchups_df['Team1_Exp'] - matchups_df['Team2_Exp']
    matchups_df['Seed_diff'] = matchups_df['Team1_Seed'] - matchups_df['Team2_Seed']
    
    # Create target variable (1 if Team1 wins, 0 if Team2 wins)
    matchups_df['Target'] = (matchups_df['Team1'] == matchups_df['Winner']).astype(int)
    
    return matchups_df

def create_advanced_features(matchups_df):
    """Create additional advanced features"""
    # Barthag squared (emphasize large differences)
    matchups_df['Barthag_diff_squared'] = matchups_df['Barthag_diff'] ** 2 * np.sign(matchups_df['Barthag_diff'])
    
    # Experience-weighted Barthag
    matchups_df['Team1_Exp_Barthag'] = matchups_df['Team1_Barthag'] * matchups_df['Team1_Exp']
    matchups_df['Team2_Exp_Barthag'] = matchups_df['Team2_Barthag'] * matchups_df['Team2_Exp']
    matchups_df['Exp_Barthag_diff'] = matchups_df['Team1_Exp_Barthag'] - matchups_df['Team2_Exp_Barthag']
    
    # Seed-Barthag interaction
    matchups_df['Team1_Seed_Barthag'] = matchups_df['Team1_Barthag'] / matchups_df['Team1_Seed'].replace(0, 0.001)
    matchups_df['Team2_Seed_Barthag'] = matchups_df['Team2_Barthag'] / matchups_df['Team2_Seed'].replace(0, 0.001)
    matchups_df['Seed_Barthag_diff'] = matchups_df['Team1_Seed_Barthag'] - matchups_df['Team2_Seed_Barthag']
    
    return matchups_df

def analyze_feature_correlations(matchups_df):
    """Analyze correlations between features and with the target variable"""
    # Select features for analysis
    diff_cols = [col for col in matchups_df.columns if col.endswith('_diff')]
    
    team_cols = [
        'Team1_Barthag', 'Team2_Barthag',
        'Team1_Exp', 'Team2_Exp',
        'Team1_Seed', 'Team2_Seed',
        'Team1_Exp_Barthag', 'Team2_Exp_Barthag',
        'Team1_Seed_Barthag', 'Team2_Seed_Barthag'
    ]
    
    feature_cols = diff_cols + team_cols
    
    # Handle any NaN values
    features_df = matchups_df[feature_cols + ['Target']].fillna(0)
    
    # Calculate correlation with target
    target_corr = features_df.corr()['Target'].sort_values(ascending=False)
    print("\nFeature Correlation with Target (Win Probability):")
    print(target_corr)
    
    # Create correlation matrix
    corr_matrix = features_df[feature_cols].corr()
    
    # Create output directory if it doesn't exist
    output_dir = "../analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save correlation matrix to CSV
    corr_matrix.to_csv(f"{output_dir}/feature_correlation_matrix.csv")
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
    
    # Find highly correlated feature pairs
    high_corr_threshold = 0.8
    high_corr_pairs = []
    
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > high_corr_threshold:
                high_corr_pairs.append((feature_cols[i], feature_cols[j], corr_value))
    
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nHighly Correlated Feature Pairs (|r| > {high_corr_threshold}):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"{feat1} & {feat2}: {corr:.4f}")
    
    return target_corr, high_corr_pairs

def calculate_vif(matchups_df, feature_cols):
    """Calculate Variance Inflation Factor to detect multicollinearity"""
    # Handle any NaN values
    X = matchups_df[feature_cols].fillna(0)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_cols
    vif_data["VIF"] = [variance_inflation_factor(X_scaled_df.values, i) for i in range(X_scaled_df.shape[1])]
    
    # Sort by highest VIF
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data)
    
    # Save VIF data to CSV
    output_dir = "../analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vif_data.to_csv(f"{output_dir}/vif_analysis.csv", index=False)
    
    return vif_data

def recommend_features(target_corr, high_corr_pairs, vif_data):
    """Recommend a set of features based on correlation and VIF analysis"""
    # Start with features most correlated with target
    top_features = target_corr.index[:10].tolist()
    
    # Remove target from the list if it's there
    if 'Target' in top_features:
        top_features.remove('Target')
    
    # Create a set of features to exclude due to high correlation
    exclude_features = set()
    for feat1, feat2, _ in high_corr_pairs:
        # If both features are in top features, keep the one with higher correlation to target
        if feat1 in top_features and feat2 in top_features:
            if abs(target_corr[feat1]) > abs(target_corr[feat2]):
                exclude_features.add(feat2)
            else:
                exclude_features.add(feat1)
    
    # Remove excluded features
    recommended_features = [f for f in top_features if f not in exclude_features]
    
    # Add some features with lower VIF if they're not already included
    low_vif_features = vif_data[vif_data['VIF'] < 5]['Feature'].tolist()
    for feature in low_vif_features:
        if feature not in recommended_features and feature not in exclude_features:
            recommended_features.append(feature)
            if len(recommended_features) >= 15:  # Limit to 15 features
                break
    
    print("\nRecommended Features for Model Training:")
    for feature in recommended_features:
        print(f"- {feature} (correlation with target: {target_corr.get(feature, 0):.4f})")
    
    return recommended_features

def main():
    print("Loading data...")
    tour_df, seeds_df, trank_df = load_data()
    print(f"Loaded tournament data with {len(tour_df)} games")
    print(f"Loaded T-Rank data with {len(trank_df)} rows")
    
    print("\nCreating matchups...")
    matchups_df = create_matchups(tour_df, seeds_df, trank_df)
    print(f"Created {len(matchups_df)} matchups")
    
    print("\nAdding features...")
    matchups_df = add_features(matchups_df, trank_df, seeds_df)
    
    print("\nCreating difference features...")
    matchups_df = create_difference_features(matchups_df)
    
    print("\nCreating advanced features...")
    matchups_df = create_advanced_features(matchups_df)
    
    print("\n" + "="*50)
    print("ANALYZING FEATURE CORRELATIONS")
    print("="*50)
    
    # Select features for analysis
    diff_cols = [col for col in matchups_df.columns if col.endswith('_diff')]
    
    team_cols = [
        'Team1_Barthag', 'Team2_Barthag',
        'Team1_Exp', 'Team2_Exp',
        'Team1_Seed', 'Team2_Seed',
        'Team1_Exp_Barthag', 'Team2_Exp_Barthag',
        'Team1_Seed_Barthag', 'Team2_Seed_Barthag'
    ]
    
    feature_cols = diff_cols + team_cols
    
    # Analyze feature correlations
    target_corr, high_corr_pairs = analyze_feature_correlations(matchups_df)
    
    print("\n" + "="*50)
    print("ANALYZING MULTICOLLINEARITY (VIF)")
    print("="*50)
    
    # Calculate VIF
    vif_data = calculate_vif(matchups_df, feature_cols)
    
    print("\n" + "="*50)
    print("FEATURE RECOMMENDATIONS")
    print("="*50)
    
    # Recommend features
    recommended_features = recommend_features(target_corr, high_corr_pairs, vif_data)
    
    # Save recommended features to a file
    output_dir = "../analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(f"{output_dir}/recommended_features.txt", "w") as f:
        f.write("Recommended Features for Model Training:\n")
        for feature in recommended_features:
            f.write(f"{feature}\n")
    
    print(f"\nAnalysis complete! Results saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main() 