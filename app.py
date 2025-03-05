import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Set page title and layout
st.set_page_config(
    page_title="March Madness Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("March Madness Prediction Tool")
st.markdown("""
This tool helps you predict the outcome of March Madness matchups using a machine learning model.
Enter the teams and their seeds to get predictions and detailed team statistics.
""")

# Load data files
@st.cache_data
def load_data():
    # Load team data
    teams_df = pd.read_csv("raw_data/MTeams.csv")
    
    # Load enhanced stats
    enhanced_stats = pd.read_csv("pre_tourney_data/EnhancedTournamentStats.csv")
    
    # Load KenPom rankings
    kenpom_rankings = pd.read_csv("pre_tourney_data/KenPom-Rankings-Updated.csv")
    kenpom_rankings = kenpom_rankings[kenpom_rankings['Season'] == 2025]
    kenpom_rankings = kenpom_rankings.rename(columns={'OrdinalRank': 'KenPom'})
    
    # Get the most recent season data
    latest_season = enhanced_stats['Season'].max()
    current_stats = enhanced_stats[enhanced_stats['Season'] == latest_season]
    
    # Merge KenPom rankings with current stats
    current_stats = current_stats.merge(
        kenpom_rankings[['TeamID', 'KenPom']],
        on='TeamID',
        how='left'
    )
    
    # Historical seed matchup data
    seed_matchups = {
        (1, 16): 0.987,
        (2, 15): 0.929,
        (3, 14): 0.853,
        (4, 13): 0.788,
        (5, 12): 0.647,
        (6, 11): 0.609,
        (7, 10): 0.613,
        (8, 9): 0.481
    }
    
    return teams_df, enhanced_stats, current_stats, latest_season, seed_matchups

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('scripts/final_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to create feature differences for prediction
def create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed):
    # Define the feature order as expected by the model
    feature_order = [
        'SeedDiff', 'KenPomDiff', 'Diff_AdjO', 'Diff_AdjD', 'Diff_AdjNetRtg',
        'Diff_SOS_NetRtg', 'Diff_Expected Win%', 'Diff_ThreePtRate', 'Diff_FTRate',
        'Diff_AstRate', 'Diff_TORate', 'Diff_ORRate', 'Diff_DRRate',
        'Diff_ScoreStdDev', 'Diff_MarginStdDev', 'Diff_ORtgStdDev',
        'Diff_DRtgStdDev', 'Diff_HomeWin%', 'Diff_AwayWin%', 'Diff_NeutralWin%',
        'Diff_Last10Win%'
    ]
    
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
        'Diff_Last10Win%': team1_stats['Last10Win%'] - team2_stats['Last10Win%']
    }
    
    # Create DataFrame with features in the correct order
    return pd.DataFrame([diffs])[feature_order]

# Main function
def main():
    # Load data
    teams_df, enhanced_stats, current_stats, latest_season, seed_matchups = load_data()
    model = load_model()
    
    # Create sidebar for inputs
    st.sidebar.header("Enter Matchup Details")
    
    # Team selection
    team_names = sorted(current_stats['TeamName'].unique())
    
    team1_name = st.sidebar.selectbox("Select Team 1", team_names, index=0)
    team1_seed = st.sidebar.number_input("Team 1 Seed", min_value=1, max_value=16, value=1)
    
    team2_name = st.sidebar.selectbox("Select Team 2", team_names, index=1)
    team2_seed = st.sidebar.number_input("Team 2 Seed", min_value=1, max_value=16, value=16)
    
    # Get team stats
    team1_stats = current_stats[current_stats['TeamName'] == team1_name].iloc[0]
    team2_stats = current_stats[current_stats['TeamName'] == team2_name].iloc[0]
    
    # Button to make prediction
    if st.sidebar.button("Predict Winner"):
        if model is None:
            st.error("Model not loaded. Please check if the model file exists.")
            return
        
        # Create features for prediction
        X = create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed)
        
        # Make prediction
        try:
            win_probability = model.predict_proba(X)[0][1]
            
            # Display prediction
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction")
                if win_probability > 0.5:
                    st.success(f"{team1_name} wins with {win_probability*100:.1f}% probability")
                else:
                    st.success(f"{team2_name} wins with {(1-win_probability)*100:.1f}% probability")
                
                st.progress(win_probability)
                
                st.write(f"**{team1_name}**: {win_probability*100:.1f}% chance to win")
                st.write(f"**{team2_name}**: {(1-win_probability)*100:.1f}% chance to win")
                
                # Add historical context
                higher_seed = min(team1_seed, team2_seed)
                lower_seed = max(team1_seed, team2_seed)
                if (higher_seed, lower_seed) in seed_matchups:
                    historical_win_rate = seed_matchups[(higher_seed, lower_seed)]
                    predicted_upset_prob = 1 - win_probability if team1_seed > team2_seed else win_probability
                    
                    st.markdown("---")
                    st.subheader("Historical Context")
                    st.write(f"Historically, #{higher_seed} seeds win {historical_win_rate*100:.1f}% of games against #{lower_seed} seeds")
                    
                    if predicted_upset_prob > (1 - historical_win_rate):
                        upset_likelihood = predicted_upset_prob - (1 - historical_win_rate)
                        st.warning(f"⚠️ Potential Upset Alert: This game has a {upset_likelihood*100:.1f}% higher chance of an upset compared to historical averages")
                    elif predicted_upset_prob < (1 - historical_win_rate) * 0.5:
                        st.info("🔒 This matchup appears to be safer than the historical average for the higher seed")
            
            with col2:
                st.subheader("Key Matchup Factors")
                
                # Display feature differences
                st.write("Most important differences:")
                feature_diffs = pd.DataFrame({
                    'Feature': X.columns,
                    'Value': X.values[0]
                })
                feature_diffs = feature_diffs.sort_values(by='Value', key=abs, ascending=False).head(5)
                
                for _, row in feature_diffs.iterrows():
                    feature = row['Feature']
                    value = row['Value']
                    if feature == 'SeedDiff':
                        st.write(f"• Seed Difference: {value:.0f}")
                    else:
                        feature_name = feature.replace('Diff_', '')
                        team_advantage = team1_name if value > 0 else team2_name
                        st.write(f"• {feature_name}: Advantage to {team_advantage}")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Display team comparisons
    st.header("Team Comparison")
    
    # Create tabs for different stat categories
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Offensive Stats", "Defensive Stats", "Performance Metrics"])
    
    with tab1:
        # Create a comparison table for overview stats
        overview_cols = ['KenPom', 'AdjO', 'AdjD', 'AdjNetRtg', 'Expected Win%', 'SOS_NetRtg', 'Last10Win%']
        overview_names = ['KenPom Rating', 'Adjusted Offensive Rating', 'Adjusted Defensive Rating', 
                         'Adjusted Net Rating', 'Expected Win %', 'Strength of Schedule', 'Last 10 Games Win %']
        
        overview_data = []
        for col, name in zip(overview_cols, overview_names):
            # For KenPom and defensive ratings, lower is better
            lower_is_better = col in ['KenPom', 'AdjD']
            team1_better = team1_stats[col] < team2_stats[col] if lower_is_better else team1_stats[col] > team2_stats[col]
            
            overview_data.append({
                'Metric': name,
                team1_name: team1_stats[col],
                team2_name: team2_stats[col],
                'Advantage': team1_name if team1_better else team2_name
            })
        
        overview_df = pd.DataFrame(overview_data)
        st.table(overview_df)
    
    with tab2:
        # Offensive stats comparison - expanded
        offensive_cols = ['AdjO', 'Score', 'ORtg', 'ThreePtRate', 'FTRate', 'AstRate', 'TORate', 'ORRate']
        offensive_names = ['Adjusted Offensive Rating', 'Points per Game', 'Offensive Rating', '3-Point Rate', 
                          'Free Throw Rate', 'Assist Rate', 'Turnover Rate', 'Offensive Rebound Rate']
        
        offensive_data = []
        for col, name in zip(offensive_cols, offensive_names):
            better_higher = col not in ['TORate']
            team1_better = team1_stats[col] > team2_stats[col] if better_higher else team1_stats[col] < team2_stats[col]
            
            offensive_data.append({
                'Metric': name,
                team1_name: team1_stats[col],
                team2_name: team2_stats[col],
                'Advantage': team1_name if team1_better else team2_name
            })
        
        offensive_df = pd.DataFrame(offensive_data)
        st.table(offensive_df)
    
    with tab3:
        # Defensive stats comparison - expanded
        defensive_cols = ['AdjD', 'DRtg', 'DRRate']
        defensive_names = ['Adjusted Defensive Rating', 'Defensive Rating', 'Defensive Rebound Rate']
        
        defensive_data = []
        for col, name in zip(defensive_cols, defensive_names):
            # For defensive ratings, lower is better, but higher is better for rebound rate
            better_lower = col in ['AdjD', 'DRtg']
            team1_better = team1_stats[col] < team2_stats[col] if better_lower else team1_stats[col] > team2_stats[col]
            
            defensive_data.append({
                'Metric': name,
                team1_name: team1_stats[col],
                team2_name: team2_stats[col],
                'Advantage': team1_name if team1_better else team2_name
            })
        
        defensive_df = pd.DataFrame(defensive_data)
        st.table(defensive_df)
    
    with tab4:
        # Performance metrics comparison - expanded
        performance_cols = ['HomeWin%', 'AwayWin%', 'NeutralWin%', 'ClutchWin%', 'Last10Win%',
                          'ScoreStdDev', 'MarginStdDev', 'ORtgStdDev', 'DRtgStdDev', 'HomeAwayORtgDiff']
        performance_names = ['Home Win %', 'Away Win %', 'Neutral Win %', 'Clutch Win %', 'Last 10 Games Win %',
                           'Scoring Consistency', 'Margin Consistency', 'Off. Rating Consistency',
                           'Def. Rating Consistency', 'Home/Away Off. Rating Difference']
        
        performance_data = []
        for col, name in zip(performance_cols, performance_names):
            better_higher = col not in ['ScoreStdDev', 'MarginStdDev', 'ORtgStdDev', 'DRtgStdDev']
            team1_better = team1_stats[col] > team2_stats[col] if better_higher else team1_stats[col] < team2_stats[col]
            
            performance_data.append({
                'Metric': name,
                team1_name: team1_stats[col],
                team2_name: team2_stats[col],
                'Advantage': team1_name if team1_better else team2_name
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.table(performance_df)

if __name__ == "__main__":
    main()
