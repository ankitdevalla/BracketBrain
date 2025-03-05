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
    
    return teams_df, enhanced_stats, current_stats, latest_season

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
    
    # Create DataFrame with features in the correct order
    return pd.DataFrame([diffs])[feature_order]

# Main function
def main():
    # Load data
    teams_df, enhanced_stats, current_stats, latest_season = load_data()
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
            overview_data.append({
                'Metric': name,
                team1_name: team1_stats[col],
                team2_name: team2_stats[col],
                'Advantage': team1_name if team1_stats[col] > team2_stats[col] else team2_name
            })
        
        overview_df = pd.DataFrame(overview_data)
        st.table(overview_df)
    
    with tab2:
        # Offensive stats comparison
        offensive_cols = ['AdjO', 'ThreePtRate', 'FTRate', 'AstRate', 'TORate', 'ORRate']
        offensive_names = ['Adjusted Offensive Rating', '3-Point Rate', 'Free Throw Rate', 
                          'Assist Rate', 'Turnover Rate', 'Offensive Rebound Rate']
        
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
        # Defensive stats comparison
        defensive_cols = ['AdjD', 'DRRate']
        defensive_names = ['Adjusted Defensive Rating', 'Defensive Rebound Rate']
        
        defensive_data = []
        for col, name in zip(defensive_cols, defensive_names):
            better_lower = col == 'AdjD'
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
        # Performance metrics comparison
        performance_cols = ['HomeWin%', 'AwayWin%', 'NeutralWin%', 'ScoreStdDev', 'MarginStdDev']
        performance_names = ['Home Win %', 'Away Win %', 'Neutral Win %', 
                            'Scoring Consistency', 'Margin Consistency']
        
        performance_data = []
        for col, name in zip(performance_cols, performance_names):
            better_higher = col not in ['ScoreStdDev', 'MarginStdDev']
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
