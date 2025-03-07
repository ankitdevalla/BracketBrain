import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
from sklearn.preprocessing import StandardScaler
from assets.basketball_logo import get_logo_html, create_basketball_logo
from PIL import Image, ImageDraw
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="BracketBrain",
    layout="wide",
    page_icon=Image.open(BytesIO(base64.b64decode(create_basketball_logo()))),
    initial_sidebar_state="expanded"
)

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# JavaScript to detect sidebar state and adjust footer
sidebar_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Function to check if sidebar is expanded
    function checkSidebarState() {
        const sidebarExpanded = document.querySelector('[data-testid="stSidebar"]').style.width !== '0px';
        if (sidebarExpanded) {
            document.body.classList.add('sidebar-expanded');
        } else {
            document.body.classList.remove('sidebar-expanded');
        }
    }
    
    // Initial check
    setTimeout(checkSidebarState, 500);
    
    // Set up a mutation observer to watch for changes to the sidebar
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === 'style') {
                checkSidebarState();
            }
        });
    });
    
    // Start observing the sidebar for style changes
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
        observer.observe(sidebar, { attributes: true });
    }
});
</script>
"""
st.markdown(sidebar_js, unsafe_allow_html=True)

# Custom header with BracketBrain logo and title
header_html = f"""
<div class="header">
    <div class="header-logo">
        {get_logo_html(size=50)}
        <div>
            <h1 class="header-title">BracketBrain</h1>
            <p class="header-subtitle">NCAA Tournament Prediction Tool</p>
        </div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# App description
st.markdown("""
Use our machine learning model to predict outcomes of games in March Madness and win your bracket challenge. 
In the matchup you want to analyze, enter the teams and their seeds to get predictions and detailed team comparisons.
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
        # Try to load the model with tempo features first
        model = joblib.load('models/final_model_py2.pkl')
        # feature_names = np.load('scripts/feature_names_with_tempo.npy', allow_pickle=True)
        st.sidebar.success("Using model v2")
        return model
    except Exception as e:
        st.sidebar.warning(f"Could not load model with tempo: {str(e)}")
        try:
            # Fall back to original model if needed
            model = joblib.load('models/final_model.pkl')
            st.sidebar.info("Using original model without tempo features")
            return model
        except Exception as e:
            st.error(f"Error loading any model: {str(e)}")
            return None

# Function to style dataframes with color highlighting
def style_comparison_table(df, team1_name, team2_name):
    # Create a copy to avoid modifying the original
    styled_df = df.copy()
    
    # Function to apply background color based on advantage
    def apply_color(row):
        # Skip the Metric and Advantage columns
        if row.name == 'Metric' or row.name == 'Advantage':
            return [''] * len(row)
        
        styles = [''] * len(row)
        
        # Get the values for both teams
        team1_val = row[team1_name]
        team2_val = row[team2_name]
        
        # Calculate the difference and normalize it
        diff = abs(team1_val - team2_val)
        max_val = max(abs(team1_val), abs(team2_val))
        if max_val == 0:
            normalized_diff = 0
        else:
            normalized_diff = diff / max_val
        
        # Cap the intensity at 0.5 (50% difference)
        intensity = min(normalized_diff * 2, 0.5)
        
        # Determine which team has the advantage
        advantage_team = row['Advantage']
        
        # Apply colors - green for better, red for worse
        for i, col_name in enumerate(row.index):
            if col_name == team1_name:
                if advantage_team == team1_name:
                    styles[i] = f'background-color: rgba(0, 255, 0, {intensity})'
                else:
                    styles[i] = f'background-color: rgba(255, 0, 0, {intensity})'
            elif col_name == team2_name:
                if advantage_team == team2_name:
                    styles[i] = f'background-color: rgba(0, 255, 0, {intensity})'
                else:
                    styles[i] = f'background-color: rgba(255, 0, 0, {intensity})'
        
        return styles
    
    # Apply the styling
    return styled_df.style.apply(apply_color, axis=1)

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
    
    # Calculate all differences consistently as (team1 - team2)
    diffs = {
        'SeedDiff': team1_seed - team2_seed,
        'KenPomDiff': team1_stats['KenPom'] - team2_stats['KenPom'],  # Now consistent direction
        'Diff_AdjO': team1_stats['AdjO'] - team2_stats['AdjO'],
        'Diff_AdjD': team1_stats['AdjD'] - team2_stats['AdjD'],  # Now consistent direction
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
        # Add tempo features
        # 'Diff_Poss': team1_stats['Poss'] - team2_stats['Poss'],
        # 'AvgTempo': (team1_stats['Poss'] + team2_stats['Poss']) / 2,
        # 'TempoDiff': abs(team1_stats['Poss'] - team2_stats['Poss'])
    }
    
    # Create DataFrame with features in the correct order
    # Only include features that exist in the feature_order list
    available_features = [f for f in feature_order if f in diffs]
    return pd.DataFrame([diffs])[available_features]

# Main function
def main():
    # Load data
    teams_df, enhanced_stats, current_stats, latest_season, seed_matchups = load_data()
    
    # Load model
    model = load_model()
    
    # Sidebar for team selection
    st.sidebar.header("Team Selection")
    
    # Get list of teams for the current season
    current_teams = current_stats['TeamName'].unique()
    
    # Team 1 selection
    team1_name = st.sidebar.selectbox("Select Team 1", current_teams, index=0)
    team1_seed = st.sidebar.number_input("Team 1 Seed", min_value=1, max_value=16, value=1, step=1)
    
    # Team 2 selection
    team2_name = st.sidebar.selectbox("Select Team 2", current_teams, index=1)
    team2_seed = st.sidebar.number_input("Team 2 Seed", min_value=1, max_value=16, value=8, step=1)
    
    # Get team stats
    team1_stats = current_stats[current_stats['TeamName'] == team1_name].iloc[0]
    team2_stats = current_stats[current_stats['TeamName'] == team2_name].iloc[0]
    
    # Button to make prediction
    if st.sidebar.button("Predict Winner"):
        if model is None:
            st.error("Model not loaded. Please check if the model file exists.")
            return
        
        # Create features for prediction with consistent ordering
        if team1_seed > team2_seed:
            # Swap teams for prediction
            X = create_matchup_features(team2_stats, team1_stats, team2_seed, team1_seed)
            # Get prediction and flip it - FLIP THE INTERPRETATION
            win_probability = model.predict_proba(X)[0][1]  # Flipped from 1-prob to prob
        else:
            # No swap needed
            X = create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed)
            win_probability = model.predict_proba(X)[0][1]  # Flipped from prob to 1-prob
        
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
            
            # Determine which team is the higher seed
            higher_seed_team = team1_name if team1_seed < team2_seed else team2_name
            lower_seed_team = team2_name if team1_seed < team2_seed else team1_name
            
            # Get model's predicted probability for the higher seed winning
            higher_seed_win_prob = win_probability if team1_seed < team2_seed else (1 - win_probability)
            
            # Calculate upset probability (lower seed winning)
            upset_prob = 1 - higher_seed_win_prob
            
            if (higher_seed, lower_seed) in seed_matchups:
                historical_win_rate = seed_matchups[(higher_seed, lower_seed)]
                historical_upset_prob = 1 - historical_win_rate
                
                st.markdown("---")
                st.subheader("Historical Context")
                st.write(f"Historically, #{higher_seed} seeds win {historical_win_rate*100:.1f}% of games against #{lower_seed} seeds")
                
                # Compare model's upset probability to historical upset probability
                if upset_prob > historical_upset_prob:
                    upset_likelihood = upset_prob - historical_upset_prob
                    
                    # Only show upset alert if it's at least 10% higher than historical average
                    if upset_likelihood >= 0.04:
                        st.warning(f"⚠️ Potential Upset Alert: This game has a {upset_likelihood*100:.1f}% higher chance of an upset compared to historical averages")
                        
                        # Add additional context if the lower seed is actually favored
                        if upset_prob > 0.5:
                            st.info(f"📊 The model actually favors {lower_seed_team} (#{lower_seed} seed) to win this game!")
                elif upset_prob < historical_upset_prob * 0.5:
                    st.info("🔒 This matchup appears to be safer than the historical average for the higher seed")
            else:
                # For later rounds where we don't have historical seed matchup data
                st.markdown("---")
                st.subheader("Upset Potential")
                
                # Show upset alert if lower seed has >25% chance to win
                if upset_prob > 0.25:
                    if upset_prob > 0.5:
                        st.warning(f"⚠️ Major Upset Alert: The model favors {lower_seed_team} (#{lower_seed} seed) to win against {higher_seed_team} (#{higher_seed} seed)!")
                    else:
                        st.warning(f"⚠️ Upset Alert: {lower_seed_team} (#{lower_seed} seed) has a {upset_prob*100:.1f}% chance to upset {higher_seed_team} (#{higher_seed} seed)")
                else:
                    st.info(f"🔒 {higher_seed_team} (#{higher_seed} seed) is strongly favored with a {higher_seed_win_prob*100:.1f}% chance to win")
        
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
            
            # Add tempo analysis if tempo features exist in the prediction
            tempo_features = [f for f in X.columns if 'Tempo' in f or 'Poss' in f]
            if tempo_features:
                st.markdown("---")
                st.subheader("Tempo Analysis")
                
                # Display team tempos
                team1_tempo = team1_stats['Poss']
                team2_tempo = team2_stats['Poss']
                avg_tempo = (team1_tempo + team2_tempo) / 2
                tempo_diff = abs(team1_tempo - team2_tempo)
                
                st.write(f"**{team1_name}**: {team1_tempo:.1f} possessions/40 min")
                st.write(f"**{team2_name}**: {team2_tempo:.1f} possessions/40 min")
                
                # Classify game pace
                if avg_tempo > 70:
                    pace = "Fast-paced"
                elif avg_tempo < 65:
                    pace = "Slow-paced"
                else:
                    pace = "Moderate-paced"
                
                st.write(f"Expected game: **{pace}** ({avg_tempo:.1f} possessions/40 min)")
                
                # Analyze tempo mismatch
                if tempo_diff > 4:
                    faster_team = team1_name if team1_tempo > team2_tempo else team2_name
                    slower_team = team2_name if team1_tempo > team2_tempo else team1_name
                    st.write(f"**Significant tempo mismatch**: {faster_team} wants to play much faster than {slower_team}")
                    
                    if faster_team == team1_name and win_probability > 0.5:
                        st.write("✓ Faster team is favored to win")
                    elif faster_team == team2_name and win_probability < 0.5:
                        st.write("✓ Faster team is favored to win")
                    else:
                        st.write("✗ Slower team is favored to win")
    
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
        st.write('<div class="comparison-table">' + style_comparison_table(overview_df, team1_name, team2_name).to_html() + '</div>', unsafe_allow_html=True)
    
    with tab2:
        # Offensive stats comparison - expanded
        offensive_cols = ['AdjO', 'Score', 'ORtg', 'ThreePtRate', 'FTRate', 'AstRate', 'TORate', 'ORRate', 'Poss']
        offensive_names = ['Adjusted Offensive Rating', 'Points per Game', 'Offensive Rating', '3-Point Rate', 
                          'Free Throw Rate', 'Assist Rate', 'Turnover Rate', 'Offensive Rebound Rate', 'Tempo (Possessions/40 min)']
        
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
        st.write('<div class="comparison-table">' + style_comparison_table(offensive_df, team1_name, team2_name).to_html() + '</div>', unsafe_allow_html=True)
    
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
        st.write('<div class="comparison-table">' + style_comparison_table(defensive_df, team1_name, team2_name).to_html() + '</div>', unsafe_allow_html=True)
    
    with tab4:
        # Performance metrics comparison - expanded
        performance_cols = ['HomeWin%', 'AwayWin%', 'NeutralWin%', 'ClutchWin%', 'Last10Win%',
                          'ScoreStdDev', 'MarginStdDev', 'ORtgStdDev', 'DRtgStdDev', 'HomeAwayORtgDiff']
        performance_names = ['Home Win %', 'Away Win %', 'Neutral Site Win %', 'Close Game Win %', 'Last 10 Games Win %',
                           'Scoring Consistency', 'Margin Consistency', 'Offensive Consistency', 'Defensive Consistency', 'Home/Away Performance Gap']
        
        performance_data = []
        for col, name in zip(performance_cols, performance_names):
            # For consistency metrics (StdDev), lower is better
            better_lower = 'StdDev' in col
            team1_better = team1_stats[col] < team2_stats[col] if better_lower else team1_stats[col] > team2_stats[col]
            
            performance_data.append({
                'Metric': name,
                team1_name: team1_stats[col],
                team2_name: team2_stats[col],
                'Advantage': team1_name if team1_better else team2_name
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.write('<div class="comparison-table">' + style_comparison_table(performance_df, team1_name, team2_name).to_html() + '</div>', unsafe_allow_html=True)
    
    # Footer at the bottom of the content
    footer_html = f"""
    <footer class="footer">
        <div class="footer-content">
            <div class="footer-logo">
                {get_logo_html()}
                <span>BracketBrain</span>
            </div>
            <div class="footer-links">
                <a href="https://ankitdevalla.com" target="_blank">About Me</a>
                <a href="https://github.com/ankitdevalla/March_Madness_Pred" target="_blank">GitHub</a>
            </div>
            <div class="footer-contact">
                <div>ankitdevalla.dev@gmail.com</div>
                <div>&copy; {datetime.date.today().year} BracketBrain</div>
            </div>
        </div>
    </footer>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
