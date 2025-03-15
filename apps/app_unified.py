import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import xgboost as xgb
from PIL import Image, ImageDraw
import base64
from io import BytesIO

import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now use regular import
from assets.basketball_logo import get_logo_html, create_basketball_logo

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="BracketBrain",
    layout="wide",
    page_icon=Image.open(BytesIO(base64.b64decode(create_basketball_logo()))),
    initial_sidebar_state="expanded"
)

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
st.write(f"Checking CSS Path: {css_path}")

if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Warning: `style.css` file not found. Skipping custom styles.")


# Additional CSS for tables
TABLE_CSS = """
<style>
.comparison-table table.dataframe {
    width: 100% !important;
}
</style>
"""
st.markdown(TABLE_CSS, unsafe_allow_html=True)

# JavaScript for sidebar state detection
sidebar_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    function checkSidebarState() {
        const sidebarExpanded = document.querySelector('[data-testid="stSidebar"]').style.width !== '0px';
        if (sidebarExpanded) {
            document.body.classList.add('sidebar-expanded');
        } else {
            document.body.classList.remove('sidebar-expanded');
        }
    }
    setTimeout(checkSidebarState, 500);
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === 'style') {
                checkSidebarState();
            }
        });
    });
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
        observer.observe(sidebar, { attributes: true });
    }
});
</script>
"""
st.markdown(sidebar_js, unsafe_allow_html=True)

# Custom header with logo and title
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

st.markdown("""
Use our machine learning model to predict outcomes of games in March Madness and win your bracket challenge. 
Enter the teams and their seeds to get predictions and detailed team comparisons.
""")

# ------------------------------
# Model Loading Functions
# ------------------------------
@st.cache_resource
def load_basic_model():
    try:
        model = joblib.load('../models/xgb_model_basic.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading basic model: {str(e)}")
        return None

@st.cache_resource
def load_enhanced_model():
    try:
        model = joblib.load('../models/final_model_py2.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading enhanced model: {str(e)}")
        return None

# ------------------------------
# Data Loading Functions
# ------------------------------
@st.cache_data
def load_data():
    # Load team data
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raw_data", "MTeams.csv"))

    # Debugging: Check if the file exists
    st.write(f"Checking CSV Path: {csv_path}")
    st.write(f"File Exists: {os.path.exists(csv_path)}")

    # Load the CSV if it exists
    if os.path.exists(csv_path):
        teams_df = pd.read_csv(csv_path)
    else:
        st.error("❌ `MTeams.csv` not found! Please check file path and deployment.")
    
    # Load basic stats
    basic_stats = pd.read_csv("../pre_tourney_data/TeamSeasonAverages_with_SoS.csv")
    
    # Load enhanced stats
    try:
        enhanced_stats = pd.read_csv("../pre_tourney_data/EnhancedTournamentStats.csv")
    except:
        enhanced_stats = basic_stats.copy()  # Fallback if enhanced stats not available
    
    # Load KenPom rankings
    try:
        kenpom_rankings = pd.read_csv("../pre_tourney_data/KenPom-Rankings-Updated.csv")
        kenpom_rankings = kenpom_rankings[kenpom_rankings['Season'] == 2025]
        kenpom_rankings = kenpom_rankings.rename(columns={'OrdinalRank': 'KenPom'})
    except:
        kenpom_rankings = None
    
    # Get the most recent season data
    latest_season = basic_stats['Season'].max()
    current_basic_stats = basic_stats[basic_stats['Season'] == latest_season]
    
    # Get enhanced stats for current season if available
    if 'Season' in enhanced_stats.columns:
        current_enhanced_stats = enhanced_stats[enhanced_stats['Season'] == latest_season]
    else:
        current_enhanced_stats = current_basic_stats.copy()
    
    # Merge KenPom rankings with current stats if available
    if kenpom_rankings is not None:
        current_enhanced_stats = current_enhanced_stats.merge(
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
    
    return teams_df, basic_stats, enhanced_stats, current_basic_stats, current_enhanced_stats, latest_season, seed_matchups

# ------------------------------
# Feature Creation Functions
# ------------------------------
def create_basic_features(team1_stats, team2_stats, team1_seed, team2_seed):
    """Create features for the basic XGBoost model"""
    season_cols = ['WinPct', 'Avg_Score', 'Avg_FGM', 'Avg_FGA', 'Avg_FGM3', 'Avg_FGA3',
                   'Avg_FTM', 'Avg_FTA', 'Avg_OR', 'Avg_DR', 'Avg_Ast', 'Avg_TO',
                   'Avg_Stl', 'Avg_Blk', 'Avg_PF', 'Avg_Opp_WinPct', 'Last30_WinRatio']
    
    features = {}
    for col in season_cols:
        features[f"{col}_diff"] = team1_stats[col] - team2_stats[col]
    
    # Add seed difference
    features['Seed_diff'] = team1_seed - team2_seed
    
    return pd.DataFrame([features])

def create_enhanced_features(team1_stats, team2_stats, team1_seed, team2_seed):
    """Create features for the enhanced model"""
    # Define the feature order as expected by the enhanced model
    features = {
        'SeedDiff': team1_seed - team2_seed,
    }
    
    # Add KenPom difference if available
    if 'KenPom' in team1_stats and 'KenPom' in team2_stats:
        features['KenPomDiff'] = team1_stats['KenPom'] - team2_stats['KenPom']
    
    # Add enhanced stats differences
    enhanced_features = [
        'AdjO', 'AdjD', 'AdjNetRtg', 'SOS_NetRtg', 'Expected Win%', 
        'ThreePtRate', 'FTRate', 'AstRate', 'TORate', 'ORRate', 'DRRate',
        'ScoreStdDev', 'MarginStdDev', 'ORtgStdDev', 'DRtgStdDev',
        'HomeWin%', 'AwayWin%', 'NeutralWin%', 'Last10Win%'
    ]
    
    for feature in enhanced_features:
        if feature in team1_stats and feature in team2_stats:
            features[f'Diff_{feature}'] = team1_stats[feature] - team2_stats[feature]
    
    # Flip the sign for metrics where lower is better
    lower_is_better = ['KenPomDiff', 'Diff_AdjD', 'Diff_TORate']
    for feature in lower_is_better:
        if feature in features:
            features[feature] = -features[feature]
    
    return pd.DataFrame([features])

# ------------------------------
# Prediction Functions
# ------------------------------
def predict_with_basic_model(model, team1_stats, team2_stats, team1_seed, team2_seed):
    """Make prediction using the basic XGBoost model"""
    # Check if we need to swap teams (basic model expects better seed first)
    swap_needed = team1_seed > team2_seed
    
    if swap_needed:
        # Swap teams so better seed is first
        X = create_basic_features(team2_stats, team1_stats, team2_seed, team1_seed)
        dmatrix = xgb.DMatrix(X)
        prob = model.predict(dmatrix)[0]
        # Return probability for team1 winning (need to flip result)
        return 1 - prob
    else:
        # No swap needed
        X = create_basic_features(team1_stats, team2_stats, team1_seed, team2_seed)
        dmatrix = xgb.DMatrix(X)
        return model.predict(dmatrix)[0]

def predict_with_enhanced_model(model, team1_stats, team2_stats, team1_seed, team2_seed):
    """Make prediction using the enhanced model"""
    # Create features
    X = create_enhanced_features(team1_stats, team2_stats, team1_seed, team2_seed)
    
    # Make prediction
    if hasattr(model, 'predict_proba'):
        # For sklearn models
        return model.predict_proba(X)[0][1]
    else:
        # For XGBoost models
        dmatrix = xgb.DMatrix(X)
        return model.predict(dmatrix)[0]

# ------------------------------
# Styling Functions
# ------------------------------
def style_comparison_table(df, team1_name, team2_name):
    """Style the comparison table with color highlighting"""
    styled_df = df.copy()
    
    # Store original numeric values for calculations
    original_values = {
        team1_name: df[team1_name].astype(float),
        team2_name: df[team2_name].astype(float)
    }
    
    def apply_color(row):
        # Skip the Metric and Advantage columns
        if row.name == 'Metric' or row.name == 'Advantage':
            return [''] * len(row)
        
        styles = [''] * len(row)
        
        # Get the values for both teams from the original values
        team1_val = original_values[team1_name][row.name]
        team2_val = original_values[team2_name][row.name]
        
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

def build_comparison_table(team1_stats, team2_stats, team1_name, team2_name, metrics, metric_names=None, lower_is_better=None):
    """Build a comparison table for the given metrics"""
    if metric_names is None:
        metric_names = metrics
    
    if lower_is_better is None:
        lower_is_better = []
    
    data = []
    for metric, name in zip(metrics, metric_names):
        if metric in team1_stats and metric in team2_stats:
            # Determine which team has the advantage
            is_lower_better = metric in lower_is_better
            team1_better = team1_stats[metric] < team2_stats[metric] if is_lower_better else team1_stats[metric] > team2_stats[metric]
            
            data.append({
                'Metric': name,
                team1_name: f"{team1_stats[metric]:.2f}",
                team2_name: f"{team2_stats[metric]:.2f}",
                'Advantage': team1_name if team1_better else team2_name
            })
    
    return pd.DataFrame(data)

# ------------------------------
# Analysis Functions
# ------------------------------
def display_tempo_analysis(team1_name, team2_name, team1_stats, team2_stats, win_probability=None):
    """Display tempo analysis for the matchup"""
    st.markdown("---")
    st.subheader("Tempo Analysis")
    
    # Check if tempo data is available
    tempo_field = 'Poss' if 'Poss' in team1_stats else 'Avg_Poss'
    
    if tempo_field in team1_stats and tempo_field in team2_stats:
        team1_tempo = team1_stats[tempo_field]
        team2_tempo = team2_stats[tempo_field]
        
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
            
            # Only show win prediction if win_probability is provided
            if win_probability is not None:
                if faster_team == team1_name and win_probability > 0.5:
                    st.write("✓ Faster team is favored to win")
                elif faster_team == team2_name and win_probability < 0.5:
                    st.write("✓ Faster team is favored to win")
                else:
                    st.write("✗ Slower team is favored to win")
    else:
        st.write("Tempo data not available for one or both teams.")

# ------------------------------
# Main Function
# ------------------------------
def main():
    # Load data
    teams_df, basic_stats, enhanced_stats, current_basic_stats, current_enhanced_stats, latest_season, seed_matchups = load_data()
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select Prediction Model",
        ["Basic Model", "Enhanced Model"],
        index=0,  # Default to Basic Model
        help="Basic Model uses season averages. Enhanced Model includes KenPom and advanced metrics."
    )
    
    # Load the selected model
    if model_choice == "Basic Model":
        model = load_basic_model()
        current_stats = current_basic_stats
        st.sidebar.info("Using basic model with season averages")
    else:
        model = load_enhanced_model()
        current_stats = current_enhanced_stats
        st.sidebar.info("Using enhanced model with KenPom and advanced metrics")
    
    # Team selection
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
        
        # Make prediction based on selected model
        if model_choice == "Basic Model":
            win_probability = predict_with_basic_model(model, team1_stats, team2_stats, team1_seed, team2_seed)
        else:
            win_probability = predict_with_enhanced_model(model, team1_stats, team2_stats, team1_seed, team2_seed)
        
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction")
            if win_probability > 0.5:
                st.success(f"{team1_name} wins with {win_probability*100:.1f}% probability")
            else:
                st.success(f"{team2_name} wins with {(1-win_probability)*100:.1f}% probability")
            
            st.progress(float(win_probability))
            
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
                    
                    # Only show upset alert if it's at least 4% higher than historical average
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
            
            # Display feature differences based on model
            if model_choice == "Basic Model":
                features = create_basic_features(team1_stats, team2_stats, team1_seed, team2_seed)
            else:
                features = create_enhanced_features(team1_stats, team2_stats, team1_seed, team2_seed)
            
            # Display top 5 most important differences
            st.write("Most important differences:")
            feature_diffs = pd.DataFrame({
                'Feature': features.columns,
                'Value': features.values[0]
            })
            feature_diffs = feature_diffs.sort_values(by='Value', key=abs, ascending=False).head(5)
            
            for _, row in feature_diffs.iterrows():
                feature = row['Feature']
                value = row['Value']
                if feature in ['SeedDiff', 'Seed_diff']:
                    st.write(f"• Seed Difference: {value:.0f}")
                elif feature == 'KenPomDiff':
                    st.write(f"• KenPom Ranking: {'Advantage to ' + team1_name if value < 0 else 'Advantage to ' + team2_name}")
                else:
                    feature_name = feature.replace('Diff_', '').replace('_diff', '')
                    team_advantage = team1_name if value > 0 else team2_name
                    st.write(f"• {feature_name}: Advantage to {team_advantage}")
            
            # Display tempo analysis
            display_tempo_analysis(team1_name, team2_name, team1_stats, team2_stats, win_probability)
    
    # Display team comparisons
    st.header("Team Comparison")
    
    # Create tabs for different stat categories
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Offensive Stats", "Defensive Stats", "Performance Metrics"])
    
    # Define metrics for each tab based on available data
    with tab1:
        # Overview metrics
        if model_choice == "Basic Model":
            overview_metrics = ['WinPct', 'Last30_WinRatio', 'Avg_Opp_WinPct']
            overview_names = ['Win Percentage', 'Last 30 Games Win Ratio', 'Opponent Win Percentage']
            lower_is_better = ['Avg_Opp_WinPct']
        else:
            overview_metrics = ['KenPom', 'AdjO', 'AdjD', 'AdjNetRtg', 'Expected Win%', 'SOS_NetRtg', 'Last10Win%']
            overview_names = ['KenPom Rating', 'Adjusted Offensive Rating', 'Adjusted Defensive Rating', 
                             'Adjusted Net Rating', 'Expected Win %', 'Strength of Schedule', 'Last 10 Games Win %']
            lower_is_better = ['KenPom', 'AdjD']
        
        overview_df = build_comparison_table(team1_stats, team2_stats, team1_name, team2_name, 
                                            overview_metrics, overview_names, lower_is_better)
        st.write('<div class="comparison-table">' + style_comparison_table(overview_df, team1_name, team2_name).to_html() + '</div>', unsafe_allow_html=True)
    
    with tab2:
        # Offensive metrics
        if model_choice == "Basic Model":
            offensive_metrics = ['Avg_Score', 'Avg_FGM', 'Avg_FGA', 'Avg_FGM3', 'Avg_FGA3', 'Avg_FTM', 'Avg_FTA', 'Avg_OR', 'Avg_Ast']
            offensive_names = ['Points per Game', 'Field Goals Made', 'Field Goals Attempted', '3-Point Field Goals Made',
                              '3-Point Field Goals Attempted', 'Free Throws Made', 'Free Throws Attempted', 
                              'Offensive Rebounds', 'Assists']
        else:
            offensive_metrics = ['AdjO', 'Score', 'ORtg', 'ThreePtRate', 'FTRate', 'AstRate', 'ORRate', 'Poss']
            offensive_names = ['Adjusted Offensive Rating', 'Points per Game', 'Offensive Rating', '3-Point Rate', 
                              'Free Throw Rate', 'Assist Rate', 'Offensive Rebound Rate', 'Tempo (Possessions/40 min)']
        
        offensive_df = build_comparison_table(team1_stats, team2_stats, team1_name, team2_name, 
                                             offensive_metrics, offensive_names)
        st.write('<div class="comparison-table">' + style_comparison_table(offensive_df, team1_name, team2_name).to_html() + '</div>', unsafe_allow_html=True)
    
    with tab3:
        # Defensive metrics
        if model_choice == "Basic Model":
            defensive_metrics = ['Avg_DR', 'Avg_Stl', 'Avg_Blk']
            defensive_names = ['Defensive Rebounds', 'Steals', 'Blocks']
            lower_is_better = []
        else:
            defensive_metrics = ['AdjD', 'DRtg', 'DRRate']
            defensive_names = ['Adjusted Defensive Rating', 'Defensive Rating', 'Defensive Rebound Rate']
            lower_is_better = ['AdjD', 'DRtg']
        
        defensive_df = build_comparison_table(team1_stats, team2_stats, team1_name, team2_name, 
                                             defensive_metrics, defensive_names, lower_is_better)
        st.write('<div class="comparison-table">' + style_comparison_table(defensive_df, team1_name, team2_name).to_html() + '</div>', unsafe_allow_html=True)
    
    with tab4:
        # Performance metrics
        if model_choice == "Basic Model":
            performance_metrics = ['Avg_TO', 'Avg_PF']
            performance_names = ['Turnovers', 'Personal Fouls']
            lower_is_better = ['Avg_TO', 'Avg_PF']
        else:
            performance_metrics = ['HomeWin%', 'AwayWin%', 'NeutralWin%', 'TORate', 'ScoreStdDev', 'MarginStdDev']
            performance_names = ['Home Win %', 'Away Win %', 'Neutral Site Win %', 'Turnover Rate', 
                                'Scoring Consistency', 'Margin Consistency']
            lower_is_better = ['TORate', 'ScoreStdDev', 'MarginStdDev']
        
        performance_df = build_comparison_table(team1_stats, team2_stats, team1_name, team2_name, 
                                               performance_metrics, performance_names, lower_is_better)
        st.write('<div class="comparison-table">' + style_comparison_table(performance_df, team1_name, team2_name).to_html() + '</div>', unsafe_allow_html=True)
    
    # Footer
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