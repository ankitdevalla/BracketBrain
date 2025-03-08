import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import xgboost as xgb
from PIL import Image
import base64
from io import BytesIO

# If you have custom logo assets
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

# Load custom CSS (for your main styling)
with open("assets/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Additional CSS to force the comparison tables to full width
TABLE_CSS = """
<style>
.comparison-table table.dataframe {
    width: 100% !important;
}
</style>
"""
st.markdown(TABLE_CSS, unsafe_allow_html=True)

# JavaScript to detect sidebar state and adjust footer
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
# Feature Lists and Utility
# ------------------------------

# Base features from your XGBoost model
BASE_FEATURES = [
    'WinPct',
    'Avg_Score',
    'Avg_FGM',
    'Avg_FGA',
    'Avg_FGM3',
    'Avg_FGA3',
    'Avg_FTM',
    'Avg_FTA',
    'Avg_OR',
    'Avg_DR',
    'Avg_Ast',
    'Avg_TO',
    'Avg_Stl',
    'Avg_Blk',
    'Avg_PF',
    'Avg_Opp_WinPct',
    'Last30_WinRatio'
]

# For some stats, "lower is better"
LOWER_IS_BETTER = {"Avg_TO", "Avg_PF", "Avg_Opp_WinPct"}

# Define metric groups for the tabs
OVERVIEW_METRICS = ['WinPct', 'Last30_WinRatio', 'Avg_Opp_WinPct']
OFFENSIVE_METRICS = ['Avg_Score', 'Avg_FGM', 'Avg_FGA', 'Avg_FGM3', 'Avg_FGA3', 'Avg_FTM', 'Avg_FTA', 'Avg_OR', 'Avg_Ast']
DEFENSIVE_METRICS = ['Avg_DR', 'Avg_Stl', 'Avg_Blk']
PERFORMANCE_METRICS = ['Avg_PF', 'Avg_TO']  # Example; add or remove as you see fit

def create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed):
    """
    Create the DataFrame of difference features that the model expects:
    e.g., WinPct_diff, Avg_Score_diff, etc.
    """
    base_features = {
        'WinPct_diff': team1_stats['WinPct'] - team2_stats['WinPct'],
        'Avg_Score_diff': team1_stats['Avg_Score'] - team2_stats['Avg_Score'],
        'Avg_FGM_diff': team1_stats['Avg_FGM'] - team2_stats['Avg_FGM'],
        'Avg_FGA_diff': team1_stats['Avg_FGA'] - team2_stats['Avg_FGA'],
        'Avg_FGM3_diff': team1_stats['Avg_FGM3'] - team2_stats['Avg_FGM3'],
        'Avg_FGA3_diff': team1_stats['Avg_FGA3'] - team2_stats['Avg_FGA3'],
        'Avg_FTM_diff': team1_stats['Avg_FTM'] - team2_stats['Avg_FTM'],
        'Avg_FTA_diff': team1_stats['Avg_FTA'] - team2_stats['Avg_FTA'],
        'Avg_OR_diff': team1_stats['Avg_OR'] - team2_stats['Avg_OR'],
        'Avg_DR_diff': team1_stats['Avg_DR'] - team2_stats['Avg_DR'],
        'Avg_Ast_diff': team1_stats['Avg_Ast'] - team2_stats['Avg_Ast'],
        'Avg_TO_diff': team1_stats['Avg_TO'] - team2_stats['Avg_TO'],
        'Avg_Stl_diff': team1_stats['Avg_Stl'] - team2_stats['Avg_Stl'],
        'Avg_Blk_diff': team1_stats['Avg_Blk'] - team2_stats['Avg_Blk'],
        'Avg_PF_diff': team1_stats['Avg_PF'] - team2_stats['Avg_PF'],
        'Avg_Opp_WinPct_diff': team1_stats['Avg_Opp_WinPct'] - team2_stats['Avg_Opp_WinPct'],
        'Last30_WinRatio_diff': team1_stats['Last30_WinRatio'] - team2_stats['Last30_WinRatio'],
        'Seed_diff': team1_seed - team2_seed
    }
    return pd.DataFrame([base_features])

def predict_matchup(model, team1_stats, team2_stats, team1_seed, team2_seed):
    """
    Predict the outcome of a matchup using the trained XGBoost model.
    Returns probability that team1 wins.
    """
    swap_needed = team1_seed > team2_seed
    if swap_needed:
        team1_stats, team2_stats = team2_stats, team1_stats
        team1_seed, team2_seed = team2_seed, team1_seed

    # Create features
    X = create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed)
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    prob_first_wins = preds[0]  # Probability that the "first" team (in features) wins

    return 1 - prob_first_wins if swap_needed else prob_first_wins

def build_advantage_table(team1_stats, team2_stats, team1_name, team2_name, metrics_list):
    """
    Returns a DataFrame with columns:
      [Metric, team1_name, team2_name, Advantage]
    for the given subset of metrics_list.
    """
    rows = []
    for col in metrics_list:
        val1 = team1_stats[col]
        val2 = team2_stats[col]
        # Decide advantage
        if col in LOWER_IS_BETTER:
            advantage = team1_name if val1 < val2 else team2_name
        else:
            advantage = team1_name if val1 > val2 else team2_name
        
        rows.append({
            "Metric": col,
            team1_name: val1,
            team2_name: val2,
            "Advantage": advantage
        })
    return pd.DataFrame(rows)

def style_comparison_table(df, team1_name, team2_name):
    """
    Color each cell for the team that has the advantage (green),
    and color the other team’s cell red.
    """
    styled_df = df.copy()
    
    def apply_color(row):
        # Skip row for "Metric" and "Advantage"
        if row.name == 'Metric' or row.name == 'Advantage':
            return [''] * len(row)
        
        styles = [''] * len(row)
        # Get numeric values for both teams
        team1_val = row[team1_name]
        team2_val = row[team2_name]
        
        # Compute difference and intensity
        diff = abs(team1_val - team2_val)
        max_val = max(abs(team1_val), abs(team2_val))
        normalized_diff = diff / max_val if max_val != 0 else 0
        intensity = min(normalized_diff * 2, 0.5)
        
        advantage_team = row['Advantage']
        
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
    
    return styled_df.style.apply(apply_color, axis=1)

# ------------------------------
# Data and Model Load
# ------------------------------
@st.cache_data
def load_data():
    teams_df = pd.read_csv("raw_data/MTeams.csv")
    enhanced_stats = pd.read_csv("scripts/TeamSeasonAverages_with_SoS.csv")
    latest_season = enhanced_stats['Season'].max()
    current_stats = enhanced_stats[enhanced_stats['Season'] == latest_season].copy()
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

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/xgb_model_basic.pkl')
        st.sidebar.success("Using basic XGBoost model")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ------------------------------
# Tempo Analysis
# ------------------------------
def display_tempo_analysis(team1_name, team2_name, team1_stats, team2_stats, win_probability=None):
    st.markdown("---")
    st.subheader("Tempo Analysis")
    
    # Display team tempos
    team1_tempo = team1_stats['Avg_Poss'] if 'Avg_Poss' in team1_stats else 0
    team2_tempo = team2_stats['Avg_Poss'] if 'Avg_Poss' in team2_stats else 0
    
    if team1_tempo > 0 and team2_tempo > 0:
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
# Main
# ------------------------------
def main():
    st.sidebar.header("Team Selection")
    teams_df, enhanced_stats, current_stats, latest_season, seed_matchups = load_data()
    current_teams = current_stats['TeamName'].unique()
    
    # Team 1
    team1_name = st.sidebar.selectbox("Select Team 1", current_teams, index=0)
    team1_seed = st.sidebar.number_input("Team 1 Seed", min_value=1, max_value=16, value=1, step=1)
    # Team 2
    team2_name = st.sidebar.selectbox("Select Team 2", current_teams, index=1)
    team2_seed = st.sidebar.number_input("Team 2 Seed", min_value=1, max_value=16, value=8, step=1)
    
    # Grab stats
    team1_stats = current_stats[current_stats['TeamName'] == team1_name].iloc[0]
    team2_stats = current_stats[current_stats['TeamName'] == team2_name].iloc[0]
    
    # Predict button
    if st.sidebar.button("Predict Winner"):
        model = load_model()
        if model is None:
            st.error("Model not loaded.")
            return
        
        # Make sure to cast to float for st.progress
        win_probability = float(predict_matchup(model, team1_stats, team2_stats, team1_seed, team2_seed))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction")
            if win_probability > 0.5:
                st.success(f"{team1_name} wins with {win_probability*100:.1f}% probability")
            else:
                st.success(f"{team2_name} wins with {(1-win_probability)*100:.1f}% probability")
            
            st.progress(win_probability)  # requires a standard float
            st.write(f"**{team1_name}**: {win_probability*100:.1f}% chance to win")
            st.write(f"**{team2_name}**: {(1-win_probability)*100:.1f}% chance to win")
            
            higher_seed = min(team1_seed, team2_seed)
            lower_seed = max(team1_seed, team2_seed)
            higher_seed_team = team1_name if team1_seed < team2_seed else team2_name
            lower_seed_team = team2_name if team1_seed < team2_seed else team1_name
            higher_seed_win_prob = win_probability if team1_seed < team2_seed else (1 - win_probability)
            upset_prob = 1 - higher_seed_win_prob
            
            # Historical context
            if (higher_seed, lower_seed) in seed_matchups:
                historical_win_rate = seed_matchups[(higher_seed, lower_seed)]
                historical_upset_prob = 1 - historical_win_rate
                st.markdown("---")
                st.subheader("Historical Context")
                st.write(f"Historically, #{higher_seed} seeds win {historical_win_rate*100:.1f}% of games against #{lower_seed} seeds")
                
                if upset_prob > historical_upset_prob:
                    upset_likelihood = upset_prob - historical_upset_prob
                    if upset_likelihood >= 0.04:
                        st.warning(f"⚠️ Upset Alert: {upset_likelihood*100:.1f}% higher upset chance than historical average")
                        if upset_prob > 0.5:
                            st.info(f"📊 Model favors {lower_seed_team} (#{lower_seed} seed) to win!")
                elif upset_prob < historical_upset_prob * 0.5:
                    st.info("🔒 Matchup appears safer than historical upset rates")
            else:
                st.markdown("---")
                st.subheader("Upset Potential")
                if upset_prob > 0.25:
                    if upset_prob > 0.5:
                        st.warning(f"⚠️ Major Upset Alert: Model favors {lower_seed_team} (#{lower_seed} seed) to win!")
                    else:
                        st.warning(f"⚠️ Upset Alert: {lower_seed_team} (#{lower_seed} seed) has {upset_prob*100:.1f}% upset chance")
                else:
                    st.info(f"🔒 {higher_seed_team} (#{higher_seed} seed) is strongly favored")
        
        with col2:
            st.subheader("Key Matchup Factors")
            # Show top 5 difference features
            diffs = create_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed).iloc[0]
            top5 = diffs.abs().nlargest(5).index
            for feature in top5:
                value = diffs[feature]
                if feature == 'Seed_diff':
                    st.write(f"• Seed Difference: {value:.0f}")
                else:
                    base_name = feature.replace('_diff','')
                    team_advantage = team1_name if value > 0 else team2_name
                    st.write(f"• {base_name}: Advantage to {team_advantage} ({value:+.2f})")
            
            display_tempo_analysis(team1_name, team2_name, team1_stats, team2_stats, win_probability)
    
    # Side-by-side comparison in TABS
    st.header("Team Comparison")
    st.markdown("Compare stats across different categories in the tabs below.")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Offensive Stats", "Defensive Stats", "Performance Metrics"])

    # Build advantage-based tables for each category
    with tab1:
        st.subheader("Overview")
        overview_df = build_advantage_table(team1_stats, team2_stats, team1_name, team2_name, OVERVIEW_METRICS)
        overview_styled = style_comparison_table(overview_df, team1_name, team2_name)
        st.write('<div class="comparison-table">' + overview_styled.to_html() + '</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Offensive Stats")
        off_df = build_advantage_table(team1_stats, team2_stats, team1_name, team2_name, OFFENSIVE_METRICS)
        off_styled = style_comparison_table(off_df, team1_name, team2_name)
        st.write('<div class="comparison-table">' + off_styled.to_html() + '</div>', unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Defensive Stats")
        def_df = build_advantage_table(team1_stats, team2_stats, team1_name, team2_name, DEFENSIVE_METRICS)
        def_styled = style_comparison_table(def_df, team1_name, team2_name)
        st.write('<div class="comparison-table">' + def_styled.to_html() + '</div>', unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Performance Metrics")
        perf_df = build_advantage_table(team1_stats, team2_stats, team1_name, team2_name, PERFORMANCE_METRICS)
        perf_styled = style_comparison_table(perf_df, team1_name, team2_name)
        st.write('<div class="comparison-table">' + perf_styled.to_html() + '</div>', unsafe_allow_html=True)
    
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
