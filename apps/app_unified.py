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
import json
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now use regular import
from assets.basketball_logo import get_logo_html, create_basketball_logo
from mobile_utils import is_mobile, inject_mobile_js

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="BracketBrain",
    layout="wide",
    page_icon=Image.open(BytesIO(base64.b64decode(create_basketball_logo()))),
    initial_sidebar_state="expanded"
)

# Add Google Search Console verification meta tag
st.markdown('<meta name="google-site-verification" content="MvzMZFiCbHt8maCA4Ev7yoOnFNDqoZcPxSWelmwRcjQ" />', unsafe_allow_html=True)

# Inject mobile detection JavaScript
inject_mobile_js()

creds_json = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
print(creds_json)
creds_dict = json.loads(creds_json)

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
# st.write(f"Checking CSS Path: {css_path}")

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
.dataframe-container {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    margin-bottom: 15px;
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
        {get_logo_html(size=50 if not is_mobile() else 40)}
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
        model = joblib.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model_no_seeds.pkl")))
        return model
    except Exception as e:
        st.error(f"Error loading basic model: {str(e)}")
        return None

@st.cache_resource
def load_basic_kenpom_model():
    try:
        model = joblib.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model_no_seeds_kenpom.pkl")))
        return model
    except Exception as e:
        st.error(f"Error loading basic model with KenPom: {str(e)}")
        return None

@st.cache_resource
def load_enhanced_model():
    try:
        model = joblib.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "final_model_py2.pkl")))
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

    # Load the CSV if it exists
    if os.path.exists(csv_path):
        teams_df = pd.read_csv(csv_path)
    else:
        st.error("❌ `MTeams.csv` not found! Please check file path and deployment.")

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pre_tourney_data", "TeamSeasonAverages_with_SoS.csv"))

    # Load the CSV if it exists
    if os.path.exists(csv_path):
        basic_stats = pd.read_csv(csv_path)
    else:
        st.error("❌ `TeamSeasonAverages_with_SoS.csv` not found! Please check file path and deployment.")

    
    # Load enhanced stats
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pre_tourney_data", "EnhancedTournamentStats.csv"))

    # Load the CSV if it exists
    if os.path.exists(csv_path):
        enhanced_stats = pd.read_csv(csv_path)
    else:
        st.error("❌ `EnhancedTournamentStats.csv` not found! Please check file path and deployment.")
            
    # Load KenPom rankings
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pre_tourney_data", "KenPom-Rankings-Updated.csv"))

    # Load the CSV if it exists
    if os.path.exists(csv_path):
        kenpom_rankings = pd.read_csv(csv_path)
        kenpom_rankings = kenpom_rankings[kenpom_rankings['Season'] == 2025]
        kenpom_rankings = kenpom_rankings.rename(columns={'OrdinalRank': 'KenPom'})
    else:
        st.error("❌ `KenPom-Rankings-Updated.csv` not found! Please check file path and deployment.")
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

@st.cache_data
def load_betting_odds_data():
    """Load NCAA tournament betting odds from CSV file"""
    try:
        odds_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pre_tourney_data", "ncaa_tournament_odds_latest.csv"))
        if os.path.exists(odds_path):
            odds_df = pd.read_csv(odds_path)
            print(f"Loaded betting odds data with {len(odds_df)} entries")
            return odds_df
        else:
            st.warning("⚠️ NCAA Tournament odds data not found. Betting analysis will not be available.")
            return None
    except Exception as e:
        st.warning(f"Could not load betting odds data: {str(e)}")
        return None

def american_odds_to_probability(american_odds):
    """Convert American odds to implied probability"""
    if pd.isna(american_odds):
        return None
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def display_betting_odds(team1_id, team1_name, team1_seed, team2_id, team2_name, team2_seed, prediction_prob=None):
    """
    Display betting odds analysis for the matchup, with model comparison
    
    Parameters:
    - team1_id: Team 1 ID
    - team1_name: Team 1 name
    - team1_seed: Team 1 seed
    - team2_id: Team 2 ID
    - team2_name: Team 2 name
    - team2_seed: Team 2 seed
    - prediction_prob: Model's prediction probability of team1 winning (optional)
    """
    st.markdown("---")
    st.subheader("💰 Betting Odds Analysis")
    
    # Load betting odds data
    odds_df = load_betting_odds_data()
    
    if odds_df is None or len(odds_df) == 0:
        st.info("📊 Betting odds data is not available. Run the `fetch_daily_odds.py` script to fetch the latest NCAA Tournament odds.")
        return
    
    # Convert team IDs to integers for comparison if they're not already
    try:
        team1_id = int(team1_id)
        team2_id = int(team2_id)
    except (TypeError, ValueError):
        # If conversion fails, continue with original values
        pass
    
    # Filter for relevant games involving either team
    team1_games = odds_df[(odds_df['team_id'] == team1_id) | 
                         (odds_df['home_team_id'] == team1_id) | 
                         (odds_df['away_team_id'] == team1_id)]
    
    team2_games = odds_df[(odds_df['team_id'] == team2_id) | 
                         (odds_df['home_team_id'] == team2_id) | 
                         (odds_df['away_team_id'] == team2_id)]
    
    # Find games where both teams are playing each other
    common_game_ids = set(team1_games['game_id']).intersection(set(team2_games['game_id']))
    
    if not common_game_ids:
        st.info(f"📊 No betting odds found for {team1_name} vs {team2_name}. The betting markets may not be open yet for this matchup.")
        return
    
    # Get the odds for this specific matchup
    matchup_odds = odds_df[odds_df['game_id'].isin(common_game_ids)]
    
    # Filter to get moneyline odds for each team
    team1_ml = matchup_odds[(matchup_odds['team_id'] == team1_id) & 
                          (matchup_odds['market_type'] == 'moneyline')]
    team2_ml = matchup_odds[(matchup_odds['team_id'] == team2_id) & 
                          (matchup_odds['market_type'] == 'moneyline')]
    
    # Filter to get spread odds for each team
    team1_spread = matchup_odds[(matchup_odds['team_id'] == team1_id) & 
                              (matchup_odds['market_type'] == 'spread')]
    team2_spread = matchup_odds[(matchup_odds['team_id'] == team2_id) & 
                              (matchup_odds['market_type'] == 'spread')]
    
    # Display game information
    if not team1_ml.empty:
        game_date = team1_ml['game_date'].iloc[0]
        game_time = team1_ml['game_time'].iloc[0]
        sportsbook = team1_ml['sportsbook'].iloc[0]
        
        # Calculate vig early if available
        vig_info = ""
        if not team1_ml.empty and not team2_ml.empty:
            try:
                team1_odds = team1_ml['price'].iloc[0]
                team2_odds = team2_ml['price'].iloc[0]
                team1_implied_prob = american_odds_to_probability(team1_odds) * 100
                team2_implied_prob = american_odds_to_probability(team2_odds) * 100
                total_implied = team1_implied_prob + team2_implied_prob
                vig = total_implied - 100
                vig_info = f" | Vig: {vig:.1f}% | Data from: {sportsbook}"
            except Exception:
                vig_info = f" | Data from: {sportsbook}"
        
        st.write(f"**Game Time:** {game_date} at {game_time}{vig_info}")
    
    # Create two columns for display
    if not is_mobile():
        col1, col2 = st.columns([1, 1])
    else:
        col1 = st.container()
        col2 = st.container()
    
    # Flag to track if we have valid odds data
    has_valid_odds = False
    
    with col1:
        st.write("**🎲 Moneyline Odds**")
        if not team1_ml.empty and not team2_ml.empty:
            try:
                # Calculate implied probabilities
                team1_odds = team1_ml['price'].iloc[0]
                team2_odds = team2_ml['price'].iloc[0]
                
                team1_implied_prob = american_odds_to_probability(team1_odds) * 100
                team2_implied_prob = american_odds_to_probability(team2_odds) * 100
                
                # Calculate the overround/vig
                total_implied = team1_implied_prob + team2_implied_prob
                vig = total_implied - 100
                
                # Calculate fair probabilities (remove the vig)
                team1_fair_prob = (team1_implied_prob / total_implied) * 100
                team2_fair_prob = (team2_implied_prob / total_implied) * 100
                
                # Format the odds for display
                team1_odds_str = f"+{team1_odds}" if team1_odds > 0 else f"{team1_odds}"
                team2_odds_str = f"+{team2_odds}" if team2_odds > 0 else f"{team2_odds}"
                
                # Create a dataframe for display
                ml_data = {
                    'Team': [team1_name, team2_name],
                    'Odds': [team1_odds_str, team2_odds_str],
                    'Implied %': [f"{team1_implied_prob:.1f}%", f"{team2_implied_prob:.1f}%"],
                    'Fair %': [f"{team1_fair_prob:.1f}%", f"{team2_fair_prob:.1f}%"]
                }
                ml_df = pd.DataFrame(ml_data)
                
                # Highlight the favorite
                def style_ml_odds(val):
                    if val in [team1_odds_str, team2_odds_str]:
                        if val == team1_odds_str and team1_odds < team2_odds:
                            return 'background-color: rgba(0, 255, 0, 0.1); font-weight: bold'
                        elif val == team2_odds_str and team2_odds < team1_odds:
                            return 'background-color: rgba(0, 255, 0, 0.1); font-weight: bold'
                    return ''
                
                styled_ml_df = ml_df.style.map(style_ml_odds, subset=['Odds'])
                st.dataframe(styled_ml_df, hide_index=True)
                
                # Add explanation
                with st.expander("How to read moneyline odds"):
                    st.write("""
                    - **American Odds** show how much you'd win on a 100 dollar bet (if positive) or how much you need to bet to win $100 (if negative).
                    - **Implied %** is the probability of winning implied by the odds.
                    - **Fair %** is the implied probability with the bookmaker's margin removed.
                    - The team with the lower (more negative) odds is the favorite.
                    """)
                
                has_valid_odds = True
            except Exception as e:
                st.warning(f"Error processing moneyline odds: {str(e)}")
        else:
            st.info("Moneyline odds not available for this matchup.")
    
    with col2:
        st.write("**📏 Point Spread**")
        if not team1_spread.empty and not team2_spread.empty:
            try:
                # Get the spread and odds
                team1_spread_line = team1_spread['handicap'].iloc[0]
                team1_spread_odds = team1_spread['price'].iloc[0]
                team2_spread_line = team2_spread['handicap'].iloc[0]
                team2_spread_odds = team2_spread['price'].iloc[0]
                
                # Format for display
                team1_spread_str = f"{team1_spread_line:+g}" if not pd.isna(team1_spread_line) else "N/A"
                team2_spread_str = f"{team2_spread_line:+g}" if not pd.isna(team2_spread_line) else "N/A"
                team1_spread_odds_str = f"+{team1_spread_odds}" if team1_spread_odds > 0 else f"{team1_spread_odds}"
                team2_spread_odds_str = f"+{team2_spread_odds}" if team2_spread_odds > 0 else f"{team2_spread_odds}"
                
                # Create a dataframe for display
                spread_data = {
                    'Team': [team1_name, team2_name],
                    'Spread': [team1_spread_str, team2_spread_str],
                    'Odds': [team1_spread_odds_str, team2_spread_odds_str]
                }
                spread_df = pd.DataFrame(spread_data)
                
                # Highlight the favorite
                def style_spread(val):
                    if val in [team1_spread_str, team2_spread_str]:
                        if val == team1_spread_str and team1_spread_line < 0:
                            return 'background-color: rgba(0, 255, 0, 0.1); font-weight: bold'
                        elif val == team2_spread_str and team2_spread_line < 0:
                            return 'background-color: rgba(0, 255, 0, 0.1); font-weight: bold'
                    return ''
                
                styled_spread_df = spread_df.style.map(style_spread, subset=['Spread'])
                st.dataframe(styled_spread_df, hide_index=True)
                
                # Add explanation of how to read the spread
                if not pd.isna(team1_spread_line) and not pd.isna(team2_spread_line):
                    favorite = team1_name if team1_spread_line < 0 else team2_name
                    underdog = team2_name if team1_spread_line < 0 else team1_name
                    spread_value = abs(min(team1_spread_line, team2_spread_line))
                
                # Add explanation
                with st.expander("How to read point spreads"):
                    st.write("""
                    - A negative spread (e.g., -5.5) means that team is favored to win by that many points.
                    - A positive spread (e.g., +5.5) means that team can lose by up to that many points and still "cover" the spread.
                    - If you bet on a -5.5 spread, your team must win by 6 or more points for you to win the bet.
                    - If you bet on a +5.5 spread, your team can lose by 5 or fewer points (or win outright) for you to win the bet.
                    """)
            except Exception as e:
                st.warning(f"Error processing spread odds: {str(e)}")
        else:
            st.info("Spread odds not available for this matchup.")
    
    # Model Edge Analysis - if we have a prediction and valid odds data
    if prediction_prob is not None and has_valid_odds:
        st.markdown("---")
        st.subheader("📊 Model vs. Market Analysis")
        
        if not team1_ml.empty and not team2_ml.empty:
            try:
                # Get model probabilities
                model_team1_prob = prediction_prob * 100
                model_team2_prob = (1 - prediction_prob) * 100
                
                # Calculate edge (model probability - fair implied probability)
                team1_edge = model_team1_prob - team1_fair_prob
                team2_edge = model_team2_prob - team2_fair_prob
                
                # Display the comparison
                st.write("Our model's prediction vs. what the betting markets imply:")
                
                # Create columns for visualization
                if not is_mobile():
                    prob_col1, prob_col2 = st.columns(2)
                else:
                    prob_col1 = st.container()
                    prob_col2 = st.container()
                
                with prob_col1:
                    st.metric(
                        label=f"{team1_name} Win Probability", 
                        value=f"{model_team1_prob:.1f}%", 
                        delta=f"{team1_edge:+.1f}%" if abs(team1_edge) > 1 else "≈ Market",
                        delta_color="normal"
                    )
                
                with prob_col2:
                    st.metric(
                        label=f"{team2_name} Win Probability", 
                        value=f"{model_team2_prob:.1f}%", 
                        delta=f"{team2_edge:+.1f}%" if abs(team2_edge) > 1 else "≈ Market",
                        delta_color="normal"
                    )
                
                # Create dataframe for comparison
                comparison_data = {
                    'Team': [team1_name, team2_name],
                    'Model Probability': [f"{model_team1_prob:.1f}%", f"{model_team2_prob:.1f}%"],
                    'Market Probability': [f"{team1_fair_prob:.1f}%", f"{team2_fair_prob:.1f}%"],
                    'Edge': [f"{team1_edge:+.1f}%", f"{team2_edge:+.1f}%"]
                }
                comparison_df = pd.DataFrame(comparison_data)
                
                # Add styling to highlight edges
                def style_edge(val):
                    if isinstance(val, str) and '+' in val:
                        edge_val = float(val.replace('+', '').replace('%', ''))
                        if edge_val > 5:
                            return 'background-color: rgba(0, 255, 0, 0.3); font-weight: bold'
                        elif edge_val > 2:
                            return 'background-color: rgba(0, 255, 0, 0.1)'
                    elif isinstance(val, str) and '-' in val:
                        edge_val = float(val.replace('%', ''))
                        if edge_val < -5:
                            return 'background-color: rgba(255, 0, 0, 0.3)'
                        elif edge_val < -2:
                            return 'background-color: rgba(255, 0, 0, 0.1)'
                    return ''
                
                # Apply styling
                styled_df = comparison_df.style.map(style_edge, subset=['Edge'])
                st.dataframe(styled_df, hide_index=True)
                
                # Find maximum edge and display betting value
                max_edge_team1 = team1_edge
                max_edge_team2 = team2_edge
                
                # Calculate implied odds for a bet with our model's probabilities
                def calculate_fair_odds(probability):
                    """Convert probability to fair American odds"""
                    if probability <= 0 or probability >= 1:
                        return None
                    if probability < 0.5:
                        return round(100 / probability - 100)
                    else:
                        return round(-100 * probability / (1 - probability))
                
                team1_fair_odds = calculate_fair_odds(prediction_prob)
                team2_fair_odds = calculate_fair_odds(1 - prediction_prob)
                
                team1_fair_odds_str = f"+{team1_fair_odds}" if team1_fair_odds > 0 else f"{team1_fair_odds}"
                team2_fair_odds_str = f"+{team2_fair_odds}" if team2_fair_odds > 0 else f"{team2_fair_odds}"
                
                # Value bet analysis
                st.markdown("### Value Bet Analysis")
                
                if max_edge_team1 > max_edge_team2 and max_edge_team1 > 2:
                    st.markdown(f"**Value Bet:** {team1_name} with a **{max_edge_team1:+.1f}%** edge")
                    if max_edge_team1 > 5:
                        st.success(f"✅ **Significant Edge Detected:** The model gives {team1_name} a strong edge over the market odds.")
                        st.write(f"**Fair odds:** {team1_fair_odds_str} (Model's estimate) vs **Market odds:** {team1_odds_str}")
                    else:
                        st.info(f"ℹ️ **Small Edge Detected:** The model gives {team1_name} a slight edge over the market odds.")
                        st.write(f"**Fair odds:** {team1_fair_odds_str} (Model's estimate) vs **Market odds:** {team1_odds_str}")
                elif max_edge_team2 > max_edge_team1 and max_edge_team2 > 2:
                    st.markdown(f"**Value Bet:** {team2_name} with a **{max_edge_team2:+.1f}%** edge")
                    if max_edge_team2 > 5:
                        st.success(f"✅ **Significant Edge Detected:** The model gives {team2_name} a strong edge over the market odds.")
                        st.write(f"**Fair odds:** {team2_fair_odds_str} (Model's estimate) vs **Market odds:** {team2_odds_str}")
                    else:
                        st.info(f"ℹ️ **Small Edge Detected:** The model gives {team2_name} a slight edge over the market odds.")
                        st.write(f"**Fair odds:** {team2_fair_odds_str} (Model's estimate) vs **Market odds:** {team2_odds_str}")
                else:
                    st.write("The model's predictions are closely aligned with the betting market odds.")
                    st.write(f"No significant betting edge detected (max edge < 2%).")
                    
                # Add interpretation of the edge
                with st.expander("Understanding the edge"):
                    st.markdown("""
                    **How to Interpret the Edge:**
                    
                    The edge represents the difference between our model's predicted probability and the market implied probability:
                    
                    - **5%+ edge:** Very strong betting opportunity
                    - **2-5% edge:** Potentially valuable betting opportunity
                    - **<2% edge:** No significant edge or value
                    
                    When a positive edge is detected, it means our model believes a team has a better chance of winning than what the betting markets suggest.
                    
                    **Fair odds** represent what the odds should be based on our model's probability. If the market odds are more favorable than our fair odds, there's potentially value in that bet.
                    """)
            except Exception as e:
                st.warning(f"Error in edge analysis: {str(e)}")
        else:
            st.info("Cannot perform edge analysis without moneyline odds.")

# ------------------------------
# Team Summary Functions
# ------------------------------
@st.cache_data
def load_team_summaries():
    """Load team summaries from JSON file"""
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pre_tourney_data", "team_sum_clean.json")), "r") as f:
            team_summaries = json.load(f)
        return team_summaries
    except Exception as e:
        st.warning(f"Could not load team summaries: {str(e)}")
        return {}

def display_team_comparison_summaries(team1_id, team1_name, team2_id, team2_name):
    """Display summaries for both teams side by side or stacked on mobile"""
    summaries = load_team_summaries()
    
    if is_mobile():
        # Stacked layout for mobile
        st.markdown(f"### {team1_name}")
        if str(team1_id) in summaries:
            st.markdown(f"<div class='team-profile'>{summaries[str(team1_id)]}</div>", unsafe_allow_html=True)
        else:
            st.info(f"No detailed profile available for {team1_name}")
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        st.markdown(f"### {team2_name}")
        if str(team2_id) in summaries:
            st.markdown(f"<div class='team-profile'>{summaries[str(team2_id)]}</div>", unsafe_allow_html=True)
        else:
            st.info(f"No detailed profile available for {team2_name}")
    else:
        # Side-by-side layout for desktop
        col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {team1_name}")
        if str(team1_id) in summaries:
            st.write(summaries[str(team1_id)])
        else:
            st.info(f"No detailed profile available for {team1_name}")
    
    with col2:
        st.markdown(f"### {team2_name}")
        if str(team2_id) in summaries:
            st.write(summaries[str(team2_id)])
        else:
            st.info(f"No detailed profile available for {team2_name}")

# ------------------------------
# Feature Creation Functions
# ------------------------------
def create_basic_features(team1_stats, team2_stats, team1_seed, team2_seed):
    """Create features for the basic XGBoost model that doesn't use seeds"""
    season_cols = ['WinPct', 'Avg_Score', 'Avg_FGM', 'Avg_FGA', 'Avg_FGM3', 'Avg_FGA3',
                   'Avg_FTM', 'Avg_FTA', 'Avg_OR', 'Avg_DR', 'Avg_Ast', 'Avg_TO',
                   'Avg_Stl', 'Avg_Blk', 'Avg_PF', 'Avg_Opp_WinPct', 'Last30_WinRatio']
    
    features = {}
    for col in season_cols:
        features[f"{col}_diff"] = team1_stats[col] - team2_stats[col]
    
    # Create SoS emphasis features
    features['SoS_squared_diff'] = features['Avg_Opp_WinPct_diff'] ** 2 * np.sign(features['Avg_Opp_WinPct_diff'])
    features['SoS_WinPct_interaction'] = team1_stats['Avg_Opp_WinPct'] * team1_stats['WinPct'] - team2_stats['Avg_Opp_WinPct'] * team2_stats['WinPct']
    features['SoS_Last30_interaction'] = team1_stats['Avg_Opp_WinPct'] * team1_stats['Last30_WinRatio'] - team2_stats['Avg_Opp_WinPct'] * team2_stats['Last30_WinRatio']
    
    return pd.DataFrame([features])

def create_basic_kenpom_features(team1_stats, team2_stats, team1_seed, team2_seed):
    """Create features for the basic XGBoost model with KenPom that doesn't use seeds"""
    # Start with the basic features
    season_cols = ['WinPct', 'Avg_Score', 'Avg_FGM', 'Avg_FGA', 'Avg_FGM3', 'Avg_FGA3',
                   'Avg_FTM', 'Avg_FTA', 'Avg_OR', 'Avg_DR', 'Avg_Ast', 'Avg_TO',
                   'Avg_Stl', 'Avg_Blk', 'Avg_PF', 'Avg_Opp_WinPct', 'Last30_WinRatio']
    
    features = {}
    for col in season_cols:
        features[f"{col}_diff"] = team1_stats[col] - team2_stats[col]
    
    # Add KenPom difference (note: lower rank is better, so we subtract Team1 from Team2)
    if 'KenPom' in team1_stats and 'KenPom' in team2_stats:
        features['KenPom_diff'] = team2_stats['KenPom'] - team1_stats['KenPom']
    else:
        features['KenPom_diff'] = 0
    
    # Create SoS emphasis features
    features['SoS_squared_diff'] = features['Avg_Opp_WinPct_diff'] ** 2 * np.sign(features['Avg_Opp_WinPct_diff'])
    features['SoS_WinPct_interaction'] = team1_stats['Avg_Opp_WinPct'] * team1_stats['WinPct'] - team2_stats['Avg_Opp_WinPct'] * team2_stats['WinPct']
    features['SoS_Last30_interaction'] = team1_stats['Avg_Opp_WinPct'] * team1_stats['Last30_WinRatio'] - team2_stats['Avg_Opp_WinPct'] * team2_stats['Last30_WinRatio']
    
    # Add KenPom-SoS interaction feature
    features['KenPom_SoS_interaction'] = features['KenPom_diff'] * features['Avg_Opp_WinPct_diff']
    
    return pd.DataFrame([features])

def create_enhanced_features(team1_stats, team2_stats, team1_seed, team2_seed):
    """Create features for the enhanced model"""
    # Define the feature order as expected by the enhanced model
    features = {
        'SeedDiff': team1_seed - team2_seed,
    }
    
    # Add KenPom difference if available
    if 'KenPom' in team1_stats and 'KenPom' in team2_stats:
        features['KenPomDiff'] = team2_stats['KenPom'] - team1_stats['KenPom']
    
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
    """Make prediction using the basic XGBoost model that doesn't use seeds"""
    # Create features
    X = create_basic_features(team1_stats, team2_stats, team1_seed, team2_seed)
    dmatrix = xgb.DMatrix(X)
    team1_win_prob = model.predict(dmatrix)[0]
    
    # Create features with teams swapped to ensure symmetry
    X_swapped = create_basic_features(team2_stats, team1_stats, team2_seed, team1_seed)
    dmatrix_swapped = xgb.DMatrix(X_swapped)
    team2_win_prob = model.predict(dmatrix_swapped)[0]
    
    # Average the two predictions (1 - team2_win_prob gives team1's win probability from swapped perspective)
    final_team1_win_prob = (team1_win_prob + (1 - team2_win_prob)) / 2
    
    # For debugging
    print(f"Team1 win prob: {team1_win_prob:.4f}")
    print(f"1 - Team2 win prob: {1 - team2_win_prob:.4f}")
    print(f"Difference: {abs(team1_win_prob - (1 - team2_win_prob)):.4f}")
    print(f"Symmetrized prob: {final_team1_win_prob:.4f}")
    
    return final_team1_win_prob

def predict_with_basic_kenpom_model(model, team1_stats, team2_stats, team1_seed, team2_seed):
    """Make prediction using the basic XGBoost model with KenPom that doesn't use seeds"""
    # Create features
    X = create_basic_kenpom_features(team1_stats, team2_stats, team1_seed, team2_seed)
    dmatrix = xgb.DMatrix(X)
    team1_win_prob = model.predict(dmatrix)[0]
    
    # Create features with teams swapped to ensure symmetry
    X_swapped = create_basic_kenpom_features(team2_stats, team1_stats, team2_seed, team1_seed)
    dmatrix_swapped = xgb.DMatrix(X_swapped)
    team2_win_prob = model.predict(dmatrix_swapped)[0]
    
    # For debugging
    print(f"Team1 win prob: {team1_win_prob:.4f}")
    print(f"1 - Team2 win prob: {1 - team2_win_prob:.4f}")
    print(f"Difference: {abs(team1_win_prob - (1 - team2_win_prob)):.4f}")
    
    # Average the two predictions
    final_team1_win_prob = (team1_win_prob + (1 - team2_win_prob)) / 2
    
    return final_team1_win_prob

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
    
    # Get the safe column names from DataFrame attributes
    safe_team1_name = df.attrs.get('safe_team1_name', "Team1")
    safe_team2_name = df.attrs.get('safe_team2_name', "Team2")
    
    # Rename the columns before styling
    styled_df = styled_df.rename(columns={
        safe_team1_name: team1_name,
        safe_team2_name: team2_name
    })
    
    # Store original numeric values for calculations
    original_values = {
        team1_name: styled_df[team1_name].astype(float),
        team2_name: styled_df[team2_name].astype(float)
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
    styled = styled_df.style.apply(apply_color, axis=1)
    
    return styled

def build_comparison_table(team1_stats, team2_stats, team1_name, team2_name, metrics, metric_names=None, lower_is_better=None):
    """Build a comparison table for the given metrics"""
    if metric_names is None:
        metric_names = metrics
    
    if lower_is_better is None:
        lower_is_better = []
    
    # For mobile, use shorter names
    if is_mobile():
        shortened_names = []
        for name in metric_names:
            # Shorten long metric names on mobile
            if len(name) > 15:
                shortened_names.append(name[:12] + "...")
            else:
                shortened_names.append(name)
        metric_names = shortened_names
    
    # Create safe column names to avoid issues with spaces or special characters
    safe_team1_name = "Team1"
    safe_team2_name = "Team2"
    
    data = []
    for metric, name in zip(metrics, metric_names):
        if metric in team1_stats and metric in team2_stats:
            # Determine which team has the advantage
            is_lower_better = metric in lower_is_better
            team1_better = team1_stats[metric] < team2_stats[metric] if is_lower_better else team1_stats[metric] > team2_stats[metric]
            
            data.append({
                'Metric': name,
                safe_team1_name: f"{team1_stats[metric]:.2f}",
                safe_team2_name: f"{team2_stats[metric]:.2f}",
                'Advantage': team1_name if team1_better else team2_name
            })
    
    df = pd.DataFrame(data)
    # Store original team names as attributes
    df.attrs['team1_name'] = team1_name
    df.attrs['team2_name'] = team2_name
    df.attrs['safe_team1_name'] = safe_team1_name
    df.attrs['safe_team2_name'] = safe_team2_name
    
    return df

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
    
    # Load bracket data
    bracket_data = load_bracket_data()
    
    # Add KenPom data to basic stats for the KenPom model
    if 'KenPom' not in current_basic_stats.columns and 'KenPom' in current_enhanced_stats.columns:
        current_basic_stats_with_kenpom = current_basic_stats.copy()
        kenpom_data = current_enhanced_stats[['TeamID', 'KenPom']].drop_duplicates()
        current_basic_stats_with_kenpom = current_basic_stats_with_kenpom.merge(
            kenpom_data, on='TeamID', how='left'
        )
        # Fill NaN values in KenPom column
        current_basic_stats_with_kenpom['KenPom'] = current_basic_stats_with_kenpom['KenPom'].fillna(0)
    else:
        current_basic_stats_with_kenpom = current_basic_stats.copy()
        if 'KenPom' not in current_basic_stats_with_kenpom.columns:
            current_basic_stats_with_kenpom['KenPom'] = 0
    
    # Sidebar for model selection and other settings
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select Prediction Model",
        ["Basic Model", "Basic Model + KenPom", "Enhanced Model"],
        index=0,  # Default to Basic Model
        help="Basic Model uses season averages and strength of schedule. Basic Model + KenPom adds KenPom rankings. Enhanced Model includes additional advanced metrics."
    )
    
    # Add option to show betting odds
    show_betting_odds = st.sidebar.checkbox("Show Betting Odds Analysis", value=False, 
                                           help="Display current betting odds and analysis for this matchup")
    
    # Load the selected model
    if model_choice == "Basic Model":
        model = load_basic_model()
        current_stats = current_basic_stats
        st.sidebar.info("Using basic model with season averages and strength of schedule")
    elif model_choice == "Basic Model + KenPom":
        model = load_basic_kenpom_model()
        current_stats = current_basic_stats_with_kenpom
        st.sidebar.info("Using basic model with season averages, strength of schedule, and KenPom rankings")
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
    
    # Get team stats
    team1_stats = current_stats[current_stats['TeamName'] == team1_name].iloc[0]
    team1_id = team1_stats['TeamID']
    
    # Check if team is in the bracket and get its seed
    team1_seed_locked = False
    if str(team1_id) in bracket_data:
        team1_seed = bracket_data[str(team1_id)]["seed"]
        team1_region = bracket_data[str(team1_id)]["region"]
        st.sidebar.info(f"{team1_name} is #{team1_seed} seed in the {team1_region} region")
        team1_seed_locked = True
    else:
        team1_seed = st.sidebar.number_input("Team 1 Seed", min_value=1, max_value=16, value=1, step=1)
    
    # Team 2 selection
    team2_name = st.sidebar.selectbox("Select Team 2", current_teams, index=1)
    
    # Get team stats
    team2_stats = current_stats[current_stats['TeamName'] == team2_name].iloc[0]
    team2_id = team2_stats['TeamID']
    
    # Check if team is in the bracket and get its seed
    team2_seed_locked = False
    if str(team2_id) in bracket_data:
        team2_seed = bracket_data[str(team2_id)]["seed"]
        team2_region = bracket_data[str(team2_id)]["region"]
        st.sidebar.info(f"{team2_name} is #{team2_seed} seed in the {team2_region} region")
        team2_seed_locked = True
    else:
        team2_seed = st.sidebar.number_input("Team 2 Seed", min_value=1, max_value=16, value=8, step=1)
    
    # Create a placeholder for team summaries
    team_summaries_placeholder = st.empty()
    
    # Button to make prediction
    if st.sidebar.button("Predict Winner"):
        if model is None:
            st.error("Model not loaded. Please check if the model file exists.")
            return
        
        # Make prediction based on selected model
        if model_choice == "Basic Model":
            win_probability = predict_with_basic_model(model, team1_stats, team2_stats, team1_seed, team2_seed)
        elif model_choice == "Basic Model + KenPom":
            win_probability = predict_with_basic_kenpom_model(model, team1_stats, team2_stats, team1_seed, team2_seed)
        else:
            win_probability = predict_with_enhanced_model(model, team1_stats, team2_stats, team1_seed, team2_seed)
        
        # Display prediction - adjust layout based on mobile or desktop
        if is_mobile():
            # Single column layout for mobile
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
            
            st.markdown("---")
            
            if (higher_seed, lower_seed) in seed_matchups:
                historical_win_rate = seed_matchups[(higher_seed, lower_seed)]
                historical_upset_prob = 1 - historical_win_rate
                
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
                
                # Show upset alert only if lower seed has >30% chance to win
                if upset_prob > 0.3:
                    if upset_prob > 0.5:
                        st.warning(f"⚠️ Major Upset Alert: The model favors {lower_seed_team} (#{lower_seed} seed) to win against {higher_seed_team} (#{higher_seed} seed)!")
                    else:
                        st.warning(f"⚠️ Potential Upset: {lower_seed_team} (#{lower_seed} seed) has a {upset_prob*100:.1f}% chance to upset {higher_seed_team} (#{higher_seed} seed)")
                elif higher_seed_win_prob > 0.85:
                    st.info(f"🔒 Lock Alert: {higher_seed_team} (#{higher_seed} seed) is heavily favored with a {higher_seed_win_prob*100:.1f}% chance to win")
                else:
                    st.info(f"{higher_seed_team} (#{higher_seed} seed) is favored with a {higher_seed_win_prob*100:.1f}% chance to win")
            
            st.markdown("---")
            
            st.subheader("Key Matchup Factors")
            
            # Display feature differences based on model
            if model_choice == "Basic Model":
                features = create_basic_features(team1_stats, team2_stats, team1_seed, team2_seed)
            elif model_choice == "Basic Model + KenPom":
                features = create_basic_kenpom_features(team1_stats, team2_stats, team1_seed, team2_seed)
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
                elif feature == 'SoS_squared_diff':
                    st.write(f"• Strength of Schedule (squared): {'Advantage to ' + team1_name if value > 0 else 'Advantage to ' + team2_name}")
                elif feature == 'SoS_WinPct_interaction':
                    st.write(f"• SoS-adjusted Win %: {'Advantage to ' + team1_name if value > 0 else 'Advantage to ' + team2_name}")
                elif feature == 'SoS_Last30_interaction':
                    st.write(f"• SoS-adjusted Recent Form: {'Advantage to ' + team1_name if value > 0 else 'Advantage to ' + team2_name}")
                elif feature == 'Avg_Opp_WinPct_diff':
                    st.write(f"• Strength of Schedule: {'Advantage to ' + team1_name if value > 0 else 'Advantage to ' + team2_name}")
                else:
                    feature_name = feature.replace('Diff_', '').replace('_diff', '')
                    team_advantage = team1_name if value > 0 else team2_name
                    st.write(f"• {feature_name}: Advantage to {team_advantage}")
            
            # Display tempo analysis
            display_tempo_analysis(team1_name, team2_name, team1_stats, team2_stats, win_probability)
        else:
            # Two column layout for desktop (original code)
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
                
                # Show upset alert only if lower seed has >30% chance to win
                if upset_prob > 0.3:
                    if upset_prob > 0.5:
                        st.warning(f"⚠️ Major Upset Alert: The model favors {lower_seed_team} (#{lower_seed} seed) to win against {higher_seed_team} (#{higher_seed} seed)!")
                    else:
                        st.warning(f"⚠️ Potential Upset: {lower_seed_team} (#{lower_seed} seed) has a {upset_prob*100:.1f}% chance to upset {higher_seed_team} (#{higher_seed} seed)")
                elif higher_seed_win_prob > 0.85:
                    st.info(f"🔒 Lock Alert: {higher_seed_team} (#{higher_seed} seed) is heavily favored with a {higher_seed_win_prob*100:.1f}% chance to win")
                else:
                    st.info(f"{higher_seed_team} (#{higher_seed} seed) is favored with a {higher_seed_win_prob*100:.1f}% chance to win")
        
        with col2:
            st.subheader("Key Matchup Factors")
            
            # Display feature differences based on model
            if model_choice == "Basic Model":
                features = create_basic_features(team1_stats, team2_stats, team1_seed, team2_seed)
            elif model_choice == "Basic Model + KenPom":
                features = create_basic_kenpom_features(team1_stats, team2_stats, team1_seed, team2_seed)
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
                elif feature == 'SoS_squared_diff':
                    st.write(f"• Strength of Schedule (squared): {'Advantage to ' + team1_name if value > 0 else 'Advantage to ' + team2_name}")
                elif feature == 'SoS_WinPct_interaction':
                    st.write(f"• SoS-adjusted Win %: {'Advantage to ' + team1_name if value > 0 else 'Advantage to ' + team2_name}")
                elif feature == 'SoS_Last30_interaction':
                    st.write(f"• SoS-adjusted Recent Form: {'Advantage to ' + team1_name if value > 0 else 'Advantage to ' + team2_name}")
                elif feature == 'Avg_Opp_WinPct_diff':
                    st.write(f"• Strength of Schedule: {'Advantage to ' + team1_name if value > 0 else 'Advantage to ' + team2_name}")
                else:
                    feature_name = feature.replace('Diff_', '').replace('_diff', '')
                    team_advantage = team1_name if value > 0 else team2_name
                    st.write(f"• {feature_name}: Advantage to {team_advantage}")
            
            # Display tempo analysis
            display_tempo_analysis(team1_name, team2_name, team1_stats, team2_stats, win_probability)
        
        # Now display team summaries after prediction
        with team_summaries_placeholder:
            st.header("Team Profiles")
            display_team_comparison_summaries(team1_id, team1_name, team2_id, team2_name)
        
        # Display betting odds analysis if selected
        if show_betting_odds:
            display_betting_odds(team1_id, team1_name, team1_seed, team2_id, team2_name, team2_seed, win_probability)
    
    # Display team comparisons
    st.header("Team Comparison")
    
    # Create tabs for different stat categories
    if is_mobile():
        # For mobile, use a more compact layout with fewer tabs
        tabs = st.tabs(["Overview", "Off/Def", "Advanced"])
        
        with tabs[0]:
            # Overview tab
            metrics = ['WinPct', 'Last30_WinRatio', 'Avg_Score', 'Avg_Opp_Score', 'Avg_Opp_WinPct', 'SoS']
            metric_names = ['Win %', 'Last 30 Win %', 'Points Scored', 'Points Allowed', 'Opponent Win %', 'Strength of Schedule']
            lower_is_better = ['Avg_Opp_Score']
            
            try:
                df = build_comparison_table(team1_stats, team2_stats, team1_name, team2_name, metrics, metric_names, lower_is_better)
                styled_df = style_comparison_table(df, team1_name, team2_name)
                
                # Wrap table in a container for mobile scrolling
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(styled_df, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying comparison table: {str(e)}")
        
        with tabs[1]:
            # Offensive/Defensive tab
            metrics = ['Avg_Score', 'Avg_Opp_Score', 'FG_Pct', 'Opp_FG_Pct', 'FG3_Pct', 'Opp_FG3_Pct']
            metric_names = ['Points Scored', 'Points Allowed', 'FG%', 'Opp FG%', '3PT%', 'Opp 3PT%']
            lower_is_better = ['Avg_Opp_Score', 'Opp_FG_Pct', 'Opp_FG3_Pct']
            
            try:
                df = build_comparison_table(team1_stats, team2_stats, team1_name, team2_name, metrics, metric_names, lower_is_better)
                styled_df = style_comparison_table(df, team1_name, team2_name)
                
                # Wrap table in a container for mobile scrolling
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(styled_df, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying comparison table: {str(e)}")
        
        with tabs[2]:
            # Advanced tab - check if KenPom is available
            if 'KenPom' in team1_stats and 'KenPom' in team2_stats:
                metrics = ['KenPom', 'SoS', 'Avg_Opp_WinPct']
                metric_names = ['KenPom Rank', 'Strength of Schedule', 'Opponent Win %']
                lower_is_better = ['KenPom']  # Lower KenPom rank is better
                
                try:
                    df = build_comparison_table(team1_stats, team2_stats, team1_name, team2_name, metrics, metric_names, lower_is_better)
                    styled_df = style_comparison_table(df, team1_name, team2_name)
                    
                    # Wrap table in a container for mobile scrolling
                    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                    st.dataframe(styled_df, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying comparison table: {str(e)}")
            else:
                st.info("Advanced metrics not available for this model selection.")
    else:
        # Desktop layout with more tabs (your original code)
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Offensive Stats", "Defensive Stats", "Performance Metrics"])
    
    with tab1:
        # Overview metrics
        if model_choice == "Basic Model":
            overview_metrics = ['WinPct', 'Last30_WinRatio', 'Avg_Opp_WinPct']
            overview_names = ['Win Percentage', 'Last 30 Games Win Ratio', 'Opponent Win Percentage']
            lower_is_better = ['Avg_Opp_WinPct']
        elif model_choice == "Basic Model + KenPom":
            overview_metrics = ['WinPct', 'Last30_WinRatio', 'Avg_Opp_WinPct']
            overview_names = ['Win Percentage', 'Last 30 Games Win Ratio', 'Opponent Win Percentage']
            if 'KenPom' in team1_stats and 'KenPom' in team2_stats:
                overview_metrics.append('KenPom')
                overview_names.append('KenPom Ranking')
            lower_is_better = ['Avg_Opp_WinPct', 'KenPom']
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
        if model_choice == "Basic Model" or model_choice == "Basic Model + KenPom":
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
        if model_choice == "Basic Model" or model_choice == "Basic Model + KenPom":
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
        if model_choice == "Basic Model" or model_choice == "Basic Model + KenPom":
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

    st.markdown("---")  # Add a separator line
    
    # Newsletter section with custom styling
    st.markdown("""
        <style>
        .newsletter-container {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 0;
        }
        .newsletter-title {
            color: #1f77b4;
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    def save_email_to_gsheet(email):
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        sheet = client.open("BracketBrain Subscribers").sheet1  
        sheet.append_row([email, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        return "Email saved!"
    
    with st.container():
        st.markdown('<div class="newsletter-container">', unsafe_allow_html=True)
        st.markdown('<div class="newsletter-title">Subscribe to Our Pre-Tournament Analysis</div>', unsafe_allow_html=True)
        
        with st.form("newsletter_form"):
            email = st.text_input("Enter your email", placeholder="your@email.com")
            submit = st.form_submit_button("Subscribe", use_container_width=True)

            if submit:
                if "@" in email and "." in email:
                    save_email_to_gsheet(email)
                    st.success("🎉 Thank you for subscribing!")
                else:
                    st.error("⚠️ Please enter a valid email.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Adjust footer based on mobile or desktop
    if is_mobile():
        footer_html = """
        <div class="footer-mobile">
            <p>BracketBrain v1.0 | Created by Your Name</p>
            <p>Data from NCAA, KenPom, and other sources</p>
            <p><a href="https://givebutter.com/LobMTv" target="_blank" style="color: #ffc107; font-weight: bold;">Support BracketBrain!</a></p>
        </div>
        """
    else:
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
                    <a href="https://givebutter.com/LobMTv" target="_blank" style="color: #ffc107; font-weight: bold;">Support BracketBrain!</a>
                </div>
                <div class="footer-contact">
                    <div>ankitdevalla.dev@gmail.com | lizach739@gmail.com</div>
                    <div>&copy; {datetime.date.today().year} BracketBrain</div>
                </div>
            </div>
        </footer>
        """
    
    st.markdown(footer_html, unsafe_allow_html=True)

# Add this function to load the bracket data
@st.cache_data
def load_bracket_data():
    """Load the 2025 NCAA Tournament bracket data"""
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pre_tourney_data", "2025_Bracket.json")), "r") as f:
            bracket_data = json.load(f)
        return bracket_data
    except Exception as e:
        st.warning(f"Could not load bracket data: {str(e)}")
        return {}

if __name__ == "__main__":
    main() 