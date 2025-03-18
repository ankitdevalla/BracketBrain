# BracketBrain: NCAA March Madness Prediction Tool

BracketBrain is a comprehensive tool for predicting NCAA basketball tournament outcomes using machine learning models. The project provides both command-line tools for model training and data processing, as well as a Streamlit web application that allows users to interactively explore team matchups, predictions, and betting odds analysis.

## Repository Structure

### Core Components

- **apps/**: Contains Streamlit web applications
  - `app_unified.py`: The main Streamlit application that provides interactive predictions, team comparisons, and betting odds analysis
  - `deployment_app.py`: A simplified version of the app for deployment
  - `app.py` & `app2.py`: Earlier versions of the application

- **scripts/**: Contains Python scripts for data processing, model training, and odds fetching
  - `fetch_daily_odds.py`: Fetches the latest NCAA basketball betting odds from sportsbooks API
  - `tournament_prediction.py`: Core script for training tournament prediction models
  - `model_without_seeds.py`: Trains a basic prediction model without using seed information
  - `no_seeds_kenpom.py`: Trains a prediction model using KenPom ratings without seed information
  - `enhanced_stats_calculator.py`: Calculates advanced statistics for teams
  - `add_tempo_predictor.py`: Adds tempo (possessions per 40 min) as a predictor in models
  - Various other scripts for feature engineering and model analysis

- **models/**: Contains trained machine learning models
  - `xgb_model_no_seeds.pkl`: Basic XGBoost model without seed information
  - `xgb_model_no_seeds_kenpom.pkl`: XGBoost model with KenPom data but no seed information
  - `final_model_py2.pkl`: Enhanced production model with advanced features

- **pre_tourney_data/**: Contains pre-processed data files for tournament teams
  - `EnhancedTournamentStats.csv`: Advanced metrics for tournament teams
  - `TeamSeasonAverages_with_SoS.csv`: Season averages with Strength of Schedule
  - `KenPom-Rankings-Updated.csv`: KenPom ratings for teams
  - `2025_Bracket.json`: Tournament bracket information
  - `ncaa_tournament_moneyline_latest.csv`: Latest moneyline betting odds
  - `ncaa_tournament_spread_latest.csv`: Latest point spread betting odds
  - `team_sum_clean.json`: Team summaries and descriptions
  - `mapping.json`: Mapping of team names to their IDs

- **raw_data/**: Contains raw data files used for training
  - `MTeams.csv`: Basic team information
  - NCAA tournament historical data

- **assets/**: Contains assets for the web application
  - `style.css`: Styling for the Streamlit app
  - `basketball_logo.py`: Code to generate basketball logo

### Utility Files

- **mobile_utils.py**: Utilities for mobile responsive design
- **requirements.txt**: Python package dependencies
- **.env**: Environment variables configuration (e.g., API keys)

## Key Features

### Prediction Models

BracketBrain uses multiple prediction models of increasing sophistication:

1. **Basic Model**: Uses team season averages and strength of schedule
2. **Basic Model + KenPom**: Adds KenPom rankings to the basic model
3. **Enhanced Model**: Incorporates advanced metrics, adjusted ratings, and more

### Web Application Features

- **Team Matchup Analysis**: Compare any two NCAA teams with detailed stats
- **Win Probability Prediction**: Get win probability predictions based on selected model
- **Historical Context**: View historical outcomes for similar seed matchups
- **Betting Odds Analysis**: View current betting odds and potential value bets
- **Team Profiles**: Read detailed team summaries and performance analysis
- **Mobile Responsive**: Adapts to both desktop and mobile viewing

### Betting Odds Integration

- **Real-time Odds Fetching**: Fetch latest NCAA basketball betting odds
- **Moneyline and Spread Odds**: View both moneyline and point spread odds
- **Model vs. Market Analysis**: Compare model predictions with implied market probabilities
- **Edge Detection**: Identify potential value bets where the model differs from the market

## Getting Started

### Prerequisites

- Python 3.8 or higher
- API key for sportsgameodds.com (for betting odds functionality)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/MMpred.git
   cd MMpred
   ```

2. Install required dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables
   ```
   # Create a .env file with your API key
   SPORTSGAMEODDS_API_KEY=your_api_key_here
   ```

### Running the Web Application

```bash
streamlit run apps/app_unified.py
```

### Fetching Latest Betting Odds

```bash
cd scripts
python fetch_daily_odds.py --start-date 2025-03-19 --end-date 2025-03-21
```

### Training Models

```bash
# Train basic model without seeds
python scripts/model_without_seeds.py

# Train model with KenPom rankings
python scripts/no_seeds_kenpom.py

# Train enhanced model with advanced stats
python scripts/tournament_prediction.py
```

## Data Files

### Team Statistics

- `TeamSeasonAverages_with_SoS.csv`: Basic team statistics with strength of schedule
- `EnhancedTournamentStats.csv`: Advanced team metrics including adjusted offensive/defensive ratings, tempo, and efficiency metrics

### Betting Odds Data

- `ncaa_tournament_moneyline_latest.csv`: Latest moneyline betting odds
- `ncaa_tournament_spread_latest.csv`: Latest point spread betting odds
- `ncaa_tournament_odds_latest.csv`: Combined betting odds data

### Tournament Structure

- `2025_Bracket.json`: Contains information about the tournament structure including teams, seeds, and regions

## Key Scripts Explained

### fetch_daily_odds.py

Fetches NCAA basketball betting odds from the sportsgameodds.com API and saves them to CSV files. Separate files are created for moneyline and spread odds, with a combined file for backward compatibility.

Features:
- Fetches odds for specified date ranges
- Processes both moneyline and spread odds
- Maps team names to team IDs for integration with prediction models
- Provides detailed logging for troubleshooting

### app_unified.py

The main Streamlit web application for interactive tournament predictions.

Features:
- Team selection and comparison
- Multiple prediction models
- Detailed statistical comparisons
- Betting odds analysis
- Mobile-responsive design

### tournament_prediction.py

Core script for training the enhanced tournament prediction model.

Features:
- Feature engineering from basic and advanced stats
- Multiple model training options
- Cross-validation for model evaluation
- Feature importance analysis

## License

This project is available for educational and research purposes. Please check the license terms for usage restrictions.

## Acknowledgments

- KenPom for advanced college basketball metrics
- NCAA for tournament historical data
- sportsgameodds.com for betting odds API

---

*Note: This README is a comprehensive guide to the repository structure and functionality. For specific questions or issues, please open an issue on the GitHub repository.*


