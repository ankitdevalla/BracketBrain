# BracketBrain

This repository contains code and data for predicting NCAA March Madness tournament outcomes using machine learning models. The project includes data preprocessing, feature engineering, model training, evaluation, and analysis.

## Repository Structure

### Folders

- **data/**: Contains raw and processed data files used for training and evaluating the models.
  - `MTeams.csv`: Contains team information.
  - `MNCAATourneyDetailedResults.csv`: Detailed results of NCAA tournament games.
  - `MNCAATourneySeeds.csv`: Seed information for NCAA tournament teams.
  - `KenPom-Rankings-Updated.csv`: KenPom rankings data.
  - `EnhancedTournamentStats.csv`: Enhanced statistics for tournament teams.

- **scripts/**: Contains Python scripts for various tasks such as data preprocessing, feature engineering, model training, and evaluation.
  - `add_tempo_predictor.py`: Script to add tempo (possessions per 40 minutes) as a predictor for tournament predictions.
  - `enhanced_stats_calculator.py`: Script to calculate enhanced statistics for teams.
  - `tournament_prediction_model.py`: Main script to build and evaluate the tournament prediction model.
  - `analyze_prediction.py`: Script to analyze why the model predicts one team over another.
  - `inspect_tempo_model.py`: Script to inspect the feature importance of the model with tempo features.
  - `extract_model.py`: Script to extract and analyze the model from the pickle file.
  - `show_model_weights.py`: Script to extract and display the weights for each variable in the model.

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis, model training, and evaluation.
  - `Tournament_Prediction.ipynb`: Main notebook for building and evaluating the tournament prediction model.
  - `Model_training.ipynb`: Notebook for training and evaluating different machine learning models.
  - `enhanced_stats_calculator.ipynb`: Notebook for calculating enhanced statistics for teams.

- **output/**: Contains output files such as model predictions, evaluation metrics, and plots.
  - `feature_importance_with_tempo.png`: Plot of feature importance for the model with tempo features.
  - `2024_first_round_predictions.csv`: Predictions for the 2024 NCAA tournament first round.

### Key Files

- **README.md**: This file, providing an overview of the project and repository structure.
- **requirements.txt**: List of Python dependencies required to run the scripts and notebooks.
- **.gitignore**: Specifies files and directories to be ignored by Git.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Install the required Python packages using the following command:
  ```bash
  pip3 install -r requirements.txt

## Running the Scripts

1. **Data Preprocessing and Feature Engineering**
    ```bash 
    python scripts/add_tempo_predictor.py

2. **Training the Model**
    ```bash 
    python scripts/tournament_prediction_model.py

3. **Analyzing Predictions**
    ```bash 
    python scripts/analyze_prediction.py

4. **Inspecting Model with Tempo Features**
    ```bash 
    python scripts/inspect_tempo_model.py

## **Usage**

### ***Jupyter Notebooks***
- Open the notebooks in the `notebooks/` folder to:
  - **Explore the data**
  - **Train models**
  - **Evaluate predictions interactively**

### ***Scripts***
- Use the scripts in the `scripts/` folder for:
  - **Automated data processing**
  - **Model training**
  - **Model evaluation**


