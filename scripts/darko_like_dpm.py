import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge


ROOT = Path(__file__).resolve().parents[1]
REGULAR_SEASON_PATH = ROOT / "raw_data" / "MRegularSeasonDetailedResults.csv"
TOURNEY_RESULTS_PATH = ROOT / "raw_data" / "MNCAATourneyDetailedResults.csv"
SEEDS_PATH = ROOT / "raw_data" / "MNCAATourneySeeds.csv"
MTEAMS_PATH = ROOT / "raw_data" / "MTeams.csv"
MODELS_DIR = ROOT / "models"
PRE_TOURNEY_DIR = ROOT / "pre_tourney_data"


def _calc_possessions(team_fga, team_fta, team_or, team_to, opp_fga, opp_fta, opp_or, opp_to):
    return 0.5 * (
        (team_fga + 0.475 * team_fta - team_or + team_to)
        + (opp_fga + 0.475 * opp_fta - opp_or + opp_to)
    )


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return float(np.std(values))
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    return float(np.sqrt(max(variance, 0.0)))


def _team_perspective_games(games: pd.DataFrame) -> pd.DataFrame:
    winner_rows = pd.DataFrame(
        {
            "Season": games["Season"],
            "DayNum": games["DayNum"],
            "TeamID": games["WTeamID"],
            "OppTeamID": games["LTeamID"],
            "PointsFor": games["WScore"],
            "PointsAgainst": games["LScore"],
            "TeamFGA": games["WFGA"],
            "TeamFTA": games["WFTA"],
            "TeamOR": games["WOR"],
            "TeamTO": games["WTO"],
            "OppFGA": games["LFGA"],
            "OppFTA": games["LFTA"],
            "OppOR": games["LOR"],
            "OppTO": games["LTO"],
            "HomeFlag": games["WLoc"].map({"H": 1.0, "A": -1.0, "N": 0.0}).fillna(0.0),
        }
    )
    loser_rows = pd.DataFrame(
        {
            "Season": games["Season"],
            "DayNum": games["DayNum"],
            "TeamID": games["LTeamID"],
            "OppTeamID": games["WTeamID"],
            "PointsFor": games["LScore"],
            "PointsAgainst": games["WScore"],
            "TeamFGA": games["LFGA"],
            "TeamFTA": games["LFTA"],
            "TeamOR": games["LOR"],
            "TeamTO": games["LTO"],
            "OppFGA": games["WFGA"],
            "OppFTA": games["WFTA"],
            "OppOR": games["WOR"],
            "OppTO": games["WTO"],
            "HomeFlag": games["WLoc"].map({"H": -1.0, "A": 1.0, "N": 0.0}).fillna(0.0),
        }
    )
    team_games = pd.concat([winner_rows, loser_rows], ignore_index=True)
    team_games["Poss"] = _calc_possessions(
        team_games["TeamFGA"],
        team_games["TeamFTA"],
        team_games["TeamOR"],
        team_games["TeamTO"],
        team_games["OppFGA"],
        team_games["OppFTA"],
        team_games["OppOR"],
        team_games["OppTO"],
    ).clip(lower=1.0)
    team_games["OffEff"] = 100.0 * team_games["PointsFor"] / team_games["Poss"]
    team_games["MarginPer100"] = 100.0 * (team_games["PointsFor"] - team_games["PointsAgainst"]) / team_games["Poss"]
    return team_games


def load_core_data():
    regular = pd.read_csv(REGULAR_SEASON_PATH)
    tourney = pd.read_csv(TOURNEY_RESULTS_PATH)
    seeds = pd.read_csv(SEEDS_PATH)
    seeds = seeds[seeds["Season"] >= 2003].copy()
    seeds["SeedValue"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
    teams = pd.read_csv(MTEAMS_PATH)[["TeamID", "TeamName"]]
    return regular, tourney, seeds, teams


@dataclass
class SeasonRatingParams:
    half_life_days: float = 14.0
    ridge_alpha: float = 35.0
    momentum_window_days: int = 12


def fit_season_ratings(team_games: pd.DataFrame, params: SeasonRatingParams) -> pd.DataFrame:
    season = int(team_games["Season"].iloc[0])
    max_day = float(team_games["DayNum"].max())
    teams = sorted(set(team_games["TeamID"]).union(set(team_games["OppTeamID"])))
    team_to_idx = {team_id: idx for idx, team_id in enumerate(teams)}
    n_teams = len(teams)

    ages = max_day - team_games["DayNum"].to_numpy(dtype=float)
    weights = np.exp(-np.log(2.0) * ages / params.half_life_days)
    y = team_games["OffEff"].to_numpy(dtype=float)

    row_idx = np.arange(len(team_games))
    team_idx = team_games["TeamID"].map(team_to_idx).to_numpy()
    opp_idx = team_games["OppTeamID"].map(team_to_idx).to_numpy()

    design = sparse.csr_matrix(
        (
            np.concatenate(
                [
                    np.ones(len(team_games)),
                    -np.ones(len(team_games)),
                    team_games["HomeFlag"].to_numpy(dtype=float),
                ]
            ),
            (
                np.concatenate([row_idx, row_idx, row_idx]),
                np.concatenate([team_idx, opp_idx + n_teams, np.full(len(team_games), 2 * n_teams)]),
            ),
        ),
        shape=(len(team_games), 2 * n_teams + 1),
    )

    ridge = Ridge(alpha=params.ridge_alpha, fit_intercept=True, solver="lsqr", random_state=42)
    ridge.fit(design, y, sample_weight=weights)
    coefs = ridge.coef_

    off_raw = coefs[:n_teams]
    def_raw = coefs[n_teams : 2 * n_teams]
    home_adv = float(coefs[-1])
    intercept = float(ridge.intercept_)

    # Re-center the latent offense and defense terms so that the average team is roughly neutral.
    off_center = float(off_raw.mean())
    def_center = float(def_raw.mean())
    off_raw = off_raw - off_center
    def_raw = def_raw - def_center
    intercept = intercept + off_center - def_center

    recent_cutoff = max_day - params.momentum_window_days
    records = []
    for team_id in teams:
        idx = team_to_idx[team_id]
        team_slice = team_games[team_games["TeamID"] == team_id].copy()
        team_weights = np.exp(-np.log(2.0) * (max_day - team_slice["DayNum"].to_numpy(dtype=float)) / params.half_life_days)
        recent_slice = team_slice[team_slice["DayNum"] >= recent_cutoff]
        recent_weights = np.exp(
            -np.log(2.0) * (max_day - recent_slice["DayNum"].to_numpy(dtype=float)) / params.half_life_days
        )

        adj_off = intercept + off_raw[idx]
        adj_def = intercept - def_raw[idx]
        adj_net = adj_off - adj_def
        schedule = float(np.average([- (intercept - def_raw[team_to_idx[opp]]) + (intercept + off_raw[team_to_idx[opp]]) for opp in team_slice["OppTeamID"]], weights=team_weights))
        records.append(
            {
                "Season": season,
                "TeamID": int(team_id),
                "DPM_AdjO": float(adj_off),
                "DPM_AdjD": float(adj_def),
                "DPM_Net": float(adj_net),
                "DPM_Momentum": float(
                    np.average(recent_slice["MarginPer100"], weights=recent_weights) if len(recent_slice) else np.average(team_slice["MarginPer100"], weights=team_weights)
                ),
                "DPM_Volatility": _weighted_std(team_slice["MarginPer100"].to_numpy(dtype=float), team_weights),
                "DPM_Schedule": schedule,
                "DPM_HCA": home_adv,
            }
        )

    return pd.DataFrame(records)


def build_all_season_ratings(params: SeasonRatingParams) -> pd.DataFrame:
    regular, _, _, teams = load_core_data()
    team_games = _team_perspective_games(regular[regular["Season"] >= 2003].copy())
    season_frames = []
    for season, season_games in team_games.groupby("Season"):
        season_frames.append(fit_season_ratings(season_games.copy(), params))
    ratings = pd.concat(season_frames, ignore_index=True)
    return ratings.merge(teams, on="TeamID", how="left")


def create_matchup_frame(games: pd.DataFrame, ratings: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    merged = games.merge(
        ratings,
        left_on=["Season", "Team1"],
        right_on=["Season", "TeamID"],
        how="left",
    ).drop(columns=["TeamID"])
    merged = merged.rename(
        columns={
            "DPM_AdjO": "Team1_DPM_AdjO",
            "DPM_AdjD": "Team1_DPM_AdjD",
            "DPM_Net": "Team1_DPM_Net",
            "DPM_Momentum": "Team1_DPM_Momentum",
            "DPM_Volatility": "Team1_DPM_Volatility",
            "DPM_Schedule": "Team1_DPM_Schedule",
        }
    )
    merged = merged.merge(
        ratings,
        left_on=["Season", "Team2"],
        right_on=["Season", "TeamID"],
        how="left",
        suffixes=("", "_Team2"),
    ).drop(columns=["TeamID"])
    merged = merged.rename(
        columns={
            "DPM_AdjO": "Team2_DPM_AdjO",
            "DPM_AdjD": "Team2_DPM_AdjD",
            "DPM_Net": "Team2_DPM_Net",
            "DPM_Momentum": "Team2_DPM_Momentum",
            "DPM_Volatility": "Team2_DPM_Volatility",
            "DPM_Schedule": "Team2_DPM_Schedule",
        }
    )

    merged = merged.merge(
        seeds[["Season", "TeamID", "SeedValue"]],
        left_on=["Season", "Team1"],
        right_on=["Season", "TeamID"],
        how="left",
    ).drop(columns=["TeamID"])
    merged = merged.rename(columns={"SeedValue": "Team1_Seed"})
    merged = merged.merge(
        seeds[["Season", "TeamID", "SeedValue"]],
        left_on=["Season", "Team2"],
        right_on=["Season", "TeamID"],
        how="left",
    ).drop(columns=["TeamID"])
    merged = merged.rename(columns={"SeedValue": "Team2_Seed"})

    merged["DPM_NetDiff"] = merged["Team1_DPM_Net"] - merged["Team2_DPM_Net"]
    merged["DPM_AdjODiff"] = merged["Team1_DPM_AdjO"] - merged["Team2_DPM_AdjO"]
    merged["DPM_DefEdge"] = merged["Team2_DPM_AdjD"] - merged["Team1_DPM_AdjD"]
    merged["DPM_MatchupSpread"] = (
        (merged["Team1_DPM_AdjO"] - merged["Team2_DPM_AdjD"])
        - (merged["Team2_DPM_AdjO"] - merged["Team1_DPM_AdjD"])
    )
    merged["DPM_MomentumDiff"] = merged["Team1_DPM_Momentum"] - merged["Team2_DPM_Momentum"]
    merged["DPM_ScheduleDiff"] = merged["Team1_DPM_Schedule"] - merged["Team2_DPM_Schedule"]
    merged["DPM_StabilityEdge"] = merged["Team2_DPM_Volatility"] - merged["Team1_DPM_Volatility"]
    merged["SeedDiff"] = merged["Team1_Seed"] - merged["Team2_Seed"]

    return merged


DPM_FEATURES = [
    "DPM_NetDiff",
    "DPM_AdjODiff",
    "DPM_DefEdge",
    "DPM_MatchupSpread",
    "DPM_MomentumDiff",
    "DPM_ScheduleDiff",
    "DPM_StabilityEdge",
    "SeedDiff",
]


def make_balanced_tournament_dataset(tourney: pd.DataFrame, ratings: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    base_games = pd.DataFrame(
        {
            "Season": tourney["Season"],
            "DayNum": tourney["DayNum"],
            "Team1": tourney["WTeamID"],
            "Team2": tourney["LTeamID"],
            "Target": 1,
        }
    )
    swapped_games = pd.DataFrame(
        {
            "Season": tourney["Season"],
            "DayNum": tourney["DayNum"],
            "Team1": tourney["LTeamID"],
            "Team2": tourney["WTeamID"],
            "Target": 0,
        }
    )
    all_games = pd.concat([base_games, swapped_games], ignore_index=True)
    return create_matchup_frame(all_games, ratings, seeds)


def train_dpm_classifier(train_df: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression(C=1.5, max_iter=500, random_state=42)
    model.fit(train_df[DPM_FEATURES], train_df["Target"])
    return model


def save_final_dpm_model(
    output_model_path: Path | None = None,
    output_ratings_path: Path | None = None,
    params: SeasonRatingParams | None = None,
):
    params = params or SeasonRatingParams()
    output_model_path = output_model_path or (MODELS_DIR / "darko_like_dpm_model_2026.pkl")
    output_ratings_path = output_ratings_path or (PRE_TOURNEY_DIR / "DPMTeamRatings.csv")

    regular, tourney, seeds, _ = load_core_data()
    ratings = build_all_season_ratings(params)
    train_df = make_balanced_tournament_dataset(tourney[tourney["Season"] >= 2003].copy(), ratings, seeds)
    train_df = train_df.dropna(subset=DPM_FEATURES)
    model = train_dpm_classifier(train_df)

    joblib.dump(
        {
            "model": model,
            "features": DPM_FEATURES,
            "params": params,
        },
        output_model_path,
    )
    ratings.to_csv(output_ratings_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return output_model_path, output_ratings_path


if __name__ == "__main__":
    model_path, ratings_path = save_final_dpm_model()
    print(f"Saved model to {model_path}")
    print(f"Saved ratings to {ratings_path}")
