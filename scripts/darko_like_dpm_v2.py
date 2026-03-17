from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from darko_like_dpm import (
    MODELS_DIR,
    PRE_TOURNEY_DIR,
    _team_perspective_games,
    _weighted_std,
    fit_season_ratings,
    load_core_data,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SeasonRatingParamsV2:
    half_life_days: float = 14.0
    ridge_alpha: float = 35.0
    momentum_window_days: int = 12
    recent_schedule_window_days: int = 18


def fit_season_ratings_v2(team_games: pd.DataFrame, params: SeasonRatingParamsV2) -> pd.DataFrame:
    # Reuse the latent offense/defense solver from v1, then layer tempo and schedule descriptors on top.
    base = fit_season_ratings(
        team_games.copy(),
        params=type("BaseParams", (), {
            "half_life_days": params.half_life_days,
            "ridge_alpha": params.ridge_alpha,
            "momentum_window_days": params.momentum_window_days,
        })(),
    )
    season = int(team_games["Season"].iloc[0])
    max_day = float(team_games["DayNum"].max())
    recent_cutoff = max_day - params.momentum_window_days
    recent_schedule_cutoff = max_day - params.recent_schedule_window_days

    base_lookup = base.set_index("TeamID")
    opp_net_lookup = base_lookup["DPM_Net"].to_dict()

    extra_rows = []
    for team_id, team_slice in team_games.groupby("TeamID"):
        ages = max_day - team_slice["DayNum"].to_numpy(dtype=float)
        weights = np.exp(-np.log(2.0) * ages / params.half_life_days)

        recent_slice = team_slice[team_slice["DayNum"] >= recent_cutoff]
        recent_weights = np.exp(
            -np.log(2.0) * (max_day - recent_slice["DayNum"].to_numpy(dtype=float)) / params.half_life_days
        )
        recent_schedule_slice = team_slice[team_slice["DayNum"] >= recent_schedule_cutoff]
        recent_schedule_weights = np.exp(
            -np.log(2.0) * (max_day - recent_schedule_slice["DayNum"].to_numpy(dtype=float)) / params.half_life_days
        )

        opp_net = np.array([opp_net_lookup.get(int(opp), 0.0) for opp in team_slice["OppTeamID"]], dtype=float)
        recent_opp_net = np.array(
            [opp_net_lookup.get(int(opp), 0.0) for opp in recent_schedule_slice["OppTeamID"]],
            dtype=float,
        )

        weighted_tempo = float(np.average(team_slice["Poss"], weights=weights))
        weighted_recent_tempo = float(
            np.average(recent_slice["Poss"], weights=recent_weights) if len(recent_slice) else weighted_tempo
        )
        weighted_recent_margin = float(
            np.average(recent_slice["MarginPer100"], weights=recent_weights)
            if len(recent_slice)
            else np.average(team_slice["MarginPer100"], weights=weights)
        )
        weighted_schedule = float(np.average(opp_net, weights=weights)) if len(opp_net) else 0.0
        weighted_recent_schedule = float(np.average(recent_opp_net, weights=recent_schedule_weights)) if len(recent_opp_net) else weighted_schedule
        weighted_recent_off = float(
            np.average(recent_slice["OffEff"], weights=recent_weights)
            if len(recent_slice)
            else np.average(team_slice["OffEff"], weights=weights)
        )

        extra_rows.append(
            {
                "Season": season,
                "TeamID": int(team_id),
                "DPM2_Tempo": weighted_tempo,
                "DPM2_RecentTempo": weighted_recent_tempo,
                "DPM2_TempoVolatility": _weighted_std(team_slice["Poss"].to_numpy(dtype=float), weights),
                "DPM2_RecentSchedule": weighted_recent_schedule,
                "DPM2_OffTrend": weighted_recent_off - float(base_lookup.loc[team_id, "DPM_AdjO"]),
                "DPM2_MarginTrend": weighted_recent_margin - float(base_lookup.loc[team_id, "DPM_Net"]),
                "DPM2_ScheduleDrift": weighted_recent_schedule - weighted_schedule,
            }
        )

    extra = pd.DataFrame(extra_rows)
    merged = base.merge(extra, on=["Season", "TeamID"], how="left")
    merged = merged.rename(
        columns={
            "DPM_AdjO": "DPM2_AdjO",
            "DPM_AdjD": "DPM2_AdjD",
            "DPM_Net": "DPM2_Net",
            "DPM_Momentum": "DPM2_Momentum",
            "DPM_Volatility": "DPM2_Volatility",
            "DPM_Schedule": "DPM2_Schedule",
            "DPM_HCA": "DPM2_HCA",
        }
    )
    return merged


def build_all_season_ratings_v2(params: SeasonRatingParamsV2) -> pd.DataFrame:
    regular, _, _, teams = load_core_data()
    team_games = _team_perspective_games(regular[regular["Season"] >= 2003].copy())
    season_frames = []
    for _, season_games in team_games.groupby("Season"):
        season_frames.append(fit_season_ratings_v2(season_games.copy(), params))
    ratings = pd.concat(season_frames, ignore_index=True)
    return ratings.merge(teams, on="TeamID", how="left")


def create_matchup_frame_v2(games: pd.DataFrame, ratings: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    team1 = ratings.add_prefix("Team1_")
    team2 = ratings.add_prefix("Team2_")

    merged = games.merge(
        team1,
        left_on=["Season", "Team1"],
        right_on=["Team1_Season", "Team1_TeamID"],
        how="left",
    ).drop(columns=["Team1_Season", "Team1_TeamID", "Team1_TeamName"], errors="ignore")
    merged = merged.merge(
        team2,
        left_on=["Season", "Team2"],
        right_on=["Team2_Season", "Team2_TeamID"],
        how="left",
    ).drop(columns=["Team2_Season", "Team2_TeamID", "Team2_TeamName"], errors="ignore")

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

    merged["DPM2_NetDiff"] = merged["Team1_DPM2_Net"] - merged["Team2_DPM2_Net"]
    merged["DPM2_AdjODiff"] = merged["Team1_DPM2_AdjO"] - merged["Team2_DPM2_AdjO"]
    merged["DPM2_DefEdge"] = merged["Team2_DPM2_AdjD"] - merged["Team1_DPM2_AdjD"]
    merged["DPM2_MatchupSpread"] = (
        (merged["Team1_DPM2_AdjO"] - merged["Team2_DPM2_AdjD"])
        - (merged["Team2_DPM2_AdjO"] - merged["Team1_DPM2_AdjD"])
    )
    merged["DPM2_MomentumDiff"] = merged["Team1_DPM2_Momentum"] - merged["Team2_DPM2_Momentum"]
    merged["DPM2_ScheduleDiff"] = merged["Team1_DPM2_Schedule"] - merged["Team2_DPM2_Schedule"]
    merged["DPM2_RecentScheduleDiff"] = merged["Team1_DPM2_RecentSchedule"] - merged["Team2_DPM2_RecentSchedule"]
    merged["DPM2_StabilityEdge"] = merged["Team2_DPM2_Volatility"] - merged["Team1_DPM2_Volatility"]
    merged["DPM2_TempoDiff"] = merged["Team1_DPM2_Tempo"] - merged["Team2_DPM2_Tempo"]
    merged["DPM2_ExpectedTempo"] = (merged["Team1_DPM2_Tempo"] + merged["Team2_DPM2_Tempo"]) / 2.0
    merged["DPM2_RecentTempoDiff"] = merged["Team1_DPM2_RecentTempo"] - merged["Team2_DPM2_RecentTempo"]
    merged["DPM2_TempoVolatilityEdge"] = (
        merged["Team2_DPM2_TempoVolatility"] - merged["Team1_DPM2_TempoVolatility"]
    )
    merged["DPM2_OffTrendDiff"] = merged["Team1_DPM2_OffTrend"] - merged["Team2_DPM2_OffTrend"]
    merged["DPM2_MarginTrendDiff"] = merged["Team1_DPM2_MarginTrend"] - merged["Team2_DPM2_MarginTrend"]
    merged["DPM2_ScheduleDriftDiff"] = merged["Team1_DPM2_ScheduleDrift"] - merged["Team2_DPM2_ScheduleDrift"]
    merged["DPM2_TempoSpreadInteraction"] = merged["DPM2_MatchupSpread"] * merged["DPM2_ExpectedTempo"] / 100.0
    merged["DPM2_MomentumScheduleInteraction"] = merged["DPM2_MomentumDiff"] * (1.0 + merged["DPM2_RecentScheduleDiff"])
    merged["SeedDiff"] = merged["Team1_Seed"] - merged["Team2_Seed"]
    return merged


DPM_V2_FEATURES = [
    "DPM2_NetDiff",
    "DPM2_AdjODiff",
    "DPM2_DefEdge",
    "DPM2_MatchupSpread",
    "DPM2_MomentumDiff",
    "DPM2_ScheduleDiff",
    "DPM2_RecentScheduleDiff",
    "DPM2_StabilityEdge",
    "DPM2_TempoDiff",
    "DPM2_ExpectedTempo",
    "DPM2_RecentTempoDiff",
    "DPM2_TempoVolatilityEdge",
    "DPM2_OffTrendDiff",
    "DPM2_MarginTrendDiff",
    "DPM2_ScheduleDriftDiff",
    "DPM2_TempoSpreadInteraction",
    "DPM2_MomentumScheduleInteraction",
    "SeedDiff",
]


def make_balanced_tournament_dataset_v2(tourney: pd.DataFrame, ratings: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    winners = pd.DataFrame(
        {
            "Season": tourney["Season"],
            "DayNum": tourney["DayNum"],
            "Team1": tourney["WTeamID"],
            "Team2": tourney["LTeamID"],
            "Target": 1,
        }
    )
    losers = pd.DataFrame(
        {
            "Season": tourney["Season"],
            "DayNum": tourney["DayNum"],
            "Team1": tourney["LTeamID"],
            "Team2": tourney["WTeamID"],
            "Target": 0,
        }
    )
    return create_matchup_frame_v2(pd.concat([winners, losers], ignore_index=True), ratings, seeds)


def train_dpm_v2_classifier(train_df: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression(C=0.9, max_iter=600, random_state=42)
    model.fit(train_df[DPM_V2_FEATURES], train_df["Target"])
    return model


def save_final_dpm_v2_model():
    regular, tourney, seeds, _ = load_core_data()
    _ = regular  # kept for symmetry and future extension
    params = SeasonRatingParamsV2()
    ratings = build_all_season_ratings_v2(params)
    full_df = make_balanced_tournament_dataset_v2(tourney[tourney["Season"] >= 2003].copy(), ratings, seeds)
    full_df = full_df.dropna(subset=DPM_V2_FEATURES)
    model = train_dpm_v2_classifier(full_df)
    model_path = MODELS_DIR / "darko_like_dpm_v2_model_2026.pkl"
    ratings_path = PRE_TOURNEY_DIR / "DPMTeamRatings_v2.csv"
    joblib.dump({"model": model, "features": DPM_V2_FEATURES, "params": params}, model_path)
    ratings.to_csv(ratings_path, index=False)
    return model_path, ratings_path


if __name__ == "__main__":
    model_path, ratings_path = save_final_dpm_v2_model()
    print(f"Saved model to {model_path}")
    print(f"Saved ratings to {ratings_path}")
