from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from darko_like_dpm import load_core_data
from darko_like_dpm_v2 import SeasonRatingParamsV2, build_all_season_ratings_v2, create_matchup_frame_v2


ROOT = Path(__file__).resolve().parents[1]
PRE_TOURNEY_DIR = ROOT / "pre_tourney_data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "2026_testing"

BASIC_FEATURES = [
    "WinPct_diff",
    "Avg_Score_diff",
    "Avg_FGM_diff",
    "Avg_FGA_diff",
    "Avg_FGM3_diff",
    "Avg_FGA3_diff",
    "Avg_FTM_diff",
    "Avg_FTA_diff",
    "Avg_OR_diff",
    "Avg_DR_diff",
    "Avg_Ast_diff",
    "Avg_TO_diff",
    "Avg_Stl_diff",
    "Avg_Blk_diff",
    "Avg_PF_diff",
    "Avg_Opp_WinPct_diff",
    "Last30_WinRatio_diff",
    "SoS_squared_diff",
    "SoS_WinPct_interaction",
    "SoS_Last30_interaction",
]

BASIC_KENPOM_FEATURES = BASIC_FEATURES + [
    "KenPom_diff",
    "KenPom_SoS_interaction",
]

ENHANCED_DIFF_FEATURES = [
    "Diff_AdjO",
    "Diff_AdjD",
    "Diff_AdjNetRtg",
    "Diff_SOS_NetRtg",
    "Diff_Expected Win%",
    "Diff_ThreePtRate",
    "Diff_FTRate",
    "Diff_AstRate",
    "Diff_TORate",
    "Diff_ORRate",
    "Diff_DRRate",
    "Diff_ScoreStdDev",
    "Diff_MarginStdDev",
    "Diff_ORtgStdDev",
    "Diff_DRtgStdDev",
    "Diff_HomeWin%",
    "Diff_AwayWin%",
    "Diff_NeutralWin%",
    "Diff_Last10Win%",
]

ENHANCED_FEATURES = ["SeedDiff", "KenPomDiff"] + ENHANCED_DIFF_FEATURES

MINIMAL_DPM_FEATURES = [
    "DPM2_NetDiff",
    "DPM2_MomentumDiff",
    "DPM2_RecentScheduleDiff",
]


def make_balanced_games(tourney: pd.DataFrame) -> pd.DataFrame:
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
    return pd.concat([winners, losers], ignore_index=True)


def build_basic_dataset(tourney: pd.DataFrame, basic: pd.DataFrame, kenpom: pd.DataFrame) -> pd.DataFrame:
    games = make_balanced_games(tourney)
    df = games.copy()

    basic1 = basic.add_prefix("Team1_")
    basic2 = basic.add_prefix("Team2_")
    df = df.merge(
        basic1,
        left_on=["Season", "Team1"],
        right_on=["Team1_Season", "Team1_TeamID"],
        how="left",
    ).drop(columns=["Team1_Season", "Team1_TeamID", "Team1_TeamName"], errors="ignore")
    df = df.merge(
        basic2,
        left_on=["Season", "Team2"],
        right_on=["Team2_Season", "Team2_TeamID"],
        how="left",
    ).drop(columns=["Team2_Season", "Team2_TeamID", "Team2_TeamName"], errors="ignore")

    kp1 = kenpom.rename(columns={"TeamID": "Team1", "OrdinalRank": "Team1_KenPom"})
    kp2 = kenpom.rename(columns={"TeamID": "Team2", "OrdinalRank": "Team2_KenPom"})
    df = df.merge(kp1[["Season", "Team1", "Team1_KenPom"]], on=["Season", "Team1"], how="left")
    df = df.merge(kp2[["Season", "Team2", "Team2_KenPom"]], on=["Season", "Team2"], how="left")

    stat_cols = [
        "WinPct",
        "Avg_Score",
        "Avg_FGM",
        "Avg_FGA",
        "Avg_FGM3",
        "Avg_FGA3",
        "Avg_FTM",
        "Avg_FTA",
        "Avg_OR",
        "Avg_DR",
        "Avg_Ast",
        "Avg_TO",
        "Avg_Stl",
        "Avg_Blk",
        "Avg_PF",
        "Avg_Opp_WinPct",
        "Last30_WinRatio",
    ]
    for col in stat_cols:
        df[f"{col}_diff"] = df[f"Team1_{col}"] - df[f"Team2_{col}"]

    df["KenPom_diff"] = df["Team2_KenPom"].fillna(400) - df["Team1_KenPom"].fillna(400)
    df["SoS_squared_diff"] = df["Avg_Opp_WinPct_diff"] ** 2 * np.sign(df["Avg_Opp_WinPct_diff"])
    df["SoS_WinPct_interaction"] = (
        df["Team1_Avg_Opp_WinPct"] * df["Team1_WinPct"]
        - df["Team2_Avg_Opp_WinPct"] * df["Team2_WinPct"]
    )
    df["SoS_Last30_interaction"] = (
        df["Team1_Avg_Opp_WinPct"] * df["Team1_Last30_WinRatio"]
        - df["Team2_Avg_Opp_WinPct"] * df["Team2_Last30_WinRatio"]
    )
    df["KenPom_SoS_interaction"] = df["KenPom_diff"] * df["Avg_Opp_WinPct_diff"]
    return df


def build_enhanced_dataset(
    tourney: pd.DataFrame, enhanced: pd.DataFrame, seeds: pd.DataFrame, kenpom: pd.DataFrame
) -> pd.DataFrame:
    games = make_balanced_games(tourney)
    df = games.copy()

    enh1 = enhanced.add_prefix("Team1_")
    enh2 = enhanced.add_prefix("Team2_")
    df = df.merge(
        enh1,
        left_on=["Season", "Team1"],
        right_on=["Team1_Season", "Team1_TeamID"],
        how="left",
    ).drop(columns=["Team1_Season", "Team1_TeamID", "Team1_TeamName"], errors="ignore")
    df = df.merge(
        enh2,
        left_on=["Season", "Team2"],
        right_on=["Team2_Season", "Team2_TeamID"],
        how="left",
    ).drop(columns=["Team2_Season", "Team2_TeamID", "Team2_TeamName"], errors="ignore")

    kp1 = kenpom.rename(columns={"TeamID": "Team1", "OrdinalRank": "Team1_KenPom"})
    kp2 = kenpom.rename(columns={"TeamID": "Team2", "OrdinalRank": "Team2_KenPom"})
    df = df.merge(kp1[["Season", "Team1", "Team1_KenPom"]], on=["Season", "Team1"], how="left")
    df = df.merge(kp2[["Season", "Team2", "Team2_KenPom"]], on=["Season", "Team2"], how="left")

    seed_values = seeds.copy()
    seed_values["SeedValue"] = seed_values["Seed"].str.extract(r"(\d+)").astype(int)
    seed1 = seed_values.rename(columns={"TeamID": "Team1", "SeedValue": "Team1_Seed"})
    seed2 = seed_values.rename(columns={"TeamID": "Team2", "SeedValue": "Team2_Seed"})
    df = df.merge(seed1[["Season", "Team1", "Team1_Seed"]], on=["Season", "Team1"], how="left")
    df = df.merge(seed2[["Season", "Team2", "Team2_Seed"]], on=["Season", "Team2"], how="left")

    df["SeedDiff"] = df["Team1_Seed"] - df["Team2_Seed"]
    df["KenPomDiff"] = -(df["Team2_KenPom"].fillna(400) - df["Team1_KenPom"].fillna(400))

    feature_map = {
        "AdjO": "AdjO",
        "AdjD": "AdjD",
        "AdjNetRtg": "AdjNetRtg",
        "SOS_NetRtg": "SOS_NetRtg",
        "Expected Win%": "Expected Win%",
        "ThreePtRate": "ThreePtRate",
        "FTRate": "FTRate",
        "AstRate": "AstRate",
        "TORate": "TORate",
        "ORRate": "ORRate",
        "DRRate": "DRRate",
        "ScoreStdDev": "ScoreStdDev",
        "MarginStdDev": "MarginStdDev",
        "ORtgStdDev": "ORtgStdDev",
        "DRtgStdDev": "DRtgStdDev",
        "HomeWin%": "HomeWin%",
        "AwayWin%": "AwayWin%",
        "NeutralWin%": "NeutralWin%",
        "Last10Win%": "Last10Win%",
    }
    for output_name, base_name in feature_map.items():
        df[f"Diff_{output_name}"] = df[f"Team1_{base_name}"] - df[f"Team2_{base_name}"]

    for col in ["Diff_AdjD", "Diff_TORate"]:
        df[col] = -df[col]

    return df


def fit_booster(train_x: pd.DataFrame, train_y: pd.Series) -> xgb.Booster:
    dtrain = xgb.DMatrix(train_x, label=train_y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    return xgb.train(params, dtrain, num_boost_round=150)


def fit_enhanced_model(train_x: pd.DataFrame, train_y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=200,
        random_state=42,
    )
    model.fit(train_x, train_y)
    return model


def predict_booster_symmetric(model: xgb.Booster, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    probs = model.predict(xgb.DMatrix(df[feature_cols]))
    swapped = df.copy()
    for feature in feature_cols:
        if feature in ["SoS_squared_diff", "TempoDiff", "AvgTempo", "DPM2_ExpectedTempo"]:
            continue
        if feature.endswith("_diff") or feature in [
            "KenPom_diff",
            "KenPom_SoS_interaction",
            "SoS_WinPct_interaction",
            "SoS_Last30_interaction",
            "DPM2_NetDiff",
            "DPM2_MomentumDiff",
            "DPM2_RecentScheduleDiff",
        ]:
            swapped[feature] = -swapped[feature]
    swapped_probs = model.predict(xgb.DMatrix(swapped[feature_cols]))
    return (probs + (1.0 - swapped_probs)) / 2.0


def evaluate_predictions(actual_games: pd.DataFrame, prob_col: str) -> dict:
    probs = actual_games[prob_col].clip(1e-6, 1 - 1e-6)
    preds = (probs >= 0.5).astype(int)
    y = actual_games["Target"].astype(int)
    return {
        "model": prob_col,
        "games": int(len(actual_games)),
        "accuracy": float(accuracy_score(y, preds)),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probs)),
    }


def upset_precision(actual_games: pd.DataFrame, prob_col: str) -> dict:
    predicted_upsets = 0
    true_upsets = 0
    false_upsets = 0
    close_upsets = 0
    actual_upsets = 0
    for _, row in actual_games.iterrows():
        if pd.isna(row["WinnerSeed"]) or pd.isna(row["LoserSeed"]) or row["WinnerSeed"] == row["LoserSeed"]:
            continue
        lower_seed_prob = row[prob_col] if row["WinnerSeed"] > row["LoserSeed"] else 1 - row[prob_col]
        actual_upset = row["WinnerSeed"] > row["LoserSeed"]
        if actual_upset:
            actual_upsets += 1
        if lower_seed_prob >= 0.5:
            predicted_upsets += 1
            if actual_upset:
                true_upsets += 1
            else:
                false_upsets += 1
        if actual_upset and lower_seed_prob >= 0.4:
            close_upsets += 1
    return {
        "model": prob_col,
        "predicted_upsets": predicted_upsets,
        "true_upsets": true_upsets,
        "false_upsets": false_upsets,
        "upset_precision": true_upsets / predicted_upsets if predicted_upsets else np.nan,
        "actual_upsets": actual_upsets,
        "upset_recall": true_upsets / actual_upsets if actual_upsets else np.nan,
        "actual_upsets_with_40pct_plus": close_upsets,
        "actual_upsets_with_40pct_plus_rate": close_upsets / actual_upsets if actual_upsets else np.nan,
    }


def build_actual_games(test_tourney: pd.DataFrame, teams: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    seed_values = seeds.copy()
    seed_values["SeedValue"] = seed_values["Seed"].str.extract(r"(\d+)").astype(int)
    name1 = teams.rename(columns={"TeamID": "Team1", "TeamName": "WinnerName"})
    name2 = teams.rename(columns={"TeamID": "Team2", "TeamName": "LoserName"})
    actual = pd.DataFrame(
        {
            "Season": test_tourney["Season"],
            "DayNum": test_tourney["DayNum"],
            "Team1": test_tourney["WTeamID"],
            "Team2": test_tourney["LTeamID"],
            "Target": 1,
        }
    )
    actual = actual.merge(name1, on="Team1", how="left").merge(name2, on="Team2", how="left")
    actual = actual.merge(
        seed_values[["Season", "TeamID", "SeedValue"]].rename(columns={"TeamID": "Team1", "SeedValue": "WinnerSeed"}),
        on=["Season", "Team1"],
        how="left",
    )
    actual = actual.merge(
        seed_values[["Season", "TeamID", "SeedValue"]].rename(columns={"TeamID": "Team2", "SeedValue": "LoserSeed"}),
        on=["Season", "Team2"],
        how="left",
    )
    return actual


def evaluate_holdout(
    holdout_season: int,
    tourney: pd.DataFrame,
    seeds: pd.DataFrame,
    teams: pd.DataFrame,
    basic_df: pd.DataFrame,
    enhanced_df: pd.DataFrame,
    dpm_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_basic = basic_df[basic_df["Season"] < holdout_season].dropna(subset=BASIC_KENPOM_FEATURES).copy()
    test_basic = basic_df[basic_df["Season"] == holdout_season].dropna(subset=BASIC_KENPOM_FEATURES).copy()

    train_enhanced = enhanced_df[enhanced_df["Season"] < holdout_season].dropna(subset=ENHANCED_FEATURES).copy()
    test_enhanced = enhanced_df[enhanced_df["Season"] == holdout_season].dropna(subset=ENHANCED_FEATURES).copy()

    train_dpm = dpm_df[dpm_df["Season"] < holdout_season].dropna(subset=BASIC_KENPOM_FEATURES + MINIMAL_DPM_FEATURES).copy()
    test_dpm = dpm_df[dpm_df["Season"] == holdout_season].dropna(subset=BASIC_KENPOM_FEATURES + MINIMAL_DPM_FEATURES).copy()

    basic_model = fit_booster(train_basic[BASIC_FEATURES], train_basic["Target"])
    basic_kenpom_model = fit_booster(train_basic[BASIC_KENPOM_FEATURES], train_basic["Target"])
    enhanced_model = fit_enhanced_model(train_enhanced[ENHANCED_FEATURES], train_enhanced["Target"])
    dpm_model = fit_booster(train_dpm[BASIC_KENPOM_FEATURES + MINIMAL_DPM_FEATURES], train_dpm["Target"])

    test_tourney = tourney[tourney["Season"] == holdout_season].copy()
    actual = build_actual_games(test_tourney, teams, seeds)

    basic_actual = (
        build_basic_dataset(test_tourney, basic_stats, kenpom_df)
        .query("Target == 1")
        .dropna(subset=BASIC_KENPOM_FEATURES)
        .copy()
    )
    enhanced_actual = (
        build_enhanced_dataset(test_tourney, enhanced_stats, seeds, kenpom_df)
        .query("Target == 1")
        .dropna(subset=ENHANCED_FEATURES)
        .copy()
    )
    dpm_actual = create_matchup_frame_v2(
        pd.DataFrame(
            {
                "Season": test_tourney["Season"],
                "DayNum": test_tourney["DayNum"],
                "Team1": test_tourney["WTeamID"],
                "Team2": test_tourney["LTeamID"],
                "Target": 1,
            }
        ),
        ratings_v2,
        seeds,
    )
    dpm_actual = dpm_actual.merge(
        build_basic_dataset(test_tourney, basic_stats, kenpom_df)[
            ["Season", "DayNum", "Team1", "Team2"] + BASIC_KENPOM_FEATURES
        ],
        on=["Season", "DayNum", "Team1", "Team2"],
        how="left",
    ).dropna(subset=BASIC_KENPOM_FEATURES + MINIMAL_DPM_FEATURES)

    actual["xgb_basic"] = predict_booster_symmetric(basic_model, basic_actual, BASIC_FEATURES)
    actual["xgb_basic_kenpom"] = predict_booster_symmetric(basic_kenpom_model, basic_actual, BASIC_KENPOM_FEATURES)
    actual["xgb_enhanced"] = enhanced_model.predict_proba(enhanced_actual[ENHANCED_FEATURES])[:, 1]
    actual["xgb_basic_kenpom_dpm_minimal"] = predict_booster_symmetric(
        dpm_model, dpm_actual, BASIC_KENPOM_FEATURES + MINIMAL_DPM_FEATURES
    )

    summary = pd.DataFrame(
        [
            evaluate_predictions(actual, "xgb_basic"),
            evaluate_predictions(actual, "xgb_basic_kenpom"),
            evaluate_predictions(actual, "xgb_enhanced"),
            evaluate_predictions(actual, "xgb_basic_kenpom_dpm_minimal"),
        ]
    ).sort_values(["accuracy", "log_loss"], ascending=[False, True])
    precision = pd.DataFrame(
        [
            upset_precision(actual, "xgb_basic"),
            upset_precision(actual, "xgb_basic_kenpom"),
            upset_precision(actual, "xgb_enhanced"),
            upset_precision(actual, "xgb_basic_kenpom_dpm_minimal"),
        ]
    ).sort_values(["upset_precision", "true_upsets"], ascending=[False, False])
    actual.insert(0, "holdout_season", holdout_season)
    return actual, summary, precision


if __name__ == "__main__":
    regular, tourney, seeds, teams = load_core_data()
    basic_stats = pd.read_csv(PRE_TOURNEY_DIR / "TeamSeasonAverages_with_SoS.csv")
    enhanced_stats = pd.read_csv(PRE_TOURNEY_DIR / "EnhancedTournamentStats.csv")
    kenpom_df = pd.read_csv(PRE_TOURNEY_DIR / "KenPom-Rankings-Updated.csv")[["Season", "TeamID", "OrdinalRank"]]
    ratings_v2 = build_all_season_ratings_v2(SeasonRatingParamsV2())

    basic_dataset = build_basic_dataset(tourney[tourney["Season"] >= 2003].copy(), basic_stats, kenpom_df)
    enhanced_dataset = build_enhanced_dataset(tourney[tourney["Season"] >= 2003].copy(), enhanced_stats, seeds, kenpom_df)
    dpm_dataset = create_matchup_frame_v2(make_balanced_games(tourney[tourney["Season"] >= 2003].copy()), ratings_v2, seeds)
    dpm_dataset = dpm_dataset.merge(
        basic_dataset[["Season", "DayNum", "Team1", "Team2"] + BASIC_KENPOM_FEATURES],
        on=["Season", "DayNum", "Team1", "Team2"],
        how="left",
    )

    holdout_matchups = []
    holdout_summaries = []
    holdout_precisions = []
    for holdout_season in [2024, 2025]:
        actual, summary, precision = evaluate_holdout(
            holdout_season, tourney, seeds, teams, basic_dataset, enhanced_dataset, dpm_dataset
        )
        holdout_matchups.append(actual)
        summary.insert(0, "holdout_season", holdout_season)
        precision.insert(0, "holdout_season", holdout_season)
        holdout_summaries.append(summary)
        holdout_precisions.append(precision)

    final_basic = fit_booster(
        basic_dataset.dropna(subset=BASIC_FEATURES)[BASIC_FEATURES],
        basic_dataset.dropna(subset=BASIC_FEATURES)["Target"],
    )
    final_basic_kenpom = fit_booster(
        basic_dataset.dropna(subset=BASIC_KENPOM_FEATURES)[BASIC_KENPOM_FEATURES],
        basic_dataset.dropna(subset=BASIC_KENPOM_FEATURES)["Target"],
    )
    final_enhanced = fit_enhanced_model(
        enhanced_dataset.dropna(subset=ENHANCED_FEATURES)[ENHANCED_FEATURES],
        enhanced_dataset.dropna(subset=ENHANCED_FEATURES)["Target"],
    )
    final_dpm = fit_booster(
        dpm_dataset.dropna(subset=BASIC_KENPOM_FEATURES + MINIMAL_DPM_FEATURES)[BASIC_KENPOM_FEATURES + MINIMAL_DPM_FEATURES],
        dpm_dataset.dropna(subset=BASIC_KENPOM_FEATURES + MINIMAL_DPM_FEATURES)["Target"],
    )

    joblib.dump(final_basic, MODELS_DIR / "xgb_model_no_seeds_2026.pkl")
    joblib.dump(final_basic_kenpom, MODELS_DIR / "xgb_model_no_seeds_kenpom_2026.pkl")
    joblib.dump(final_basic_kenpom, MODELS_DIR / "xgb_model_no_seeds_kenpom_2026_clean.pkl")
    joblib.dump(final_enhanced, MODELS_DIR / "final_model_2026.pkl")
    joblib.dump(final_dpm, MODELS_DIR / "xgb_model_no_seeds_kenpom_dpm_minimal_2026.pkl")
    ratings_v2.to_csv(PRE_TOURNEY_DIR / "DPMTeamRatings_v2.csv", index=False)

    (MODELS_DIR / "no_seeds_model_features.txt").write_text("\n".join(BASIC_FEATURES))
    (MODELS_DIR / "no_seeds_kenpom_features.txt").write_text("\n".join(BASIC_KENPOM_FEATURES))
    (MODELS_DIR / "final_model_2026_features.txt").write_text("\n".join(ENHANCED_FEATURES))
    (MODELS_DIR / "no_seeds_kenpom_dpm_minimal_features.txt").write_text(
        "\n".join(BASIC_KENPOM_FEATURES + MINIMAL_DPM_FEATURES)
    )

    matchup_report = pd.concat(holdout_matchups, ignore_index=True)
    summary_report = pd.concat(holdout_summaries, ignore_index=True)
    precision_report = pd.concat(holdout_precisions, ignore_index=True)
    matchup_report.to_csv(OUTPUT_DIR / "model_holdout_matchups_2024_2025.csv", index=False)
    summary_report.to_csv(OUTPUT_DIR / "model_holdout_summary_2024_2025.csv", index=False)
    precision_report.to_csv(OUTPUT_DIR / "model_holdout_upset_precision_2024_2025.csv", index=False)

    print("Saved 2026 model artifacts and holdout reports.")
    print(summary_report.to_string(index=False))
    print("\nUpset precision:")
    print(precision_report.to_string(index=False))
