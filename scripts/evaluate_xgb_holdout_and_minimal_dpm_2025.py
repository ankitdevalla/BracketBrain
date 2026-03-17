from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from darko_like_dpm import load_core_data
from darko_like_dpm_v2 import (
    SeasonRatingParamsV2,
    build_all_season_ratings_v2,
    create_matchup_frame_v2,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "2025_testing"
MODELS_DIR = ROOT / "models"

BASE_FEATURES = [
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
    "KenPom_diff",
    "SoS_squared_diff",
    "SoS_WinPct_interaction",
    "SoS_Last30_interaction",
    "KenPom_SoS_interaction",
]

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


def build_dataset(tourney: pd.DataFrame, basic: pd.DataFrame, kenpom: pd.DataFrame, seeds: pd.DataFrame, ratings_v2: pd.DataFrame) -> pd.DataFrame:
    games = make_balanced_games(tourney)
    df = create_matchup_frame_v2(games, ratings_v2, seeds)

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

    df["KenPom_diff"] = df["Team2_KenPom"] - df["Team1_KenPom"]
    df["SoS_squared_diff"] = df["Avg_Opp_WinPct_diff"] ** 2 * np.sign(df["Avg_Opp_WinPct_diff"])
    df["SoS_WinPct_interaction"] = df["Team1_Avg_Opp_WinPct"] * df["Team1_WinPct"] - df["Team2_Avg_Opp_WinPct"] * df["Team2_WinPct"]
    df["SoS_Last30_interaction"] = (
        df["Team1_Avg_Opp_WinPct"] * df["Team1_Last30_WinRatio"]
        - df["Team2_Avg_Opp_WinPct"] * df["Team2_Last30_WinRatio"]
    )
    df["KenPom_SoS_interaction"] = df["KenPom_diff"] * df["Avg_Opp_WinPct_diff"]
    return df


def fit_xgb(train_x: pd.DataFrame, train_y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=150,
        random_state=42,
    )
    model.fit(train_x, train_y)
    return model


def summarize(df: pd.DataFrame, prob_col: str) -> dict:
    probs = df[prob_col].clip(1e-6, 1 - 1e-6)
    preds = (probs >= 0.5).astype(int)
    y = df["Target"].astype(int)
    return {
        "model": prob_col,
        "games": int(len(df)),
        "accuracy": float(accuracy_score(y, preds)),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probs)),
    }


def upset_precision(actual_games: pd.DataFrame, prob_col: str) -> dict:
    predicted_upsets = 0
    true_upsets = 0
    false_upsets = 0
    for _, row in actual_games.iterrows():
        if row["WinnerSeed"] == row["LoserSeed"]:
            continue
        lower_seed_prob = row[prob_col] if row["WinnerSeed"] > row["LoserSeed"] else 1 - row[prob_col]
        actual_upset = row["WinnerSeed"] > row["LoserSeed"]
        if lower_seed_prob > 0.5:
            predicted_upsets += 1
            if actual_upset:
                true_upsets += 1
            else:
                false_upsets += 1
    return {
        "model": prob_col,
        "predicted_upsets": predicted_upsets,
        "true_upsets": true_upsets,
        "false_upsets": false_upsets,
        "upset_precision": true_upsets / predicted_upsets if predicted_upsets else np.nan,
    }


def main():
    _, tourney, seeds, teams = load_core_data()
    basic = pd.read_csv(ROOT / "pre_tourney_data" / "TeamSeasonAverages_with_SoS.csv")
    kenpom = pd.read_csv(ROOT / "pre_tourney_data" / "KenPom-Rankings-Updated.csv")[["Season", "TeamID", "OrdinalRank"]]
    ratings_v2 = build_all_season_ratings_v2(SeasonRatingParamsV2())

    dataset = build_dataset(tourney[tourney["Season"] >= 2003].copy(), basic, kenpom, seeds, ratings_v2)
    dataset = dataset.dropna(subset=BASE_FEATURES + MINIMAL_DPM_FEATURES).copy()

    train_df = dataset[dataset["Season"] < 2025].copy()
    test_df = dataset[dataset["Season"] == 2025].copy()

    baseline_model = fit_xgb(train_df[BASE_FEATURES], train_df["Target"])
    minimal_model = fit_xgb(train_df[BASE_FEATURES + MINIMAL_DPM_FEATURES], train_df["Target"])

    joblib.dump(baseline_model, MODELS_DIR / "xgb_model_no_seeds_kenpom_2025_holdout.pkl")
    joblib.dump(minimal_model, MODELS_DIR / "xgb_model_no_seeds_kenpom_dpm_minimal_2025_holdout.pkl")

    final_baseline_model = fit_xgb(dataset[BASE_FEATURES], dataset["Target"])
    final_minimal_model = fit_xgb(dataset[BASE_FEATURES + MINIMAL_DPM_FEATURES], dataset["Target"])
    joblib.dump(final_baseline_model, MODELS_DIR / "xgb_model_no_seeds_kenpom_2026_clean.pkl")
    joblib.dump(final_minimal_model, MODELS_DIR / "xgb_model_no_seeds_kenpom_dpm_minimal_2026.pkl")
    (MODELS_DIR / "no_seeds_kenpom_dpm_minimal_features.txt").write_text(
        "\n".join(BASE_FEATURES + MINIMAL_DPM_FEATURES)
    )

    results = test_df[["Season", "DayNum", "Team1", "Team2", "Target", "Team1_Seed", "Team2_Seed"]].copy()
    results["xgb_no_seeds_kenpom_holdout"] = baseline_model.predict_proba(test_df[BASE_FEATURES])[:, 1]
    results["xgb_no_seeds_kenpom_dpm_minimal"] = minimal_model.predict_proba(test_df[BASE_FEATURES + MINIMAL_DPM_FEATURES])[:, 1]

    name1 = teams.rename(columns={"TeamID": "Team1", "TeamName": "WinnerName"})
    name2 = teams.rename(columns={"TeamID": "Team2", "TeamName": "LoserName"})
    actual = results[results["Target"] == 1].copy()
    actual = actual.merge(name1, on="Team1", how="left").merge(name2, on="Team2", how="left")
    actual["WinnerSeed"] = actual["Team1_Seed"]
    actual["LoserSeed"] = actual["Team2_Seed"]

    summary = pd.DataFrame(
        [
            summarize(actual, "xgb_no_seeds_kenpom_holdout"),
            summarize(actual, "xgb_no_seeds_kenpom_dpm_minimal"),
        ]
    ).sort_values(["accuracy", "log_loss"], ascending=[False, True])
    precision = pd.DataFrame(
        [
            upset_precision(actual, "xgb_no_seeds_kenpom_holdout"),
            upset_precision(actual, "xgb_no_seeds_kenpom_dpm_minimal"),
        ]
    ).sort_values(["upset_precision", "predicted_upsets"], ascending=[False, False])

    results.to_csv(OUTPUT_DIR / "model_comparison_2025_matchups_holdout_clean.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "model_comparison_2025_summary_holdout_clean.csv", index=False)
    precision.to_csv(OUTPUT_DIR / "model_comparison_2025_upset_precision_holdout_clean.csv", index=False)

    print("Clean 2025 holdout results:")
    print(summary.to_string(index=False))
    print("\nClean 2025 holdout upset precision:")
    print(precision.to_string(index=False))
    print("\nSaved final 2026 models:")
    print(MODELS_DIR / "xgb_model_no_seeds_kenpom_2026_clean.pkl")
    print(MODELS_DIR / "xgb_model_no_seeds_kenpom_dpm_minimal_2026.pkl")


if __name__ == "__main__":
    main()
