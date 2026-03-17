from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from darko_like_dpm import load_core_data
from darko_like_dpm_v2 import (
    DPM_V2_FEATURES,
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

ADDED_DPM_FEATURES = [feature for feature in DPM_V2_FEATURES if feature != "SeedDiff"]
HYBRID_FEATURES = BASE_FEATURES + ADDED_DPM_FEATURES


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


def build_hybrid_dataset(
    tourney: pd.DataFrame,
    basic_stats: pd.DataFrame,
    kenpom: pd.DataFrame,
    seeds: pd.DataFrame,
    ratings_v2: pd.DataFrame,
) -> pd.DataFrame:
    games = make_balanced_games(tourney)
    df = create_matchup_frame_v2(games, ratings_v2, seeds)

    basic_team1 = basic_stats.add_prefix("Team1_")
    basic_team2 = basic_stats.add_prefix("Team2_")
    df = df.merge(
        basic_team1,
        left_on=["Season", "Team1"],
        right_on=["Team1_Season", "Team1_TeamID"],
        how="left",
    ).drop(columns=["Team1_Season", "Team1_TeamID", "Team1_TeamName"], errors="ignore")
    df = df.merge(
        basic_team2,
        left_on=["Season", "Team2"],
        right_on=["Team2_Season", "Team2_TeamID"],
        how="left",
    ).drop(columns=["Team2_Season", "Team2_TeamID", "Team2_TeamName"], errors="ignore")

    kenpom_team1 = kenpom.rename(columns={"TeamID": "Team1", "OrdinalRank": "Team1_KenPom"})
    kenpom_team2 = kenpom.rename(columns={"TeamID": "Team2", "OrdinalRank": "Team2_KenPom"})
    df = df.merge(kenpom_team1[["Season", "Team1", "Team1_KenPom"]], on=["Season", "Team1"], how="left")
    df = df.merge(kenpom_team2[["Season", "Team2", "Team2_KenPom"]], on=["Season", "Team2"], how="left")

    base_cols = [
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
    for col in base_cols:
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


def main():
    _, tourney, seeds, teams = load_core_data()
    basic = pd.read_csv(ROOT / "pre_tourney_data" / "TeamSeasonAverages_with_SoS.csv")
    kenpom = pd.read_csv(ROOT / "pre_tourney_data" / "KenPom-Rankings-Updated.csv")[["Season", "TeamID", "OrdinalRank"]]
    ratings_v2 = build_all_season_ratings_v2(SeasonRatingParamsV2())

    hybrid_df = build_hybrid_dataset(tourney[tourney["Season"] >= 2003].copy(), basic, kenpom, seeds, ratings_v2)
    hybrid_df = hybrid_df.dropna(subset=HYBRID_FEATURES).copy()

    train_df = hybrid_df[hybrid_df["Season"] < 2025].copy()
    test_df = hybrid_df[hybrid_df["Season"] == 2025].copy()

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=250,
        random_state=42,
    )
    model.fit(train_df[HYBRID_FEATURES], train_df["Target"])

    holdout_path = MODELS_DIR / "xgb_model_no_seeds_kenpom_dpm_v2_2025_holdout.pkl"
    joblib.dump(model, holdout_path)

    # Train the final 2026-ready model on all available seasons.
    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=250,
        random_state=42,
    )
    final_model.fit(hybrid_df[HYBRID_FEATURES], hybrid_df["Target"])
    final_model_path = MODELS_DIR / "xgb_model_no_seeds_kenpom_dpm_v2_2026.pkl"
    joblib.dump(final_model, final_model_path)
    (MODELS_DIR / "no_seeds_kenpom_dpm_v2_features.txt").write_text("\n".join(HYBRID_FEATURES))

    baseline = pd.read_csv(OUTPUT_DIR / "model_comparison_2025_matchups.csv")
    results = test_df[["Season", "DayNum", "Team1", "Team2", "Target", "Team1_Seed", "Team2_Seed"]].copy()
    results["xgb_model_no_seeds_kenpom_dpm_v2"] = model.predict_proba(test_df[HYBRID_FEATURES])[:, 1]
    results = results.merge(
        baseline[
            [
                "Season",
                "DayNum",
                "Team1",
                "Team2",
                "Target",
                "xgb_model_no_seeds_kenpom",
                "darko_like_dpm_v2",
                "final_model_with_tempo2",
                "final_model_py2",
            ]
        ],
        on=["Season", "DayNum", "Team1", "Team2", "Target"],
        how="left",
    )

    teams_lookup = teams.rename(columns={"TeamID": "Team1", "TeamName": "Team1Name"})
    results = results.merge(teams_lookup, on="Team1", how="left")
    teams_lookup = teams.rename(columns={"TeamID": "Team2", "TeamName": "Team2Name"})
    results = results.merge(teams_lookup, on="Team2", how="left")

    actual_results = results[results["Target"] == 1].copy()
    summary = pd.DataFrame(
        [
            summarize(actual_results, "xgb_model_no_seeds_kenpom_dpm_v2"),
            summarize(actual_results, "xgb_model_no_seeds_kenpom"),
            summarize(actual_results, "darko_like_dpm_v2"),
            summarize(actual_results, "final_model_with_tempo2"),
            summarize(actual_results, "final_model_py2"),
        ]
    ).sort_values(["accuracy", "log_loss"], ascending=[False, True])

    seeds_2025 = seeds[seeds["Season"] == 2025].copy()
    seed_lookup = seeds_2025.set_index("TeamID")["SeedValue"]
    actual_results["WinnerSeed"] = actual_results["Team1"].map(seed_lookup)
    actual_results["LoserSeed"] = actual_results["Team2"].map(seed_lookup)
    actual_results["SeedGap"] = actual_results["WinnerSeed"] - actual_results["LoserSeed"]
    upsets = actual_results[actual_results["SeedGap"] > 0].copy()

    upsets_report = upsets[
        [
            "DayNum",
            "Team1Name",
            "Team2Name",
            "WinnerSeed",
            "LoserSeed",
            "SeedGap",
            "xgb_model_no_seeds_kenpom_dpm_v2",
            "xgb_model_no_seeds_kenpom",
            "darko_like_dpm_v2",
            "final_model_with_tempo2",
        ]
    ].sort_values(["SeedGap", "xgb_model_no_seeds_kenpom_dpm_v2"], ascending=[False, False])

    results.to_csv(OUTPUT_DIR / "model_comparison_2025_matchups_hybrid.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "model_comparison_2025_summary_hybrid.csv", index=False)
    upsets_report.to_csv(OUTPUT_DIR / "model_comparison_2025_upsets_hybrid.csv", index=False)

    print(summary.to_string(index=False))
    print("\nTop upset calls:")
    print(upsets_report.head(12).to_string(index=False))
    print(f"\nSaved holdout model: {holdout_path}")
    print(f"Saved final model: {final_model_path}")


if __name__ == "__main__":
    main()
