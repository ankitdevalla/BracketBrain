from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from darko_like_dpm import (
    DPM_FEATURES,
    SeasonRatingParams,
    build_all_season_ratings,
    create_matchup_frame,
    load_core_data,
    make_balanced_tournament_dataset,
    train_dpm_classifier,
)
from darko_like_dpm_v2 import (
    DPM_V2_FEATURES,
    SeasonRatingParamsV2,
    build_all_season_ratings_v2,
    make_balanced_tournament_dataset_v2,
    train_dpm_v2_classifier,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "2025_testing"


def create_enhanced_matchup_features(team1_stats, team2_stats, team1_seed, team2_seed, include_tempo):
    features = {
        "SeedDiff": team1_seed - team2_seed,
        "KenPomDiff": team1_stats["KenPom"] - team2_stats["KenPom"],
        "Diff_AdjO": team1_stats["AdjO"] - team2_stats["AdjO"],
        "Diff_AdjD": team1_stats["AdjD"] - team2_stats["AdjD"],
        "Diff_AdjNetRtg": team1_stats["AdjNetRtg"] - team2_stats["AdjNetRtg"],
        "Diff_SOS_NetRtg": team1_stats["SOS_NetRtg"] - team2_stats["SOS_NetRtg"],
        "Diff_Expected Win%": team1_stats["Expected Win%"] - team2_stats["Expected Win%"],
        "Diff_ThreePtRate": team1_stats["ThreePtRate"] - team2_stats["ThreePtRate"],
        "Diff_FTRate": team1_stats["FTRate"] - team2_stats["FTRate"],
        "Diff_AstRate": team1_stats["AstRate"] - team2_stats["AstRate"],
        "Diff_TORate": team1_stats["TORate"] - team2_stats["TORate"],
        "Diff_ORRate": team1_stats["ORRate"] - team2_stats["ORRate"],
        "Diff_DRRate": team1_stats["DRRate"] - team2_stats["DRRate"],
        "Diff_ScoreStdDev": team1_stats["ScoreStdDev"] - team2_stats["ScoreStdDev"],
        "Diff_MarginStdDev": team1_stats["MarginStdDev"] - team2_stats["MarginStdDev"],
        "Diff_ORtgStdDev": team1_stats["ORtgStdDev"] - team2_stats["ORtgStdDev"],
        "Diff_DRtgStdDev": team1_stats["DRtgStdDev"] - team2_stats["DRtgStdDev"],
        "Diff_HomeWin%": team1_stats["HomeWin%"] - team2_stats["HomeWin%"],
        "Diff_AwayWin%": team1_stats["AwayWin%"] - team2_stats["AwayWin%"],
        "Diff_NeutralWin%": team1_stats["NeutralWin%"] - team2_stats["NeutralWin%"],
        "Diff_Last10Win%": team1_stats["Last10Win%"] - team2_stats["Last10Win%"],
    }
    if include_tempo:
        poss1 = team1_stats.get("Poss", 0.0)
        poss2 = team2_stats.get("Poss", 0.0)
        features["Diff_Poss"] = poss1 - poss2
        features["AvgTempo_scaled"] = ((poss1 + poss2) / 2.0) * 0.2
        features["TempoDiff_scaled"] = abs(poss1 - poss2) * 0.2
    return pd.DataFrame([features])


def create_no_seeds_kenpom_features(team1_stats, team2_stats, team1_seed, team2_seed, team1_kenpom, team2_kenpom):
    features = {}
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
        features[f"{col}_diff"] = team1_stats[col] - team2_stats[col]
    features["KenPom_diff"] = team2_kenpom - team1_kenpom
    features["SoS_squared_diff"] = features["Avg_Opp_WinPct_diff"] ** 2 * np.sign(features["Avg_Opp_WinPct_diff"])
    features["SoS_WinPct_interaction"] = (
        team1_stats["Avg_Opp_WinPct"] * team1_stats["WinPct"] - team2_stats["Avg_Opp_WinPct"] * team2_stats["WinPct"]
    )
    features["SoS_Last30_interaction"] = (
        team1_stats["Avg_Opp_WinPct"] * team1_stats["Last30_WinRatio"]
        - team2_stats["Avg_Opp_WinPct"] * team2_stats["Last30_WinRatio"]
    )
    features["KenPom_SoS_interaction"] = features["KenPom_diff"] * features["Avg_Opp_WinPct_diff"]

    feature_order = [
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
    return pd.DataFrame([{feature: features.get(feature, 0.0) for feature in feature_order}])


def predict_existing_models(eval_games, enhanced_2025, basic_2025, kenpom_2025, seeds_2025):
    pipeline_with_tempo = joblib.load(ROOT / "models" / "final_model_with_tempo2.pkl")
    pipeline_no_tempo = joblib.load(ROOT / "models" / "final_model_py2.pkl")
    booster = joblib.load(ROOT / "models" / "xgb_model_no_seeds_kenpom.pkl")

    enhanced_lookup = enhanced_2025.set_index("TeamID")
    basic_lookup = basic_2025.set_index("TeamID")
    kenpom_lookup = kenpom_2025.set_index("TeamID")["OrdinalRank"]
    seed_lookup = seeds_2025.set_index("TeamID")["SeedValue"]

    rows = []
    for _, game in eval_games.iterrows():
        team1 = int(game["Team1"])
        team2 = int(game["Team2"])
        team1_seed = int(seed_lookup.loc[team1])
        team2_seed = int(seed_lookup.loc[team2])

        enhanced_x_tempo = create_enhanced_matchup_features(
            enhanced_lookup.loc[team1], enhanced_lookup.loc[team2], team1_seed, team2_seed, include_tempo=True
        )
        enhanced_x = create_enhanced_matchup_features(
            enhanced_lookup.loc[team1], enhanced_lookup.loc[team2], team1_seed, team2_seed, include_tempo=False
        )
        xgb_x = create_no_seeds_kenpom_features(
            basic_lookup.loc[team1],
            basic_lookup.loc[team2],
            team1_seed,
            team2_seed,
            float(kenpom_lookup.loc[team1]),
            float(kenpom_lookup.loc[team2]),
        )

        rows.append(
            {
                "Season": int(game["Season"]),
                "DayNum": int(game["DayNum"]),
                "Team1": team1,
                "Team2": team2,
                "Target": int(game["Target"]),
                "final_model_with_tempo2": float(pipeline_with_tempo.predict_proba(enhanced_x_tempo)[0][1]),
                "final_model_py2": float(pipeline_no_tempo.predict_proba(enhanced_x)[0][1]),
                "xgb_model_no_seeds_kenpom": float(booster.predict(xgb.DMatrix(xgb_x))[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_model(df: pd.DataFrame, prob_col: str) -> dict:
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


def build_upset_report(results: pd.DataFrame, seeds_2025: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    seed_lookup = seeds_2025.set_index("TeamID")["SeedValue"]
    name_lookup = teams.set_index("TeamID")["TeamName"]

    actual_winners = results[results["Target"] == 1].copy()
    actual_winners["WinnerSeed"] = actual_winners["Team1"].map(seed_lookup)
    actual_winners["LoserSeed"] = actual_winners["Team2"].map(seed_lookup)
    actual_winners["WinnerName"] = actual_winners["Team1"].map(name_lookup)
    actual_winners["LoserName"] = actual_winners["Team2"].map(name_lookup)
    actual_winners["SeedGap"] = actual_winners["WinnerSeed"] - actual_winners["LoserSeed"]
    upset_games = actual_winners[actual_winners["SeedGap"] > 0].copy()

    keep_cols = [
        "Season",
        "DayNum",
        "WinnerName",
        "LoserName",
        "WinnerSeed",
        "LoserSeed",
        "SeedGap",
        "darko_like_dpm",
        "darko_like_dpm_v2",
        "final_model_with_tempo2",
        "final_model_py2",
        "xgb_model_no_seeds_kenpom",
    ]
    return upset_games[keep_cols].sort_values(["SeedGap", "darko_like_dpm"], ascending=[False, False])


def main():
    regular, tourney, seeds, teams = load_core_data()
    params = SeasonRatingParams()
    ratings = build_all_season_ratings(params)
    ratings_v2 = build_all_season_ratings_v2(SeasonRatingParamsV2())

    train_tourney = tourney[(tourney["Season"] >= 2003) & (tourney["Season"] < 2025)].copy()
    test_tourney = tourney[tourney["Season"] == 2025].copy()

    train_df = make_balanced_tournament_dataset(train_tourney, ratings, seeds).dropna(subset=DPM_FEATURES)
    test_df = make_balanced_tournament_dataset(test_tourney, ratings, seeds).dropna(subset=DPM_FEATURES)
    train_df_v2 = make_balanced_tournament_dataset_v2(train_tourney, ratings_v2, seeds).dropna(subset=DPM_V2_FEATURES)
    test_df_v2 = make_balanced_tournament_dataset_v2(test_tourney, ratings_v2, seeds).dropna(subset=DPM_V2_FEATURES)

    dpm_model = train_dpm_classifier(train_df)
    dpm_v2_model = train_dpm_v2_classifier(train_df_v2)

    dpm_results = test_df[["Season", "DayNum", "Team1", "Team2", "Target", "Team1_Seed", "Team2_Seed"]].copy()
    dpm_results["darko_like_dpm"] = dpm_model.predict_proba(test_df[DPM_FEATURES])[:, 1]
    dpm_results["darko_like_dpm_v2"] = dpm_v2_model.predict_proba(test_df_v2[DPM_V2_FEATURES])[:, 1]

    enhanced = pd.read_csv(ROOT / "pre_tourney_data" / "EnhancedTournamentStats.csv")
    enhanced_2025 = enhanced[enhanced["Season"] == 2025].copy()
    kenpom = pd.read_csv(ROOT / "pre_tourney_data" / "KenPom-Rankings-Updated.csv")
    kenpom_2025 = kenpom[kenpom["Season"] == 2025][["TeamID", "OrdinalRank"]].copy()
    enhanced_2025 = enhanced_2025.merge(kenpom_2025, on="TeamID", how="left").rename(columns={"OrdinalRank": "KenPom"})

    basic = pd.read_csv(ROOT / "pre_tourney_data" / "TeamSeasonAverages_with_SoS.csv")
    basic_2025 = basic[basic["Season"] == 2025].copy()
    seeds_2025 = seeds[seeds["Season"] == 2025].copy()

    baseline_results = predict_existing_models(dpm_results, enhanced_2025, basic_2025, kenpom_2025, seeds_2025)

    results = dpm_results.merge(
        baseline_results,
        on=["Season", "DayNum", "Team1", "Team2", "Target"],
        how="left",
    )
    results = results.merge(teams.rename(columns={"TeamID": "Team1", "TeamName": "Team1Name"}), on="Team1", how="left")
    results = results.merge(teams.rename(columns={"TeamID": "Team2", "TeamName": "Team2Name"}), on="Team2", how="left")

    balanced_summary = pd.DataFrame(
        [
            summarize_model(results, "darko_like_dpm"),
            summarize_model(results, "darko_like_dpm_v2"),
            summarize_model(results, "final_model_with_tempo2"),
            summarize_model(results, "final_model_py2"),
            summarize_model(results, "xgb_model_no_seeds_kenpom"),
        ]
    ).sort_values(["accuracy", "log_loss"], ascending=[False, True])

    actual_results = results[results["Target"] == 1].copy()
    actual_summary = pd.DataFrame(
        [
            summarize_model(actual_results, "darko_like_dpm"),
            summarize_model(actual_results, "darko_like_dpm_v2"),
            summarize_model(actual_results, "final_model_with_tempo2"),
            summarize_model(actual_results, "final_model_py2"),
            summarize_model(actual_results, "xgb_model_no_seeds_kenpom"),
        ]
    ).sort_values(["accuracy", "log_loss"], ascending=[False, True])

    upsets = build_upset_report(results, seeds_2025, teams)

    results.to_csv(OUTPUT_DIR / "model_comparison_2025_matchups.csv", index=False)
    balanced_summary.to_csv(OUTPUT_DIR / "model_comparison_2025_summary_balanced.csv", index=False)
    actual_summary.to_csv(OUTPUT_DIR / "model_comparison_2025_summary.csv", index=False)
    upsets.to_csv(OUTPUT_DIR / "model_comparison_2025_upsets.csv", index=False)

    joblib.dump(
        {
            "model": dpm_model,
            "features": DPM_FEATURES,
            "params": params,
        },
        ROOT / "models" / "darko_like_dpm_model_2025_holdout.pkl",
    )
    joblib.dump(
        {
            "model": dpm_v2_model,
            "features": DPM_V2_FEATURES,
            "params": SeasonRatingParamsV2(),
        },
        ROOT / "models" / "darko_like_dpm_v2_model_2025_holdout.pkl",
    )

    # Also train the final ready-for-2026 version on all completed tournament seasons.
    full_df = make_balanced_tournament_dataset(tourney[tourney["Season"] >= 2003].copy(), ratings, seeds).dropna(subset=DPM_FEATURES)
    final_dpm = train_dpm_classifier(full_df)
    full_df_v2 = make_balanced_tournament_dataset_v2(tourney[tourney["Season"] >= 2003].copy(), ratings_v2, seeds).dropna(subset=DPM_V2_FEATURES)
    final_dpm_v2 = train_dpm_v2_classifier(full_df_v2)
    joblib.dump(
        {
            "model": final_dpm,
            "features": DPM_FEATURES,
            "params": params,
        },
        ROOT / "models" / "darko_like_dpm_model_2026.pkl",
    )
    joblib.dump(
        {
            "model": final_dpm_v2,
            "features": DPM_V2_FEATURES,
            "params": SeasonRatingParamsV2(),
        },
        ROOT / "models" / "darko_like_dpm_v2_model_2026.pkl",
    )
    ratings.to_csv(ROOT / "pre_tourney_data" / "DPMTeamRatings.csv", index=False)
    ratings_v2.to_csv(ROOT / "pre_tourney_data" / "DPMTeamRatings_v2.csv", index=False)

    print("Actual 2025 tournament games:")
    print(actual_summary.to_string(index=False))
    print("\nBalanced mirrored evaluation:")
    print(balanced_summary.to_string(index=False))
    print("\nTop upset calls:")
    print(upsets.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
