import joblib
import pandas as pd


MATCHUP_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

BASE_FEATURE_ORDER = [
    'SeedDiff', 'KenPomDiff', 'Diff_AdjO', 'Diff_AdjD', 'Diff_AdjNetRtg',
    'Diff_SOS_NetRtg', 'Diff_Expected Win%', 'Diff_ThreePtRate', 'Diff_FTRate',
    'Diff_AstRate', 'Diff_TORate', 'Diff_ORRate', 'Diff_DRRate',
    'Diff_ScoreStdDev', 'Diff_MarginStdDev', 'Diff_ORtgStdDev',
    'Diff_DRtgStdDev', 'Diff_HomeWin%', 'Diff_AwayWin%', 'Diff_NeutralWin%',
    'Diff_Last10Win%'
]


def load_latest_team_stats():
    stats = pd.read_csv("../pre_tourney_data/EnhancedTournamentStats.csv")
    stats = stats[stats["Season"] == stats["Season"].max()].copy()

    kenpom = pd.read_csv("../pre_tourney_data/KenPom-Rankings-Updated.csv")
    kenpom = kenpom[kenpom["Season"] == kenpom["Season"].max()].copy()
    kenpom = kenpom.rename(columns={"OrdinalRank": "KenPom"})

    return stats.merge(kenpom[["TeamID", "KenPom"]], on="TeamID", how="left")


def build_features(team1, team2, seed1, seed2, include_tempo=False):
    feature_values = {
        "SeedDiff": seed1 - seed2,
        "KenPomDiff": team1["KenPom"] - team2["KenPom"],
        "Diff_AdjO": team1["AdjO"] - team2["AdjO"],
        "Diff_AdjD": team1["AdjD"] - team2["AdjD"],
        "Diff_AdjNetRtg": team1["AdjNetRtg"] - team2["AdjNetRtg"],
        "Diff_SOS_NetRtg": team1["SOS_NetRtg"] - team2["SOS_NetRtg"],
        "Diff_Expected Win%": team1["Expected Win%"] - team2["Expected Win%"],
        "Diff_ThreePtRate": team1["ThreePtRate"] - team2["ThreePtRate"],
        "Diff_FTRate": team1["FTRate"] - team2["FTRate"],
        "Diff_AstRate": team1["AstRate"] - team2["AstRate"],
        "Diff_TORate": team1["TORate"] - team2["TORate"],
        "Diff_ORRate": team1["ORRate"] - team2["ORRate"],
        "Diff_DRRate": team1["DRRate"] - team2["DRRate"],
        "Diff_ScoreStdDev": team1["ScoreStdDev"] - team2["ScoreStdDev"],
        "Diff_MarginStdDev": team1["MarginStdDev"] - team2["MarginStdDev"],
        "Diff_ORtgStdDev": team1["ORtgStdDev"] - team2["ORtgStdDev"],
        "Diff_DRtgStdDev": team1["DRtgStdDev"] - team2["DRtgStdDev"],
        "Diff_HomeWin%": team1["HomeWin%"] - team2["HomeWin%"],
        "Diff_AwayWin%": team1["AwayWin%"] - team2["AwayWin%"],
        "Diff_NeutralWin%": team1["NeutralWin%"] - team2["NeutralWin%"],
        "Diff_Last10Win%": team1["Last10Win%"] - team2["Last10Win%"],
    }

    feature_order = list(BASE_FEATURE_ORDER)
    if include_tempo:
        poss1 = team1.get("Poss", 0)
        poss2 = team2.get("Poss", 0)
        feature_values["Diff_Poss"] = poss1 - poss2
        feature_values["AvgTempo_scaled"] = ((poss1 + poss2) / 2) * 0.2
        feature_values["TempoDiff_scaled"] = abs(poss1 - poss2) * 0.2
        feature_order += ["Diff_Poss", "AvgTempo_scaled", "TempoDiff_scaled"]

    return pd.DataFrame([[feature_values[column] for column in feature_order]], columns=feature_order)


def predict_matchup(model, team1, team2, seed1, seed2, include_tempo=False):
    swapped = seed1 > seed2
    if swapped:
        team1, team2 = team2, team1
        seed1, seed2 = seed2, seed1

    features = build_features(team1, team2, seed1, seed2, include_tempo=include_tempo)
    probability = model.predict_proba(features)[0][1]
    return 1 - probability if swapped else probability


def main():
    stats = load_latest_team_stats()
    bracket = pd.read_csv("bracket.csv")

    enhanced_model = joblib.load("../models/final_model_py2.pkl")
    tempo_model = joblib.load("../models/final_model_with_tempo2.pkl")

    rows = []
    for region, group in bracket.groupby("Region"):
        seed_map = {int(row.Seed): row for _, row in group.iterrows()}

        for seed_a, seed_b in MATCHUP_PAIRS:
            team_a = seed_map[seed_a]["Team"]
            team_b = seed_map[seed_b]["Team"]

            if "/" in team_a or "/" in team_b:
                rows.append({
                    "Region": region,
                    "SeedA": seed_a,
                    "TeamA": team_a,
                    "SeedB": seed_b,
                    "TeamB": team_b,
                    "status": "pending_play_in",
                })
                continue

            team_a_stats = stats[stats["TeamName"] == team_a]
            team_b_stats = stats[stats["TeamName"] == team_b]

            if team_a_stats.empty or team_b_stats.empty:
                rows.append({
                    "Region": region,
                    "SeedA": seed_a,
                    "TeamA": team_a,
                    "SeedB": seed_b,
                    "TeamB": team_b,
                    "status": "missing_team_stats",
                })
                continue

            team_a_stats = team_a_stats.iloc[0]
            team_b_stats = team_b_stats.iloc[0]

            enhanced_prob = predict_matchup(enhanced_model, team_a_stats, team_b_stats, seed_a, seed_b)
            tempo_prob = predict_matchup(tempo_model, team_a_stats, team_b_stats, seed_a, seed_b, include_tempo=True)

            rows.append({
                "Region": region,
                "SeedA": seed_a,
                "TeamA": team_a,
                "SeedB": seed_b,
                "TeamB": team_b,
                "status": "ok",
                "enhanced_teamA_win_prob": round(float(enhanced_prob), 4),
                "tempo_teamA_win_prob": round(float(tempo_prob), 4),
                "enhanced_pick": team_a if enhanced_prob >= 0.5 else team_b,
                "tempo_pick": team_a if tempo_prob >= 0.5 else team_b,
            })

    output = pd.DataFrame(rows)
    output.to_csv("first_round_predictions.csv", index=False)
    print(output.to_string(index=False))
    print("\nSaved to first_round_predictions.csv")


if __name__ == "__main__":
    main()
