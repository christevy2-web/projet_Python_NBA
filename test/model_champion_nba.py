
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.special import expit

from test.modele import load_data as modele_load_data, compute_player_rating, get_team_strengths, project_rosters, simulate_tournament


def read_csv_smart(path: str) -> pd.DataFrame:
    attempts = [
        dict(encoding="utf-8", sep=None, engine="python"),
        dict(encoding="latin1", sep=None, engine="python"),
        dict(encoding="cp1252", sep=None, engine="python"),
        dict(encoding="utf-8", sep=";", engine="python"),
        dict(encoding="latin1", sep=";", engine="python"),
        dict(encoding="cp1252", sep=";", engine="python"),
        dict(encoding="utf-8", sep=","),
        dict(encoding="latin1", sep=","),
    ]
    last_err = None
    for kw in attempts:
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Impossible de lire {path}") from last_err


def load_data():
    teams = read_csv_smart("Team_Complete.csv")  
    players = read_csv_smart("Player Totals.csv")               
    games = read_csv_smart("Games.csv")  

    for df in (teams, players, games):
        df.columns = [c.strip() for c in df.columns]

    teams = teams[teams["team"].ne("League Average")].copy()
    teams["season"] = teams["season"].astype(int)
    players["season"] = players["season"].astype(int)

    games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"], dayfirst=True, errors="coerce")
    games["season"] = np.where(
        games["gameDateTimeEst"].dt.month >= 7,
        games["gameDateTimeEst"].dt.year + 1,
        games["gameDateTimeEst"].dt.year,
    ).astype(int)
    games["home_team"] = games["hometeamCity"].astype(str).str.strip() + " " + games["hometeamName"].astype(str).str.strip()
    games["away_team"] = games["awayteamCity"].astype(str).str.strip() + " " + games["awayteamName"].astype(str).str.strip()

    reg_games = games[(games["gameType"] == "Regular Season") & games["homeScore"].gt(0) & games["awayScore"].gt(0)].copy()
    return teams, players, reg_games


def build_elo(reg_games: pd.DataFrame, home_adv: float = 65.0, k: float = 20.0, season_regr: float = 0.75):
    def expected(r_a, r_b):
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))

    def mov_multiplier(mov, rd):
        return math.log(abs(mov) + 1.0) * (2.2 / ((abs(rd) * 0.001) + 2.2))

    ratings = {}
    season_end = {}
    prev_season = None

    for season, g in reg_games.sort_values("gameDateTimeEst").groupby("season", sort=True):
        if prev_season is not None:
            for team in list(ratings.keys()):
                ratings[team] = season_regr * ratings[team] + (1 - season_regr) * 1500.0
        prev_season = season

        for _, row in g.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            hs = float(row["homeScore"])
            aas = float(row["awayScore"])

            ratings.setdefault(home, 1500.0)
            ratings.setdefault(away, 1500.0)

            r_home = ratings[home]
            r_away = ratings[away]

            p_home = expected(r_home + home_adv, r_away)
            home_win = 1.0 if hs > aas else 0.0
            delta = k * mov_multiplier(hs - aas, r_home - r_away) * (home_win - p_home)

            ratings[home] = r_home + delta
            ratings[away] = r_away - delta

        season_end[season] = ratings.copy()

    elo_rows = [
        {"season": season, "team": team, "elo_end": rating}
        for season, season_map in season_end.items()
        for team, rating in season_map.items()
    ]
    return pd.DataFrame(elo_rows)


def build_player_aggregates(players: pd.DataFrame):
    players_clean = players[~players["team"].isin(["2TM", "3TM", "4TM", "5TM"])].copy()
    for c in ["mp", "pts", "ast", "trb", "stl", "blk", "tov", "pf", "fg", "fga", "x3p", "x3pa", "x2p", "x2pa", "ft", "fta", "orb", "drb", "gs", "g", "age"]:
        players_clean[c] = pd.to_numeric(players_clean[c], errors="coerce")

    def weighted_age(g):
        mp = g["mp"].fillna(0)
        age = g["age"].fillna(g["age"].mean())
        s = mp.sum()
        return float(np.average(age, weights=mp)) if s > 0 else np.nan

    base = players_clean.groupby(["season", "team"]).agg(
        players=("player_id", "count"),
        total_mp=("mp", "sum"),
        total_pts=("pts", "sum"),
        total_ast=("ast", "sum"),
        total_trb=("trb", "sum"),
        total_stl=("stl", "sum"),
        total_blk=("blk", "sum"),
        total_tov=("tov", "sum"),
        avg_age=("age", "mean"),
    ).reset_index()
    base["wavg_age"] = players_clean.groupby(["season", "team"]).apply(weighted_age, include_groups=False).values

    def topk_summary(g):
        out = {}
        for col, ks in [("pts", [1, 3, 5]), ("mp", [1, 3, 5])]:
            vals = pd.to_numeric(g[col], errors="coerce").fillna(0).sort_values(ascending=False)
            total = vals.sum()
            for k in ks:
                out[f"top{k}_{col}"] = vals.head(k).sum()
            out[f"{col}_share_top3"] = vals.head(3).sum() / total if total > 0 else np.nan
        out["rotation_players_500mp"] = int((pd.to_numeric(g["mp"], errors="coerce").fillna(0) >= 500).sum())
        out["rotation_players_1000mp"] = int((pd.to_numeric(g["mp"], errors="coerce").fillna(0) >= 1000).sum())
        return pd.Series(out)

    top = players_clean.groupby(["season", "team"]).apply(topk_summary, include_groups=False).reset_index()
    return base.merge(top, on=["season", "team"], how="left")


def build_model_table(teams: pd.DataFrame, elo_df: pd.DataFrame, player_agg: pd.DataFrame):
    df = teams.copy()
    df["win_pct"] = df["w"] / (df["w"] + df["l"])
    for c in ["pts", "trb", "ast", "tov", "fg", "fga", "x3p", "x3pa", "ft", "fta", "orb", "drb", "stl", "blk"]:
        df[f"{c}_pg"] = df[c] / df["g"]

    df = df.merge(elo_df, on=["season", "team"], how="left")
    df = df.merge(player_agg, left_on=["season", "abbreviation_x"], right_on=["season", "team"], how="left", suffixes=("", "_player"))

    for c in ["total_mp", "total_pts", "total_ast", "total_trb", "total_stl", "total_blk", "total_tov",
              "top1_pts", "top3_pts", "top5_pts", "top1_mp", "top3_mp", "top5_mp"]:
        df[f"{c}_pg"] = df[c] / df["g"]

    df["pts_per_player"] = df["total_pts"] / df["players"]
    df["mp_per_player"] = df["total_mp"] / df["players"]
    return df


def train_champion_model(model_df: pd.DataFrame):
    feature_cols = [
        "win_pct", "g", "elo_end",
        "pts_pg", "trb_pg", "ast_pg", "tov_pg", "fg_percent", "x3p_percent", "ft_percent",
        "orb_pg", "drb_pg", "stl_pg", "blk_pg",
        "players", "avg_age", "wavg_age",
        "total_pts_pg", "total_ast_pg", "total_trb_pg", "total_tov_pg",
        "top1_pts_pg", "top3_pts_pg", "top5_pts_pg", "pts_share_top3",
        "top1_mp_pg", "top3_mp_pg", "top5_mp_pg", "mp_share_top3",
        "rotation_players_500mp", "rotation_players_1000mp",
        "pts_per_player", "mp_per_player",
    ]

    train_df = model_df[(model_df["season"] <= 2025) & model_df["is_champion"].notna()].copy()
    train_df["target"] = train_df["is_champion"].astype(int)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=5000, class_weight="balanced", C=0.7, solver="liblinear")),
    ])
    clf.fit(train_df[feature_cols], train_df["target"])
    return clf, feature_cols


def project_season(df: pd.DataFrame, feature_cols: list[str], model_df: pd.DataFrame, years_ahead: int = 1):
    proj = df.copy()
    hist_means = model_df[feature_cols].mean(numeric_only=True)
    carry = 0.78 ** years_ahead

    for col in feature_cols:
        if col == "g":
            proj[col] = 82.0
        elif col in ["avg_age", "wavg_age"]:
            proj[col] = pd.to_numeric(proj[col], errors="coerce") + years_ahead
        elif col == "win_pct":
            proj[col] = carry * pd.to_numeric(proj[col], errors="coerce") + (1 - carry) * 0.50
        elif col == "elo_end":
            proj[col] = carry * pd.to_numeric(proj[col], errors="coerce") + (1 - carry) * 1500.0
        else:
            proj[col] = carry * pd.to_numeric(proj[col], errors="coerce") + (1 - carry) * hist_means[col]
    return proj


def simulate_bracket(df: pd.DataFrame, top_n: int = 16, n_sim: int = 3000, seed: int = 7):
    score_map = df.set_index("team")["score"].to_dict()
    teams_sorted = df.sort_values("score", ascending=False).head(top_n)["team"].tolist()
    counts = {t: 0 for t in teams_sorted}

    def series_winner(a, b):
        p = expit(score_map[a] - score_map[b])
        wa = wb = 0
        while wa < 4 and wb < 4:
            if np.random.rand() < p:
                wa += 1
            else:
                wb += 1
        return a if wa > wb else b

    for i in range(n_sim):
        np.random.seed(seed + i)
        bracket = teams_sorted.copy()
        while len(bracket) > 1:
            nxt = []
            for j in range(0, len(bracket), 2):
                nxt.append(series_winner(bracket[j], bracket[j + 1]))
            bracket = nxt
        counts[bracket[0]] += 1

    return pd.DataFrame({"team": list(counts.keys()), "champion_prob": [v / n_sim for v in counts.values()]})


def predict_year(df, clf, feature_cols, model_df, years_ahead: int = 0, n_sim: int = 3000):
    if years_ahead > 0:
        df = project_season(df, feature_cols, model_df, years_ahead=years_ahead)

    df = df.copy()
    df["score"] = clf.decision_function(df[feature_cols])
    probs = simulate_bracket(df, top_n=16, n_sim=n_sim)
    out = df[["team", "season", "score", "win_pct", "elo_end"]].merge(probs, on="team", how="left").fillna({"champion_prob": 0})
    return out.sort_values("champion_prob", ascending=False)


def plot_predictions_histogram(pred_df, year, top_n=10):
    """Crée un histogramme des probabilités de champion avec pourcentages"""
    top_teams = pred_df.head(top_n).copy()
    top_teams["champion_prob_pct"] = top_teams["champion_prob"] * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    
    bars = ax.barh(top_teams["team"], top_teams["champion_prob_pct"], 
                    color='#58a6ff', edgecolor='white', linewidth=1.5)
    
    # Ajouter les pourcentages sur les barres
    for i, (bar, pct) in enumerate(zip(bars, top_teams["champion_prob_pct"])):
        ax.text(pct + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', color='white', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Probabilité du titre (%)', color='white', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} favoris pour le titre NBA {year}\n(Prédictions du modèle)', 
                 color='white', fontsize=12, fontweight='bold', pad=15)
    ax.tick_params(colors='white', labelsize=10)
    ax.set_xlim(0, max(top_teams["champion_prob_pct"]) * 1.15)
    
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    
    plt.tight_layout()
    plt.savefig(f'champions_histogram_{year}.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f"✅ Sauvegardé : champions_histogram_{year}.png")


def main():
    teams, players, reg_games = load_data()
    elo_df = build_elo(reg_games)
    player_agg = build_player_aggregates(players)
    model_df = build_model_table(teams, elo_df, player_agg)
    clf, feature_cols = train_champion_model(model_df)

    current_2026 = model_df[model_df["season"] == 2026].copy()

    pred_2026 = predict_year(current_2026, clf, feature_cols, model_df, years_ahead=0, n_sim=3000)
    pred_2027 = predict_year(current_2026, clf, feature_cols, model_df, years_ahead=1, n_sim=3000)
    pred_2028 = predict_year(current_2026, clf, feature_cols, model_df, years_ahead=2, n_sim=3000)

    pred_2026["season_pred"] = 2026
    pred_2027["season_pred"] = 2027
    pred_2028["season_pred"] = 2028

    all_preds = pd.concat([pred_2026, pred_2027, pred_2028], ignore_index=True)
    all_preds.to_csv("predictions_2026_2028.csv", index=False)

    print("\n" + "="*60)
    print("PRÉDICTIONS - MODÈLE DE RÉGRESSION LOGISTIQUE")
    print("="*60)
    
    print("\n🏀 Top 10 - 2026")
    print(pred_2026.head(10).to_string(index=False))
    print("\n🏀 Top 10 - 2027")
    print(pred_2027.head(10).to_string(index=False))
    print("\n🏀 Top 10 - 2028")
    print(pred_2028.head(10).to_string(index=False))
    
    # Créer les histogrammes avec pourcentages
    print("\n📊 Création des histogrammes...")
    plot_predictions_histogram(pred_2026, 2026, top_n=10)
    plot_predictions_histogram(pred_2027, 2027, top_n=10)
    plot_predictions_histogram(pred_2028, 2028, top_n=10)


if __name__ == "__main__":
    main()