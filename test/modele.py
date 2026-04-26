import pandas as pd
import numpy as np
from scipy.special import expit
import os

# ==========================================
# LECTURE ROBUSTE DES CSV
# ==========================================
def read_csv_smart(path: str) -> pd.DataFrame:
    """Essaie plusieurs encodages et séparateurs pour lire le CSV"""
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

# ==========================================
# 1. CHARGEMENT SIMPLIFIÉ (VIA TON FICHIER COMPLET)
# ==========================================
def load_data():
    # On utilise ton fichier déjà fusionné
    path_teams = 'Team_complete_avec_champions.csv'
    path_players = 'Player Totals.csv'
    
    if not os.path.exists(path_teams) or not os.path.exists(path_players):
        raise FileNotFoundError("Assure-tu d'avoir 'Team_complete_avec_champions.csv' et 'Player Totals.csv' dans le dossier.")

    # Lecture avec détection automatique du séparateur et de l'encodage
    teams = read_csv_smart(path_teams)
    players = read_csv_smart(path_players)
    
    # Nettoyage
    teams.columns = [c.strip() for c in teams.columns]
    players.columns = [c.strip() for c in players.columns]
    
    # On s'assure que is_champion est bien en booléen/numérique
    if 'is_champion' in teams.columns:
        teams['is_champion'] = teams['is_champion'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0)
    
    return teams, players

# ==========================================
# 2. LOGIQUE JOUEUR & VIEILLISSEMENT
# ==========================================
def compute_player_rating(players_df):
    df = players_df.copy()
    cols = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'mp', 'g', 'age']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['games'] = df['g'].replace(0, 1)
    # Score de performance individuelle
    df['player_rating'] = (
        df['pts'] + (1.2 * df['trb']) + (1.5 * df['ast']) - (1.2 * df['tov'])
    ) / df['games'] * ( (df['mp']/df['games']) / 30)
    
    return df

def project_rosters(players_df, years_ahead):
    df = players_df.copy()
    
    def aging_logic(age):
        target_age = age + years_ahead
        if target_age < 24: return np.random.normal(1.07, 0.03) # Progression
        if target_age <= 29: return np.random.normal(1.02, 0.02) # Peak
        return np.random.normal(0.90, 0.05) # Déclin

    df['projected_rating'] = df.apply(lambda x: x['player_rating'] * aging_logic(x['age']), axis=1)
    return df

# ==========================================
# 3. CALCUL DE FORCE & SIMULATION
# ==========================================
def get_team_strengths(projected_players):
    strengths = {}
    weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3] # Poids Top 8
    
    for team in projected_players['team'].unique():
        # On utilise l'abréviation ou le nom pour matcher
        p_team = projected_players[projected_players['team'] == team]
        top_8 = p_team.sort_values('projected_rating', ascending=False).head(8)['projected_rating'].values
        
        if len(top_8) < 8:
            top_8 = np.pad(top_8, (0, 8 - len(top_8)), 'constant')
        
        strengths[team] = np.sum(top_8 * weights)
    return strengths

def simulate_tournament(strengths):
    # Top 16 équipes basées sur la force
    contenders = sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:16]
    bracket = [c[0] for c in contenders]
    
    while len(bracket) > 1:
        next_round = []
        for i in range(0, len(bracket), 2):
            t1, t2 = bracket[i], bracket[-i-1]
            p1 = expit((strengths[t1] - strengths[t2]) / 15)
            # Simulation Best-of-7
            next_round.append(t1 if np.random.binomial(7, p1) >= 4 else t2)
        bracket = next_round
    return bracket[0]

# ==========================================
# 4. EXECUTION DES PRÉDICTIONS
# ==========================================
def run_simulation_pipeline(n_sim=1000):
    teams, players = load_data()
    players_base = compute_player_rating(players)
    
    # On commence la projection à partir de la dernière saison du fichier (2026)
    last_season = teams['season'].max()
    current_rosters = players_base[players_base['season'] == last_season]

    results = []
    for year in [2026, 2027, 2028]:
        print(f"Simulation de l'année {year}...")
        winners = {}
        years_ahead = year - 2025 # Décalage par rapport à la base
        
        for _ in range(n_sim):
            proj = project_rosters(current_rosters, years_ahead)
            strengths = get_team_strengths(proj)
            winner = simulate_tournament(strengths)
            winners[winner] = winners.get(winner, 0) + 1
            
        for t, count in winners.items():
            results.append({'season': year, 'team': t, 'prob': count/n_sim})
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    final_preds = run_simulation_pipeline()
    
    for y in [2026, 2027, 2028]:
        print(f"\n--- Favoris {y} ---")
        print(final_preds[final_preds['season']==y].sort_values('prob', ascending=False).head(5))