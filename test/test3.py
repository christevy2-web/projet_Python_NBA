import pandas as pd
import numpy as np

# --- CORRECTION 1 : Ajout de low_memory=False ---
df_games = pd.read_csv('Games.csv', sep=';', low_memory=False)
df_teams = pd.read_csv('Team_Complete.csv')
df_players = pd.read_csv('Player Per Game.csv')

# Paramètres du modèle
HOME_ADVANTAGE = 100
K_FACTOR = 20
elo_ratings = {}

def predict_winner(home_elo, away_elo, home_stars, away_stars):
    elo_h_adj = home_elo + HOME_ADVANTAGE + (home_stars - away_stars) * 2
    prob_home = 1 / (1 + 10 ** ((away_elo - elo_h_adj) / 400))
    return prob_home

# Calcul du Star Power
df_players['impact'] = df_players['pts_per_game'] + (df_players['trb_per_game'] * 0.5)
star_power = df_players.groupby(['season', 'team'])['impact'].apply(lambda x: x.nlargest(3).sum()).to_dict()

# --- CORRECTION 2 : Gérer les années futures ---
# On trouve la dernière année présente dans ton fichier (ex: 2024 ou 2026)
last_known_season_teams = df_teams['season'].max()
last_known_season_players = df_players['season'].max()

future_seasons = [2026, 2027, 2028]
predictions = []

for season in future_seasons:
    # On prend les équipes de la dernière année connue pour faire la projection
    teams_this_season = df_teams[df_teams['season'] == last_known_season_teams]['abbreviation_x'].unique()
    
    season_ratings = []
    for team in teams_this_season:
        # On utilise les données des joueurs de la dernière année connue si on est dans le futur
        search_season_players = season if season <= last_known_season_players else last_known_season_players
        stars = star_power.get((search_season_players, team), 40)
        
        # On récupère le Win Rate de la dernière année connue
        team_data = df_teams[(df_teams['season'] == last_known_season_teams) & (df_teams['abbreviation_x'] == team)]
        win_rate = team_data['w'].iloc[0] / team_data['g'].iloc[0] if not team_data.empty else 0.5
        
        # Calcul de l'Elo projeté
        current_elo = 1500 + (win_rate - 0.5) * 400 + (stars - 45) * 5
        season_ratings.append({'team': team, 'elo': current_elo, 'stars': stars})

    # --- SÉCURITÉ ---
    # On vérifie que la liste n'est pas vide avant de créer le DataFrame
    if not season_ratings:
        print(f"Aucune donnée trouvée pour simuler l'année {season}.")
        continue

    # Simulation du Top 4 pour le titre
    df_rank = pd.DataFrame(season_ratings).sort_values(by='elo', ascending=False)
    top_1 = df_rank.iloc[0]
    top_2 = df_rank.iloc[1]
    
    prob_final = predict_winner(top_1['elo'], top_2['elo'], top_1['stars'], top_2['stars'])
    
    predictions.append({
        'Saison': season,
        'Favori n°1': top_1['team'],
        'Challenger': top_2['team'],
        'Prob Titre (%)': round(prob_final * 100, 2)
    })

# Affichage des résultats
df_future = pd.DataFrame(predictions)
print("--- PRÉDICTIONS DU MODÈLE (2026-2028) ---")
print(df_future)