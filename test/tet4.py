import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Chargement et Nettoyage (Correction Encodage incluse)
try:
    df = pd.read_csv('Player Totals.csv', sep=';', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('Player Totals.csv', sep=';', encoding='latin-1')

cols = ['pts', 'trb', 'ast', 'stl', 'blk', 'g']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- ANALYSE 1 : HISTOGRAMME TOP 10 ALL TIME ---
top_10_all_time = df.groupby('player')['pts'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
top_10_all_time.plot(kind='bar', color='midnightblue', edgecolor='black')
plt.title('Top 10 Meilleurs Marqueurs de l\'Histoire', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right') # Noms inclinés
plt.ylabel('Points Totaux')
plt.tight_layout()
plt.show()

# --- ANALYSE 2 : PRÉPARATION DONNÉES 2026 ---
df_2026 = df[df['season'] == 2026].copy()
# Efficacité globale (MVT/All-Star)
df_2026['eff_pg'] = (df_2026['pts'] + df_2026['trb'] + df_2026['ast'] + df_2026['stl'] + df_2026['blk']) / df_2026['g'].replace(0, 1)

# --- ANALYSE 3 : HISTOGRAMME TOP DÉFENSEURS ---
df_2026['defense'] = (df_2026['stl'] + df_2026['blk']) / df_2026['g'].replace(0, 1)
top_5_def = df_2026.sort_values(by='defense', ascending=False).head(5)
print("\n🛡️ Top 5 défenseurs 2026 (steals + blocks par match) :")
print(top_5_def[['player','defense']].to_string(index=False))

plt.figure(figsize=(10, 6))
plt.bar(top_5_def['player'], top_5_def['defense'], color='darkgreen')
plt.title('Top 5 Défenseurs 2026 (Interceptions + Contres / Match)', fontsize=13)
plt.xticks(rotation=45, ha='right') # Noms inclinés
plt.ylabel('Stats Défensives Moyennes')
plt.tight_layout()
plt.show()

# --- ANALYSE 4 : RADAR PAR POSTE + VICTOR WEMBANYAMA ---
def plot_radar_with_victor():
    positions = ['C', 'PF', 'SF', 'SG', 'PG']
    labels = ['PTS', 'TRB', 'AST', 'STL', 'BLK']
    ref_scales = [30, 15, 12, 3, 3] # Normalisation indicative
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Couleurs distinctes pour les postes (Pastels)
    colors = ['#f4a582', '#92c5de', '#a6d96a', '#d7191c', '#2c7bb6']
    
    # Trace les 5 meilleurs par poste
    for i, pos in enumerate(positions):
        # Meilleur joueur à ce poste pur
        best_in_pos = df_2026[df_2026['pos'] == pos].sort_values('eff_pg', ascending=False).iloc[0]
        
        # Calcul et Normalisation
        values = [best_in_pos['pts']/best_in_pos['g'], best_in_pos['trb']/best_in_pos['g'], 
                  best_in_pos['ast']/best_in_pos['g'], best_in_pos['stl']/best_in_pos['g'], 
                  best_in_pos['blk']/best_in_pos['g']]
        norm_values = [min(v / s, 1.2) for v, s in zip(values, ref_scales)]
        norm_values += norm_values[:1]
        
        ax.plot(angles, norm_values, color=colors[i], linewidth=2.5, linestyle=':', label=f"Leader {pos}: {best_in_pos['player']}")
        ax.fill(angles, norm_values, color=colors[i], alpha=0.05)

    # AJOUT SPÉCIFIQUE DE VICTOR WEMBANYAMA
    wemb_data = df_2026[df_2026['player'] == 'Victor Wembanyama'].iloc[0]
    values_w = [wemb_data['pts']/wemb_data['g'], wemb_data['trb']/wemb_data['g'], 
                wemb_data['ast']/wemb_data['g'], wemb_data['stl']/wemb_data['g'], 
                wemb_data['blk']/wemb_data['g']]
    norm_w = [min(v / s, 1.2) for v, s in zip(values_w, ref_scales)] + [min(values_w[0]/ref_scales[0], 1.2)]
    
    # Victor est en trait plein NOIR pour se démarquer
    ax.plot(angles, norm_w, color='black', linewidth=4, label=f"C: Victor Wembanyama", marker='o', markersize=8)
    ax.fill(angles, norm_w, color='black', alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_yticklabels([])
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), title="Légende Comparaison")
    plt.title("Profil unique de Victor Wembanyama vs Leaders par Poste (2026)", size=16, y=1.1, fontweight='bold')
    plt.show()

plot_radar_with_victor()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- ANALYSE 5 : GRAPHIQUE DU TERRAIN AVEC LES ALL-STARS 2026 ---

# 1. Préparation des données : Top 12 basé sur le score 'composite'
# Note : Assure-toi que la colonne 'composite' existe dans df_2026
all_stars = df_2026[df_2026['season'] == 2026].nlargest(12, 'composite')

print("\n⭐ Classement All-Star 2026 (Top 12) :")
print(all_stars[['player', 'pts_g', 'trb_g', 'ast_g', 'stl_g', 'blk_g', 'composite']].to_string(index=False))

def dessiner_terrain(ax):
    """Trace les lignes d'un terrain de basket standard"""
    # Contour du terrain (Dimensions NBA : 94x50 pieds, ajustées ici à 100x94 pour ton code)
    ax.plot([0, 100, 100, 0, 0], [0, 0, 94, 94, 0], color="black")
    
    # Cercle central
    ax.add_patch(patches.Circle((50, 47), 6, fill=False))
    
    # Arcs de la ligne à 3 points (haut et bas)
    ax.add_patch(patches.Arc((50, 5.25), 47.5, 47.5, theta1=0, theta2=180, fill=False))
    ax.add_patch(patches.Arc((50, 88.75), 47.5, 47.5, theta1=180, theta2=360, fill=False))
    
    # Masquer les axes pour un rendu propre
    ax.axis('off')

# 2. Création de la figure
fig, ax = plt.subplots(figsize=(7, 9))
dessiner_terrain(ax)

# 3. Placement des joueurs sur le terrain
for i, (idx, row) in enumerate(all_stars.iterrows()):
    # Calcul des coordonnées pour répartir les joueurs en deux colonnes
    x = 25 if i % 2 == 0 else 75
    y = 10 + (i // 2) * 14
    
    # Dessiner le point du joueur
    ax.scatter(x, y, s=500, color='gold', edgecolors='black', zorder=3)
    
    # Ajouter le nom et le rang
    nom_joueur = f"{i+1}. {row['player']}"
    ax.text(x, y + 3, nom_joueur, ha='center', fontweight='bold', fontsize=9)

plt.title('Positionnement des All-Stars 2026 sur le terrain', pad=20, fontsize=14)
plt.tight_layout()
plt.show()