import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

output_dir = "images_players"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Chargement et Nettoyage
try:
    df = pd.read_csv('Player Totals.csv', sep=';', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('Player Totals.csv', sep=';', encoding='latin-1')

# Conversion en numérique et gestion des valeurs manquantes
cols = ['pts', 'trb', 'ast', 'stl', 'blk', 'g']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- ANALYSE 1 : HISTOGRAMME TOP 10 ALL TIME ---
top_10_all_time = df.groupby('player')['pts'].sum().sort_values(ascending=False).head(10)
print("🏆 Top 10 Meilleurs Marqueurs de l'Histoire :")
print(top_10_all_time.to_string())

plt.figure(figsize=(12, 6))
top_10_all_time.plot(kind='bar', color='midnightblue', edgecolor='black')
plt.title("Top 10 Meilleurs Marqueurs de l'Histoire", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Points Totaux')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_10_all_time.pdf"), bbox_inches='tight')
plt.show()

# --- ANALYSE 2 : PRÉPARATION DONNÉES 2026 ---
df_2026 = df[df['season'] == 2026].copy()

# Calcul des moyennes par match (indispensable pour l'affichage final)
df_2026['pts_g'] = df_2026['pts'] / df_2026['g'].replace(0, 1)
df_2026['trb_g'] = df_2026['trb'] / df_2026['g'].replace(0, 1)
df_2026['ast_g'] = df_2026['ast'] / df_2026['g'].replace(0, 1)
df_2026['stl_g'] = df_2026['stl'] / df_2026['g'].replace(0, 1)
df_2026['blk_g'] = df_2026['blk'] / df_2026['g'].replace(0, 1)

# CRÉATION DE LA COLONNE MANQUANTE 'composite' 
# On définit le score All-Star comme la somme des stats moyennes
df_2026['composite'] = df_2026['pts_g'] + df_2026['trb_g'] + df_2026['ast_g'] + df_2026['stl_g'] + df_2026['blk_g']

# --- ANALYSE 3 : TOP DÉFENSEURS ---
df_2026['defense'] = df_2026['stl_g'] + df_2026['blk_g']
top_defenders = df_2026.nlargest(5, 'defense')
print("\n🛡️ Top 5 défenseurs 2026 (steals + blocks par match) :")
print(top_defenders[['player', 'stl_g', 'blk_g', 'defense']].to_string(index=False))

plt.figure(figsize=(10, 6))
plt.bar(top_defenders['player'], top_defenders['defense'], color='darkgreen')
plt.title('Top 5 Défenseurs 2026 (Interceptions + Contres / Match)', fontsize=13)
plt.ylabel('Stats Défensives Moyennes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "TOP 5 DÉFENSEURS 2026.pdf"), bbox_inches='tight')
plt.show()

# --- ANALYSE 4 : RADAR AVEC JOUEURS SUPPLÉMENTAIRES ---
def plot_radar_with_extra():
    positions = ['C', 'PF', 'SF', 'SG', 'PG']
    labels = ['PTS', 'REB', 'AST', 'STL', 'BLK']  # rebonds abrégé
    ref_scales = [30, 15, 12, 3, 3]
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors_pos = ['#f4a582', '#92c5de', '#a6d96a', '#d7191c', '#2c7bb6']
    
    # Leaders par poste (pointillés)
    for i, pos in enumerate(positions):
        subset = df_2026[df_2026['pos'] == pos]
        if not subset.empty:
            best = subset.sort_values('composite', ascending=False).iloc[0]
            values = [best['pts_g'], best['trb_g'], best['ast_g'], best['stl_g'], best['blk_g']]
            norm_vals = [min(v / s, 1.2) for v, s in zip(values, ref_scales)]
            norm_vals += norm_vals[:1]
            ax.plot(angles, norm_vals, color=colors_pos[i], linestyle=':', linewidth=2,
                    label=f"Leader {pos}: {best['player']}")
    
    # Victor Wembanyama
    wemb = df_2026[df_2026['player'] == 'Victor Wembanyama']
    if not wemb.empty:
        w = wemb.iloc[0]
        poste_w = w['pos']
        w_vals = [w['pts_g'], w['trb_g'], w['ast_g'], w['stl_g'], w['blk_g']]
        w_norm = [min(v / s, 1.2) for v, s in zip(w_vals, ref_scales)]
        w_norm += w_norm[:1]
        ax.plot(angles, w_norm, color='black', linewidth=3, marker='o', label='Victor Wembanyama')
        ax.fill(angles, w_norm, color='black', alpha=0.1)
    
    # Joueurs additionnels
    extra_players = ['Cason Wallace', 'Tyrese Maxey', 'Cade Cunningham', 'Karl-Anthony Towns']
    extra_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    # Joueurs additionnels
    extra_players = ['Cason Wallace', 'Tyrese Maxey', 'Cade Cunningham', 'Karl-Anthony Towns']
    extra_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    for idx, name in enumerate(extra_players):
        player = df_2026[df_2026['player'] == name]
        if not player.empty:
            p = player.iloc[0]
            poste = p['pos'] 
            vals = [p['pts_g'], p['trb_g'], p['ast_g'], p['stl_g'], p['blk_g']]
            norm_vals = [min(v / s, 1.2) for v, s in zip(vals, ref_scales)]
            norm_vals += norm_vals[:1]
            ax.plot(angles, norm_vals, color=extra_colors[idx], linewidth=2, marker='s',
                    label=f"{poste}: {name}")
    
    # Personnalisation du radar
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Légende optimisée : placée à droite, sans chevauchement
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize=10)
    legend.get_frame().set_alpha(0.9)
    
    # Titre et sous-titre
    plt.subplots_adjust(bottom=0.2, right=0.8)

    plt.title("Comparaison radar : Leaders 2026, Wembanyama, et joueurs ciblés\n", 
              fontsize=14, fontweight='bold')
    plt.figtext(0.80, 0.05, 
                "LÉGENDE: PTS=Points, TRB=Rebonds, AST=Passes, STL=Interceptions, BLK=Contres\n"
                "POSTES: C=Pivot, PF=Ailier Fort, SF=Ailier, SG=Arrière, PG=Meneur",
                ha='right', fontsize=9, style='italic', 
                bbox=dict(facecolor='oldlace', alpha=0.5, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/radar_comparaison_2026.pdf", bbox_inches='tight')
    plt.show()

plot_radar_with_extra()


# --- ANALYSE 5 : TERRAIN ALL-STAR ---
all_stars = df_2026[df_2026['season'] == 2026].nlargest(12, 'composite')

print("\n⭐ Classement All-Star 2026 (Top 12) :")
print(all_stars[['player', 'pts_g', 'trb_g', 'ast_g', 'stl_g', 'blk_g', 'composite']].to_string(index=False))


def dessiner_terrain(ax):
    ax.plot([0, 100, 100, 0, 0], [0, 0, 94, 94, 0], color="black")
    ax.add_patch(patches.Circle((50, 47), 6, fill=False))
    ax.add_patch(patches.Arc((50, 5.25), 47.5, 47.5, theta1=0, theta2=180, fill=False))
    ax.add_patch(patches.Arc((50, 88.75), 47.5, 47.5, theta1=180, theta2=360, fill=False))
    ax.axis('off')

# Sélection du Top 12 (Le KeyError 'composite' est réglé ici)
all_stars = df_2026.nlargest(12, 'composite')

fig, ax = plt.subplots(figsize=(7, 9))
dessiner_terrain(ax)

for i, (idx, row) in enumerate(all_stars.iterrows()):
    x = 25 if i % 2 == 0 else 75
    y = 10 + (i // 2) * 14
    ax.scatter(x, y, s=500, color='gold', edgecolors='black', zorder=3)
    ax.text(x, y + 3, f"{i+1}. {row['player']}", ha='center', fontweight='bold', fontsize=9)

plt.title('Positionnement des All-Stars 2026 sur le terrain', pad=20, fontsize=14)
plt.savefig(f"{output_dir}/terrain_all_star_2026.pdf", bbox_inches='tight')
plt.show()