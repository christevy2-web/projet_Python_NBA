import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration du style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Charger les données
df = pd.read_csv('Team Totals.csv', sep=';')

# 2. Nettoyer et préparer les données
# Filtrer uniquement la NBA (optionnel)
df = df[df['lg'] == 'NBA']

# Agréger les données par saison (totaux de la ligue)
season_stats = df.groupby('season').agg({
    'x2p': 'sum',      # Tirs à 2 points marqués
    'x3p': 'sum',      # Tirs à 3 points marqués
    'ft': 'sum',       # Lancers-francs marqués
    'stl': 'sum',      # Interceptions
    'blk': 'sum',      # Contres
    'pf': 'sum',       # Fautes personnelles
    'orb': 'sum',      # Rebonds offensifs
    'drb': 'sum',      # Rebonds défensifs
    'g': 'sum'         # Nombre de matchs
}).reset_index()

# Calculer les moyennes par match pour la ligue
season_stats['x2p_per_game'] = season_stats['x2p'] / season_stats['g']
season_stats['x3p_per_game'] = season_stats['x3p'] / season_stats['g']
season_stats['ft_per_game'] = season_stats['ft'] / season_stats['g']
season_stats['stl_per_game'] = season_stats['stl'] / season_stats['g']
season_stats['blk_per_game'] = season_stats['blk'] / season_stats['g']
season_stats['pf_per_game'] = season_stats['pf'] / season_stats['g']
season_stats['orb_per_game'] = season_stats['orb'] / season_stats['g']
season_stats['drb_per_game'] = season_stats['drb'] / season_stats['g']

# Trier par saison
season_stats = season_stats.sort_values('season')

# 3. Créer la figure avec 6 sous-graphiques
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Évolution des statistiques NBA (moyenne par match)', fontsize=16, fontweight='bold')

# Graphique 1 : Tirs marqués (2 pts, 3 pts, LF)
ax1 = axes[0, 0]
ax1.plot(season_stats['season'], season_stats['x2p_per_game'], 
         marker='o', linewidth=2, markersize=4, label='Tirs à 2 pts', color='#1f77b4')
ax1.plot(season_stats['season'], season_stats['x3p_per_game'], 
         marker='s', linewidth=2, markersize=4, label='Tirs à 3 pts', color='#ff7f0e')
ax1.plot(season_stats['season'], season_stats['ft_per_game'], 
         marker='^', linewidth=2, markersize=4, label='Lancers-francs', color='#2ca02c')
ax1.set_title('Tirs marqués par match', fontsize=12, fontweight='bold')
ax1.set_xlabel('Saison')
ax1.set_ylabel('Moyenne par match')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Graphique 2 : Interceptions
ax2 = axes[0, 1]
ax2.plot(season_stats['season'], season_stats['stl_per_game'], 
         marker='o', linewidth=2, markersize=4, color='#9467bd')
ax2.fill_between(season_stats['season'], 0, season_stats['stl_per_game'], alpha=0.3, color='#9467bd')
ax2.set_title('Interceptions par match', fontsize=12, fontweight='bold')
ax2.set_xlabel('Saison')
ax2.set_ylabel('Moyenne par match')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Graphique 3 : Contres
ax3 = axes[0, 2]
ax3.plot(season_stats['season'], season_stats['blk_per_game'], 
         marker='s', linewidth=2, markersize=4, color='#d62728')
ax3.fill_between(season_stats['season'], 0, season_stats['blk_per_game'], alpha=0.3, color='#d62728')
ax3.set_title('Contres par match', fontsize=12, fontweight='bold')
ax3.set_xlabel('Saison')
ax3.set_ylabel('Moyenne par match')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Graphique 4 : Fautes personnelles
ax4 = axes[1, 0]
ax4.plot(season_stats['season'], season_stats['pf_per_game'], 
         marker='^', linewidth=2, markersize=4, color='#e377c2')
ax4.fill_between(season_stats['season'], 0, season_stats['pf_per_game'], alpha=0.3, color='#e377c2')
ax4.set_title('Fautes personnelles par match', fontsize=12, fontweight='bold')
ax4.set_xlabel('Saison')
ax4.set_ylabel('Moyenne par match')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# Graphique 5 : Rebonds offensifs
ax5 = axes[1, 1]
ax5.plot(season_stats['season'], season_stats['orb_per_game'], 
         marker='D', linewidth=2, markersize=4, color='#8c564b')
ax5.fill_between(season_stats['season'], 0, season_stats['orb_per_game'], alpha=0.3, color='#8c564b')
ax5.set_title('Rebonds offensifs par match', fontsize=12, fontweight='bold')
ax5.set_xlabel('Saison')
ax5.set_ylabel('Moyenne par match')
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45)

# Graphique 6 : Rebonds défensifs
ax6 = axes[1, 2]
ax6.plot(season_stats['season'], season_stats['drb_per_game'], 
         marker='p', linewidth=2, markersize=4, color='#17becf')
ax6.fill_between(season_stats['season'], 0, season_stats['drb_per_game'], alpha=0.3, color='#17becf')
ax6.set_title('Rebonds défensifs par match', fontsize=12, fontweight='bold')
ax6.set_xlabel('Saison')
ax6.set_ylabel('Moyenne par match')
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 4. Graphique combiné des statistiques défensives (Interceptions + Contres)
fig2, ax = plt.subplots(figsize=(12, 6))

ax.plot(season_stats['season'], season_stats['stl_per_game'], 
        marker='o', linewidth=2, markersize=4, label='Interceptions', color='#9467bd')
ax.plot(season_stats['season'], season_stats['blk_per_game'], 
        marker='s', linewidth=2, markersize=4, label='Contres', color='#d62728')
ax.plot(season_stats['season'], season_stats['pf_per_game'], 
        marker='^', linewidth=2, markersize=4, label='Fautes personnelles', color='#e377c2', linestyle='--')

ax.set_title('Évolution des statistiques défensives (moyenne par match)', fontsize=14, fontweight='bold')
ax.set_xlabel('Saison')
ax.set_ylabel('Moyenne par match')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 5. Graphique combiné des rebonds
fig3, ax = plt.subplots(figsize=(12, 6))

ax.plot(season_stats['season'], season_stats['orb_per_game'], 
        marker='D', linewidth=2, markersize=4, label='Rebonds offensifs', color='#8c564b')
ax.plot(season_stats['season'], season_stats['drb_per_game'], 
        marker='p', linewidth=2, markersize=4, label='Rebonds défensifs', color='#17becf')
ax.fill_between(season_stats['season'], season_stats['orb_per_game'], 
                season_stats['drb_per_game'] + season_stats['orb_per_game'], 
                alpha=0.2, color='gray', label='Total des rebonds')

ax.set_title('Évolution des rebonds (moyenne par match)', fontsize=14, fontweight='bold')
ax.set_xlabel('Saison')
ax.set_ylabel('Moyenne par match')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 6. Graphique avec toutes les statistiques (sauf les tirs)
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# Interceptions
axes[0, 0].plot(season_stats['season'], season_stats['stl_per_game'], linewidth=2, color='#9467bd')
axes[0, 0].fill_between(season_stats['season'], 0, season_stats['stl_per_game'], alpha=0.3, color='#9467bd')
axes[0, 0].set_title('Interceptions par match', fontsize=12)
axes[0, 0].set_ylabel('Moyenne')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Contres
axes[0, 1].plot(season_stats['season'], season_stats['blk_per_game'], linewidth=2, color='#d62728')
axes[0, 1].fill_between(season_stats['season'], 0, season_stats['blk_per_game'], alpha=0.3, color='#d62728')
axes[0, 1].set_title('Contres par match', fontsize=12)
axes[0, 1].set_ylabel('Moyenne')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# Fautes
axes[1, 0].plot(season_stats['season'], season_stats['pf_per_game'], linewidth=2, color='#e377c2')
axes[1, 0].fill_between(season_stats['season'], 0, season_stats['pf_per_game'], alpha=0.3, color='#e377c2')
axes[1, 0].set_title('Fautes personnelles par match', fontsize=12)
axes[1, 0].set_xlabel('Saison')
axes[1, 0].set_ylabel('Moyenne')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# Rebonds totaux
axes[1, 1].plot(season_stats['season'], season_stats['orb_per_game'] + season_stats['drb_per_game'], 
                linewidth=2, color='gray', label='Rebonds totaux')
axes[1, 1].fill_between(season_stats['season'], 0, 
                        season_stats['orb_per_game'] + season_stats['drb_per_game'], 
                        alpha=0.2, color='gray')
axes[1, 1].set_title('Rebonds totaux par match', fontsize=12)
axes[1, 1].set_xlabel('Saison')
axes[1, 1].set_ylabel('Moyenne')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

fig4.suptitle('Évolution des statistiques NBA (hors tirs)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. Afficher les statistiques clés
print("\n=== STATISTIQUES CLÉS ===\n")

# Saison avec le plus d'interceptions
max_stl = season_stats.loc[season_stats['stl_per_game'].idxmax()]
print(f"Saison avec le plus d'interceptions : {int(max_stl['season'])} - {max_stl['stl_per_game']:.1f} par match")

# Saison avec le plus de contres
max_blk = season_stats.loc[season_stats['blk_per_game'].idxmax()]
print(f"Saison avec le plus de contres : {int(max_blk['season'])} - {max_blk['blk_per_game']:.1f} par match")

# Saison avec le plus de rebonds offensifs
max_orb = season_stats.loc[season_stats['orb_per_game'].idxmax()]
print(f"Saison avec le plus de rebonds offensifs : {int(max_orb['season'])} - {max_orb['orb_per_game']:.1f} par match")

# Saison avec le plus de rebonds défensifs
max_drb = season_stats.loc[season_stats['drb_per_game'].idxmax()]
print(f"Saison avec le plus de rebonds défensifs : {int(max_drb['season'])} - {max_drb['drb_per_game']:.1f} par match")

# Évolution globale
print("\n=== ÉVOLUTIONS (première → dernière saison) ===\n")
first = season_stats.iloc[0]
last = season_stats.iloc[-2]

stats_to_show = [
    ('Interceptions', 'stl_per_game'),
    ('Contres', 'blk_per_game'),
    ('Fautes personnelles', 'pf_per_game'),
    ('Rebonds offensifs', 'orb_per_game'),
    ('Rebonds défensifs', 'drb_per_game'),
]

for name, col in stats_to_show:
    evolution = last[col] - first[col]
    pct = (evolution / first[col]) * 100
    print(f"{name:20} : {first[col]:.1f} → {last[col]:.1f}  ({evolution:+.1f}, {pct:+.1f}%)")

# 8. Graphique des tendances avec lissage (moyenne mobile)
fig5, ax = plt.subplots(figsize=(14, 7))

# Calcul des moyennes mobiles sur 5 saisons
window = 5
season_stats['stl_ma'] = season_stats['stl_per_game'].rolling(window=window, min_periods=1).mean()
season_stats['blk_ma'] = season_stats['blk_per_game'].rolling(window=window, min_periods=1).mean()
season_stats['orb_ma'] = season_stats['orb_per_game'].rolling(window=window, min_periods=1).mean()
season_stats['drb_ma'] = season_stats['drb_per_game'].rolling(window=window, min_periods=1).mean()
season_stats['pf_ma'] = season_stats['pf_per_game'].rolling(window=window, min_periods=1).mean()

# Tracer les moyennes mobiles
ax.plot(season_stats['season'], season_stats['stl_ma'], linewidth=2, label='Interceptions (MA5)', color='#9467bd')
ax.plot(season_stats['season'], season_stats['blk_ma'], linewidth=2, label='Contres (MA5)', color='#d62728')
ax.plot(season_stats['season'], season_stats['pf_ma'], linewidth=2, label='Fautes (MA5)', color='#e377c2')
ax.plot(season_stats['season'], season_stats['orb_ma'], linewidth=2, label='Rebonds offensifs (MA5)', color='#8c564b')
ax.plot(season_stats['season'], season_stats['drb_ma'], linewidth=2, label='Rebonds défensifs (MA5)', color='#17becf')

ax.axvline(x=1980, color='red', linestyle='--', alpha=0.5, label='Introduction du tir à 3 pts')
ax.axvline(x=2000, color='gray', linestyle='--', alpha=0.5, label='Année 2000')

ax.set_title('Tendances des statistiques NBA (moyenne mobile sur 5 saisons)', fontsize=14, fontweight='bold')
ax.set_xlabel('Saison')
ax.set_ylabel('Moyenne par match')
ax.legend(loc='best', ncol=2)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()