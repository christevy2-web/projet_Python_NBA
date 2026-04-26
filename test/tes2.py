import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Charger les données
df = pd.read_csv('Team_Complete.csv', sep=',')

# 2. Filtrer les saisons d'intérêt et la NBA uniquement
seasons_of_interest = [1996, 2006, 2016, 2025]
df_filtered = df[(df['lg'] == 'NBA') & (df['season'].isin(seasons_of_interest))].copy()

# 3. Créer la colonne 'is_playoff' (déjà présente dans les données)
# Vérifier que la colonne 'playoffs' existe
if 'playoffs' in df_filtered.columns:
    df_filtered['is_playoff'] = df_filtered['playoffs'].astype(bool)
else:
    # Si la colonne n'existe pas, utiliser 'w' pour déterminer les playoffs
    # (les équipes avec un certain nombre de victoires)
    df_filtered['is_playoff'] = df_filtered['w'] >= 37  # Seuil approximatif

# 4. Sélectionner les statistiques à analyser
stats_columns = [
    'pts',           # Points totaux
    'pf',            # Fautes personnelles
    'tov',           # Pertes de balle
    'blk',           # Contres
    'stl',           # Interceptions
    'ast',           # Passes décisives
    'orb',           # Rebonds offensifs
    'drb',           # Rebonds défensifs
    'age',           # Âge moyen
    'x3p_percent',   # Pourcentage à 3-points
    'x2p_percent',   # Pourcentage à 2-points
    'ft_percent'     # Pourcentage de lancers-francs
]

# Vérifier que toutes les colonnes existent
existing_stats = [col for col in stats_columns if col in df_filtered.columns]
missing_stats = [col for col in stats_columns if col not in df_filtered.columns]
if missing_stats:
    print(f"Colonnes manquantes : {missing_stats}")

# 5. Calculer les moyennes par saison pour les équipes en playoffs
playoff_stats = df_filtered[df_filtered['is_playoff']].groupby('season')[existing_stats].mean()
non_playoff_stats = df_filtered[~df_filtered['is_playoff']].groupby('season')[existing_stats].mean()

# 6. Calculer le ratio d'importance (playoff / non-playoff)
importance_matrix = (playoff_stats / non_playoff_stats - 1) * 100

# 7. Créer la matrice de scores pour la visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Importance des statistiques pour les équipes en playoffs\n(Écart en % par rapport aux équipes non-playoffs)', 
             fontsize=14, fontweight='bold')

# Noms français des statistiques
stat_names_fr = {
    'pts': 'Points totaux',
    'pf': 'Fautes personnelles',
    'tov': 'Pertes de balle',
    'blk': 'Contres',
    'stl': 'Interceptions',
    'ast': 'Passes décisives',
    'orb': 'Rebonds offensifs',
    'drb': 'Rebonds défensifs',
    'age': 'Âge moyen',
    'x3p_percent': '% à 3 points',
    'x2p_percent': '% à 2 points',
    'ft_percent': '% LF'
}

# Palette de couleurs
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in importance_matrix.values.flatten()]

for i, (ax, season) in enumerate(zip(axes.flat, seasons_of_interest)):
    if season in importance_matrix.index:
        data = importance_matrix.loc[season].sort_values()
        
        # Créer les labels en français
        labels = [stat_names_fr.get(idx, idx) for idx in data.index]
        
        # Créer les couleurs (vert si positif, rouge si négatif)
        bar_colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in data.values]
        
        # Barres horizontales
        bars = ax.barh(labels, data.values, color=bar_colors, edgecolor='white', linewidth=1)
        
        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, data.values):
            ax.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}%', va='center', ha='left' if val >= 0 else 'right', 
                   fontsize=9, fontweight='bold')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_title(f'Saison {season}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Écart (%)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Ajuster les limites
        max_val = max(abs(data.values)) + 5
        ax.set_xlim(-max_val, max_val)

plt.tight_layout()
plt.show()

# 8. Matrice de chaleur (heatmap) comparative
fig2, ax = plt.subplots(figsize=(12, 10))

# Préparer les données pour la heatmap
heatmap_data = importance_matrix.T  # Transposer pour avoir les stats en lignes
heatmap_data = heatmap_data[seasons_of_interest]  # Réorganiser les colonnes

# Remplacer les noms des colonnes par les noms français
heatmap_data.index = [stat_names_fr.get(idx, idx) for idx in heatmap_data.index]

# Créer la heatmap
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Écart (%)'})

ax.set_title('Importance des statistiques pour les équipes en playoffs\n(Écart en % par rapport aux équipes non-playoffs)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Saison')
ax.set_ylabel('Statistiques')

plt.tight_layout()
plt.show()

# 9. Analyse des tendances sur les 4 saisons
fig3, ax = plt.subplots(figsize=(14, 8))

# Sélectionner les statistiques les plus importantes en moyenne
avg_importance = importance_matrix.mean().sort_values(ascending=False)
top_stats = avg_importance.head(8).index

for stat in top_stats:
    ax.plot(importance_matrix.index, importance_matrix[stat], 
            marker='o', linewidth=2, markersize=6, 
            label=stat_names_fr.get(stat, stat))

ax.set_title('Évolution de l\'importance des principales statistiques pour les playoffs\n(Écart % par rapport aux non-playoffs)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Saison')
ax.set_ylabel('Écart (%)')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3)
ax.legend(loc='best')

plt.tight_layout()
plt.show()

# 10. Tableau récapitulatif
print("\n" + "="*100)
print("MATRICE D'IMPORTANCE DES STATISTIQUES POUR LES ÉQUIPES EN PLAYOFFS")
print("(Écart en % par rapport aux équipes non-playoffs)")
print("="*100)

# Créer un dataframe pour l'affichage
display_matrix = importance_matrix[seasons_of_interest].round(1)
display_matrix.index = [stat_names_fr.get(idx, idx) for idx in display_matrix.index]

print(display_matrix.to_string())

# 11. Statistiques clés
print("\n" + "="*100)
print("STATISTIQUES CLÉS PAR SAISON")
print("="*100)

for season in seasons_of_interest:
    if season in importance_matrix.index:
        print(f"\n--- Saison {season} ---")
        data = importance_matrix.loc[season].sort_values(ascending=False)
        print("Top 3 des avantages des équipes en playoffs :")
        for i, (stat, val) in enumerate(data.head(3).items()):
            stat_name = stat_names_fr.get(stat, stat)
            print(f"  {i+1}. {stat_name}: {val:.1f}%")
        
        print("\nTop 3 des désavantages (ou points faibles) :")
        for i, (stat, val) in enumerate(data.tail(3).items()):
            stat_name = stat_names_fr.get(stat, stat)
            print(f"  {i+1}. {stat_name}: {val:.1f}%")

# 12. Analyse de corrélation avec les victoires
print("\n" + "="*100)
print("CORRÉLATION AVEC LE NOMBRE DE VICTOIRES")
print("="*100)

# Calculer les corrélations pour chaque saison
for season in seasons_of_interest:
    season_data = df_filtered[df_filtered['season'] == season]
    if len(season_data) > 0:
        print(f"\n--- Saison {season} ---")
        corr_with_wins = season_data[existing_stats + ['w']].corr()['w'].sort_values(ascending=False)
        corr_with_wins = corr_with_wins.drop('w')
        
        for stat, corr in corr_with_wins.head(5).items():
            stat_name = stat_names_fr.get(stat, stat)
            print(f"  {stat_name}: {corr:.3f}")

# 13. Sauvegarder la matrice (optionnel)
importance_matrix.to_csv('importance_playoffs_matrix.csv')
print("\n✅ Matrice d'importance sauvegardée dans 'importance_playoffs_matrix.csv'")