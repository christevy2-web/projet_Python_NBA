import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Chargement des données ─────────────────────────────────────────────────
df = pd.read_csv("Team_Complete.csv")

# ── Filtrage depuis 1996 ───────────────────────────────────────────────────
df96 = df[(df['season'] >= 1996)].dropna(subset=['confi']).copy()
df96['win_rate'] = df96['w'] / (df96['w'] + df96['l'])

# ── Colonnes d'intérêt ────────────────────────────────────────────────────
feat_cols   = ['w','pts','fg_percent','x3p_percent','ft_percent',
               'orb','drb','ast','stl','blk','tov','pf','age']
feat_labels = ['Victoires','Points','%Champ','%3pts','%LF',
               'RebOff','RebDéf','Passes','Interc.','Contres','Pertes','Fautes','Âge']

# ── Matrice de corrélation ────────────────────────────────────────────────
corr = df96[feat_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
    center=0, vmin=-1, vmax=1, linewidths=.5,
    xticklabels=feat_labels, yticklabels=feat_labels,
    ax=ax, annot_kws={'size': 8.5, 'color': 'white'},
    cbar_kws={'shrink': 0.8}
)

ax.set_title(
    'Matrice de corrélation\n(Prédicteurs de victoires – depuis 1996)',
    fontsize=14, fontweight='bold', color='white', pad=15
)
plt.xticks(color='white', fontsize=9, rotation=45, ha='right')
plt.yticks(color='white', fontsize=9, rotation=0)
ax.figure.axes[-1].tick_params(colors='white')

plt.tight_layout()
plt.savefig('fig1_correlation_matrix.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ Sauvegardé : fig1_correlation_matrix.png")