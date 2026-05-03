# 🏀 Projet NBA – Analyse de données
## 📌 Description

Ce projet a pour objectif d’analyser des données liées à la NBA afin d’extraire des insights pertinents sur les performances des équipes et des joueurs.

L’analyse est réalisée à l’aide de Python dans un notebook Jupyter, en utilisant des bibliothèques de data science.

## 🎯 Objectifs
Explorer un dataset NBA
Nettoyer et préparer les données
Analyser les performances (joueurs / équipes)
Visualiser les résultats
Identifier des tendances ou patterns intéressants
## 🧰 Technologies utilisées
Python
Jupyter Notebook
Pandas (manipulation de données)
NumPy
Matplotlib / Seaborn (visualisation)
# 📂 Structure du projet
projet/
│── find_ft_2026.py
│── projet_NBA.ipynb
│── data/
│   ├── Team_Complete.csv
│   ├── Player Totals.csv
│   ├── nba_champions.csv
│── NBA_Analysis_results/
│   ├── champions_histogram_2026.png
│   ├── .png
│   ├── ...
│── predictions_2026_2028.csv
│── champions_histogram_*.png
│── README.md
├── Team Summaire.csv
├── Team Total.csv
├── Player Totals.csv
├── Games.csv       # Documentation du projet
## ⚙️ Installation
Cloner le projet :
git clone <url-du-repo>
cd projet
Installer les dépendances :
pip install pandas numpy matplotlib seaborn jupyter
Lancer le notebook :
jupyter notebook
## 📊 Contenu de l’analyse

Le notebook contient généralement :

## 1. Chargement des données
Import du dataset NBA
Aperçu des données
## 2. Nettoyage des données
Gestion des valeurs manquantes
Conversion des types
## 3. Analyse exploratoire (EDA)
Statistiques descriptives
Corrélations
Distribution des performances
## 4. Visualisation
Graphiques (histogrammes, scatter plots, etc.)
Comparaison joueurs / équipes
## 5. Conclusions
Résumé des insights principaux
Interprétation des résultats
📈 Exemples d’analyses possibles
Meilleurs joueurs selon certaines statistiques
Évolution des performances dans le temps
Comparaison entre équipes
Impact de certaines variables sur les résultats
🚀 Améliorations possibles
Ajouter un modèle de machine learning (prédiction)
Intégrer des données en temps réel (API NBA)
Créer un dashboard interactif (Streamlit / Dash)

# 🏀 NBA Championship Prediction Model
## 📌 Description

Ce projet implémente un modèle avancé de prédiction du champion NBA basé sur :

statistiques d’équipes
performances individuelles des joueurs
rating Elo dynamique
régression logistique
simulation Monte Carlo des playoffs

Le modèle intègre une projection dynamique des joueurs (progression, vieillissement, blessures), permettant de générer des prédictions réalistes pour les saisons futures (2026–2028).

#"  🧠 Architecture du modèle

### Pipeline global :

JOUEURS → player_value → projection → force équipe → Elo ajusté → modèle ML → simulation playoffs → champion
⚙️ Fonctionnalités
## 📊 Lecture des données
Lecture robuste des CSV (multi-encodage)
Nettoyage automatique des colonnes
Gestion des dates et saisons
# 🏀 Calcul Elo
Mise à jour match par match
Avantage du terrain
Régression entre saisons
# 👤 Modélisation des joueurs
Player Value

### Score synthétique basé sur la production :

player_value =
0.4 * points +
0.2 * rebounds +
0.2 * assists +
0.2 * (points / minutes)
Projection des joueurs

### Le modèle simule :

progression des jeunes joueurs
pic de performance
déclin des vétérans
blessures aléatoires
bruit statistique

Exemple de logique :

< 24 ans  → progression forte  
24–28     → progression modérée  
29–31     → stable  
32+       → déclin  
### 🏀 Force d’équipe

Calcul basée sur les 8 meilleurs joueurs :

team_strength = somme(player_value pondéré)

Poids appliqués :

1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3
### ⚡ Elo ajusté
adjusted_rating = Elo + α × team_strength
α ≈ 0.1–0.3
### 🤖 Modèle Machine Learning
Régression logistique
Standardisation des données
Imputation des valeurs manquantes

## Features :

statistiques d’équipe
Elo
production joueurs
structure du roster
### 🎲 Simulation des playoffs
Format NBA (Best-of-7)
Probabilités basées sur ratings
Monte Carlo (3000 simulations)
### 🔮 Simulation multi-années

## Pour chaque simulation :

évolution des joueurs
recalcul des équipes
ajustement Elo
simulation des playoffs

Fichiers générés :

predictions_2026_2028.csv
champions_histogram_2026.png
champions_histogram_2027.png
champions_histogram_2028.png

## Contenu :

probabilités de titre
score du modèle
Elo
win percentage
### 🔥 Améliorations apportées
intégration des joueurs dans le modèle
projection multi-années réaliste
simulation dynamique complète
variabilité des résultats
réduction du biais des équipes statiques
### ⚠️ Limites
pas de simulation de trades
pas de lineups détaillés
minutes approximées
modèle simplifié des blessures
### 🚀 Améliorations possibles
simulation de trades
modèle de blessures avancé
pondération par lineups
dashboard interactif
modèle bayésien

### 🧠 Conclusion

Ce projet propose une approche avancée de prédiction NBA en combinant :

statistiques historiques
modélisation joueurs
simulation probabiliste

Il permet de produire des prédictions crédibles et dynamiques sur plusieurs saisons.
