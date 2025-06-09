import base64
import io
from django.shortcuts import render
import pandas as pd


def remplir_valeurs_vides(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:  # S'il y a des NaN dans cette colonne
            if df[col].dtype in ['int64', 'float64']:
                # Calculer la moyenne et arrondir à l'entier le plus proche
                moyenne_arrondie = int(round(df[col].mean()))
                df[col].fillna(moyenne_arrondie, inplace=True)
            else:
                # Remplacer les NaN par la valeur la plus fréquente (mode)
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def supprimer_doublons(df):
    """
    Supprime les lignes dupliquées du DataFrame.
    """
    avant = df.shape[0]  # Nombre de lignes avant
    df = df.drop_duplicates()
    apres = df.shape[0]  # Nombre de lignes après
    print(f"{avant - apres} doublon(s) supprimé(s).")
    return df

def one_hot_encode(df, verbose=False):
    """
    Applique un One-Hot Encoding sur toutes les colonnes catégorielles du DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame à encoder.
        verbose (bool): Affiche les colonnes encodées si True.

    Returns:
        pd.DataFrame: Le DataFrame encodé.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_cols:
        if verbose:
            print(f"Colonnes catégorielles encodées : {categorical_cols}")
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        return df_encoded
    else:
        if verbose:
            print("Aucune colonne catégorielle à encoder.")
        return df
    

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

def correlation_avec_cible(df, target_col):
    """
    Calcule et retourne un graphique de corrélation avec la colonne cible.

    Args:
        df (pd.DataFrame): Le DataFrame.
        target_col (str): La colonne cible.

    Returns:
        str: Image du graphique en base64.
    """
    if target_col not in df.columns:
        raise ValueError("La colonne cible n'existe pas dans le DataFrame")

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if target_col not in numeric_df.columns:
        raise ValueError("La colonne cible doit être numérique pour le calcul de la corrélation")

    corr_series = numeric_df.corr()[target_col].drop(target_col).sort_values(ascending=False)

    # Tracer le graphique
    plt.figure(figsize=(10, 5))
    sns.barplot(x=corr_series.values, y=corr_series.index, palette='viridis')
    plt.title(f"Corrélation avec la cible : {target_col}")
    plt.xlabel("Coefficient de corrélation")

    # Convertir l’image en base64
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graphique = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    plt.close()

    return graphique

import seaborn as sns
import matplotlib.pyplot as plt

def afficher_correlation_avec_target(df, verbose=True):
    """
    Affiche la corrélation entre les colonnes numériques et la colonne cible probable (prix, charges, valeur).
    """
    # Essayer d'identifier automatiquement une colonne cible probable
    possible_targets = ['prix', 'valeur', 'charges', 'target']
    target_col = None

    for col in df.columns:
        if col.lower() in possible_targets:
            target_col = col
            break

    if not target_col:
        print("❌ Aucune colonne cible identifiée automatiquement (ex: prix, valeur, charges).")
        print("Corrélation globale sera affichée.")
        correlation_matrix = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Matrice de corrélation")
        plt.show()
        return

    if verbose:
        print(f"✅ Colonne cible détectée : {target_col}")

    # Calcul des corrélations avec la cible
    corr_target = df.corr(numeric_only=True)[target_col].drop(target_col).sort_values(ascending=False)

    # Affichage en heatmap
    plt.figure(figsize=(6, len(corr_target) * 0.5 + 1))
    sns.heatmap(corr_target.to_frame(), annot=True, cmap='coolwarm')
    plt.title(f"Corrélation avec {target_col}")
    plt.show()




def preprocess_dataset(file_path, verbose=True):
    """
    Charge et prétraite un dataset CSV :
    - Remplit les valeurs manquantes
    - Supprime les doublons
    - Encode les variables catégorielles
    """
    try:
        df = pd.read_csv(file_path)

        if verbose:
            print("✅ Dataset chargé.")
            print(f"Forme initiale : {df.shape}")
            print("\nAperçu du dataset :")
            print(df.head())  # ⬅️ Afficher le dataset d'abord

        df = remplir_valeurs_vides(df)
        df = supprimer_doublons(df)
        df = one_hot_encode(df, verbose=verbose)

        


        afficher_correlation_avec_target(df, verbose=True)

        if verbose:
            print(f"Forme finale : {df.shape}")

        return df

    except Exception as e:
        print(f"❌ Erreur pendant le prétraitement : {e}")
        return None