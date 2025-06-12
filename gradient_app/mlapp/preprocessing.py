import base64
import io
from django.shortcuts import render
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64



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
    

import matplotlib.pyplot as plt
import seaborn as sns

def generate_correlation_heatmap(df, taille=(20, 10), annot=True, cmap="coolwarm", title="Matrice de corrélation"):
    """
    Affiche une heatmap de la matrice de corrélation d'un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        taille (tuple): Taille de la figure (largeur, hauteur).
        annot (bool): Affiche les coefficients de corrélation sur la heatmap.
        cmap (str): Palette de couleurs.
        title (str): Titre de la heatmap.
    """
    corr = df.corr()
    plt.figure(figsize=taille)
    sns.heatmap(data=corr, annot=annot, cmap=cmap, fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()





def min_max_normalisation(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    return df
def convertir_booleens(df, verbose=False):
    """
    Convertit les colonnes contenant des valeurs booléennes (True/False)
    ou chaînes de texte 'true'/'false' en 0 et 1.

    Args:
        df (pd.DataFrame): Le DataFrame à traiter.
        verbose (bool): Affiche les colonnes converties si True.

    Returns:
        pd.DataFrame: Le DataFrame modifié.
    """
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
            if verbose:
                print(f"Colonne booléenne convertie : {col}")
        elif df[col].dtype == 'object':
            valeurs_uniques = df[col].dropna().unique()
            valeurs_bools = set(str(val).lower() for val in valeurs_uniques)
            if valeurs_bools.issubset({'true', 'false'}):
                df[col] = df[col].str.lower().map({'true': 1, 'false': 0})
                if verbose:
                    print(f"Colonne texte booléenne convertie : {col}")
    return df

def preprocess_dataset_without_normalization(file_path, verbose=True):
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
        df = convertir_booleens(df, verbose=verbose) 

        if verbose:
            print(f"Forme finale : {df.shape}")

        return df

    except Exception as e:
        print(f"❌ Erreur pendant le prétraitement : {e}")
        return None
def preprocess_dataset(df, verbose=True):
    """
    Prend un DataFrame déjà chargé et fait le prétraitement :
    - Remplit les valeurs manquantes
    - Supprime les doublons
    - Encode les variables catégorielles
    - Normalise les features
    """
    try:
        if verbose:
            print("✅ Dataset reçu.")
            print(f"Forme initiale : {df.shape}")
            print("\nAperçu du dataset :")
            print(df.head())  # Affiche le dataset d'abord

        df = remplir_valeurs_vides(df)
        df = supprimer_doublons(df)
        df = one_hot_encode(df, verbose=verbose)
        df = convertir_booleens(df, verbose=verbose)
        df = min_max_normalisation(df)  # Normalisation
        

        if verbose:
            print(f"Forme finale : {df.shape}")

        return df

    except Exception as e:
        print(f"❌ Erreur pendant le prétraitement : {e}")
        return None




















    from sklearn.model_selection import train_test_split

def split_dataset(dataset, target_column='target', train_size=0.7, random_state=0):
    """
    Divise un dataset en features (X) et target (y), puis en ensembles d'entraînement et de test.
    
    Args:
        dataset (pandas.DataFrame): Le dataset complet.
        target_column (str): Le nom de la colonne cible (target).
        train_size (float): La proportion du jeu d'entraînement (ex: 0.7 pour 70%).
        random_state (int): Graine pour la reproductibilité.
    
    Returns:
        X_train, X_test, y_train, y_test : Les jeux de données divisés.
    """
    X = dataset.drop(columns=target_column)
    y = dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test




