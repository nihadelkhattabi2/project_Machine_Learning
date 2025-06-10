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
    


def generate_correlation_heatmap(file_path, verbose=True):
    """
    Charge un dataset CSV, le prétraite,
    puis génère une heatmap de corrélation et renvoie l'image en base64.
    
    Args:
        file_path (str): chemin vers le fichier CSV.
        verbose (bool): affiche des infos pendant le traitement.

    Returns:
        str: image PNG encodée en base64, ou None en cas d'erreur.
    """
    try:
        # Chargement du dataset
        df = pd.read_csv(file_path)


        # Prétraitement (fonctions à définir ailleurs)
        df = remplir_valeurs_vides(df)
        df = supprimer_doublons(df)
        df = one_hot_encode(df, verbose=verbose)



        # Garder que les colonnes numériques pour corrélation
        numeric_df = df.select_dtypes(include=['number'])

        # Calcul de la matrice de corrélation
        corr = numeric_df.corr()

        # Génération de la heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        plt.title('Matrice de corrélation')
        plt.tight_layout()

        # Sauvegarde dans un buffer en PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()

        # Encodage base64
        graphic = base64.b64encode(image_png).decode('utf-8')
        return graphic

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None





def min_max_normalisation(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    return df


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
        
        df = min_max_normalisation(df)  #  normalisation des _features
        


       # afficher_correlation_avec_target(df, verbose=True)

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
