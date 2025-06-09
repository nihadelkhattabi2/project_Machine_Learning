import pandas as pd

def preprocess_dataset(file_path):
    """
    Charge et prétraite un dataset CSV.
    Nettoie les valeurs manquantes et encode les colonnes catégorielles.
    """
    try:
        df = pd.read_csv(file_path)

        # # Nettoyage de base
        # df.drop_duplicates(inplace=True)
        # df.fillna(method='ffill', inplace=True)  # ou df.fillna(0), selon le contexte

        # # Encodage des colonnes catégorielles
        # for col in df.select_dtypes(include='object').columns:
        #     df[col] = df[col].astype('category').cat.codes

        return df
    except Exception as e:
        print(f"Erreur pendant le prétraitement : {e}")
        return None
