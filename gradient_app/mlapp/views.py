import uuid
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from .forms import FeatureSelectionForm
import os
from django.conf import settings
from django.utils.html import escape
import json

from .preprocessing import generate_correlation_heatmap, preprocess_dataset  # <- si ta fonction est dans un fichier utils.py
# ou bien, tu l'appelles localement si elle est dans views.py

def import_dataset(request):
    dataset_preview = None
    filename = ""

    if request.method == 'POST' and 'dataset_file' in request.FILES:
        file = request.FILES['dataset_file']
        filename = file.name

        try:
            # 🎯 Appel direct au prétraitement
            df = preprocess_dataset(file, verbose=True)

            if df is None or df.empty:
                raise ValueError("Le dataset est vide ou mal formaté.")

            # 👁️ Aperçu du dataset nettoyé
            dataset_preview = df.head(20).to_html(classes='table', index=False)

        except Exception as e:
            dataset_preview = f"<p style='color:red;'>Erreur : {escape(str(e))}</p>"

    return render(request, 'mlapp/import_dataset.html', {
        'dataset_preview': dataset_preview,
        'filename': filename,
    })

from django.shortcuts import render
import pandas as pd

def correlation_view(request):
    # Exemple : charger un dataset (à adapter à ton cas)
    file_path = 'chemin/vers/ton_dataset.csv'
    df = preprocess_dataset(file_path, verbose=False)  # ta fonction de prétraitement

    if df is None:
        return render(request, 'mlapp/error.html', {'message': 'Erreur chargement dataset'})

    heatmap_img = generate_correlation_heatmap(df)

    return render(request, 'mlapp/correlation.html', {'heatmap': heatmap_img})


    # df = get_last_uploaded_dataset()  # Charge ton DataFrame déjà importé
    # fig = plot_correlation_heatmap(df)
    # image_uri = fig_to_base64(fig)
    # return render(request, 'correlation.html', {'image_uri': image_uri})

# def afficher_correlation_target(request):
#     df = get_last_uploaded_dataset()
#     columns = df.select_dtypes(include='number').columns.tolist()

#     if request.method == 'POST':
#         target = request.POST.get('target_col')
#         fig = plot_correlation_with_target(df, target)
#         image_uri = fig_to_base64(fig)
#         return render(request, 'correlation_target.html', {
#             'image_uri': image_uri,
#             'target': target,
#             'columns': columns
#         })

#     return render(request, 'correlation_target.html', {'columns': columns})


# def afficher_correlation_target(request):
#     df_json = request.session.get('dataset', None)
#     if not df_json:
#         return render(request, 'mlapp/error.html', {'message': 'Aucun dataset trouvé.'})

#     df = pd.read_json(df_json)

#     if request.method == 'POST':
#         target = request.POST.get('target_col')

#         try:
#             graphique = correlation_avec_cible(df, target)
#             return render(request, 'mlapp/correlation_target.html', {
#                 'graphique': graphique,
#                 'target': target
#             })
#         except Exception as e:
#             return render(request, 'mlapp/error.html', {'message': str(e)})

#     columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     return render(request, 'mlapp/import_dataset.html', {'columns': columns})


def select_features(request):
    df = pd.read_csv("media/housingCalifornia.csv")
    columns = df.columns.tolist()

    if request.method == 'POST':
        form = FeatureSelectionForm(request.POST, columns=columns)
        # On récupère la target pour la retirer des features
        selected_target = request.POST.get('target')
        
        target_choices = [(col, col) for col in columns]
        feature_choices = [(col, col) for col in columns if col != selected_target]

        form.fields['target'].choices = target_choices
        form.fields['features'].choices = feature_choices

        if form.is_valid():
            target = form.cleaned_data['target']
            features = form.cleaned_data['features']
            request.session['target'] = target
            request.session['features'] = features
            return redirect('configure')

    else:
        form = FeatureSelectionForm(columns=columns)
        form.fields['target'].choices = [(col, col) for col in columns]
        form.fields['features'].choices = [(col, col) for col in columns]

    return render(request, 'mlapp/select_features.html', {
        'form': form,
        'table': df.head().to_html(classes='table table-striped')
    })

from django.shortcuts import render, redirect
from django.contrib import messages  # pour afficher les erreurs

def configure_model_view(request):
    if request.method == 'POST':
        task_type = request.POST.get('task_type')       # récupère le choix régression ou classification
        alpha = request.POST.get('alpha')               # récupère alpha
        iterations = request.POST.get('iterations')     # récupère le nombre d’itérations

        # Vérifie que le choix est valide
        if task_type not in ['regression', 'classification']:
            messages.error(request, "Choisis un type d’apprentissage valide.")
            return render(request, 'mlapp/configure_model.html')

        # Vérifie que alpha est un nombre positif
        try:
            alpha = float(alpha)
            if alpha <= 0:
                raise ValueError()
        except:
            messages.error(request, "Le taux d’apprentissage doit être un nombre positif.")
            return render(request, 'mlapp/configure_model.html')

        # Vérifie que iterations est un entier >= 1
        try:
            iterations = int(iterations)
            if iterations < 1:
                raise ValueError()
        except:
            messages.error(request, "Le nombre d’itérations doit être un entier positif.")
            return render(request, 'mlapp/configure_model.html')

        # Tout est OK : on stocke dans la session
        request.session['task_type'] = task_type
        request.session['alpha'] = alpha
        request.session['iterations'] = iterations

        # Puis on redirige vers la page d’entraînement
        return redirect('train_model')

    # Si méthode GET, on affiche juste le formulaire
    return render(request, 'mlapp/configure_model.html')

def train_model(request):
    if request.method == 'POST':
        # Récupération du dataset prétraité en JSON dans la session
        df = pd.read_json(request.session.get('preprocessed_dataset'))
        features = request.session.get('features')
        target = request.session.get('target')

        # Extraction des variables
        X = df[features].values
        y = df[target].values

        # Paramètres du modèle
        alpha = float(request.POST.get('alpha', 0.01))
        iterations = int(request.POST.get('iterations', 1000))

        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        theta = np.zeros(n + 1)

        cost_history = []

        # Gradient Descent
        for _ in range(iterations):
            gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
            theta -= alpha * gradients
            cost = ((X_b.dot(theta) - y)**2).mean()
            cost_history.append(cost)

        # Création de la figure de la courbe d'apprentissage
        fig, ax = plt.subplots()
        ax.plot(range(iterations), cost_history, label='Coût')
        ax.set_xlabel('Itérations')
        ax.set_ylabel('Coût (Erreur)')
        ax.set_title('Courbe d’apprentissage (Gradient Descent)')
        ax.legend()

        # Sauvegarde de l'image
        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join(settings.BASE_DIR, 'mlapp/static/plots', filename)
        plt.savefig(path)
        plt.close()

        request.session['plot_filename'] = filename

        return render(request, 'mlapp/train_model.html', {
            'theta': theta,
            'cost': round(cost_history[-1], 4),
            'plot_image': f"plots/{filename}",
        })

    else:
        # Si GET, afficher la page de configuration du modèle
        return render(request, 'mlapp/configure_model.html')
    
def predict_view(request):
    prediction = None  # Initialisation de la variable prédiction

    if request.method == 'POST':
        try:
            # Récupération des valeurs envoyées par le formulaire
            f1 = float(request.POST.get('feature1', 0))
            f2 = float(request.POST.get('feature2', 0))
            f3 = float(request.POST.get('feature3', 0))

            # Exemple simple de prédiction : moyenne des trois valeurs
            X = np.array([[f1, f2, f3]])
            prediction = round((f1 + f2 + f3) / 3, 2)
        except:
            # En cas d'erreur (valeur non numérique par exemple)
            prediction = "Erreur lors de la saisie des valeurs."

    # Que la méthode soit GET ou POST, on retourne toujours la page HTML
    return render(request, 'mlapp/predict.html', {'prediction': prediction})

