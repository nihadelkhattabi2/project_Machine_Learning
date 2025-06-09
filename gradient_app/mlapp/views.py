import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from .forms import FeatureSelectionForm
import os
from django.conf import settings
from django.utils.html import escape
import json

from .preprocessing import preprocess_dataset  # <- si ta fonction est dans un fichier utils.py
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

def correlation_view(request):
    df = get_last_uploaded_dataset()  # Charge ton DataFrame déjà importé
    fig = plot_correlation_heatmap(df)
    image_uri = fig_to_base64(fig)
    return render(request, 'correlation.html', {'image_uri': image_uri})

def afficher_correlation_target(request):
    df = get_last_uploaded_dataset()
    columns = df.select_dtypes(include='number').columns.tolist()

    if request.method == 'POST':
        target = request.POST.get('target_col')
        fig = plot_correlation_with_target(df, target)
        image_uri = fig_to_base64(fig)
        return render(request, 'correlation_target.html', {
            'image_uri': image_uri,
            'target': target,
            'columns': columns
        })

    return render(request, 'correlation_target.html', {'columns': columns})


def afficher_correlation_target(request):
    df_json = request.session.get('dataset', None)
    if not df_json:
        return render(request, 'mlapp/error.html', {'message': 'Aucun dataset trouvé.'})

    df = pd.read_json(df_json)

    if request.method == 'POST':
        target = request.POST.get('target_col')

        try:
            graphique = correlation_avec_cible(df, target)
            return render(request, 'mlapp/correlation_target.html', {
                'graphique': graphique,
                'target': target
            })
        except Exception as e:
            return render(request, 'mlapp/error.html', {'message': str(e)})

    columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return render(request, 'mlapp/import_dataset.html', {'columns': columns})


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

def configure_model_view(request):
    if request.method == 'POST':
        task_type = request.POST.get('task_type')
        if task_type in ['regression', 'classification']:
            request.session['task_type'] = task_type
            return redirect('predict')  # Redirection vers la page de prédiction
    return render(request, 'mlapp/configure_model.html')