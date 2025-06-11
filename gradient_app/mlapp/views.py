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
import matplotlib.pyplot as plt
import io, base64
from django.contrib import messages # pour afficher les erreurs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

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
            
            request.session['dataset'] = df.to_json()

            # 👁️ Aperçu du dataset nettoyé
            dataset_preview = df.head(20).to_html(classes='table', index=False)

        except Exception as e:
            dataset_preview = f"<p style='color:red;'>Erreur : {escape(str(e))}</p>"

    return render(request, 'mlapp/import_dataset.html', {
        'dataset_preview': dataset_preview,
        'filename': filename,
        'request': request,
    })


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
    # Étape 1 : Récupérer le dataset prétraité depuis la session
    dataset_json = request.session.get('dataset', None)
    
    if dataset_json is None:
        # Si aucun dataset en session, rediriger vers l'importation
        return redirect('import_dataset')  # mets ici le nom correct de la vue d'import
    
    df = pd.read_json(dataset_json)
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
        'table': df.head(100).to_html(classes='table table-striped')
    })


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta, task_type):
    m = len(y)
    h = X.dot(theta)
    if task_type == "regression":
        return (1/(2*m)) * np.sum((h - y) ** 2)
    elif task_type == "classification":
        h = sigmoid(h)
        return (-1/m) * np.sum(y*np.log(h+1e-10) + (1-y)*np.log(1 - h + 1e-10))
    

def gradient_descent(X, y, theta, alpha, iterations, task_type):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        if task_type == "regression":
            h = X.dot(theta)
            gradient = (1/m) * X.T.dot(h - y)
        elif task_type == "classification":
            h = sigmoid(X.dot(theta))
            gradient = (1/m) * X.T.dot(h - y)
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta, task_type)
        cost_history.append(cost)
    return theta, cost_history


def configure_model_view(request):
    if request.method == 'POST':
        task_type = request.POST.get('task_type')
        alpha = request.POST.get('alpha')
        iterations = request.POST.get('iterations')

        try:
            alpha = float(alpha)
            iterations = int(iterations)
            if task_type not in ['regression', 'classification'] or alpha <= 0 or iterations < 1:
                raise ValueError()
        except:
            messages.error(request, "Entrée invalide.")
            return render(request, 'mlapp/configure_model.html')

        # Charger le dataset depuis la session
        dataset_json = request.session.get('dataset', None)
        if dataset_json is None:
            return redirect('import_dataset')  # ou autre redirection logique

        df = pd.read_json(dataset_json)  # Assure-toi que 'dataset_path' est bien stocké en session
        target_col = request.session['target']
        feature_cols = request.session['features']

        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)

        # Prétraitement
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = np.c_[np.ones(X.shape[0]), X]  # Ajouter biais

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        theta = np.zeros((X.shape[1], 1))
        theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, iterations, task_type)

        # Calcul de la performance
        y_pred = X_test.dot(theta) if task_type == "regression" else sigmoid(X_test.dot(theta)) >= 0.5

        if task_type == "regression":
            performance = np.sqrt(np.mean((y_pred - y_test) ** 2))  # RMSE
        else:
            performance = np.mean(y_pred == y_test)  # Accuracy

        # Courbe d'apprentissage
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, iterations+1), cost_history, color='blue')
        plt.title('Courbe d’apprentissage')
        plt.xlabel('Itérations')
        plt.ylabel('Coût (Cost)')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        learning_curve = base64.b64encode(image_png).decode('utf-8')

        return render(request, 'mlapp/result_model.html', {
            'performance': performance,
            'learning_curve': learning_curve,
            'task_type': task_type
        })

    return render(request, 'mlapp/configure_model.html')

    
@csrf_exempt
def predict_view(request):
    dataset_json = request.session.get('dataset')
    features = request.session.get('features', [])
    target = request.session.get('target', 'valeur')

    if not dataset_json or not features or not target:
        return JsonResponse({'error': 'Missing dataset or configuration.'})

    df = pd.read_json(dataset_json)

    if request.method == 'POST':
        try:
            input_data = json.loads(request.body.decode('utf-8'))
            input_values = [float(input_data[feature]) for feature in features]

            # Mini gradient descent "maison" pour faire la prédiction
            X = df[features].values
            y = df[target].values

            # Normalisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            input_scaled = scaler.transform([input_values])

            # Ajout de biais
            X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
            theta = np.zeros(X_b.shape[1])

            # Hyperparamètres
            alpha = 0.01
            iterations = 1000

            # Entraînement simple du modèle
            for _ in range(iterations):
                gradients = 2 / len(X_b) * X_b.T.dot(X_b.dot(theta) - y)
                theta -= alpha * gradients

            # Prédiction
            input_b = np.c_[np.ones((1, 1)), input_scaled]
            prediction = float(input_b.dot(theta))

            return JsonResponse({'prediction': round(prediction, 3)})

        except Exception as e:
            return JsonResponse({'error': str(e)})

    return render(request, 'mlapp/predict.html', {
        'target': target,
        'features': json.dumps(features) #convertir features en JSON 
    })

