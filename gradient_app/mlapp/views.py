import pandas as pd
from django.shortcuts import render, redirect
from .forms import FeatureSelectionForm


def import_dataset(request):
    return render(request, 'mlapp/import_dataset.html')  # Ce chemin est correct


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
