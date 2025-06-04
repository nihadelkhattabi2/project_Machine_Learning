import pandas as pd
from django.shortcuts import render, redirect
from .forms import FeatureSelectionForm

# Create your views here.
def select_features(request):
    df = pd.read_csv("media/housingCalifornia.csv") #les datasets dans media

    #options des colonnes
    columns = df.columns.tolist()
    form = FeatureSelectionForm()

    #remplir dynamiquement les choix
    form.fields['target'].choices = [(col, col) for col in columns]
    form.fields['features'].choices = [(col, col) for col in columns]

    if request.method == 'POST':
        form = FeatureSelectionForm(request.POST)
        form.fields['target'].choices = [(col, col) for col in columns]
        form.fields['features'].choices = [(col, col) for col in columns]
        if form.is_valid():
            target = form.cleaned_data['target']
            features = form.cleaned_data['features']

            #sauvegarder dans la session pour l'utiliser dans l'interface 3 de configuration
            request.session['target'] = target
            request.session['features'] = features
            return redirect('configure')
        
    return render(request, 'mlapp/select_features.html', {
        'form': form,
        'table': df.head().to_html(classes = 'table table-striped')
    })

