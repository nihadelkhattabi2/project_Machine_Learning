from django import forms

class FeatureSelectionForm(forms.Form):
    target = forms.ChoiceField(label="Target", choices=[])
    features = forms.MultipleChoiceField(
        label="Features",
        widget=forms.SelectMultiple(attrs={'class': 'form-control'}),
        choices=[]
    )

    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', [])
        super().__init__(*args, **kwargs)

        self.fields['target'].choices = [(col, col) for col in columns]

        selected_target = self.data.get('target')
        if selected_target:
            self.fields['features'].choices = [(col, col) for col in columns if col != selected_target]
        else:
            self.fields['features'].choices = [(col, col) for col in columns]

