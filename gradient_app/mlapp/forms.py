from django import forms

class FeatureSelectionForm(forms.Form):
    target = forms.ChoiceField(label = "Target", choices = [])
    features = forms.MultipleChoiceField(
        label = "Features",
        widget = forms.CheckboxSelectMultiple,
        choices = []
    )

