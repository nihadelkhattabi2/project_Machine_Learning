from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    #path('', views.upload_dataset, name='upload'),
    path('select/', views.select_features, name='select_features'),
    path('', views.import_dataset, name='import_dataset'),
    #path('correlation/', views.correlation_view, name='correlation'),
    # path('correlation/', views.correlation_view, name='correlation_view'),
    # path('correlation_target/', views.afficher_correlation_target, name='afficher_correlation_target'),
    path('configue/',  views.configure_model_view, name='configure'),

    path('predict/', views.predict_view, name='predict_view'),



    
]
 