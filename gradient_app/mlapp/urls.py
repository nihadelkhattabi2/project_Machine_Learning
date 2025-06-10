from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    #path('', views.upload_dataset, name='upload'),
    #path('', views.select_features, name='select'),
    path('', views.import_dataset, name='import_dataset'),
    path('correlation/', views.correlation_view, name='correlation'),
    # path('correlation/', views.correlation_view, name='correlation_view'),
    # path('correlation_target/', views.afficher_correlation_target, name='afficher_correlation_target'),
    #path('configure/', views.configure_model, name='configure'),
    #path('predict/', views.predict_view, name='predict'),
]
 