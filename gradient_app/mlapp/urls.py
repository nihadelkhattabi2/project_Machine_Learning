from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    #path('', views.upload_dataset, name='upload'),
    #path('', views.select_features, name='select'),
    #path('', views.predict, name='pre'),
    path('predict/', views.predict, name='predict'),
    
]
