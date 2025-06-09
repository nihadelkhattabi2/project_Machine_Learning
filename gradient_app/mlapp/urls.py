from django.urls import path
from . import views


from django.shortcuts import render

urlpatterns = [
    #path('', views.upload_dataset, name='upload'),
    #path('', views.select_features, name='select'),
    #path('', views.predict_view, name='pre'),
    path('', views.configure_model_view, name='configure'),
    path('predict/', views.predict, name='predict'),
]
