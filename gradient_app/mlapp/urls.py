from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    #path('', views.upload_dataset, name='upload'),
    #path('', views.select_features, name='select'),
<<<<<<< HEAD
    #path('', views.predict, name='pre'),
    path('predict/', views.predict, name='predict'),
    
=======
    path('', views.import_dataset, name='import_dataset'),
    #path('configure/', views.configure_model, name='configure'),
    #path('predict/', views.predict_view, name='predict'),
>>>>>>> 32e82582a6f0451e9d365be65e64448e22d57335
]
