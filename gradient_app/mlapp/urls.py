from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_dataset, name='upload'),
    path('select/', views.select_features, name='select'),
    path('configure/', views.configure_model, name='configure'),
    path('predict/', views.predict_view, name='predict'),
]
