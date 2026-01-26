from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('profile/', views.profile, name='profile'),
    path('camera/', views.camera, name='camera'),
    path('upload/', views.upload, name='upload'),
]
