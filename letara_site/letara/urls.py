from django.urls import path
from . import views

urlpatterns = [
    path('', views.document_scanner, name='document_scanner'),
    path('documents/', views.DocumentListView.as_view(), name='document_list'),
    path('document/<int:pk>/', views.DocumentDetailView.as_view(), name='document_detail'),
]
