from django.urls import path
from . import views

urlpatterns = [
    path('', views.document_scanner, name='document_scanner'),
    path('adjust/<int:pk>/', views.adjust_boundaries, name='adjust_boundaries'),
    path('apply-crop/<int:pk>/', views.apply_crop, name='apply_crop'),
    path('skip-crop/<int:pk>/', views.skip_crop, name='skip_crop'),
    path('documents/', views.DocumentListView.as_view(), name='document_list'),
    path('document/<int:pk>/', views.DocumentDetailView.as_view(), name='document_detail'),
    path('document/<int:pk>/delete/', views.delete_document, name='delete_document'),
]
