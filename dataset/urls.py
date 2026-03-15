from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('list/', views.list_datasets, name='list_datasets'),
    path('view/<int:pk>/', views.view_dataset, name='view_dataset'),
    path('delete/<int:pk>/', views.delete_dataset, name='delete_dataset'),
]
