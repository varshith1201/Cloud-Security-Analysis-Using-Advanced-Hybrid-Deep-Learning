from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.train_view, name='train_model'),
    path('result/<int:pk>/', views.model_result, name='model_result'),
    path('results/', views.all_results, name='all_results'),
    path('predict/', views.predict_view, name='predict'),
    path('compare/', views.compare_models, name='compare_models'),
]
