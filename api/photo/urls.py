from django.urls import path
from .views import PhotoListView, PhotoCreateView

urlpatterns = [
    path('', PhotoListView.as_view(), name='photo_list'),
    path('create/', PhotoCreateView.as_view(), name='create_photo'),
    
]