from django.urls import path
from .views import PhotoListView, PhotoCreateView, SinglePhotoView, PhotoUploadedView

urlpatterns = [
    path('', PhotoListView.as_view(), name='photo_list'),
    path('uploads/', PhotoUploadedView.as_view(), name='uploaded_photo'),
    path('create/', PhotoCreateView.as_view(), name='create_photo'),
    path('<int:pk>/', SinglePhotoView.as_view(), name='single_photo'),
]