from django.urls import path
from .views import PhotoListView, SearchPhotoView, SinglePhotoView, PhotoUploadedView, DemoView, AddPhotoView

urlpatterns = [
    path('', PhotoListView.as_view(), name='photo_list'),
    path('uploads/', PhotoUploadedView.as_view(), name='uploaded_photo'),
    path('create/', AddPhotoView.as_view(), name='create_photo'),
    path('<int:pk>/', SinglePhotoView.as_view(), name='single_photo'),
    path('demo/', DemoView.as_view(), name='demo'),
    path('search/', SearchPhotoView.as_view(), name='search_photo'),

]