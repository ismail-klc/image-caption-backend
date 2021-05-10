from .models import Photo
from rest_framework import generics
from rest_framework import permissions, authentication
from .serializers import PhotoSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from .pagination import MyCursorPagination

class PhotoListView(generics.ListAPIView):
    queryset = Photo.objects.all()
    serializer_class = PhotoSerializer
    permission_classes = [permissions.AllowAny]
    pagination_class = MyCursorPagination

class PhotoUploadedView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, format=None):
        photos = Photo.objects.all().filter(user=request.user.id)
        serializer = PhotoSerializer(photos, many=True)
        return Response(serializer.data)

class PhotoCreateView(generics.CreateAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = PhotoSerializer
    queryset = Photo.objects.all()

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class SinglePhotoView(generics.RetrieveAPIView):
    permission_classes = [permissions.AllowAny]
    queryset = Photo.objects.all()
    serializer_class = PhotoSerializer