from .models import Photo
from rest_framework import generics
from rest_framework import permissions, authentication
from .serializers import PhotoSerializer
from rest_framework.views import APIView
from rest_framework.response import Response

class PhotoListView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, format=None):
        photos = Photo.objects.all()
        serializer = PhotoSerializer(photos, many=True)
        return Response(serializer.data)

class PhotoCreateView(generics.CreateAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = PhotoSerializer
    queryset = Photo.objects.all()

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
