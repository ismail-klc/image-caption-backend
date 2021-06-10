from .models import Photo
from rest_framework import generics
from rest_framework import permissions, authentication
from .serializers import PhotoSerializer
from django.core import serializers
from rest_framework.views import APIView
from rest_framework.response import Response
from .pagination import MyCursorPagination
from rest_framework.parsers import FormParser, MultiPartParser, FileUploadParser
# from .predict import get_caption
from .transformer import get_caption
from PIL import Image
import io
from rest_framework import status
from urllib.request import urlopen
from rest_framework import filters


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
    parser_classes = (FormParser, MultiPartParser, FileUploadParser)
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = PhotoSerializer
    queryset = Photo.objects.all()

    def perform_create(self, serializer):
        print(serializer.validated_data)

        image = serializer.validated_data.get('image')
        image = image.read()
        image = Image.open(io.BytesIO(image))

        newCaption = get_caption(image)
        print(newCaption)
        serializer.save(
            user=self.request.user,
        )

class AddPhotoView(APIView):
    parser_classes = (MultiPartParser, FileUploadParser)
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, format=None):
        serializer = PhotoSerializer(data=request.data)

        if serializer.is_valid():
            serializer.validated_data["user"] = request.user
            serializer.save()
            pid = serializer.data.get('id')
            
            # construct an caption of image
            picture = serializer.data.get('image')
            fd = urlopen(picture)
            image_file = io.BytesIO(fd.read())
            cPicture = Image.open(image_file)
            caption = get_caption(cPicture)

            # update caption of image on database
            photo = Photo.objects.get(id=pid)
            photo.caption = caption
            serializer.data["caption"] = caption
            photo.save()

            serialized_data = serializer.data
            serialized_data["caption"] = caption
            return Response( serialized_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SinglePhotoView(generics.RetrieveAPIView):
    permission_classes = [permissions.AllowAny]
    queryset = Photo.objects.all()
    serializer_class = PhotoSerializer


class DemoView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def post(self, request, format=None):
        picture = request.FILES['picture']
        picture = picture.read()
        picture = Image.open(io.BytesIO(picture))

        caption = get_caption(picture)
        data = {
            'caption': caption
        }

        return Response(status=200, data=data)

class SearchPhotoView(generics.ListAPIView):
    serializer_class = PhotoSerializer
    permission_classes = [permissions.AllowAny]
    queryset = Photo.objects.all()
    filter_backends = [filters.SearchFilter]
    search_fields = ['caption']