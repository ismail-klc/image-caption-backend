from rest_framework import serializers
from .models import Photo
from django.contrib.auth.models import User
from authentication.serializers import UserSerializer


class PhotoSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = Photo
        fields = ('id', 'caption','image', 'user', 'created_at')
        