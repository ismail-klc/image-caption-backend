from rest_flex_fields import FlexFieldsModelSerializer
from rest_framework import serializers
from .models import Photo
from django.contrib.auth.models import User


class PhotoSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField()
    
    class Meta:
        model = Photo
        fields = ('id', 'caption','image', 'user', 'created_at')
        