from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Photo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="photos",default=None)
    caption = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.caption