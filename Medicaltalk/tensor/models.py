from django.db import models

# Create your models here.
class EcgImageDataBase(models.Model):
   image = models.ImageField(upload_to='ECG_images/', null=True)
