from django import forms
from .models import EcgImageDataBase 

# Create your forms here.
class ImageUploadForm(forms.ModelForm):

    class Meta:
        model = EcgImageDataBase
        fields = ['image']