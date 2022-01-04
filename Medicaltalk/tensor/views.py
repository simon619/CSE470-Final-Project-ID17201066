from django.shortcuts import render, redirect
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
from .forms import ImageUploadForm
from .models import EcgImageDataBase
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
from django.core.files.storage import FileSystemStorage, default_storage
import numpy as np
from django.contrib.auth.decorators import login_required

model = tf.keras.models.load_model('tensor\ECG_Classificaton_CNN_Model.h5')

CATEGORIES = ['MI Patient', 'Normal']
@login_required
def classfication(request):
	if request.method == "POST":
		form = ImageUploadForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			file = request.FILES["image"]
			file_name = default_storage.save(file.name, file)
			file_url = default_storage.path(file_name)
			prediction = model.predict([prepare(file_url)])
			x = CATEGORIES[int(prediction[0][0])]
			
		return render(request=request, template_name="tensor/ecg.html", context={'form':form, 'x':x})
	form = ImageUploadForm()
    
	return render(request=request, template_name="tensor/ecg.html", context={'form':form})

def prepare(filepath):
    IMG_SIZE = 100  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
