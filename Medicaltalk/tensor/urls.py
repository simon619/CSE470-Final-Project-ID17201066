from django.urls import path
from . import views
from .views import classfication
  
urlpatterns = [
    path('', views.classfication, name='tensor-ecg'),
]