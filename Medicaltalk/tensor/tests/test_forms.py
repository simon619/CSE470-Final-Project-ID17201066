from logging import error
from django.test import TestCase
from django.contrib.auth.models import User
from tensor.forms import ImageUploadForm
import tempfile

class TestForms(TestCase):

    def test_image_update_from_valid(self):
        user = User.objects.create(username='simon', email='honolulu@gmail.com', password='iamgonadie12')
        self.image = tempfile.NamedTemporaryFile(suffix=".jpg").name
        form_data = {
            'image' : self.image
        }

        form = ImageUploadForm(data=form_data)
        print(form.errors)
        self.assertFalse(form.is_valid())

