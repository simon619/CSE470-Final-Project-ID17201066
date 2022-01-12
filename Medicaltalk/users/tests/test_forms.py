from logging import error
from django.test import TestCase
from django.contrib.auth.models import User
from users.forms import  UserRegisterForm, UserUpdateForm, ProfileUpdateForm
import tempfile

class TestForms(TestCase):

    def test_register_from_valid(self):
        user = User.objects.create(username='simon', email='honolulu@gmail.com', password='iamgonadie12')

        form_data = {
            'username' : user.pk,
            'first_name' : 'Funny',
            'last_name' : 'man',
            'email' : 'simon@gmail.com',
            'password1' : 'iamgonadie12',
            'password2' : 'iamgonadie12'

        }
        form = UserRegisterForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_register_from_invalid(self):
        form = UserRegisterForm(data ={})
        self.assertFalse(form.is_valid())
        self.assertEquals(len(form.errors), 4)


    def test_update_from_valid(self):
        user = User.objects.create(username='simon', email='honolulu@gmail.com', password='iamgonadie12')

        form_data = {
            'username' : user.pk,
            'first_name' : 'Funny',
            'last_name' : 'man',
            'email' : 'simon@gmail.com'
        }
        form = UserUpdateForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_update_from_invalid(self):
        form = UserUpdateForm(data ={})
        self.assertFalse(form.is_valid())
        self.assertEquals(len(form.errors), 2)

    def test_profile_update_from_valid(self):
        user = User.objects.create(username='simon', email='honolulu@gmail.com', password='iamgonadie12')
        self.image = tempfile.NamedTemporaryFile(suffix=".jpg").name

        form_data = {
            'image' : self.image,
            'speciality' : 'Funniness',
            'bio' : 'Kai dai gumai'
        }
        form = ProfileUpdateForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_profile_update_from_invalid(self):
        form = ProfileUpdateForm(data ={})
        self.assertFalse(form.is_valid())
        self.assertEquals(len(form.errors), 2)
    