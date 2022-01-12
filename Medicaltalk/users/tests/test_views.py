from django.contrib.auth.models import User
from django.test import TestCase, Client
from django.urls import reverse
from users.models import Profile
import tempfile

class TestViews(TestCase):

    def setUp(self) -> None:
        self.username = 'simon'
        self.email = 'simon@gmail.com'        
        self.password = 'simon1264'
        self.image = tempfile.NamedTemporaryFile(suffix=".jpg").name
        self.speciality = 'Nothing'
        self.bio = 'Kai dai gumai'
        self.register_url = reverse('register')
        self.profile_url = reverse('profile')
        self.login_url = reverse('login')
        self.logout_url = reverse('logout')

    
    def test_register(self):
        response = self.client.post(self.register_url, data={
            'username': self.username,
            'email': self.email,
            'password1': self.password,
            'password2': self.password
        })
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, template_name='users/register.html')
        

    def test_profile(self):
        response = self.client.post(self.profile_url, data={
            'username': self.username,
            'email': self.email,
            'speciality': self.password,
            'bio': self.password
        })
        self.assertEqual(response.status_code, 302)

    def test_login(self):
        response = self.client.post(self.login_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, template_name='users/login.html')

    def test_logout(self):
        response = self.client.post(self.logout_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, template_name='users/logout.html')



    


    




