from django.test import SimpleTestCase
from django.contrib.auth import views as auth_views
from django.urls import reverse, resolve
from users import views as user_views
from users.views import register, profile

class TestUrls(SimpleTestCase):
    def test_login_url_is_resolves(self):
        url = reverse('login')
        self.assertEqual(resolve(url).func.view_class, auth_views.LoginView)

    def test_logout_url_is_resolves(self):
        url = reverse('logout')
        self.assertEqual(resolve(url).func.view_class, auth_views.LogoutView)

    def test_register_url_is_resolves(self):
        url = reverse('register')
        self.assertEqual(resolve(url).func, user_views.register)

    def test_profile_url_is_resolves(self):
        url = reverse('profile')
        self.assertEqual(resolve(url).func, user_views.profile)
