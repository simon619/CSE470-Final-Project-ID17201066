from django.test import SimpleTestCase
from django.urls import reverse, resolve
from tensor.views import classfication

class TestUrls(SimpleTestCase):
    def test_login_url_is_resolves(self):
        url = reverse('tensor-ecg')
        self.assertEqual(resolve(url).func, classfication)