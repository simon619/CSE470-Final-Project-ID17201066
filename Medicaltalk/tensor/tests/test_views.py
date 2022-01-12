from django.test import TestCase
from django.urls import reverse
import tempfile

class TensorTest(TestCase):

    def setUp(self):
        self.image = tempfile.NamedTemporaryFile(suffix=".jpg").name
        self.search_url = reverse('tensor-ecg')

    def test_tensor(self):
        response = self.client.get(self.search_url, {'query' : self.image})
        self.assertEquals(response.status_code, 302)

