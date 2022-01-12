from django.test import TestCase
from talk.models import Post
from django.contrib.auth.models import User
from django.urls import reverse

class TestModel(TestCase):

    def setUp(self):
        user = User.objects.create(username='simon')
        self.post1 = Post.objects.create(title = 'Testing', content = 'Tired of testing', date_posted='2020-09-12', author=user)
        self.search_url = reverse('post-detail', args = [1])
    
    
    def test_model(self):
        response = self.client.get(self.search_url)
        self.assertEquals(response.status_code, 200)

    
    
