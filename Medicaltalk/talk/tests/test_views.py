from django.db.models import query
from django.test import TestCase, Client
from django.urls import reverse
from talk.models import Post

class TestViews(TestCase):

    def setUp(self):
        self.client = Client()
        self.home_url = reverse('talk-home')
        self.post_list_url = reverse('talk-home')
        self.post_detail_url = reverse('post-detail', args=[1])
        self.post_create_url = reverse('post-create')
        self.post_update_url = reverse('post-update', args=[1])
        self.post_delete_url = reverse('post-delete', args=[1])
        self.user_post_list_url = reverse('user-posts', args=['simon'])
        self.about_url = reverse('talk-about')
        self.calender_url = reverse('talk-calender')
        self.search_url = reverse('search')
  

    def test_home(self):
        response = self.client.get(self.home_url)
        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'talk/home.html')

    def test_post_list_view(self):
        response = self.client.get(self.post_list_url)
        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'talk/home.html')

    def test_post_detail_view(self):
        response = self.client.get(self.post_detail_url)
        self.assertEquals(response.status_code, 404)

    def test_post_create_view(self):
        response = self.client.get(self.post_create_url)
        self.assertEquals(response.status_code, 302)

    def test_post_upadate_view(self):
        response = self.client.get(self.post_update_url)
        self.assertEquals(response.status_code, 302)

    def test_post_delete_view(self):
        response = self.client.get(self.post_delete_url)
        self.assertEquals(response.status_code, 302)

    def test_user_post_list_view(self):
        response = self.client.get(self.user_post_list_url)
        self.assertEquals(response.status_code, 404)

    def test_about(self):
        response = self.client.get(self.about_url)
        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'talk/about.html')

    def test_calender(self):
        response = self.client.get(self.calender_url)
        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'talk/calender.html')

    def test_search(self):
        response = self.client.get(self.search_url, {'query' : 'hello'})
        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'talk/search.html')


    



    

