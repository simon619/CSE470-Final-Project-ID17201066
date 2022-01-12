from django.test import SimpleTestCase
from django.test import TransactionTestCase
from talk.views import PostListView, PostDetailView, PostCreateView, PostDeleteView, PostUpdateView, UserPostListView, about, calender, search
from django.urls import reverse, resolve
from users.models import User


class TestUrls(SimpleTestCase):

    def test_about_url_is_resolves(self):
        url = reverse('talk-about')
        self.assertEqual(resolve(url).func, about)

    def test_search_url_is_resolves(self):
        url = reverse('search')
        self.assertEqual(resolve(url).func, search)

    def test_calender_url_is_resolves(self):
        url = reverse('talk-calender')
        self.assertEqual(resolve(url).func, calender)

    def test_home_url_is_resolves(self):
        url = reverse('talk-home')
        self.assertEqual(resolve(url).func.view_class, PostListView)

    def test_post_create_views_url_is_resolves(self):
        url = reverse('post-create')
        self.assertEqual(resolve(url).func.view_class, PostCreateView)

    def test_post_detail_views_url_is_resolves(self):
        url = reverse('post-detail', args=[1])
        self.assertEqual(resolve(url).func.view_class, PostDetailView)
    
    def test_post_Delete_views_url_is_resolves(self):
        url = reverse('post-delete', args=[1])
        self.assertEqual(resolve(url).func.view_class, PostDeleteView)   

    def test_post_update_views_url_is_resolves(self):
        url = reverse('post-update', args=[1])
        self.assertEqual(resolve(url).func.view_class, PostUpdateView)

    def test_User_post_list_views_url_is_resolves(self):
        url = reverse('user-posts', args=['Simon'])
        self.assertEqual(resolve(url).func.view_class, UserPostListView)



    