from django.db.models import query
from django.shortcuts import render, get_object_or_404
from .models import Post
from talk.models import Post
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from django.contrib import messages


def home(request):
    context = {
        'posts' : Post.objects.all()
    }
    return render(request, 'talk/home.html', context)

class PostListView(ListView):
    model = Post
    template_name = 'talk/home.html' #<app>/<model>_<viewtype>.html
    context_object_name = 'posts'
    ordering = ['-date_posted']
    paginate_by = 5

class PostDetailView(DetailView):
    model = Post

class PostDetailView(DetailView):
    model = Post

class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    fields = ['title', 'content']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    fields = ['title', 'content']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
    
    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    success_url = '/'
    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

class UserPostListView(ListView):
    model = Post
    template_name = 'talk/user_posts.html' #<app>/<model>_<viewtype>.html
    context_object_name = 'posts'
    paginate_by = 5

    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        return Post.objects.filter(author=user).order_by('-date_posted')


def calender(request):
    return render(request, 'talk/calender.html')

def about(request):
    return render(request, 'talk/about.html', {'title' : 'About'})

def search(request):
    query = request.GET['query']
    if len(query) > 66:
        allposts = Post.objects.none()
    else :
        allpostsTitle = Post.objects.filter(title__icontains=query)
        allpostsContent = Post.objects.filter(content__icontains=query)
        allposts = allpostsTitle.union(allpostsContent)

    if allposts.count() == 0:
        messages.warning(request, "No Search Result Found. Please Refine Your Query")
    
    params = {'allposts' : allposts, 'query' : query}
    return render(request, 'talk/search.html', params)
