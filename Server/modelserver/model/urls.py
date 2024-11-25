from django.urls import path
from .views import HelloWorld, FraminghamModel


urlpatterns = [
    path("hello_world/", HelloWorld.as_view()),
    path("framingham_model/", FraminghamModel.as_view())
]