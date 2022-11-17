"""carom_api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# django base
from django.contrib import admin
from django.urls import path, include
from .views import CaromDataViewSet, CaromImageViewSet, CaromTableViewSet

#define ViewSet
carom_data_upload = CaromDataViewSet.as_view({
    'get' : 'list',
    'post' : 'create',
})
carom_img_upload = CaromImageViewSet.as_view({
    'get' : 'list',
    'post' : 'create',
})

carom_list= CaromTableViewSet.as_view({
    'get' : 'list',
})

carom_detail = CaromTableViewSet.as_view({
    'get' : 'list',
})

#defin url pattern
urlpatterns = [
    path("", carom_list),
    path("<int:id>/", carom_detail),
    path("img/", carom_img_upload),
    path("data/", carom_data_upload),
]



