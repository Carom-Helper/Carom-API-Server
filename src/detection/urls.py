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
from .views import CoordViewSet, ProjectionViewSet, DetectRequestAPIView

#define ViewSet
coord_list = CoordViewSet.as_view({
    'get' : 'list',
})
coord_detail = CoordViewSet.as_view({
    'get' : 'retrieve',
    'delete' : 'destroy'
})

projection_list = ProjectionViewSet.as_view({
    'get' : 'list',
})
projection_detail = ProjectionViewSet.as_view({
    'get' : 'retrieve',
    'delete' : 'destroy'
})

#defin url pattern
urlpatterns = [
    path("balls-coord/<int:carom_id>/<str:usr>/", DetectRequestAPIView.as_view()),
    path("balls-coord/result/", coord_list),
    path("balls-coord/result/<int:id>/", coord_detail),
    path("projection-img/", projection_list),
    path("projection-img/<int:id>/", projection_detail),
]