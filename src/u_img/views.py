# django base
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view

# add base
import urllib.parse as uparse

# add our project
from .serializers import *


# Create your views here.
class CaromViewSet(viewsets.ModelViewSet):
    lookup_field = 'id'
    queryset = carom.objects.all()
    serializer_class = CaromSerializer