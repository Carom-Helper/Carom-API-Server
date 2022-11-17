# django base
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
# add base
import urllib.parse as uparse

# add our project
from .serializers import *


# Create your views here.
class CaromDataViewSet(viewsets.ModelViewSet):
    lookup_field = 'id'
    queryset = carom_data.objects.all()
    serializer_class = CaromDataSerializer
    
class CaromImageViewSet(viewsets.ModelViewSet):
    lookup_field = 'id'
    queryset = carom_img.objects.all()
    serializer_class = CaromImageSerializer
    parser_classes=[MultiPartParser]

class CaromTableViewSet(viewsets.ModelViewSet):
    lookup_field = 'id'
    queryset = CaromTable.objects.all()
    serializer_class = CaromTableSerializer


