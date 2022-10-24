from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view

# Create your views here.
@api_view()
def hello(request):
    return Response({"message":"hellow u-img"})