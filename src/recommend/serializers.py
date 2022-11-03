from rest_framework import serializers
from .models import *

class PositionSerializer(serializers.ModelSerializer):
    class Meta:
        model = position
        fields = '__all__'

class RouteSerializer(serializers.ModelSerializer):
    class Meta:
        model = soultion_route
        fields = '__all__'
