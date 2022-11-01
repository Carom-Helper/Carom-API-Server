from rest_framework import serializers
from .models import *

class CoordSerializer(serializers.ModelSerializer):
    class Meta:
        model = balls_coord
        fields = '__all__'
    
        
class ProjectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = projection
        fields = '__all__'

class RequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = detect_request
        fields = '__all__'