from rest_framework import serializers
from .models import *

class CaromDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = carom_data
        fields = '__all__'
class CaromImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = carom_img
        fields = '__all__'
        
class CaromTableSerializer(serializers.ModelSerializer):
    class Meta:
        model = CaromTable
        fields = '__all__'
