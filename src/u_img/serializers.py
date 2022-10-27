from rest_framework import serializers
from .models import *

class CaromSerializer(serializers.ModelSerializer):
    class Meta:
        model = carom
        fields = '__all__'