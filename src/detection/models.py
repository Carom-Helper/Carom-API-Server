from ast import Mod
from django.db import models
from django.db import models
# from django_db_views.db_view import DBView
from django.db.models import CASCADE, Model
# Create your models here.

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class projection_method(Model):
    name = models.CharField(max_length=100, verbose_name="NAME")
    value = models.PositiveSmallIntegerField(verbose_name="VALUE", null=True)