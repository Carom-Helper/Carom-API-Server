from django.db import models
from django.db import models
from django.db.models import CASCADE, Model
# from django_db_views.db_view import DBView

from carom_api.settings import BASE_DIR, MEDIA_ROOT

# Create your models here.

from pathlib import Path

class projection_method(Model):
    name = models.CharField(max_length=100, verbose_name="NAME")
    value = models.PositiveSmallIntegerField(verbose_name="VALUE", null=True)

class carom(Model):
    img = models.ImageField(upload_to="carom/%Y/%m/%d/", verbose_name="Image")
    method = models.ForeignKey(to="projection_method", on_delete=CASCADE, verbose_name="Type")
    table_size = models.DecimalField(verbose_name="Persent", max_digits=4, decimal_places=1)