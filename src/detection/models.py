from ast import Mod
from django.db import models
from django.db import models
# from django_db_views.db_view import DBView
from django.db.models import CASCADE, Model

from u_img.models import carom
from carom_api.settings import BASE_DIR, MEDIA_ROOT

# Create your models here.    
class projection(Model):
    origin = models.ForeignKey(to="u_img.carom", on_delete=CASCADE, verbose_name="Original Image ID")
    img = models.ImageField(upload_to="projections/%Y/%m/%d/", verbose_name="Image")
    
class balls_coord(Model):
    carom = models.ForeignKey(to="u_img.carom", on_delete=CASCADE, verbose_name="Carom Image ID")
    coord = models.FileField(max_length=200, upload_to="detect/", verbose_name="COORDINATE JSON", null=True)