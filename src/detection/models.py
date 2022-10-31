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
    img = models.ImageField(upload_to="projections/", verbose_name="Image")
    
class balls_coord(Model):
    carom = models.ForeignKey(to="u_img.carom", on_delete=CASCADE, verbose_name="Carom Image ID")
    coord = models.FileField(max_length=200, upload_to="detect/", verbose_name="COORDINATE JSON", null=True)
    
    
class detect_request(Model):
    WORK_STATE = [
        ('N', "None"),
        ('C', "Create"),
        ('P', "Progress"),
        ('D', "Done"),
    ]
    
    carom = models.ForeignKey(to="u_img.carom", on_delete=CASCADE, verbose_name="Carom Image ID")
    requester = models.CharField(max_length=20, verbose_name="User Name")
    state = models.CharField(max_length=1, choices=WORK_STATE, verbose_name="Work State")
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["carom"],
                name = "unique request"
            )
        ]
    