from django.db import models
from django.db.models import CASCADE, Model

from u_img.models import carom_img
from carom_api.settings import BASE_DIR, MEDIA_ROOT

# Create your models here.    
# class projection(Model):
#     origin = models.ForeignKey(to="u_img.carom", on_delete=CASCADE, verbose_name="Original Image ID")
#     img = models.ImageField(upload_to="projections/", verbose_name="Image")
    
class balls_coord(Model):
    carom = models.ForeignKey(to="u_img.carom_img", on_delete=CASCADE, verbose_name="Carom Image ID")
    coord = models.JSONField(default=dict, verbose_name="COORDINATE JSON")
    
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["carom"],
                name = "unique detect solution"
            )
        ]
    
    
class detect_request(Model):
    carom = models.ForeignKey(to="u_img.carom_img", on_delete=CASCADE, verbose_name="Carom Image ID")
    requester = models.CharField(max_length=20, verbose_name="User Name")
    
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["carom", "requester"],
                name = "unique detect request"
            )
        ]
    