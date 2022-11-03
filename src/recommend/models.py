from django.db import models
from django.db.models import CASCADE, Model

from detection.models import balls_coord
from carom_api.settings import BASE_DIR, MEDIA_ROOT

# Create your models here.
class position(Model):
    WORK_STATE = [
        ('N', "None"),
        ('A', "Accepted"),
        ('P', "Progress"),
        ('D', "Done"),
    ]
    coord = models.JSONField(default=dict, verbose_name="Coordination JSON")
    state = models.CharField(max_length=1, choices=WORK_STATE, verbose_name="Work State", default="N")

class soultion_route(Model):
    issue = models.ForeignKey(to="position", on_delete=CASCADE, verbose_name="Ball Position")
    route = models.JSONField(default=dict, verbose_name="Route JSON")
    algorithm_ver = models.CharField(max_length=10, verbose_name="Version", default="ver0")
    

class compare_detect(Model):
    ai = models.ForeignKey(to="detection.balls_coord", on_delete=CASCADE, verbose_name="AI Soulution")
    usr = models.ForeignKey(to="position", on_delete=CASCADE, verbose_name="USER Soulution")
    
    
class route_request(Model):
    issue = models.ForeignKey(to="position", on_delete=CASCADE, verbose_name="Issue ID")
    requester = models.CharField(max_length=20, verbose_name="User Name")
    
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["issue", "requester"],
                name = "unique route request"
            )
        ]
    