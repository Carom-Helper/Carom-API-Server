# Generated by Django 4.1.2 on 2022-11-23 14:34

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("detection", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="position",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "coord",
                    models.JSONField(default=dict, verbose_name="Coordination JSON"),
                ),
                (
                    "state",
                    models.CharField(
                        choices=[
                            ("N", "None"),
                            ("A", "Accepted"),
                            ("P", "Progress"),
                            ("D", "Done"),
                        ],
                        default="N",
                        max_length=1,
                        verbose_name="Work State",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="soultion_route",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("route", models.JSONField(default=dict, verbose_name="Route JSON")),
                (
                    "algorithm_ver",
                    models.CharField(max_length=10, verbose_name="Version"),
                ),
                (
                    "issue",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="recommend.position",
                        verbose_name="Ball Position",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="route_request",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "requester",
                    models.CharField(max_length=20, verbose_name="User Name"),
                ),
                (
                    "issue",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="recommend.position",
                        verbose_name="Issue ID",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="compare_detect",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "ai",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="detection.balls_coord",
                        verbose_name="AI Soulution",
                    ),
                ),
                (
                    "usr",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="recommend.position",
                        verbose_name="USER Soulution",
                    ),
                ),
            ],
        ),
        migrations.AddConstraint(
            model_name="route_request",
            constraint=models.UniqueConstraint(
                fields=("issue", "requester"), name="unique route request"
            ),
        ),
    ]
