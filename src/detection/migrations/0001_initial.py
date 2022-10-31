# Generated by Django 4.1.2 on 2022-10-31 08:40

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("u_img", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="projection",
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
                    "img",
                    models.ImageField(upload_to="projections/", verbose_name="Image"),
                ),
                (
                    "origin",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="u_img.carom",
                        verbose_name="Original Image ID",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="detect_request",
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
                    "state",
                    models.CharField(
                        choices=[
                            ("N", "None"),
                            ("C", "Create"),
                            ("P", "Progress"),
                            ("D", "Done"),
                        ],
                        max_length=1,
                        verbose_name="Work State",
                    ),
                ),
                (
                    "carom",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="u_img.carom",
                        verbose_name="Carom Image ID",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="balls_coord",
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
                    models.FileField(
                        max_length=200,
                        null=True,
                        upload_to="detect/",
                        verbose_name="COORDINATE JSON",
                    ),
                ),
                (
                    "carom",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="u_img.carom",
                        verbose_name="Carom Image ID",
                    ),
                ),
            ],
        ),
        migrations.AddConstraint(
            model_name="detect_request",
            constraint=models.UniqueConstraint(
                fields=("carom",), name="unique request"
            ),
        ),
    ]
