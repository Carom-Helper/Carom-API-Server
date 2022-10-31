# Generated by Django 4.1.2 on 2022-10-31 05:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("detection", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="balls_coord",
            name="coord",
            field=models.FileField(
                max_length=200,
                null=True,
                upload_to="detect/",
                verbose_name="COORDINATE JSON",
            ),
        ),
    ]
