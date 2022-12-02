# Generated by Django 4.1.3 on 2022-12-02 06:13

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CaromTable',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('guide', models.JSONField(default=dict, null=True, verbose_name='Guide Point')),
                ('img', models.ImageField(default='/', null=True, upload_to='', verbose_name='Image')),
                ('detect_state', models.CharField(choices=[('N', 'None'), ('A', 'Accepted'), ('P', 'Progress'), ('D', 'Done')], default='N', max_length=1, null=True, verbose_name='Work State')),
            ],
            options={
                'db_table': 'CaromTable',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='carom_img',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('img', models.ImageField(upload_to='carom/', verbose_name='Image')),
            ],
        ),
        migrations.CreateModel(
            name='carom_data',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('guide', models.JSONField(default=dict, verbose_name='Guide JSON')),
                ('detect_state', models.CharField(choices=[('N', 'None'), ('A', 'Accepted'), ('P', 'Progress'), ('D', 'Done')], default='N', max_length=1, verbose_name='Work State')),
                ('img', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='u_img.carom_img', verbose_name='Image ID')),
            ],
        ),
        migrations.AddConstraint(
            model_name='carom_data',
            constraint=models.UniqueConstraint(fields=('img',), name='unique img id'),
        ),
    ]
