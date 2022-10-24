# Carom-API-Server
This project uses Django REST API and Pythorch. It provides an API that tells you the coordinates when the user takes a picture of billiards. When the user gives the initial placement coordinates of billiards, it provides an API that informs the recommended route.

# How to install
```bash
git clone https://github.com/Carom-Helper/Carom-API-Server.git
git submodule update --init --recursive
```
##### Next Step. Set ROOT/src/secrets.json
```json
# example
{"SECRET_KEY":"django-insecure-d*upt!(-*)wA#3^cdc-e9ac3s4s8afd9d4m=_2(!a+2v&@1avs2s4v="}
```

# How to Set Development Environment with Anaconda(or PIP)
```bash
conda create -n carom-api python=3.8.2
conda activate carom-api
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
```

# How to initialize ENV with Anaconda(or PIP)
```bash
cd src
pip install -r requirements.txt
```

# How to run width Anaconda(or PIP)
```bash
cd src
python manage.py runserver
```
