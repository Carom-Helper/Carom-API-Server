# Carom-API-Server
This project uses Django REST API and Pythorch. It provides an API that tells you the coordinates when the user takes a picture of billiards. When the user gives the initial placement coordinates of billiards, it provides an API that informs the recommended route.

# How to install
```bash
git clone https://github.com/Carom-Helper/Carom-API-Server.git
cd Carom-API-Server
# git submodule update --init --recursive --remote
git submodule update --init --recursive
cd src/detection/detect/npu_yolov5/utils/box_decode/cbox_decode
python setup.py build_ext --inplace
cd ../../../../../../

```
#### Next Step. Set {$ROOT}/src/secrets.json && {$ROOT}/src/settings.json
```bash
# set +H
# FRAME_WORK('furiosa', '0', 'cpu', 'onnx')
echo '{"FRAME_WORK":"furiosa" ,"HOST_NAME":"118.36.223.138", "PORT_NUM":"7576"}' > settings.json
echo '{json contents}' > secrets.json
cd ..
```
###### json contents
```json
# example
{"SECRET_KEY":"django-insecure-d*upt!(-*)wA#3^cdc-e9ac3s4s8afd9d4m=_2(!a+2v&@1avs2s4v="}
```

# How to Set Development Environment with Anaconda(or PIP)
https://pytorch.org/get-started/locally/
```bash
conda create -n carom-api python=3.8.2 -y
conda activate carom-api
# GPU - CONDA
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
# CPU - CONDA
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

# How to initialize ENV with Anaconda(or PIP)
```bash
pip install -r requirements.txt
```

# How to run with Anaconda(or PIP)
```bash
# Check if the django works well
cd src
python manage.py runserver
```
```bash
# Check if the pytorch works well
cd detection/detect
python DetectObjectPipe.py
cd npu_yolov5
python demo.py --calib-data /caromapi/src/media/test --framework furiosa --input /caromapi/src/media/test2/sample.jpg --model /caromapi/src/detection/detect/weights/npu_yolo_ball --no-display
```

# How to Set  Development Environment with Docker
1. Add USR in Makefile
```Makefile
USR={#ADD Your name (ex - jylee}
PORT_NUM=3213
SRC_NAME=caromapi
```
2. Docker build
```bash
make build
```

# How to run with Docker
```bash
make run
```

# How to setting In Docker Container (attach shell)
#### For Window
```bash
python clear_migrate.py & python manage.py makemigrations & python manage.py makeviewmigrations & python manage.py migrate & echo import init_setter | python manage.py shell_plus
```
#### Fro Linux
```bash
python clear_migrate.py && python manage.py makemigrations && python manage.py makeviewmigrations && python manage.py migrate && echo "import init_setter" | python manage.py shell_plus
```

# How to test
```bash
# error npu_yolov5/utils/inference_engine.py InferenceEngineFuriosa.__init__
# compile_config change to compiler_config
python manage.py test
python manage.py runserver 0.0.0.0:7576
```

# How to stop Docker container
```bash
# Ctrl + c
make rm
```

#furiosa compile 
```bash
 # furiosa compile 파일명.onnx -o 파일명.enf
 NPU_COMPILER_CONFIG_PATH=compiler_config.yaml furiosa compile weights_i8.onnx -o weights.enf
```
