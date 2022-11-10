import os
import sys
from pathlib import Path
from glob import glob
import shutil
BASE_DIR = Path(__file__).resolve().parent

MIGRATIONS_DIR = "migrations" # 타겟 폴더
CACHE_DIR = "__pycache__" # 신경 안쓰는 폴더더
INIT_PY = "__init__.py" # 안 삭제 할것
DB_FILE = "db.sqlite3" # 삭제할 것

def read_all_file(path):
    # 폴더를 하나씩 조회
    dir_list = os.listdir(path)
    file_list = []

    for dir_file in dir_list:
        if dir_file == CACHE_DIR:
            continue
        elif dir_file == MIGRATIONS_DIR:
            file_list.extend(migrate_file(path / dir_file))
        elif os.path.isdir(path / dir_file): 
            file_list.extend(read_all_file(path / dir_file)) 
    return file_list

def migrate_file(path):
    # 폴더 안에 migrations 폴더 탐색
    dir_list = os.listdir(path)
    migrate_file_list = []
    
    for dir_file in dir_list:
        if dir_file == INIT_PY or dir_file == CACHE_DIR:
            continue
        else:
            migrate_file_list.append(path / dir_file)
    
    return migrate_file_list

# 폴더를 하나씩 조회
# 폴더 안에 migrations 폴더 탐색
file_list = read_all_file(BASE_DIR)
file_list.append(BASE_DIR / DB_FILE)

# migrations 폴더 내부에 py 파일들을 삭제할 것인데, __init__.py는 남김

for file in file_list:
    if os.path.exists(file):
        os.remove(file)
        
# python clear_migrate.py & python manage.py makemigrations & python manage.py makeviewmigrations & python manage.py migrate & ECHO import init_setting | python manage.py shell_plus