from time import time
from urllib import request
import datetime
import json
import sys
import os

def loadJson(dir :str) -> list:
    """_summary_

    Args:
        dir (str): _description_

    Returns:
        list: _description_
    """
    """Json 파일에 저장된 데이터 불러오기"""
    with open(dir, 'r') as file:
        ls = json.load(file)
    return ls

def saveJson(dir :str, data) -> None:
    """_summary_

    Args:
        dir (str): _description_
        data (_type_): _description_
    """
    # """데이터를 Json 파일에 저장하기"""
    with open(dir, 'w', encoding="utf-8") as file:
        json.dump(data, file, indent="\t")

def createDirectory(directory :str) -> None:
    """_summary_

    Args:
        directory (str): _description_
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def workingDirectory() -> str:
    """_summary_

    Returns:
        str: _description_
    """
    if getattr(sys, 'frozen', False):
        #test.exe로 실행한 경우,test.exe를 보관한 디렉토리의 full path를 취득
        currDir = os.path.dirname(os.path.abspath(sys.executable))    
    else:
        #python test.py로 실행한 경우,test.py를 보관한 디렉토리의 full path를 취득
        currDir = os.path.dirname(os.path.abspath(__file__))
    return currDir

def filesInFolder(dir :str, extention=False) -> list:
    """_summary_

    Args:
        dir (str): _description_
        extention (bool, optional): _description_. Defaults to False.

    Returns:
        list[str]: _description_
    """
    if extention:
        return [file.replace(f".{extention}", "") for file in os.listdir(dir) if file.endswith(f".{extention}")]
    else:
        return [file for file in os.listdir(dir)]

def download_file(url :str, filename :str) -> None:
    """_summary_

    Args:
        url (str): _description_
        filename (str): _description_
    """
    try:
        request.urlretrieve(url, filename)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def log(text :str) -> None:
    """_summary_

    Args:
        text (str): _description_
    """
    currDir = os.path.dirname(os.path.realpath(__file__))
    with open(f'{currDir}/log.txt', 'a') as file:
        file.write(f"{datetime.datetime.now()} - {text}\n")

def inputType(func) -> dict:
    """_summary_

    Args:
        func (_type_): _description_

    Returns:
        dict: _description_
    """
    return func.__annotations__

def checkTime(func, *params) -> float:
    """_summary_

    Args:
        func (_type_): _description_

    Returns:
        float: _description_
    """
    start = time()
    output = func(*params)
    return (time()-start, output)

def parentDirectory(dir :str, separator :str="\\", n :int=1) -> str:
    """_summary_

    Args:
        dir (str): _description_
        separator (str): _description_. Defaults to "\".
        n (int): _description_. Defaults to 1.

    Returns:
        str: _description_
    """
    return separator.join(dir.split(separator)[:-n])

currDir = os.path.dirname(os.path.realpath(__file__))