import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import json
import time
from src import models

"""
Se implementan funciones basicas
"""

def get_empty():
    """
    Esta funcion no pide nada y devuelve una lista vacia
    """
    out = []
    return out

def triplicate(input):
    """
    Esta funcion pide un input y lo multiplica por 3
    """
    out = input*3
    return out