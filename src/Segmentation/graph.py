import os
from src.utils import graph

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    graph(BASE_DIR)

if __name__ == "__main__":
    main()