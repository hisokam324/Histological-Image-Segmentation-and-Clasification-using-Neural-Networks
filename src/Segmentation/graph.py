import os
from src.utils import graph

"""
Auxiliary module to graph last loss function of each model in this directory
"""

def main():
    """
    Run code
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    graph(BASE_DIR)

if __name__ == "__main__":
    main()