"""
Module to train all models in this directory
"""

from src.ClasificationLynfoma import trainAllSegmentation
from src.ClasificationLynfoma import trainAllClasification

def main():
    """
    Run code
    """
    trainAllSegmentation.main()
    trainAllClasification.main()

if __name__ == "__main__":
    main()