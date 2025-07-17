from src.ClasificationLynfoma import trainAllSegmentation
from src.ClasificationLynfoma import trainAllClasification

def main():
    trainAllSegmentation.main()
    trainAllClasification.main()

if __name__ == "__main__":
    main()