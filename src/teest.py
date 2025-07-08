import matplotlib.pyplot as plt

"""
Esto es sencillamente una prueba
"""

def calcular(x):
    """
    Metodo de calculo correcto
    """
    y = (x-5+1)//2
    y = (y-5+1)//2
    return y

def calcular2(x):
    """
    Metodo alternativo a probar
    """
    return x//4-3

def main():
    """
    Cuerpo
    """
    X = []
    Y = []
    Z = []

    for i in range(32):
        X.append(i)
        Y.append(calcular(i))
        Z.append(calcular2(i))

    plt.figure()
    plt.plot(X, Y, label = "Y")
    plt.plot(X, Z, label = "Z")
    plt.legend()
    plt.show()

#if __name__ == "__main__":
#    main()