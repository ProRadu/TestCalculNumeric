import numpy as np
import matplotlib.pyplot as plt


# Ex1
# Pentru a rezolva ecuatia x^2-29=0 putem aproxima solutia cu ajutorul metodei Newton-Raphson aplicata functiei f(x)=x^2-29.
def func1(x):
    y = x ** 2 - 29
    return y

def dfunc1(x):
    y = 2 * x
    return y

def newton_raphson(f, df, x0, eps, max_iter):
    i = 1
    while i <= max_iter:
        x1 = x0 - f(x0) / df(x0)
        if np.abs(x1 - x0) < eps:
            return x1
        i += 1
        x0 = x1

def ex1():
    # Subpunctul a)
    X0 = 2.
    EPS = 1e-3
    MAX_ITER = 1000
    x_aprox= newton_raphson(f=func1, df=dfunc1, x0=X0, eps=EPS, max_iter=MAX_ITER)
    print(f'Solutiile aproximate ale ecuatiei x^2-29=0 sunt {np.abs(x_aprox)} si {-np.abs(x_aprox)}')
    print('\n')
    # Subpunctul b)
    plt.figure(0)
    f=func1
    # Discretizarea domeiului
    domeniu = np.linspace(start=-10, stop=10, num=100)
    # Plotarea functiei
    plt.plot(domeniu, f(domeniu),label='f(x)=x^2-29', c='red')
    # Adaugarea solutiilor
    plt.scatter(np.abs(x_aprox), func1(np.abs(x_aprox)), color='black')
    plt.scatter(-np.abs(x_aprox), func1(np.abs(x_aprox)), color='black')
    # Adaugarea axelor
    plt.axhline(0, c='black')
    plt.axvline(0, c='black')
    # Etichetele axelor
    plt.xlabel('Axa OX')
    plt.ylabel('Axa OY')
    # Adaugare grid
    plt.grid()
    # Afisare legenda
    plt.legend()
    # Afisare titlu
    plt.title('Graficul functiei f')
    # Afisare grafic
    plt.show()


# Ex2
# Subpunctul a)
def interp_neville(X, Y, z):
    n = np.shape(X)[0]
    Q = np.full(shape=(n,n), fill_value=np.nan)
    # Pas1
    Q[:,0] = Y
    for i in range(1,n):
        for j in range(1,i+1):
            Q[i][j] = (Q[i][j-1] * (z - X[i-j]) - Q[i-1][j-1] * (z - X[i])) / (X[i] - X[i-j])
    # Pas2
    t = Q[n-1][n-1]
    return t


def func2(x):
    y = np.e ** (2*x)
    return y


def error(val_aprox, val_true):
    return np.abs(val_aprox - val_true)


def ex2():
    # Generare date client si vizualizarea acestora
    X = np.linspace(-1, 1, 24)
    Y = func2(X)
    plt.figure(1)
    plt.scatter(X, Y, label='Date client')
    # Adaugarea axelor
    plt.axhline(0, c='black')
    plt.axvline(0, c='black')
    # Etichetele axelor
    plt.xlabel('points')
    plt.ylabel('values')
    # Adaugare grid
    plt.grid()

    # Subpunctul b)
    # Aproximarea valorilor lipsa
    # Debug
    print("True value for f(2) = ", func2(2))
    aprox_value = interp_neville(X, Y, 2)
    print("Aprox value for f(2) = ", aprox_value)
    # Discretizeaza domeniul
    points_interp = np.linspace(-1, 1, 75)
    # Calculeaza aproximarea in fiecare punct din domeniu
    values_interp = np.zeros(75)
    for i in range(0, 75):
        values_interp[i] = interp_neville(X, Y, points_interp[i])
    print(points_interp)
    print(values_interp)
    # Generare grafic (verificare)
    plt.plot(points_interp, func2(points_interp), 'k', label='Functia exacta')
    plt.plot(points_interp, values_interp, 'r:', label='Aproximare Lagrange met. Neville')
    plt.title('Interpolare Lagrange met. Neville')
    plt.legend()
    plt.show()

    # Subpunctul c)
    # Calculeaza eroarea
    err = error(values_interp, func1(points_interp))

    # Genereaza o figura noua si afiseaza graficul erorii
    plt.figure(2)
    # Adaugarea axelor
    plt.axhline(0, c='black')
    plt.axvline(0, c='black')
    # Etichetele axelor
    plt.xlabel('points')
    plt.ylabel('error')
    # Plotarea erorii
    plt.plot(points_interp, err, 'b:', label='Error')
    # Afisare titlu
    plt.title('Eroarea interpolarii Lagrange metoda Neville')
    # Adaugare grid
    plt.grid()
    # Afisare grafic
    plt.show()


# Ex3
# Subpucntul a)
def fact_qr_new(A):
    # Verifica daca matricea A este patratica
    assert np.shape(A)[0] == np.shape(A)[1], "A nu este matrice patratica!"
    n = np.shape(A)[0]  # Salvam in variabila n dimensiunea matricei
    # Verifica daca A este inversabila
    # Determinant matrice superior triunghiulara = Produs elemente diagonala
    assert np.abs(np.prod(A.diagonal())) > 0, "Matricea A nu este inversabila!"
    Q = np.zeros(n,n)
    R = np.zeros(n,n)
    # Pas1
    for i in range(n):
        R[0][0] += A[i][0]**2
    R[0][0] = np.sqrt(R[0][0])
    for i in range(n):
        Q[i][0] = A[i][0] / R[0][0]
    for j in range(1,n):
        for s in range(n):
            R[0][j] += Q[s][0] * A[s][j]
    # Pas2
    for k in range(1,n):
        sum1 = 0
        sum2 = 0
        for i in range(n):
            sum1 += A[i][k]**2
        for s in range(k-1):
            sum2 += R[s][k]**2
        R[k][k] = np.sqrt((sum1-sum2))
        sum3 = 0
        for i in range(n):
            for s in range(k-1):
                sum3 += Q[i][s] * R[s][k]
            Q[i][k] = 1 / (A[i][k] - sum3)
        for j in range(k+1,n):
            sum4 = 0
            for s in range(n):
                sum4 += Q[s][k] * A[s][k]
            R[k][j] = sum4
    # Pas3
    return Q, R


def ex3():
    # Subpucntul b)
    # Declararea tablourilor
    A = np.array([
        [0, -1, -4, -7],
        [5, 7, 5, -8],
        [0, -1, 5, -8],
        [-5, -3, -4, 5]
    ])
    b = np.array([
        [-54],
        [11],
        [-23],
        [-10]
    ])
    pass


if __name__ == '__main__':
        ex1()
        ex2()
        ex3()