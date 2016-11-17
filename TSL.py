# coding: utf-8

# Teoria Secular Linear
# =====================
#
# O objetivo deste programa é aplicar a Teoria Secular Linear a um
# planetario

# Definições
#      m: Massa
#      a: Semieixo maior
#      e: Excentricidade
#      I: Inclinação
#  capom: Longitude do nodo ascendente
#  omega: Argumento do periélio
#      M: Anomalia média
#  varpi: Capom + omega, Longitude do periélio
# lambda: M + varpi, Longitude média

# ## Procedimento para obter as autofrequências e os modos próprios
# * Calcular alpha_{ij} para todos os corpos envolvidos
# * Calcular os coeficientes de Laplace
# * Calcular os coeficientes das matrizes A e B
# * Calcular os autovetores e autovalores das matrizes A e B
# * Calcular os autovetores G_{ij} e S_{ij}

# %% Importando módulos
import sys
import os.path
import numpy as np
from scipy.integrate import quad

# %% Definições de funções
# Função alpha
def al(a1, a2):
    """ Função que calcula a razão entre o mínimo e o máximo de dois semieixos
        maiores

        Input: a1, a2: dois semieixo maiores

        Output: min(a1, a2) / max(a1, a2)
    """
    return min(a1, a2) / max(a1, a2)

# %% Coeficientes de Laplace $b_{\psi}^{k}(\alpha_{i,j})$
def coeff_laplace(K, al):
    """ Função que calcula os coeficientes de Laplace"""
    f = lambda x: (1 - 2 * al * np.cos(x) + al**2)**(-3/2) * np.cos(K * x)
    coeff = quad(f, 0, 2*np.pi)
    return (1.0 / np.pi) * coeff[0]

# %% Definição das funções parciais para o cálculo das matrizes
def parte1(i, G, n, a):
    """ Função que calcula a equação G/(4 * n[i] * a[i]**2)
        como parte da determinação da matrizes A e B
    """
    return G/(4 * n[i] * a[i]**2)

def parte2(i, j, m, a):
    """ Função que calcula a equação (m[j] * al(a[i], a[j])) / max(a[i], a[j])
        como parte da determinação da matrizes A e B
    """
    return (m[j] * al(a[i], a[j])) / max(a[i], a[j])

def parte3(K, i, j, a):
    """ Função que calcula a equação coeff_laplace(K, al(a[i], a[j]))
        como parte da determinação da matrizes A e B
    """
    return coeff_laplace(K, al(a[i], a[j]))

def FreqProp(G, a, m, n):
    """
       Função que calcula as frequências próprias
       
       Input
           G: Constante de gravitação [SS AU^3 d^-2 M_sol^-1]
           a: Semieixo maior
           m: Massa [massas solares]
           n: Movimentos médios
           
       Output: 
           Autovalores e autovetores
    """
    # %% Determinando a matriz A e B
    # Matriz A
        # Comprimento do vetor de semieixos.
    len_a = len(a)
    
    A = np.zeros((len_a, len_a))
    for i in range(0, len_a):
        for j in range(0,len_a):
             if j != i:
                A[i, j] = -parte1(i, G, n, a) * parte2(i, j, m, a) * \
                    parte3(2, i, j, a)
             else:
                parcial = 0
                for l in range(0, len_a):
                    if l != i:
                        parcial = parcial + parte2(i, l, m, a) * \
                            parte3(1, i, l, a)
                A[i, i] = parte1(i, G, n, a) * parcial

    # Matriz B
    B = np.zeros((len_a, len_a))
    for i in range(0, len_a):
        for j in range(0,len_a):
            if j != i:
                B[i, j] =  parte1(i, G, n, a) * parte2(i, j, m, a) * \
                    parte3(1, i, j, a)
            else:
                B[i, i] = -A[i, i]

    # Convertendo a matriz A para unidades de deg / yr
    A = A * (180/np.pi) * 365.25

    # Convertendo a matriz B para unidades de deg / yr
    B = B * (180/np.pi) * 365.25

    # Obtendo os autovetores e autovalres de A
    A_eigenValues, A_eigenVectors = np.linalg.eig(A)

    # Obtendo os autovetores e autovalres de B
    B_eigenValues, B_eigenVectors = np.linalg.eig(B)

    return(A_eigenValues, A_eigenVectors, B_eigenValues, B_eigenVectors)
    
# %% Função main - programa principal
def main():
    # Constante da gravitação universal
    G = (0.01720209895)**2 # para o SS AU^3 d^-2 M_sol^-1

    # Massa do Sol
    M = 1.0 # Massa 'unitária' para o Sol

    # Leitura dos dados
    #input_file = sys.argv[1]
    input_file = "planetas.txt"
    
    # Lendo nome dos planetas
    planetas = np.genfromtxt(input_file, dtype='str', usecols=0,\
        skip_header=10)
    
    # Lendo dados numéricos
    pl = np.genfromtxt(input_file, dtype='float',\
        usecols=(1, 2, 3, 4, 5), skip_header=10)

    # Semieixo maior
    a = pl[:,1] # unidade AU

    # Massa
    m = pl[:,0] # Considerando massa unitária para o Sol

    # Dados dos movimentos médios n calculado
    #n = (G * (M + m)/a**3)**(1/2) # unidade rad/day

    # Dados dos movimentos médios n lidos
    n = pl[:,4] # deg/yr
    n = (n * np.pi / 180) / 365.25 # unidade rad/day

    # Excentricidade
    e = pl[:,2]

    # Inclinação
    inc = pl[:,3] # unidade deg
    
    # Cálculo dos autovalores e autovetores
    Aval, Avec, Bval, Bvec = FreqProp (G, a, m, n)
    
    print('Auto frequências [deg/yr]')
    print(Aval)
    print(Bval)


# %% Executando a função main
if __name__ == '__main__':
    main()
    
#    # Salvando no arquivo
#    output_file = 'resultados-' + str(input_file)
#
#    f = open(output_file, 'w+')
#
#    f.write('Matriz A associada a Excentricidade em [deg/yr]\n')
#    f.write(str(A) + '\n')
#    f.write('\nAutovalores de A\n')
#    f.write(str(A_eigenValues) + '\n')
#    f.write('\nAutovetores de A\n')
#    f.write(str(A_eigenVectors) + '\n')
#
#    f.write('\nMatriz B associada a Inclinação em [deg/yr]\n')
#    f.write(str(B) + '\n')
#    f.write('\nAutovalores de B\n')
#    f.write(str(B_eigenValues) + '\n')
#    f.write('\nAutovetores de B\n')
#    f.write(str(B_eigenVectors) + '\n')
#
#
#    f.close()
#
#    # Informação ao usuário
#    os.system('clear')
#    print("TSL - Teoria Secular Linear")
#    print("---------------------------")
#
#    try:
#        os.path.isfile(output_file)
#        print()
#        print('O resultado foi gravado no arquivo', output_file)
#        print()
#    except:
#        print('O arquivo não foi criado')

