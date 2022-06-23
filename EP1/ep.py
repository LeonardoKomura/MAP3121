import matplotlib.pyplot as plt
import numpy as np
import math
import sys

np.set_printoptions(threshold=sys.maxsize) #Assistencia para o print de matrizes
np.set_printoptions(formatter={'all': lambda x: " {:.6f} ".format(x)})

# Rotacao de Givens
def givens(alpha, beta, gama, i, j, n):
    novoAlpha = alpha.copy()
    novoBeta = beta.copy()
    novoGama = gama.copy()
    ak = alpha[i-1]
    bk = beta[j-2]

    if (abs(ak) > abs(bk)):
        t = -bk/ak
        ck = 1/math.sqrt(1+(t**2))
        sk = ck*t
    else:
        t = -ak/bk
        sk = 1/math.sqrt(1+(t**2))
        ck = sk*t

    novoAlpha[i-1] = (ck*alpha[i-1]) - (sk*beta[j-2])
    novoGama[i-1] = (ck*gama[i-1]) - (sk*alpha[j-1])
    novoBeta[j-2] = (sk*alpha[i-1]) + (ck*beta[j-2])
    novoAlpha[j-1] = (sk*gama[i-1]) + (ck*alpha[j-1])
    if (j != n):
        novoGama[j-1] = (ck*gama[j-1])
    return novoAlpha, novoBeta, novoGama, ck, sk

# Funcao que calcula o R*Qt
def assist(ck, sk, alpha, beta, gama, i):
    novoAlpha = alpha.copy()
    novoBeta = beta.copy()
    
    novoAlpha[i-1] = (ck*alpha[i-1]) - (sk*gama[i-1])
    novoBeta[i-1] = -sk*alpha[i]
    novoAlpha[i] = ck*alpha[i]
    return novoAlpha, novoBeta

# Funca que calcula os autovetores (VQt)
def autovetor(V, ck, sk, i, n):
    ident = np.eye(n)
    Q = ident.copy()
    for k in range(n):
        Q[i][k] = (ck*ident[i][k]) - (sk*ident[i+1][k])
        Q[i+1][k] = (sk*ident[i][k]) + (ck*ident[i+1][k])
    
    Qt = np.transpose(Q)
    VQt = np.matmul(V, Qt)
    return VQt

# Algoritmo QR
def QR(a, b, g, n, erro):
    c = np.zeros(n)
    s = np.zeros(n)
    k = 0
    alphaR = a.copy() 
    betaR = b.copy()
    gamaR = g.copy()
    V = np.eye(n)
    VQ = np.eye(n)
    for m in range(n, 1, -1):
        while(abs(betaR[m-2]) > erro):   
            for i in range(m-1):
                alpha_R, beta_R, gama_R, c[i], s[i] = givens(alphaR, betaR, gamaR, i+1, i+2, n)
                alphaR = alpha_R.copy()
                betaR = beta_R.copy()
                gamaR = gama_R.copy()  

            alphaRQ = alphaR.copy()
            betaRQ = betaR.copy()
            gamaRQ = gamaR.copy()
            for i in range(m-1):
                alpha_RQ, beta_RQ = assist(c[i], s[i], alphaRQ, betaRQ, gamaRQ, i+1)
                alphaRQ = alpha_RQ.copy()
                betaRQ = beta_RQ.copy()
            gamaRQ = betaRQ.copy()

            alphaR = alphaRQ.copy()
            betaR = betaRQ.copy()
            gamaR = gamaRQ.copy() 

            for i in range(n-1):
                VQ = autovetor(V, c[i], s[i], i, n)
                V = VQ.copy()

            k = k + 1
    return VQ, alphaR, k

# Algoritmo QR com deslocamento espectral
def QR_deslocamento(a, b, g, n, erro):
    c = np.zeros(n)
    s = np.zeros(n)
    k = 0
    alphaR = a.copy() 
    betaR = b.copy()
    gamaR = g.copy()
    alphaRQ = a.copy()
    betaRQ = b.copy()
    gamaRQ = g.copy()
    V = np.eye(n)
    VQ = np.eye(n)
    mi_k = np.zeros(n)
    oldAlpha = a.copy()
    oldBeta = b.copy()
    for m in range(n, 1, -1):    
        while(abs(betaR[m-2]) > erro):    
            if (k > 0):    
                for j in range(m):
                    dk = (oldAlpha[m-1] - alphaRQ[m-1])/2
                    if (dk >= 0):
                        mi_k[j] = alphaRQ[m-1] + dk - np.sqrt((dk**2) + oldBeta[m-2]**2)
                    else:
                        mi_k[j] = alphaRQ[m-1] + dk + np.sqrt((dk**2) + oldBeta[m-2]**2)

            oldAlpha = alphaRQ.copy()
            oldBeta = betaRQ.copy()
            alphaR = np.subtract(alphaRQ, mi_k)

            for i in range(m-1): 
                alpha_R, beta_R, gama_R, c[i], s[i] = givens(alphaR, betaR, gamaR, i+1, i+2, n)
                alphaR = alpha_R.copy()
                betaR = beta_R.copy()
                gamaR = gama_R.copy()  

            alphaRQ = alphaR.copy()
            betaRQ = betaR.copy()
            gamaRQ = gamaR.copy()
            for i in range(m-1):
                alpha_RQ, beta_RQ = assist(c[i], s[i], alphaRQ, betaRQ, gamaRQ, i+1)
                alphaRQ = alpha_RQ.copy()
                betaRQ = beta_RQ.copy()
            gamaRQ = betaRQ.copy()
            
            alphaRQ = np.add(alpha_RQ, mi_k) 

            for i in range(n-1):
                VQ = autovetor(V, c[i], s[i], i, n)
                V = VQ.copy()
            
            betaR = betaRQ.copy()
            gamaR = gamaRQ.copy()

            k = k + 1
        
    return VQ, alphaRQ, k

# Interface de usuario
print("\n\n")
print("   ===============================================================================================")
print("                         EXERCÍCIO PROGRAMA 1 - MÉTODOS NUMÉRICOS (MAP3121)                       ")
print("   ===============================================================================================")
print("   André Lucas Pierote Rodrigues Vasconcelos - NUSP: 11356540 - Engenharia Elétrica - Turma: 01")
print("   Leonardo Isao Komura                      - NUSP: 11261656 - Engenharia Elétrica - Turma: 03")
print("   -----------------------------------------------------------------------------------------------")
print("   Qual operacao voce deseja realizar?\n")
print("   1. Teste do algoritmo QR (com e sem deslocamento espectral)")
print("   2. Resolucao de um sistema massa-mola\n")
oper = int(input("   Operacao = "))
print("   -----------------------------------------------------------------------------------------------")
print("\n")

if oper == 1:
    print("   TESTE DO ALGORITMO\n")
    n = int(input("   Digite o tamanho da matriz (nxn): "))
    Alpha = np.zeros(n)
    Beta = np.zeros(n-1)
    Gama = np.zeros(n-1)
    equal = input("   Os valores dos elementos de cada diagonal sao iguais (p.e. alpha_i = alpha_i+1)? (s/n): ")
    if equal == "s":
        x = input("   Digite o valor dos elementos da diagonal principal: ")
        y = input("   Digite o valor dos da diagonal secundaria: ")
        for i in range(n):
            Alpha[i] = x
        for i in range(n-1):
            Beta[i] = y
    else:
        print("   Digite os valores da diagonal principal:")
        for i in range(n):
            Alpha[i] = input("   Elemento %d da diagonal principal: " %(i+1))
        print("\n   Digite os valores das diagonais secundarias:")
        for i in range(n-1):
            Beta[i] = input("   Elemento %d da diagonal secundaria: " %(i+1))
    print("   Diagonal principal (Alphas): ")
    with np.printoptions(precision=3, suppress=True):
        print("   ", Alpha)
    print("   Diagonais secundarias (Betas e Gamas): ")
    with np.printoptions(precision=3, suppress=True):
        print("   ", Beta)

    Gama = Beta.copy()

    erro = int(input("\n   Digite o erro desejado: 10^(-(valor a ser digitado)): "))
    epsilon = 10**(-erro)

    V, alphas, k = QR(Alpha, Beta, Gama, n, epsilon)
    print("\n   Sem deslocamento: ")
    print("   Autovetor = ")
    print(np.matrix(V))
    print("   Autovalor = ", alphas)
    print("   Numero de iteracoes: ", k)

    V, alphas, k = QR_deslocamento(Alpha, Beta, Gama, n, epsilon)
    print("\n   Com deslocamento: ")
    print("   Autovetor = ")
    print(np.matrix(V))
    print("   Autovalor = ", alphas)
    print("   Numero de iteracoes: ", k)

    # Resposta teorica
    AutoVetores = np.zeros((n, n))
    AutoValores = np.zeros(n)
    for i in range(n):
        AutoValores[i] = 2*(1-math.cos((i+1)*math.pi/(n+1)))
        for j in range(n):
            AutoVetores[j][i] =  math.sin((i+1)*(j+1)*math.pi/(n+1))

    print("\nAutovetores teoricos = ")
    print(np.matrix(AutoVetores))
    print("Autovalores teoricos = ", AutoValores)

else:
    print("   SISTEMA MASSA-MOLA")
    print("   Voce deseja utilizar 5 ou 10 massas: ")
    print("   1. 5 massas")
    print("   2. 10 massas")
    opt = int(input("   Opcao (1/2): "))
    if opt == 1:
        n = 5
    else:
        n = 10
    m = float(input("   Digite a massa das massas (kg): "))

    erro = int(input("\n   Digite o erro desejado: 10^(-(valor a ser digitado)): "))
    epsilon = 10**(-erro)

    print("\n   Qual constante elastica voce deseja utilizar: ")
    print("    1. k(i) = 40 + 2i")
    print("    2. k(i) = 40 + 2*(-1^i)")
    cte = int(input("   Constante (1/2): "))

    #Calculo das constantes elasticas
    k = np.zeros(n+1)
    if cte==1:
        for i in range(n+1):
            k[i] = (40 + 2*(i+1))
    else:
        for i in range(n+1):
            k[i] = (40 + 2*(-1**(i+1)))
    
    #Calculo da matriz A
    Alpha = np.zeros(n)
    Beta = np.zeros(n-1)
    Gama = np.zeros(n-1)
    for i in range(n):
        Alpha[i] = (k[i] + k[i+1])/m
    for i in range(n-1):
        Beta[i] = -k[i+2]/m
    Gama = Beta.copy()

    #Calculo da matriz Qt e dos autovalores
    Lambda = np.zeros(n)
    Qt, Lambda, iter = QR_deslocamento(Alpha, Beta, Gama, n, epsilon) 

    print("\n   Voce deseja inserir as posicoes iniciais?")
    maxfreq = input("   (Caso nao deseje, elas serao aquelas que geram maior frequencia) (s/n): ")
    
    X_init = np.zeros(n)
    Y_init = np.zeros(n)
    if maxfreq == "s":
        print("\n   Digite as posicoes iniciais de cada massa: ")
        for i in range(5):
            X_init[i] = input("   Posicao X_0(%d) = " %(i+1))
        if n==10:
            for i in range(5, n, 1):
                X_init[i] = X_init[i-5]
    #Obtencao do autovetor relacionado ao maior autovalor
    else:
        maxvalue=max(Lambda)
        minvalue=min(Lambda)
        if abs(maxvalue) > abs(minvalue):
            maximo = maxvalue
        else:
            maximo = minvalue
        pos = np.where(Lambda == maximo)

        for i in range(n):
            X_init[i] = Qt[pos, i]

    Xinit = np.transpose(np.array([X_init]))    #Transpondo X(0)

    Y_init = np.matmul(Qt, Xinit)               #Calculando Y(0)
    ay = Y_init.copy()                          #Calculando os coeficientes que multiplicam os cossenos

    freqs = np.zeros(n)
    for i in range(n):
        freqs[i] = np.sqrt(Lambda[i])           #Calculando as frequencias dos cossenos

    t = int(input("   Digite o tempo de simulacao do sistema (em segundos): t = "))

    X = np.zeros((n, 40*t))
    
    Q = np.transpose(Qt)

    #Calculo da matriz X(t)
    for i in range(n):
        for q in range(40*t):
            for j in range(n):
                X[i][q] = X[i][q] + (Q[i][j]*ay[j]*math.cos(freqs[j]*(q/40)))

    #Coordenadas da abcissa 
    tempo = np.zeros(40*t)
    for i in range(40*t):
        tempo[i] = i*0.025
    
    #Construcao do grafico unico
    plt.figure(1)
    x_plot = np.zeros(40*t)
    for i in range(n):
        for j in range(40*t):
            x_plot[j] = X[i,j]
        lbl = 'Massa' + str(i + 1)
        plt.plot(tempo, x_plot, label=lbl) 
    
    plt.xlabel('Tempo(s)')
    plt.ylabel('Posicao(m)')
    plt.title("Posicao da massa em funcao do tempo")
    plt.legend()

    #Construcao dos graficos separados
    if n==5:
        x_plot1 = np.zeros(40*t)
        x_plot2 = np.zeros(40*t)
        x_plot3 = np.zeros(40*t)
        x_plot4 = np.zeros(40*t)
        x_plot5 = np.zeros(40*t)
        for j in range(40*t):
            x_plot1[j] = X[0,j]
        for j in range(40*t):
            x_plot2[j] = X[1,j] 
        for j in range(40*t):
            x_plot3[j] = X[2,j] 
        for j in range(40*t):
            x_plot4[j] = X[3,j] 
        for j in range(40*t):
            x_plot5[j] = X[4,j]    
    
        plt.figure(2)
        
        plt.subplot(231)
        plt.plot(tempo, x_plot1, color='b', label='Massa 1')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        plt.title("Posicao da massa 1 em funcao do tempo")
        
        plt.subplot(232)
        plt.plot(tempo, x_plot2, color='g', label='Massa 2')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        plt.title("Posicao da massa 2 em funcao do tempo")
        
        plt.subplot(233)
        plt.plot(tempo, x_plot3, color='r', label='Massa 3')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        plt.title("Posicao da massa 3 em funcao do tempo")
        
        plt.subplot(234)
        plt.plot(tempo, x_plot4, color='c', label='Massa 4')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        plt.title("Posicao da massa 4 em funcao do tempo")
        
        plt.subplot(235)
        plt.plot(tempo, x_plot5, color='m', label='Massa 5')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        plt.title("Posicao da massa 5 em funcao do tempo")
        
        plt.show()

    if n==10:
        x_plot1 = np.zeros(40*t)
        x_plot2 = np.zeros(40*t)
        x_plot3 = np.zeros(40*t)
        x_plot4 = np.zeros(40*t)
        x_plot5 = np.zeros(40*t)
        x_plot6 = np.zeros(40*t)
        x_plot7 = np.zeros(40*t)
        x_plot8 = np.zeros(40*t)
        x_plot9 = np.zeros(40*t)
        x_plot10 = np.zeros(40*t)
        for j in range(40*t):
            x_plot1[j] = X[0,j]
        for j in range(40*t):
            x_plot2[j] = X[1,j] 
        for j in range(40*t):
            x_plot3[j] = X[2,j] 
        for j in range(40*t):
            x_plot4[j] = X[3,j] 
        for j in range(40*t):
            x_plot5[j] = X[4,j]
        for j in range(40*t):
            x_plot6[j] = X[5,j]   
        for j in range(40*t):
            x_plot7[j] = X[6,j]   
        for j in range(40*t):
            x_plot8[j] = X[7,j]   
        for j in range(40*t):
            x_plot9[j] = X[8,j]   
        for j in range(40*t):
            x_plot10[j] = X[9,j]       
        
        plt.figure(2)
        
        plt.subplot(431)
        plt.plot(tempo, x_plot1, color='b', label='Massa 1')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        
        plt.subplot(432)
        plt.plot(tempo, x_plot2, color='g', label='Massa 2')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        
        plt.subplot(433)
        plt.plot(tempo, x_plot3, color='r', label='Massa 3')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        
        plt.subplot(434)
        plt.plot(tempo, x_plot4, color='c', label='Massa 4')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        
        plt.subplot(435)
        plt.plot(tempo, x_plot5, color='m', label='Massa 5')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        
        plt.subplot(436)
        plt.plot(tempo, x_plot6, color='y', label='Massa 6')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        
        plt.subplot(437)
        plt.plot(tempo, x_plot7, color='k', label='Massa 7')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        
        plt.subplot(438)
        plt.plot(tempo, x_plot8, color='tab:gray', label='Massa 8')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        
        plt.subplot(439)
        plt.plot(tempo, x_plot9, color='salmon', label='Massa 9')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
        
        plt.subplot(4,3,10)
        plt.plot(tempo, x_plot10, color='gold', label='Massa 10')
        plt.xlabel('Tempo(s)')
        plt.ylabel('Posicao(m)')
    
        plt.show()