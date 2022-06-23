import matplotlib.pyplot as plt
import numpy as np
import math
import sys

np.set_printoptions(threshold=10) #Assistencia para o print de matrizes
np.set_printoptions(formatter={'all': lambda x: " {:.5f} ".format(x)})

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

# Algoritmo QR com deslocamento espectral
def QR_deslocamento(a, b, g, n, erro, Vinput):
    c = np.zeros(n)
    s = np.zeros(n)
    k = 0
    alphaR = a.copy() 
    betaR = b.copy()
    gamaR = g.copy()
    alphaRQ = a.copy()
    betaRQ = b.copy()
    gamaRQ = g.copy()
    V = Vinput.copy()
    VQ = V.copy()
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

            for i in range(m-1):
                VQ = autovetor(V, c[i], s[i], i, n)
                V = VQ.copy()
            
            betaR = betaRQ.copy()
            gamaR = gamaRQ.copy()

            k = k + 1
    return VQ, alphaRQ, k

# Transformacao de Householder
def Householder(n, matrix):
    A = matrix.copy()
    Ht = np.eye(n)
    for i in range(n-2):
        HtFinal = Ht.copy()
        Nova = A.copy()
        a = np.zeros(n)
        for j in range(n-1-i):
            a[j+1+i] = Nova[j+1+i][i]
        
        #print("a = ", a)

        alpha = np.zeros(n-1-i)
        for j in range(n-1-i):
            alpha[j] = a[j+1+i]

        #print("alpha = ",alpha)

        norm2 = np.dot(a,a)
        norm = math.sqrt(norm2)

        e = np.zeros(n)
        e[i+1] = norm
        
        #print("e = ", e)

        w = np.zeros(n-i)
        if A[i+1][i] > 0:
            w = np.add(np.array(a),np.array(e))
        else:
            w = np.subtract(np.array(a),np.array(e))
        
        #print("w = ", w)

        w_barra = np.zeros(n-1-i)
        for j in range(n-1-i):
            w_barra[j] = w[j+1+i]
        #print("wbarra = ",w_barra)

        Hwa = np.zeros(n-1-i)
        Hwa = np.subtract(alpha, w_barra)

        for k in range(n-1-i):
            Nova[k+1+i][i] = Hwa[k]
        for j in range(n-1-i):
            Nova[i][j+1+i] = Nova[j+1+i][i]
        
        #HwA - Submatriz
        col = np.zeros(n-1-i)
        for j in range(n-1-i):
            for k in range(n-1-i):
                col[k] = A[k+1+i][j+1+i]
            for k in range(n-1-i):
                Nova[k+1+i][j+1+i] = A[k+1+i][j+1+i] - (2*np.dot(w_barra,col)/np.dot(w_barra,w_barra))*w_barra[k]

        A = Nova.copy()

        #HwAHw - Submatriz
        lin = np.zeros(n-1-i)
        for j in range(n-1-i):
            for k in range(n-1-i):
                lin[k] = A[j+1+i][k+1+i]
            for k in range(n-1-i):
                Nova[j+1+i][k+1+i] = A[j+1+i][k+1+i] - (2*np.dot(w_barra,lin)/np.dot(w_barra,w_barra))*w_barra[k]

        #Ht = IHw1Hw2...
        lin = np.zeros(n-1-i)
        for j in range(n-1):
            for k in range(n-1-i):
                lin[k] = Ht[j+1][k+1+i]
            for k in range(n-1-i):
                HtFinal[j+1][k+1+i] = Ht[j+1][k+1+i] - (2*np.dot(w_barra,lin)/np.dot(w_barra,w_barra))*w_barra[k]

        Ht = HtFinal.copy()
        A = Nova.copy()
        #print("Ht %d: "%(i+1))
        #print(Ht)

    return Nova, HtFinal


# Interface de usuario
print("\n\n")
print("   ===============================================================================================")
print("                         EXERCÍCIO PROGRAMA 2 - MÉTODOS NUMÉRICOS (MAP3121)                       ")
print("   ===============================================================================================")
print("   André Lucas Pierote Rodrigues Vasconcelos - NUSP: 11356540 - Engenharia Elétrica - Turma: 01")
print("   Leonardo Isao Komura                      - NUSP: 11261656 - Engenharia Elétrica - Turma: 03")
print("   -----------------------------------------------------------------------------------------------")

print("   Qual operacao voce deseja realizar?\n")
print("   1. Teste da Transformacao de Householder (exercicios a) e b))")
print("   2. Resolucao do sistema de trelicas planas (exercicio c))\n")
oper = int(input("   Operacao = "))
print("   -----------------------------------------------------------------------------------------------")
print("\n")

if oper==1:
    print("   TESTE DA TRANSFORMACAO DE HOUSEHOLDER")
    print("   Voce deseja escrever a matriz ou carregar de um dos inputs (a ou b)?\n")
    print("   1. Escrever a matriz termo a termo")
    print("   2. Carregar um dos inputs\n")
    option = int(input("   Opcao = "))
    print("\n")
    print("   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("\n")

    if option==1:
        n = int(input("   Digite o tamanho da matriz (nxn): n = "))
        ident = np.eye(n)
        A = np.eye(n)
        print("\n")
        for i in range(n):
            for j in range(n):
                A[i][j] = float(input("   Digite o valor da linha %d e coluna %d: " %(i+1,j+1)))
    
    else:
        print("   Escolha qual arquivo você deseja carregar: \n")
        print("   1. input-a (+ exercicio a))")
        print("   2. input-b (+ exercicio b))\n")
        arquivo = int(input("   Carregar arquivo: "))
        if arquivo==1:
            with open('input-a', 'r') as f:
                l = [[float(num) for num in line.split()] for line in f]

            pos1 = l[0]
            n = int(pos1[0])
            A = np.eye(n)

            for i in range(n):
                vetor = l[i+1]
                for j in range(n):
                    A[i][j] = vetor[j]

        else:     
            with open('input-b', 'r') as f:
                l = [[float(num) for num in line.split()] for line in f]

            pos1 = l[0]
            n = int(pos1[0])
            A = np.eye(n)

            for i in range(n):
                vetor = l[i+1]
                for j in range(n):
                    A[i][j] = vetor[j]
    
    ident = np.eye(n)
    
    print("\n")
    print("   Matriz A:")
    print(np.matrix(A))
    print("\n")

    # Aplicacao de Householder
    print("   Aplicacao de Householder: ")
    T, Ht = Householder(n, A)
    print("   Matriz T = H*A*Ht: ")
    print(T)
    print("\n")
    print("   Matriz Ht: ")
    print(Ht)

    # Aplicacao do algoritmo QR com deslocamento espectral
    a = np.zeros(n)
    b = np.zeros(n-1)
    for i in range(n):
        a[i] = T[i][i]
    for i in range(n-1):
        b[i] = T[i+1][i]
    g = b.copy()

    V, Autovalores, k = QR_deslocamento(a, b, g, n, 10**(-6), Ht)

    print("\n")
    print("   Aplicacao do algoritmo QR na matriz T: ")
    print("   Autovetores de T: ")
    print(V)
    print("\n")

    print("   Autovalores de T: ", Autovalores)

    print("\n")
    print("   Matriz Ht*V (autovetores ortogonais de A): ")
    HtV = np.matmul(Ht,V)
    print(HtV)

    Lambda = np.eye(n)
    for i in range(n):
        Lambda[i][i] = Autovalores[i]

    Vlambda = np.matmul(V, Lambda)
    T_QR = np.matmul(Vlambda, np.transpose(V))
    print("\n")
    print("   Forma diagonal semelhante: ")
    print("   Matriz V*Lambda*Vt: ")
    print(T_QR)

    if arquivo==1:
        print("\n")
        print("   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
        print("   Restante da resolucao do exercicio a): ")
        print("\n")
        print("   Autovalores esperados: (7, 2, -1 e -2)")
        print("   Autovalores obtidos pela matriz T (H*A*Ht): ", Autovalores)
        print("\n")
        print("   Verificacao se A*v = Lambda*v: ")
        for i in range(n):
            vtrans = np.zeros(n)
            for j in range(n):
                vtrans[j] = V[j][i]
            vetor = np.transpose(vtrans)
            valor = Autovalores[i]
            Av = np.matmul(A, vetor)
            LambdaV = valor*np.array(vetor)
            print("   v = ", vtrans)
            print("   Lambda = ", valor)
            print("   A*v    =   ", np.transpose(Av))
            print("   Lambda*v = ", np.transpose(LambdaV))
            print("\n")
        print("   Verificacao se V é ortogonal (V * Vt = I) (erro absoluto de 1e-5): ")
        I = np.matmul(V, np.transpose(V))
        print("   Matriz V*Vt: ")
        print(I)
        check = True
        for i in range(n):
            for j in range(n):
                check = check * math.isclose(I[i][j], ident[i][j], abs_tol=1e-5)
        if(check == True):
            print("   Ela é ortogonal")
        else:
            print("   Ela não é ortogonal")
    
    else:
        print("\n")
        print("   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
        print("   Restante da resolucao do exercicio b): ")
        print("\n")
        valores = np.zeros(n)
        for i in range(n):
            valores[i] = 0.5/(1-math.cos(((2*(i+1)-1)*math.pi)/((2*n)+1)))
        print("   Autovalores esperados: ", valores)
        print("   Autovalores obtidos: ", Autovalores)
        print("\n")
        print("   Verificacao se A*v = Lambda*v: ")
        for i in range(n):
            vtrans = np.zeros(n)
            for j in range(n):
                vtrans[j] = V[j][i]
            vetor = np.transpose(vtrans)
            valor = Autovalores[i]
            Av = np.matmul(A, vetor)
            LambdaV = valor*np.array(vetor)
            print("   v = ", vtrans)
            print("   Lambda = ", valor)
            print("   A*v    =   ", np.transpose(Av))
            print("   Lambda*v = ", np.transpose(LambdaV))
            print("\n")
        print("   Verificacao se V é ortogonal (V * Vt = I) (erro absoluto de 1e-5): ")
        I = np.matmul(V, np.transpose(V))
        print("   Matriz V*Vt: ")
        print(I)
        check = True
        for i in range(n):
            for j in range(n):
                check = check * math.isclose(I[i][j], ident[i][j], abs_tol=1e-5)
        if(check == True):
            print("   Ela é ortogonal")
        else:
            print("   Ela não é ortogonal")

        

else:
    print("   APLICACAO PARA TRELIÇAS PLANAS")
    print("   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    # Leitura do arquivo de input
    with open('input-c', 'r') as f:
        l = [[float(num) for num in line.split()] for line in f]

    linha1 = l[0]
    linha2 = l[1]
    linha3 = l[2]

    n_nos = int(linha1[0])
    n_nfixos = int(linha1[1])
    n_barras = int(linha1[2])
    rho = float(linha2[0])
    secao = float(linha2[1])
    E = float(linha2[2])*(10**9)

    theta = np.zeros((14, 14))
    comprimento = np.zeros((14, 14))

    print("   Número de nós: ", n_nos)
    print("   Número de nós não fixos: ", n_nfixos)
    print("   Número de barras: ", n_barras)
    print("   Densidade de massa (rho): ", rho)
    print("   Seção transversal (A): ", secao)
    print("   Módulo de elasticidade (E): ", E)
    print("\n")

    for k in range(n_barras):
        linha = l[k+2]
        i = int(linha[0])
        j = int(linha[1])
        theta[i-1][j-1] = linha[2]
        theta[j-1][i-1] = linha[2]
        comprimento[i-1][j-1] = linha[3]
        comprimento[j-1][i-1] = linha[3]

    # Definindo a matriz de rigidez K (excluindo as barras conectadas aos nos 13 e 14)
    n = 2*n_nfixos
    Kij = np.zeros((4, 4))
    K = np.zeros((n, n))
    for i in range(n_nfixos):
        for j in range(i):
            if (comprimento[i][j] != 0):
                Kij[0][0] = (secao*E/comprimento[i][j]) * (math.cos(np.radians(theta[i][j])))**2
                Kij[1][1] = (secao*E/comprimento[i][j]) * (math.sin(np.radians(theta[i][j])))**2
                Kij[2][2] = (secao*E/comprimento[i][j]) * (math.cos(np.radians(theta[i][j])))**2
                Kij[3][3] = (secao*E/comprimento[i][j]) * (math.sin(np.radians(theta[i][j])))**2
                Kij[1][0] = (secao*E/comprimento[i][j]) * math.cos(np.radians(theta[i][j]))*math.sin(np.radians(theta[i][j]))
                Kij[0][1] = Kij[1][0]
                Kij[2][0] = (secao*E/comprimento[i][j]) * -(math.cos(np.radians(theta[i][j])))**2
                Kij[0][2] = Kij[2][0]
                Kij[2][1] = (secao*E/comprimento[i][j]) * -(math.cos(np.radians(theta[i][j])))*math.sin(np.radians(theta[i][j]))
                Kij[1][2] = Kij[2][1]
                Kij[3][0] = (secao*E/comprimento[i][j]) * -(math.cos(np.radians(theta[i][j])))*math.sin(np.radians(theta[i][j]))
                Kij[0][3] = Kij[3][0]
                Kij[3][1] = (secao*E/comprimento[i][j]) * -(math.sin(np.radians(theta[i][j])))**2
                Kij[1][3] = Kij[3][1]
                Kij[3][2] = (secao*E/comprimento[i][j]) * math.cos(np.radians(theta[i][j]))*math.sin(np.radians(theta[i][j]))
                Kij[2][3] = Kij[3][2]
            
                K[(2*(i+1))-2][(2*(i+1))-2] += Kij[0][0]
                K[(2*(i+1))-2][2*(i+1)-1]   += Kij[0][1]
                K[(2*(i+1))-2][(2*(j+1))-2] += Kij[0][2]
                K[(2*(i+1))-2][2*(j+1)-1]   += Kij[0][3]
                K[2*(i+1)-1][(2*(i+1))-2]   += Kij[1][0]
                K[2*(i+1)-1][2*(i+1)-1]     += Kij[1][1]
                K[2*(i+1)-1][(2*(j+1))-2]   += Kij[1][2]
                K[2*(i+1)-1][2*(j+1)-1]     += Kij[1][3]
                K[(2*(j+1))-2][(2*(i+1))-2] += Kij[2][0]
                K[(2*(j+1))-2][2*(i+1)-1]   += Kij[2][1]
                K[(2*(j+1))-2][(2*(j+1))-2] += Kij[2][2]
                K[(2*(j+1))-2][2*(j+1)-1]   += Kij[2][3]
                K[2*(j+1)-1][(2*(i+1))-2]   += Kij[3][0]
                K[2*(j+1)-1][2*(i+1)-1]     += Kij[3][1]
                K[2*(j+1)-1][(2*(j+1))-2]   += Kij[3][2]
                K[2*(j+1)-1][2*(j+1)-1]     += Kij[3][3]
                
    # Adicionando a contribuição das barras {11,14} e {12, 13}
    for i in range(n_nfixos):
        for j in range(12,14,1):
            if (comprimento[i][j] != 0):
                Kij[0][0] = (secao*E/comprimento[i][j]) * (math.cos(np.radians(theta[i][j])))**2
                Kij[1][1] = (secao*E/comprimento[i][j]) * (math.sin(np.radians(theta[i][j])))**2
                Kij[2][2] = (secao*E/comprimento[i][j]) * (math.cos(np.radians(theta[i][j])))**2
                Kij[3][3] = (secao*E/comprimento[i][j]) * (math.sin(np.radians(theta[i][j])))**2
                Kij[1][0] = (secao*E/comprimento[i][j]) * math.cos(np.radians(theta[i][j]))*math.sin(np.radians(theta[i][j]))
                Kij[0][1] = Kij[1][0]
                Kij[2][0] = (secao*E/comprimento[i][j]) * -(math.cos(np.radians(theta[i][j])))**2
                Kij[0][2] = Kij[2][0]
                Kij[2][1] = (secao*E/comprimento[i][j]) * -(math.cos(np.radians(theta[i][j])))*math.sin(np.radians(theta[i][j]))
                Kij[1][2] = Kij[2][1]
                Kij[3][0] = (secao*E/comprimento[i][j]) * -(math.cos(np.radians(theta[i][j])))*math.sin(np.radians(theta[i][j]))
                Kij[0][3] = Kij[3][0]
                Kij[3][1] = (secao*E/comprimento[i][j]) * -(math.sin(np.radians(theta[i][j])))**2
                Kij[1][3] = Kij[3][1]
                Kij[3][2] = (secao*E/comprimento[i][j]) * math.cos(np.radians(theta[i][j]))*math.sin(np.radians(theta[i][j]))
                Kij[2][3] = Kij[3][2]

                K[(2*(i+1))-2][(2*(i+1))-2] += Kij[0][0]
                K[(2*(i+1))-2][2*(i+1)-1]   += Kij[0][1]
                K[2*(i+1)-1][(2*(i+1))-2]   += Kij[1][0]
                K[2*(i+1)-1][2*(i+1)-1]     += Kij[1][1]
    
    # Definindo a matriz M
    M = np.zeros((n,n))
    for i in range(n_nfixos):
        mi = 0.00
        for j in range(n_nos):
            mij = secao*comprimento[i][j]*rho
            mi += (mij/2)
        M[(2*(i+1))-2][(2*(i+1))-2] = mi
        M[(2*(i+1))-1][(2*(i+1))-1] = mi

    # Definindo a matriz K~
    ident = np.eye(n)
    a = np.zeros(n)
    b = np.zeros(n-1)
    for i in range(n):
        a[i] = M[i][i]
    g = b.copy()

    # Calculando M^(-1/2)
    B, Lambda, k = QR_deslocamento(a, b, g, n, 10**(-6), ident)
    C = np.zeros((n,n))
    for i in range(n):
        C[i][i] = Lambda[i]**(-0.5)

    BC = np.matmul(B,C)
    Minv = np.matmul(BC, np.transpose(B)) #Minv = M^(-1/2)

    MK = np.matmul(Minv,K)
    Ktil = np.matmul(MK, Minv)

    # Definindo a matriz tridiagonal simetrica semelhante
    Kdiagonal, Ht = Householder(n, Ktil)

    # Aplicacao do algoritmo QR com deslocamento espectral
    a = np.zeros(n)
    b = np.zeros(n-1)
    for i in range(n):
        a[i] = Kdiagonal[i][i]
    for i in range(n-1):
        b[i] = Kdiagonal[i][i+1]
    g = b.copy()

    V, Autovalores, k = QR_deslocamento(a, b, g, n, 10**(-6), ident)

    freq = np.zeros(n)
    for i in range(n):
        freq[i] = math.sqrt(abs(Autovalores[i]))

    minfreqs = np.zeros(5)
    for i in range(5):
        minimo = min(freq)
        minfreqs[i] = minimo
        index = np.argwhere(freq==minimo)
        freq = np.delete(freq, index)
    print("   5 menores frequencias (rad/s): ", minfreqs)

    Y = np.zeros((n,5))
    for j in range(5):
        value = minfreqs[j]
        pos = np.where(minfreqs == value)
        for i in range(n):
            Y[i][j] = V[i][pos]
    
    Z = np.zeros((n,5))
    autovector = np.zeros(n)
    for j in range(5):
        for i in range(n):
            autovector[i] = Y[i][j]
        autovectorT = np.transpose(autovector)
        vector = np.matmul(Minv, autovectorT)
        for i in range(n):
            Z[i][j] = vector[i]
    
    print("\n")
    print("   Matriz dos modos de vibração: ")
    print("   (Cada coluna refere-se a uma das 5 menores frequência em ordem crescente da esquerda para direita)")
    print("   (Cada linha par (0, 2, 4, ...) refere-se ao deslocamento horizontal, enquanto cada linha ímpar (1, 3, 5, ...) refere-se ao vertical)")
    print("\n")
    with np.printoptions(threshold=sys.maxsize, linewidth = 200):
        print("   Z = ")
        print(Z)
   
    #t = int(input("   Digite o tempo de simulacao: "))
    #X1 = np.zeros((n,400*t))
    #X2 = np.zeros((n,400*t))
    #X3 = np.zeros((n,400*t))
    #X4 = np.zeros((n,400*t))
    #X5 = np.zeros((n,400*t))
#
    #for j in range(400*t):
    #    for i in range(n):
    #        X1[i][j] = Z[i][0] * math.cos(minfreqs[0]*j/400)
    #        X2[i][j] = Z[i][1] * math.cos(minfreqs[1]*j/400)
    #        X3[i][j] = Z[i][2] * math.cos(minfreqs[2]*j/400)
    #        X4[i][j] = Z[i][3] * math.cos(minfreqs[3]*j/400)
    #        X5[i][j] = Z[i][4] * math.cos(minfreqs[4]*j/400)
    #
    #H1 = np.zeros((n_nfixos, 400*t))
    #V1 = np.zeros((n_nfixos, 400*t))
    #H2 = np.zeros((n_nfixos, 400*t))
    #V2 = np.zeros((n_nfixos, 400*t))
    #H3 = np.zeros((n_nfixos, 400*t))
    #V3 = np.zeros((n_nfixos, 400*t))
    #H4 = np.zeros((n_nfixos, 400*t))
    #V4 = np.zeros((n_nfixos, 400*t))
    #H5 = np.zeros((n_nfixos, 400*t))
    #V5 = np.zeros((n_nfixos, 400*t))
#
#
    #for j in range(400*t):
    #    for i in range(n_nfixos):
    #        H1[i][j] = X1[2*i][j]
    #        H2[i][j] = X2[2*i][j]
    #        H3[i][j] = X3[2*i][j]
    #        H4[i][j] = X4[2*i][j]
    #        H5[i][j] = X5[2*i][j]
#
    #        V1[i][j] = X1[(2*i)+1][j]
    #        V2[i][j] = X2[(2*i)+1][j]
    #        V3[i][j] = X3[(2*i)+1][j]
    #        V4[i][j] = X4[(2*i)+1][j]
    #        V5[i][j] = X5[(2*i)+1][j]


    ##Coordenadas da abcissa 
    #tempo = np.zeros(400*t)
    #for i in range(400*t):
    #    tempo[i] = i*0.0025
#
    ##Construcao de H1
    #plt.figure(1)
    #x_plot = np.zeros(400*t)
    #for i in range(n_nfixos):
    #    for j in range(400*t):
    #        x_plot[j] = H1[i,j]
    #    lbl = 'Nó' + str(i + 1)
    #    plt.plot(tempo, x_plot, label=lbl) 
    #
    #plt.xlabel('Tempo(s)')
    #plt.ylabel('Posicao(m)')
    #plt.title("Posição horizontal do Nó em função do tempo - Frequência 1")
    #plt.legend()
    #plt.show()
#
    ##Construcao de H2
    #plt.figure(2)
    #x_plot = np.zeros(400*t)
    #for i in range(n_nfixos):
    #    for j in range(400*t):
    #        x_plot[j] = H2[i,j]
    #    lbl = 'Nó' + str(i + 1)
    #    plt.plot(tempo, x_plot, label=lbl) 
    #
    #plt.xlabel('Tempo(s)')
    #plt.ylabel('Posicao(m)')
    #plt.title("Posição horizontal do Nó em função do tempo - Frequência 2")
    #plt.legend()
    #plt.show()
#
    ##Construcao de H3
    #plt.figure(3)
    #x_plot = np.zeros(400*t)
    #for i in range(n_nfixos):
    #    for j in range(400*t):
    #        x_plot[j] = H3[i,j]
    #    lbl = 'Nó' + str(i + 1)
    #    plt.plot(tempo, x_plot, label=lbl) 
    #
    #plt.xlabel('Tempo(s)')
    #plt.ylabel('Posicao(m)')
    #plt.title("Posição horizontal do Nó em função do tempo - Frequência 3")
    #plt.legend()
    #plt.show()
#
    ##Construcao de H4
    #plt.figure(4)
    #x_plot = np.zeros(400*t)
    #for i in range(n_nfixos):
    #    for j in range(400*t):
    #        x_plot[j] = H4[i,j]
    #    lbl = 'Nó' + str(i + 1)
    #    plt.plot(tempo, x_plot, label=lbl) 
    #
    #plt.xlabel('Tempo(s)')
    #plt.ylabel('Posicao(m)')
    #plt.title("Posição horizontal do Nó em função do tempo - Frequência 4")
    #plt.legend()
    #plt.show()
#
    ##Construcao de H5
    #plt.figure(5)
    #x_plot = np.zeros(400*t)
    #for i in range(n_nfixos):
    #    for j in range(400*t):
    #        x_plot[j] = H5[i,j]
    #    lbl = 'Nó' + str(i + 1)
    #    plt.plot(tempo, x_plot, label=lbl) 
    #
    #plt.xlabel('Tempo(s)')
    #plt.ylabel('Posicao(m)')
    #plt.title("Posição horizontal do Nó em função do tempo - Frequência 5")
    #plt.legend()
    #plt.show()
   