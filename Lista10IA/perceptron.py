import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from itertools import product



def gerar_dados(n_entradas, tipo='AND'):
    """
    Gera dados de entrada (X) e saída (y) para funções lógicas.

    Args:
        n_entradas (int): Número de variáveis de entrada booleanas.
        tipo (str): Tipo de função lógica ('AND', 'OR', 'XOR').

    Returns:
        tuple: (X, y) onde X são as entradas e y as saídas esperadas.
    """
    # Gera todas as combinações possíveis de entradas booleanas (0 ou 1)
    X = np.array(list(product([0, 1], repeat=n_entradas)))

    if tipo == 'AND':
        y = np.array([int(np.all(x)) for x in X])
    elif tipo == 'OR':
        y = np.array([int(np.any(x)) for x in X])
    elif tipo == 'XOR':
        # XOR para n entradas: 1 se o número de entradas '1' for ímpar, 0 caso contrário.
        # Para o caso clássico de 2 entradas, é (x1+x2) % 2.
        # Para n > 2, pode ser generalizado como a soma dos bits módulo 2.
        y = np.array([np.sum(x) % 2 for x in X])
    else:
        raise ValueError("Função lógica inválida. Escolha entre 'AND', 'OR', 'XOR'.")
    return X, y

def plotar_hiperplano(X, y, clf, titulo):
    """
    Plota os dados e o hiperplano de separação aprendido pelo Perceptron.

    Args:
        X (np.array): Dados de entrada.
        y (np.array): Saídas esperadas.
        clf (Perceptron): Modelo Perceptron treinado.
        titulo (str): Título do gráfico.
    """
    n_dim = X.shape[1]

    if n_dim == 2:
        # Caso 2D: plotamos os pontos e a linha de separação
        plt.figure(figsize=(6, 6))
        for xi, yi in zip(X, y):
            plt.scatter(xi[0], xi[1], c='red' if yi == 1 else 'blue', s=100, edgecolors='k', alpha=0.7)

        # w1*x1 + w2*x2 + b = 0  => x2 = -(w1*x1 + b) / w2
        x_vals = np.array(plt.gca().get_xlim())
        if clf.coef_[0][1] != 0: # Evita divisão por zero se w2 for 0
            y_vals = -(clf.intercept_[0] + clf.coef_[0][0] * x_vals) / clf.coef_[0][1]
            plt.plot(x_vals, y_vals, '--', c='green', label='Hiperplano de Separação')
        elif clf.coef_[0][0] != 0: # Se w2 é 0, mas w1 não, a linha é vertical
             x_val_const = -clf.intercept_[0] / clf.coef_[0][0]
             plt.axvline(x=x_val_const, linestyle='--', color='green', label='Hiperplano de Separação')

        plt.title(titulo + " (2D)")
        plt.xlabel("Entrada x1")
        plt.ylabel("Entrada x2")
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        plt.legend()
        plt.grid(True)
        plt.show()

    elif n_dim == 3:
        # Caso 3D: plotamos os pontos e o plano de separação
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        for xi, yi in zip(X, y):
            ax.scatter(xi[0], xi[1], xi[2], c='red' if yi == 1 else 'blue', s=100, edgecolors='k', alpha=0.7)

        # w1*x1 + w2*x2 + w3*x3 + b = 0 => x3 = -(w1*x1 + w2*x2 + b) / w3
        # Criando um meshgrid para o plano
        xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.1, X[:,0].max()+0.1, 10),
                             np.linspace(X[:,1].min()-0.1, X[:,1].max()+0.1, 10))

        w0 = clf.intercept_[0]
        w1, w2, w3 = clf.coef_[0]

        if w3 != 0: # Evita divisão por zero se w3 for 0
            zz = -(w0 + w1 * xx + w2 * yy) / w3
            ax.plot_surface(xx, yy, zz, alpha=0.4, color='green', rstride=100, cstride=100)
        elif w2 != 0: # Plano paralelo ao eixo x2 (y)
            yy_const = -(w0 + w1 * xx)/w2
            ax.plot_surface(xx, yy_const, np.zeros_like(xx)+X[:,2].mean(), alpha=0.4, color='green') # Exemplo, pode precisar de ajuste
        elif w1 != 0: # Plano paralelo ao eixo x1 (x)
            xx_const = -(w0 + w2 * yy)/w1
            ax.plot_surface(xx_const, yy, np.zeros_like(yy)+X[:,2].mean(), alpha=0.4, color='green')


        ax.set_xlabel('Entrada x1')
        ax.set_ylabel('Entrada x2')
        ax.set_zlabel('Entrada x3')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 1])
        ax.set_title(titulo + " (3D)")
        plt.show()

    else:
        # Caso n_dim > 3: usamos PCA para reduzir a 2D para visualização
        pca = PCA(n_components=2)
        X_reduzido = pca.fit_transform(X)

        plt.figure(figsize=(7, 6))
        for xi_r, yi_orig in zip(X_reduzido, y):
            plt.scatter(xi_r[0], xi_r[1], c='red' if yi_orig == 1 else 'blue', s=100, edgecolors='k', alpha=0.7)

        # O hiperplano de separação original está em n_dim.
        # A projeção 2D via PCA é apenas para visualização dos dados.
        # Plotar o hiperplano projetado corretamente é complexo e pode não ser muito informativo.
        # Vamos focar na separabilidade visual dos dados projetados.
        plt.title(titulo + f" (PCA de {n_dim}D → 2D)")
        plt.xlabel("Componente Principal 1 (PC1)")
        plt.ylabel("Componente Principal 2 (PC2)")
        plt.grid(True)
        plt.show()
        print(f"Nota: Para {n_dim} entradas, o hiperplano de separação não é diretamente plotado. "
              f"O gráfico mostra a projeção dos dados em 2D usando PCA.")


def treinar_perceptron(X, y, titulo=""):
    """
    Treina um modelo Perceptron e exibe os resultados e o gráfico.

    Args:
        X (np.array): Dados de entrada.
        y (np.array): Saídas esperadas.
        titulo (str): Título para o gráfico e saídas.
    """
    # max_iter: número máximo de épocas (passagens sobre os dados de treinamento)
    # tol: critério de parada. Se a melhoria da perda for menor que tol por n_iter_no_change épocas, o treinamento para.
    # random_state: para reprodutibilidade
    clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42, eta0=0.1) # eta0 é a taxa de aprendizado
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acuracia = clf.score(X, y)

    print("\n" + "="*40)
    print(f"Resultados para: {titulo}")
    print("="*40)
    print(f"Pesos (coeficientes w1, w2, ..., wn): {clf.coef_}")
    print(f"Bias (interceptação w0): {clf.intercept_}")
    print(f"Saídas esperadas (y): {y}")
    print(f"Saídas previstas (ŷ): {y_pred}")
    print(f"Acurácia do Perceptron: {acuracia:.4f} ({acuracia*100:.2f}%)")

    if acuracia < 1.0 and X.shape[1] == 2 and np.array_equal(y, np.array([np.sum(x) % 2 for x in X])): # Verifica se é XOR de 2 entradas
        print("\nO Perceptron de camada única não conseguiu separar linearmente os dados do XOR.")
        print("Isso é esperado, pois XOR não é linearmente separável.")
    elif acuracia == 1.0:
        print("\nO Perceptron aprendeu a função lógica perfeitamente.")
    else:
        print("\nO Perceptron não aprendeu a função perfeitamente. "
              "Isso pode ocorrer se os dados não forem linearmente separáveis ou se o treinamento não convergiu.")
    print("="*40 + "\n")

    plotar_hiperplano(X, y, clf, titulo)


if __name__ == "__main__":
    # --- Testes ---
    testes = [
        {'n': 2, 'tipo': 'AND'},
        {'n': 3, 'tipo': 'AND'},
        {'n': 10, 'tipo': 'AND'},
        {'n': 2, 'tipo': 'OR'},
        {'n': 3, 'tipo': 'OR'},
        {'n': 10, 'tipo': 'OR'},
        {'n': 2, 'tipo': 'XOR'},  # Mostrar que Perceptron não resolve XOR
        # Teste adicional para XOR com 3 entradas (também não linearmente separável)
        {'n': 3, 'tipo': 'XOR'},
    ]

    for teste in testes:
        num_entradas = teste['n']
        tipo_funcao = teste['tipo']

        print(f"\n--- Treinando Perceptron para {tipo_funcao} com {num_entradas} entradas ---")
        X_dados, y_dados = gerar_dados(num_entradas, tipo=tipo_funcao)
        treinar_perceptron(X_dados, y_dados, titulo=f"Função {tipo_funcao} com {num_entradas} entradas")