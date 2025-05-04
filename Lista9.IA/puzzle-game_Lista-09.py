

import tkinter as tk
from tkinter import messagebox
import random
import heapq
from collections import deque
import time

# --------- Classe do Jogo ---------
class Puzzle8:
    def __init__(self):
        self.estado_objetivo = list(range(1, 9)) + [0]
        self.estado_atual = self.embaralhar_tabuleiro()

    def embaralhar_tabuleiro(self):
        estado = self.estado_objetivo[:]
        while True:
            random.shuffle(estado)
            if self.eh_solucionavel(estado) and estado != self.estado_objetivo:
                return estado

    def eh_solucionavel(self, estado):
        inversoes = 0
        for i in range(8):
            for j in range(i + 1, 9):
                if estado[i] and estado[j] and estado[i] > estado[j]:
                    inversoes += 1
        return inversoes % 2 == 0

    def mover(self, direcao):
        indice = self.estado_atual.index(0)
        linha, coluna = divmod(indice, 3)
        if direcao == "Up" and linha > 0:
            self.trocar(indice, indice - 3)
        elif direcao == "Down" and linha < 2:
            self.trocar(indice, indice + 3)
        elif direcao == "Left" and coluna > 0:
            self.trocar(indice, indice - 1)
        elif direcao == "Right" and coluna < 2:
            self.trocar(indice, indice + 1)

    def trocar(self, i, j):
        self.estado_atual[i], self.estado_atual[j] = self.estado_atual[j], self.estado_atual[i]

    def resolvido(self):
        return self.estado_atual == self.estado_objetivo

# --------- Heurísticas ---------

def heuristica_manhattan(estado):
    distancia = 0
    for i, val in enumerate(estado):
        if val == 0: continue
        linha_destino, col_destino = divmod(val - 1, 3)
        linha_atual, col_atual = divmod(i, 3)
        distancia += abs(linha_destino - linha_atual) + abs(col_destino - col_atual)
    return distancia

# Nova heurística: número de peças fora do lugar

def heuristica_mal_colocado(estado):
    return sum(1 for i in range(9) if estado[i] != 0 and estado[i] != i + 1)

# --------- Funções Auxiliares ---------

def gerar_vizinhos(estado):
    vizinhos = []
    indice = estado.index(0)
    linha, coluna = divmod(indice, 3)
    direcoes = [("Up", -3), ("Down", 3), ("Left", -1), ("Right", 1)]

    for direcao, delta in direcoes:
        novo_indice = indice + delta
        if direcao == "Up" and linha == 0: continue
        if direcao == "Down" and linha == 2: continue
        if direcao == "Left" and coluna == 0: continue
        if direcao == "Right" and coluna == 2: continue
        novo_estado = list(estado)
        novo_estado[indice], novo_estado[novo_indice] = novo_estado[novo_indice], novo_estado[indice]
        vizinhos.append((novo_estado, direcao))
    return vizinhos

# --------- Algoritmos ---------

def busca_a_estrela(estado_inicial, heuristica):
    objetivo = list(range(1, 9)) + [0]
    fila = [(heuristica(estado_inicial), 0, estado_inicial, [])]
    visitados = set()
    while fila:
        _, custo, estado, caminho = heapq.heappop(fila)
        if tuple(estado) in visitados:
            continue
        visitados.add(tuple(estado))
        if estado == objetivo:
            return caminho
        for vizinho, direcao in gerar_vizinhos(estado):
            if tuple(vizinho) not in visitados:
                novo_custo = custo + 1
                estimativa = novo_custo + heuristica(vizinho)
                heapq.heappush(fila, (estimativa, novo_custo, vizinho, caminho + [direcao]))
    return []

def busca_em_largura(estado_inicial):
    objetivo = list(range(1, 9)) + [0]
    fila = deque([(estado_inicial, [])])
    visitados = set()
    while fila:
        estado, caminho = fila.popleft()
        if estado == objetivo:
            return caminho
        visitados.add(tuple(estado))
        for vizinho, direcao in gerar_vizinhos(estado):
            if tuple(vizinho) not in visitados:
                fila.append((vizinho, caminho + [direcao]))
    return []

def busca_em_profundidade(estado_inicial, max_depth=50):
    objetivo = list(range(1, 9)) + [0]
    pilha = [(estado_inicial, [], 0)]  # A pilha agora inclui a profundidade
    visitados = set()

    while pilha:
        estado, caminho, profundidade = pilha.pop()

        # Limite de profundidade para evitar loops infinitos
        if profundidade > max_depth:
            continue

        if tuple(estado) in visitados:
            continue

        visitados.add(tuple(estado))

        if estado == objetivo:
            return caminho

        # Adiciona vizinhos à pilha com a profundidade aumentada
        vizinhos = gerar_vizinhos(estado)
        for vizinho, direcao in reversed(vizinhos):
            if tuple(vizinho) not in visitados:
                pilha.append((vizinho, caminho + [direcao], profundidade + 1))

    return []

def busca_gulosa(estado_inicial, heuristica):
    objetivo = list(range(1, 9)) + [0]
    fila = [(heuristica(estado_inicial), estado_inicial, [])]
    visitados = set()
    while fila:
        _, estado, caminho = heapq.heappop(fila)
        if tuple(estado) in visitados:
            continue
        visitados.add(tuple(estado))
        if estado == objetivo:
            return caminho
        for vizinho, direcao in gerar_vizinhos(estado):
            if tuple(vizinho) not in visitados:
                estimativa = heuristica(vizinho)
                heapq.heappush(fila, (estimativa, vizinho, caminho + [direcao]))
    return []

# --------- Interface Tkinter ---------
class InterfacePuzzle:
    def __init__(self, root):
        self.jogo = Puzzle8()
        self.root = root
        self.root.title("8-Puzzle com Inteligência Artificial")

        self.frame_tabuleiro = tk.Frame(root, padx=20, pady=20)
        self.frame_tabuleiro.grid(row=0, column=0, columnspan=3)

        self.botoes = []
        for i in range(9):
            botao = tk.Button(self.frame_tabuleiro, text="", font=("Helvetica", 24, "bold"), width=4, height=2,
                              command=lambda i=i: self.clique_bloco(i))
            botao.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.botoes.append(botao)

        self.frame_controles = tk.Frame(root, pady=10)
        self.frame_controles.grid(row=1, column=0, columnspan=3)

        self.btn_shuffle = tk.Button(root, text="Embaralhar", command=self.embaralhar)
        self.btn_shuffle.grid(row=2, column=0, pady=10)

        self.btn_astar = tk.Button(root, text="A* Manhattan", command=self.resolver_astar_manhattan)
        self.btn_astar.grid(row=2, column=1, pady=10)

        self.btn_astar2 = tk.Button(root, text="A* Mal Pos", command=self.resolver_astar_mal_colocado)
        self.btn_astar2.grid(row=2, column=2, pady=10)

        self.btn_bfs = tk.Button(root, text="BFS", command=self.resolver_bfs)
        self.btn_bfs.grid(row=3, column=0, pady=10)

        self.btn_gulosa = tk.Button(root, text="Gulosa", command=self.resolver_gulosa)
        self.btn_gulosa.grid(row=3, column=1, pady=10)

        self.atualizar_tabuleiro()

    def atualizar_tabuleiro(self):
        for i in range(9):
            num = self.jogo.estado_atual[i]
            self.botoes[i].config(text="" if num == 0 else str(num), bg="#e0e0e0" if num == 0 else "#ffffff")

    def clique_bloco(self, indice):
        branco = self.jogo.estado_atual.index(0)
        linha1, col1 = divmod(branco, 3)
        linha2, col2 = divmod(indice, 3)
        if abs(linha1 - linha2) + abs(col1 - col2) == 1:
            self.jogo.trocar(branco, indice)
            self.atualizar_tabuleiro()

    def embaralhar(self):
        self.jogo.estado_atual = self.jogo.embaralhar_tabuleiro()
        self.atualizar_tabuleiro()

    def salvar_resultado(self, nome_algoritmo, caminho, duracao):
        with open("resultado_resolucao.txt", "w") as f:
            f.write(f"Algoritmo: {nome_algoritmo}\n")
            f.write(f"Estado inicial: {self.jogo.estado_atual}\n")
            f.write(f"Movimentos: {len(caminho)}\n")
            f.write(f"Tempo: {duracao:.4f} segundos\n")
            f.write(f"Caminho: {' -> '.join(caminho)}\n")

    def animar_solucao(self, solucao, duracao, nome_algoritmo):
        if not solucao:
            messagebox.showinfo("Erro", "Nenhuma solução encontrada.")
            return

        total = len(solucao)
        self.salvar_resultado(nome_algoritmo, solucao.copy(), duracao)

        def passo():
            if solucao:
                self.jogo.mover(solucao.pop(0))
                self.atualizar_tabuleiro()
                self.root.after(300, passo)
            else:
                messagebox.showinfo("Resolvido!", f"Movimentos: {total}\nTempo: {duracao:.4f}s")

        passo()

    def resolver_astar_manhattan(self):
        inicio = time.time()
        caminho = busca_a_estrela(self.jogo.estado_atual[:], heuristica_manhattan)
        fim = time.time()
        self.animar_solucao(caminho, fim - inicio, "A* Manhattan")

    def resolver_astar_mal_colocado(self):
        inicio = time.time()
        caminho = busca_a_estrela(self.jogo.estado_atual[:], heuristica_mal_colocado)
        fim = time.time()
        self.animar_solucao(caminho, fim - inicio, "A* Mal Pos")

    def resolver_bfs(self):
        inicio = time.time()
        caminho = busca_em_largura(self.jogo.estado_atual[:])
        fim = time.time()
        self.animar_solucao(caminho, fim - inicio, "BFS")

    def resolver_gulosa(self):
        inicio = time.time()
        caminho = busca_gulosa(self.jogo.estado_atual[:], heuristica_manhattan)  # Usando a heurística de Manhattan
        fim = time.time()
        self.animar_solucao(caminho, fim - inicio, "Busca Gulosa")

# --------- Executar Interface ---------
if __name__ == "__main__":
    root = tk.Tk()
    app = InterfacePuzzle(root)
    root.mainloop()
