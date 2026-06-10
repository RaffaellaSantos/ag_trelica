import numpy as np

class Utils:
    """Fornece as ferramentas matemáticas e físicas para a avaliação da treliça Howe."""

    def calcular_comprimentos_howe(self, p, h):
        """Calcula a conectividade, coordenadas, comprimentos e ângulos da treliça Howe."""
        p_total = np.sum(p)
        if p_total == 0:
            p_total = 1.0
        p_norm = (p / p_total) * 35.0
        x = [0.0, p_norm[0], p_norm[0] + p_norm[1], p_norm[0] + p_norm[1] + p_norm[2], 35.0]
        nos = np.array([
            [x[0], 0.0], [x[1], 0.0], [x[2], 0.0], [x[3], 0.0], [x[4], 0.0],
            [x[0], h[0]], [x[1], h[1]], [x[2], h[2]], [x[3], h[3]], [x[4], h[4]]
        ])
        barras = np.array([
            [0, 1], [1, 2], [2, 3], [3, 4],
            [5, 6], [6, 7], [7, 8], [8, 9],
            [0, 5], [6, 1], [7, 2], [8, 3], [4, 9],
            [5, 1], [6, 2], [8, 2], [9, 3]
        ])
        comprimentos = []
        angulos = []
        for i, j in barras:
            dx = nos[j][0] - nos[i][0]
            dy = nos[j][1] - nos[i][1]
            L = np.sqrt(dx**2 + dy**2)
            comprimentos.append(L)
            angulos.append((dx / L, dy / L))
        return nos, barras, comprimentos, angulos

    def calcular_reacoes_e_forcas(self, nos, barras, angulos, carga_kg):
        """Avalia os esforços axiais reais nas barras usando o método dos nós e álgebra linear."""
        num_nos = len(nos)
        num_barras = len(barras)
        num_reacoes = 3
        A = np.zeros((2 * num_nos, num_barras + num_reacoes))
        for k, (i, j) in enumerate(barras):
            cx, cy = angulos[k]
            A[2 * i, k] = cx
            A[2 * i + 1, k] = cy
            A[2 * j, k] = -cx
            A[2 * j + 1, k] = -cy
        A[0, num_barras] = 1.0
        A[1, num_barras + 1] = 1.0
        A[2 * 4 + 1, num_barras + 2] = 1.0
        P_real = np.zeros(2 * num_nos)
        carga_newtons = carga_kg * 9.81
        P_real[2 * 2 + 1] = -carga_newtons
        try:
            solucao_real = np.linalg.solve(A, -P_real)
            f_real = solucao_real[:num_barras]
            return f_real
        except np.linalg.LinAlgError:
            return np.zeros(num_barras)