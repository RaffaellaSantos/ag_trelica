import numpy as np

class Utils:
    def calcular_comprimentos_howe_3d(self, l_bases, l_verticais):
        p1, p2, p3, p4 = l_bases
        h1, h2, h3, h4, h5 = l_verticais

        # Coordenadas X acumuladas para garantir alinhamento perfeito vertical entre banzos
        x0 = 0.0
        x1 = p1
        x2 = p1 + p2
        x3 = p1 + p2 + p3
        x4 = p1 + p2 + p3 + p4

        nos = np.array([
            [x0, 0.0],  # 0: A
            [x1, 0.0],  # 1: B
            [x2, 0.0],  # 2: C
            [x3, 0.0],  # 3: D
            [x4, 0.0],  # 4: E
            [x0, h1],   # 5: F
            [x1, h2],   # 6: G
            [x2, h3],   # 7: H
            [x3, h4],   # 8: I
            [x4, h5],   # 9: J
        ])

        barras = np.array([
            [0, 1], [1, 2], [2, 3], [3, 4],          # 0-3:  horizontais inferiores
            [5, 6], [6, 7], [7, 8], [8, 9],          # 4-7:  horizontais superiores
            [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],  # 8-12: verticais
            [5, 1], [6, 2], [8, 2], [9, 3],          # 13-16: diagonais Howe
        ])

        comprimentos = []
        angulos = []
        for i, j in barras:
            dx = nos[j][0] - nos[i][0]
            dy = nos[j][1] - nos[i][1]
            L = np.sqrt(dx**2 + dy**2)
            # proteção contra divisão por zero se nós colidirem
            if L < 1e-6:
                L = 1e-6
            comprimentos.append(L)
            angulos.append((dx / L, dy / L))

        return nos, barras, np.array(comprimentos), angulos

    def resolver_estatica(self, nos, barras, angulos, forcas_externas):
        num_nos     = len(nos)      # 10 nós
        num_barras  = len(barras)   # 17 barras
        num_reacoes = 3             # 3 forças de reação nos apoios

        A = np.zeros((2 * num_nos, num_barras + num_reacoes))
        for k, (i, j) in enumerate(barras):
            # cx projeta força na barra horizontal, cy projeta força na barra vertical
            cx, cy = angulos[k]
            A[2*i,     k] =  cx
            A[2*i + 1, k] =  cy
            A[2*j,     k] = -cx
            A[2*j + 1, k] = -cy
       
        # condições de apoio: nó A (0) engastado, nó E (4) roliço
        A[0,       num_barras]     = 1.0  # Rax nó A
        A[1,       num_barras + 1] = 1.0  # Ray nó A
        A[2*4 + 1, num_barras + 2] = 1.0  # Rey nó E

        P = np.zeros(2 * num_nos)
        for no, fx, fy in forcas_externas:
            P[2*no]     += fx
            P[2*no + 1] += fy

        # calcula as forças usando o método dos mínimos quadrados
        solucao, _, rank, _ = np.linalg.lstsq(A, -P, rcond=None)
       
        # verifica se a estrutura é geometricamente instável
        if rank < (num_barras + num_reacoes):
            raise np.linalg.LinAlgError("Estrutura geometricamente instável")

        return solucao[:num_barras]

    def calcular_deslocamento_ptv(self, barras, comprimentos, vetor_areas, f_real, f_virtual, E_Pa):
        deslocamento_total = 0.0
        for k in range(len(barras)):
            deslocamento_total += (f_real[k] * f_virtual[k] * comprimentos[k]) / (vetor_areas[k] * E_Pa)
        return deslocamento_total