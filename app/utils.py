import numpy as np


class Utils():

    def calcular_comprimentos_howe(self, l_bases, l_verticais):
        nos = np.array([
            [0.0, 0.0],
            [l_bases[0], 0.0],
            [l_bases[0] + l_bases[1], 0.0],
            [l_bases[0] + l_bases[1] + l_bases[2], 0.0],
            [sum(l_bases), 0.0],
            [l_bases[0], l_verticais[0]],
            [l_bases[0] + l_bases[1], l_verticais[1]],
            [l_bases[0] + l_bases[1] + l_bases[2], l_verticais[2]]
        ])

        barras = np.array([
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5],
            [1, 5],
            [5, 2],
            [2, 6],
            [2, 7],
            [3, 7],
            [7, 4],
            [5, 6],
            [6, 7]
        ])

        comprimentos = []
        angulos = []

        for i, j in barras:
            dx = nos[j][0] - nos[i][0]
            dy = nos[j][1] - nos[i][1]
            L = np.sqrt(dx**2 + dy**2)
            comprimentos.append(L)
            angulos.append((dx/L, dy/L))

        return nos, barras, comprimentos, angulos

    def gerar_vetor_areas(self, num_barras, area):
        return [area for _ in range(num_barras)]
    
    def calcular_reacoes_e_forcas(self, nos, barras, angulos, forcas_externas, forcas_virtuais):
        num_nos = len(nos)
        num_barras = len(barras)
        num_reacoes = 3

        A = np.zeros((2 * num_nos, num_barras + num_reacoes))

        for k, (i, j) in enumerate(barras):
            cx, cy = angulos[k]
            A[2*i, k] = cx
            A[2*i+1, k] = cy
            A[2*j, k] = -cx
            A[2*j+1, k] = -cy

        A[0, num_barras] = 1.0
        A[1, num_barras + 1] = 1.0
        A[2*4+1, num_barras + 2] = 1.0

        P_real = np.zeros(2 * num_nos)
        for no, fx, fy in forcas_externas:
            P_real[2*no] = fx
            P_real[2*no+1] = fy

        P_virtual = np.zeros(2 * num_nos)
        for no, fx, fy in forcas_virtuais:
            P_virtual[2*no] = fx
            P_virtual[2*no+1] = fy

        solucao_real = np.linalg.solve(A, -P_real)
        solucao_virtual = np.linalg.solve(A, -P_virtual)

        f_real = solucao_real[:num_barras]
        reacoes_reais = solucao_real[num_barras:]
        f_virtual = solucao_virtual[:num_barras]

        return f_real, reacoes_reais, f_virtual
    
    def calcular_deslocamento_virtual(self, barras, comprimentos, areas, f_real, f_virtual, E_GPa=200.0):
        dados = []
        deslocamento_total = 0.0
        E_kNm2 = E_GPa * 1e6

        for k in range(len(barras)):
            L = comprimentos[k]
            A_sec = areas[k]
            F_kN = f_real[k]
            F_v = f_virtual[k]

            delta_m = (F_kN * F_v * L) / (E_kNm2 * A_sec)
            delta_mm = delta_m * 1000.0
            deslocamento_total += delta_mm

            dados.append({
                'Barra': f"{barras[k][0]}-{barras[k][1]}",
                'Comprimento (m)': round(L, 4),
                'Força Real (kN)': round(F_kN, 2),
                'Força Virtual (kN)': round(F_v, 2),
                'Deslocamento (mm)': round(delta_mm, 3)
            })

        return dados, deslocamento_total