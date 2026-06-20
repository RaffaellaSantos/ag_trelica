import numpy as np
import random
import math
from app.utils import Utils


class Individuo:
    def __init__(self, p, h, w, n):
        self.p = p
        self.h = h
        self.w = w
        self.n = n
        self.massa_g = float("inf")
        self.tensao_max_mpa = float("inf")
        self.inv_tensao = 0.0
        self.fator_seguranca = 0.0
        self.eficiencia = 0.0
        self.violacao = 0.0
        self.valido = False
        self.rank = -1
        self.distancia_aglomeracao = 0.0
        self.dominados = []
        self.num_dominantes = 0
        self.dados_extras = {}


class TrussAG:
    def __init__(self, tam_populacao=150, num_geracoes=150, carga_projeto_kg=15.0):
        self.tam_populacao = tam_populacao
        self.num_geracoes = num_geracoes
        self.carga_projeto_kg = carga_projeto_kg
        self.utils = Utils()
        self.L_PALITO_CM = 14.0
        self.H_SECTION_M = 0.0135
        self.T_SECTION_M = 0.0017
        self.SOBREPOSICAO_CM = 4.0
        self.RHO = 510.0
        self.E_PA = 3.5e9
        self.SIGMA_T_YIELD = 55e6
        self.SIGMA_C_YIELD = 35e6
        self.P0_MIN, self.P0_MAX = (
            10.0,
            12.0,
        )  # painéis externos AB e DE: corte = p + 2 ≤ 14
        self.P1_MIN, self.P1_MAX = (
            7.0,
            10.0,
        )  # painéis internos BC e CD: corte = p + 4 ≤ 14
        self.H_MIN, self.H_MAX = 3.0, 7.8
        self.W_MIN, self.W_MAX = 7.0, 14.0
        self.Sg = 2.5  # fator de calibração: reduz Euler para imperfeições reais (coluna real vs ideal)
        self.populacao = [self.gerar_individuo() for _ in range(self.tam_populacao)]
        self.historico_fronteira = []
        self.historico_melhor = []
        self.historico_populacao = []
        self.arquivo_pareto_global = []
        self.melhor_global = None  # Variável para rastrear o Campeão Absoluto

    def impor_simetria(self, p, h, n):
        p[3], p[2] = p[0], p[1]
        h[4], h[3] = h[0], h[1]
        n[3], n[2] = n[0], n[1]
        n[7], n[6] = n[4], n[5]
        n[12], n[11] = n[8], n[9]
        n[16], n[15] = n[13], n[14]
        return p, h, n

    def gerar_individuo(self):
        p = np.array(
            [
                random.uniform(self.P0_MIN, self.P0_MAX),
                random.uniform(self.P1_MIN, self.P1_MAX),
                random.uniform(self.P1_MIN, self.P1_MAX),
                random.uniform(self.P0_MIN, self.P0_MAX),
            ]
        )
        h = np.random.uniform(self.H_MIN, self.H_MAX, 5)
        w = random.uniform(self.W_MIN, self.W_MAX)
        n = np.random.randint(1, 3, 17)
        p, h, n = self.impor_simetria(p, h, n)
        return Individuo(p, h, w, n)

    def calcular_fator_seguranca(self, tensao_atuante_pa, eh_tracao):
        if tensao_atuante_pa <= 1e-9:
            return float("inf")
        limite = self.SIGMA_T_YIELD if eh_tracao else self.SIGMA_C_YIELD
        return limite / tensao_atuante_pa

    def calcular_capacidade_compressao(self, n_camadas, length_m, length_lat_m=None):
        base_m = n_camadas * self.T_SECTION_M
        altura_m = self.H_SECTION_M
        inercia_yy = (altura_m * (base_m**3)) / 12.0  # eixo lateral (fraco)
        inercia_xx = (base_m * (altura_m**3)) / 12.0  # eixo no plano (forte)
        area = base_m * altura_m

        # Flambagem no plano (usa I_min)
        I_min = min(inercia_yy, inercia_xx)
        euler_ip = (math.pi**2) * self.E_PA * I_min / (length_m**2)

        # Flambagem lateral (banzo superior): comprimento efetivo = w
        if length_lat_m is not None:
            euler_op = (math.pi**2) * self.E_PA * inercia_yy / (length_lat_m**2)
            forca_euler = min(euler_ip, euler_op)
        else:
            forca_euler = euler_ip

        # Sg reduz Euler para compensar imperfeições reais (coluna ideal vs real)
        forca_esmagamento = self.SIGMA_C_YIELD * area
        return min(forca_euler / self.Sg, forca_esmagamento) / area

    def avaliar(self, ind):
        nos, barras, comprimentos_cm, angulos = self.utils.calcular_comprimentos_howe(
            ind.p, ind.h
        )
        carga_por_portico = self.carga_projeto_kg / 2.0
        esforcos_n = self.utils.calcular_reacoes_e_forcas(
            nos, barras, angulos, carga_por_portico
        )
        esforcos_virtuais = self.utils.calcular_reacoes_e_forcas(
            nos, barras, angulos, 1.0 / 9.81
        )

        if np.all(esforcos_n == 0) or np.isnan(esforcos_n).any():
            ind.violacao = 10000.0
            ind.massa_g = 10000.0
            ind.tensao_max_mpa = 9999.0
            ind.inv_tensao = 0.0
            ind.eficiencia = 0.0
            ind.valido = False
            return

        comprimentos_geometricos_cm = np.array(comprimentos_cm, dtype=float).copy()
        comprimentos_fisicos_cm = comprimentos_geometricos_cm.copy()
        # apoios A=0 e E=4 não têm junta interna — só os demais nós somam meia sobreposição
        nos_apoio = {0, 4}
        meia_sob = self.SOBREPOSICAO_CM / 2  # 2 cm por extremidade interna
        for i, (ni, nj) in enumerate(barras):
            n_juntas = (0 if ni in nos_apoio else 1) + (0 if nj in nos_apoio else 1)
            comprimentos_fisicos_cm[i] += n_juntas * meia_sob

        l_util = self.L_PALITO_CM - 2 * self.SOBREPOSICAO_CM
        if l_util <= 0:
            l_util = 1.0

        massa_linear_g_cm = self.RHO * self.H_SECTION_M * self.T_SECTION_M * 10.0

        massa_g = 0.0
        total_palitos = 0
        deslocamento_m = 0.0

        for i in range(17):
            L_cm = comprimentos_fisicos_cm[i]

            A_m2 = ind.n[i] * self.T_SECTION_M * self.H_SECTION_M
            L_m = comprimentos_cm[i] / 100.0
            deslocamento_m += (esforcos_n[i] * esforcos_virtuais[i] * L_m) / (
                self.E_PA * A_m2
            )

            massa_g += 2 * (L_cm * ind.n[i] * massa_linear_g_cm)

            if L_cm <= self.L_PALITO_CM:
                k = 1
            else:
                k = math.ceil((L_cm - self.L_PALITO_CM) / l_util) + 1
                massa_g += 2 * (
                    (k - 1) * self.SOBREPOSICAO_CM * ind.n[i] * massa_linear_g_cm
                )

            total_palitos += 2 * (k * ind.n[i])

        massa_g += 10 * (ind.w * massa_linear_g_cm)
        palitos_travamento = 1
        if ind.w > self.L_PALITO_CM:
            k_w = math.ceil((ind.w - self.L_PALITO_CM) / l_util) + 1
            palitos_travamento = k_w
            massa_g += 10 * ((k_w - 1) * self.SOBREPOSICAO_CM * massa_linear_g_cm)

        total_palitos += 10 * palitos_travamento

        ind.massa_g = massa_g
        ind.dados_extras = {
            "comprimentos_geometricos_cm": comprimentos_geometricos_cm,
            "comprimentos_cm": comprimentos_fisicos_cm,
            "total_palitos": total_palitos,
            "esforcos_n": esforcos_n,
            "deslocamento_mm": deslocamento_m * 1000.0,
            "nos": nos, 
        }

        piores_tensoes = []
        fs_minimo = float("inf")
        for i, forca in enumerate(esforcos_n):
            area = ind.n[i] * self.T_SECTION_M * self.H_SECTION_M
            tensao_pa = abs(forca) / area
            comprimento_m = comprimentos_cm[i] / 100.0
            if forca > 1e-9:
                piores_tensoes.append(tensao_pa)
                fs = self.calcular_fator_seguranca(tensao_pa, True)
            elif forca < -1e-9:
                # Banzos superiores (índices 4-7): verifica flambagem lateral com L_eff = w
                w_m = ind.w / 100.0 if i in range(4, 8) else None
                limite_comp_pa = self.calcular_capacidade_compressao(
                    ind.n[i], comprimento_m, w_m
                )
                tensao_efetiva_pa = (
                    tensao_pa * (self.SIGMA_C_YIELD / limite_comp_pa)
                    if limite_comp_pa > 0
                    else tensao_pa * 1e3
                )
                piores_tensoes.append(tensao_efetiva_pa)
                fs = limite_comp_pa / tensao_pa if tensao_pa > 0 else float("inf")
            else:
                piores_tensoes.append(0.0)
                fs = float("inf")
            if fs < fs_minimo:
                fs_minimo = fs

        ind.tensao_max_mpa = (
            (max(piores_tensoes) / 1e6) if piores_tensoes else float("inf")
        )
        if ind.tensao_max_mpa > 1e-6 and ind.tensao_max_mpa != float("inf"):
            ind.inv_tensao = 1.0 / ind.tensao_max_mpa
        else:
            ind.inv_tensao = 0.0

        ind.fator_seguranca = fs_minimo

        v = 0.0
        if massa_g > 600.0:
            v += (massa_g - 600.0) * 100.0

        p_total = np.sum(ind.p)
        # comprimento da treliça = sum(p), alvo 43–51 cm
        if p_total < 43.0:
            v += (43.0 - p_total) * 500.0
        if p_total > 51.0:
            v += (p_total - 51.0) * 500.0

        if any(ind.n[i] == 1 for i in range(4)):
            v += 200.0

        if fs_minimo < 1.0:
            v += (1.0 - fs_minimo) * 1000.0

        if total_palitos > 150:
            v += (total_palitos - 150) * 10.0

        ind.violacao = v
        ind.valido = v == 0.0

        # Atribui a eficiência (Fitness Real) se for válida
        if ind.valido:
            ind.eficiencia = (ind.fator_seguranca * self.carga_projeto_kg) / ind.massa_g
        else:
            ind.eficiencia = 0.0

    def classificar_pareto(self, populacao):
        fronteiras = [[]]
        for p in populacao:
            p.dominados = []
            p.num_dominantes = 0
            for q in populacao:
                dominates_q = False
                if p.violacao < q.violacao:
                    dominates_q = True
                elif p.violacao == q.violacao:
                    p_melhor_ou_igual = (
                        p.massa_g <= q.massa_g and p.inv_tensao >= q.inv_tensao
                    )
                    p_estritamente_melhor = (
                        p.massa_g < q.massa_g or p.inv_tensao > q.inv_tensao
                    )
                    if p_melhor_ou_igual and p_estritamente_melhor:
                        dominates_q = True

                q_dominates_p = False
                if q.violacao < p.violacao:
                    q_dominates_p = True
                elif q.violacao == p.violacao:
                    q_melhor_ou_igual = (
                        q.massa_g <= p.massa_g and q.inv_tensao >= p.inv_tensao
                    )
                    q_estritamente_melhor = (
                        q.massa_g < p.massa_g or q.inv_tensao > p.inv_tensao
                    )
                    if q_melhor_ou_igual and q_estritamente_melhor:
                        q_dominates_p = True

                if dominates_q:
                    p.dominados.append(q)
                elif q_dominates_p:
                    p.num_dominantes += 1

            if p.num_dominantes == 0:
                p.rank = 0
                fronteiras[0].append(p)

        i = 0
        while len(fronteiras[i]) > 0:
            proxima_fronteira = []
            for p in fronteiras[i]:
                for q in p.dominados:
                    q.num_dominantes -= 1
                    if q.num_dominantes == 0:
                        q.rank = i + 1
                        proxima_fronteira.append(q)
            i += 1
            fronteiras.append(proxima_fronteira)
        return fronteiras[:-1]

    def calcular_crowding_distance(self, front):
        comprimento = len(front)
        if comprimento == 0:
            return
        for p in front:
            p.distancia_aglomeracao = 0.0

        front.sort(key=lambda x: x.massa_g)
        front[0].distancia_aglomeracao = float("inf")
        front[-1].distancia_aglomeracao = float("inf")
        m_min, m_max = front[0].massa_g, front[-1].massa_g
        if m_max > m_min:
            for i in range(1, comprimento - 1):
                front[i].distancia_aglomeracao += (
                    front[i + 1].massa_g - front[i - 1].massa_g
                ) / (m_max - m_min)

        front.sort(key=lambda x: x.inv_tensao)
        front[0].distancia_aglomeracao = float("inf")
        front[-1].distancia_aglomeracao = float("inf")
        t_min, t_max = front[0].inv_tensao, front[-1].inv_tensao
        if t_max > t_min:
            for i in range(1, comprimento - 1):
                front[i].distancia_aglomeracao += (
                    front[i + 1].inv_tensao - front[i - 1].inv_tensao
                ) / (t_max - t_min)

    def cruzar_e_mutar(self, p1, p2):
        alpha = 0.35
        c1_p = p1.p * alpha + p2.p * (1 - alpha)
        c2_p = p2.p * alpha + p1.p * (1 - alpha)
        c1_h = p1.h * alpha + p2.h * (1 - alpha)
        c2_h = p2.h * alpha + p1.h * (1 - alpha)
        c1_w, c2_w = p1.w, p2.w
        mask = np.random.rand(17) > 0.5
        c1_n = np.where(mask, p1.n, p2.n)
        c2_n = np.where(mask, p2.n, p1.n)
        filhos = [Individuo(c1_p, c1_h, c1_w, c1_n), Individuo(c2_p, c2_h, c2_w, c2_n)]

        for f in filhos:
            if random.random() < 0.2:
                f.p += np.random.normal(0, 0.5, 4)
                f.h += np.random.normal(0, 0.5, 5)
                f.w += np.random.normal(0, 0.4)
            for j in range(17):
                if random.random() < 0.1:
                    f.n[j] = random.choice([1, 2, 3])

            f.p[0] = np.clip(f.p[0], self.P0_MIN, self.P0_MAX)
            f.p[1] = np.clip(f.p[1], self.P1_MIN, self.P1_MAX)
            f.p[2] = np.clip(f.p[2], self.P1_MIN, self.P1_MAX)
            f.p[3] = np.clip(f.p[3], self.P0_MIN, self.P0_MAX)
            f.h = np.clip(f.h, self.H_MIN, self.H_MAX)
            f.w = np.clip(f.w, self.W_MIN, self.W_MAX)
            f.p, f.h, f.n = self.impor_simetria(f.p, f.h, f.n)

        return filhos[0], filhos[1]

    def iniciar(self):
        for ind in self.populacao:
            self.avaliar(ind)

        for gen in range(self.num_geracoes):
            nova_populacao = []
            while len(nova_populacao) < self.tam_populacao:
                p1, p2 = random.sample(self.populacao, 2)
                f1, f2 = self.cruzar_e_mutar(p1, p2)
                self.avaliar(f1)
                self.avaliar(f2)
                nova_populacao.extend([f1, f2])

            populacao_combinada = self.populacao + nova_populacao
            fronteiras = self.classificar_pareto(populacao_combinada)
            self.populacao = []
            para_proxima = self.tam_populacao

            for front in fronteiras:
                self.calcular_crowding_distance(front)
                if len(front) <= para_proxima:
                    self.populacao.extend(front)
                    para_proxima -= len(front)
                else:
                    front.sort(key=lambda x: -x.distancia_aglomeracao)
                    self.populacao.extend(front[:para_proxima])
                    break

            candidatos_globais = self.arquivo_pareto_global + self.populacao
            fronts_globais = self.classificar_pareto(candidatos_globais)
            self.calcular_crowding_distance(fronts_globais[0])
            fronts_globais[0].sort(key=lambda x: -x.distancia_aglomeracao)

            self.arquivo_pareto_global = [p for p in fronts_globais[0] if p.valido][
                :150
            ]

            front_atual = [
                [p.massa_g, p.inv_tensao] for p in self.arquivo_pareto_global
            ]
            pop_atual = [
                [p.massa_g, p.inv_tensao] for p in self.populacao if p.massa_g < 1000
            ]

            self.historico_fronteira.append(front_atual)
            self.historico_populacao.append(pop_atual)

            if self.arquivo_pareto_global:
                melhor_da_geracao = max(
                    self.arquivo_pareto_global, key=lambda x: x.eficiencia
                )
                if (
                    self.melhor_global is None
                    or melhor_da_geracao.eficiencia > self.melhor_global.eficiencia
                ):
                    self.melhor_global = melhor_da_geracao
            else:
                melhor_da_geracao = min(self.populacao, key=lambda x: x.violacao)
                if (
                    self.melhor_global is None
                    or melhor_da_geracao.violacao < self.melhor_global.violacao
                ):
                    self.melhor_global = melhor_da_geracao

            self.historico_melhor.append(self.melhor_global)

        return (
            self.historico_melhor[-1],
            self.historico_fronteira,
            self.historico_melhor,
            self.historico_populacao,
        )
