import numpy as np
import random
from app.utils import Utils

class TrussPalitoAG:
    def __init__(self, tam_populacao=150, num_geracoes=200, p_mutacao=0.20, p_crossover=0.85):
        self.tam_populacao  = tam_populacao
        self.num_geracoes   = num_geracoes
        self.p_mutacao      = p_mutacao
        self.p_crossover    = p_crossover
        self.utils          = Utils()
        self.num_barras     = 17
        self.num_nos        = 10  # travamentos: 1 por nó
        self.tam_cromossomo = 27  # 4 bases + 5 alturas + 17 seções + 1 trav

        # Constantes físicas
        self.E_Pa        = 3500 * 1e6
        self.rho         = 510.0
        self.sigma_t_adm = (55.0 * 1e6) / 2.0
        self.sigma_c_adm = (35.0 * 1e6) / 2.0

        # Geometria do palito (metros)
        self.L_max_palito   = 0.114
        self.area_palito_m2 = 0.010 * 0.002  # 2e-5 m²

    def gerar_cromossomo(self):
        for _ in range(10000):
            geom_bases = [random.uniform(0.03, 0.114) for _ in range(4)]
            soma = sum(geom_bases)
            if soma > 0:
                geom_bases = [(v / soma) * 0.40 for v in geom_bases]
            # calcula via Pitágoras a altura máxima que cada montante pode assumir sem estourar o tamanho do palito
            h_maximos = [
                np.sqrt(max(0.0, self.L_max_palito**2 - geom_bases[0]**2)),
                np.sqrt(max(0.0, self.L_max_palito**2 - geom_bases[1]**2)),
                self.L_max_palito,
                np.sqrt(max(0.0, self.L_max_palito**2 - geom_bases[2]**2)),
                np.sqrt(max(0.0, self.L_max_palito**2 - geom_bases[3]**2)),
            ]

            if any(hm < 0.03 for hm in h_maximos):
                continue
           
            # sorteia alturas reais abaixo do limite calculado
            geom_alturas = [random.uniform(0.03, hm) for hm in h_maximos]
            # escolha discreta de palitos por barra: 1, 2 ou 3 palitos
            secoes       = [float(random.choice([1, 2, 3])) for _ in range(17)]
            # espessura do travamento tridimensional entre as duas faces da ponte
            n_trav       = [float(random.choice([1, 2, 3]))]
            return np.array(geom_bases + geom_alturas + secoes + n_trav, dtype=float)
        # segurança estrutural caso falhe o sorteio
        return np.array([0.10, 0.10, 0.10, 0.10, 0.04, 0.04, 0.04, 0.04, 0.04] + [1.0] * 17 + [1.0], dtype=float)

    def decodificar(self, cromossomo):
        l_bases     = cromossomo[0:4].copy()
        l_verticais = cromossomo[4:9].copy()
        secoes_num  = np.round(cromossomo[9:26]).astype(int)
        secoes_num  = np.clip(secoes_num, 1, 3)
        n_trav      = int(np.clip(round(cromossomo[26]), 1, 3))

        # soma do vão = 40 cm
        soma_p = sum(l_bases)
        if soma_p > 0:
            l_bases = (l_bases / soma_p) * 0.40
        else:
            l_bases = np.array([0.10, 0.10, 0.10, 0.10])

        return l_bases, l_verticais, secoes_num, n_trav

    def avaliar_indivíduo(self, cromossomo):
        l_bases, l_verticais, secoes_num, n_trav = self.decodificar(cromossomo)
       
        # garantir que nenhuma base ou altura esteja fora dos limites físicos do palito
        if any(p < 0.03 or p > 0.114 for p in l_bases):
            return 0.0, 9999.0, 0.0, False
        if any(h < 0.03 or h > 0.114 for h in l_verticais):
            return 0.0, 9999.0, 0.0, False

        nos, barras, comprimentos, angulos = \
            self.utils.calcular_comprimentos_howe_3d(l_bases, l_verticais)

        if any(L > self.L_max_palito + 1e-9 for L in comprimentos):
            return 0.0, 9999.0, 0.0, False

        # total de palitos = 2 faces + travamentos (10 nós × n_trav)
        total_palitos = sum(secoes_num) * 2 + self.num_nos * n_trav
        # máximo de 150 palitos
        if total_palitos > 150:
            return 0.0, 9999.0, 0.0, False

        vetor_areas   = secoes_num * self.area_palito_m2
        largura_w_m = n_trav * 0.010 # largura do travamento (1 cm por palito)

        # massa: 2 faces + travamentos
        massa_faces_kg = self.rho * sum(L * A for L, A in zip(comprimentos, vetor_areas))
        area_trav      = n_trav * self.area_palito_m2
        massa_trav_kg  = self.rho * self.num_nos * largura_w_m * area_trav
        massa_total_g  = (massa_faces_kg * 2.0 + massa_trav_kg) * 1000.0

        # elimina soluções muito pesadas
        if massa_total_g > 400.0:
            return 0.0, 9999.0, 0.0, False

        try:
            # forças reais e virtuais para o cálculo do deslocamento no nó C (0.5N em cada face)
            f_reais    = self.utils.resolver_estatica(nos, barras, angulos, [(2, 0.0, -0.5)])
            f_virtuais = self.utils.resolver_estatica(nos, barras, angulos, [(2, 0.0, -0.5)])
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 9999.0, 0.0, False
       
        # encontra qual das 17 barras vai estourar primeiro por tensão admissível
        P_ruptura_sistema_N = np.inf
        for k in range(self.num_barras):
            f_u = f_reais[k]
            if abs(f_u) < 1e-6:
                continue
            # escolhe o limite de ruptura baseado no sentido do esforço (Tração vs Compressão)
            F_adm   = self.sigma_t_adm * vetor_areas[k] if f_u > 0 \
                      else self.sigma_c_adm * vetor_areas[k]
            P_barra = F_adm / abs(f_u)
            if P_barra < P_ruptura_sistema_N:
                P_ruptura_sistema_N = P_barra # o sistema quebra no elo mais fraco

        P_ruptura_kg = (P_ruptura_sistema_N * 2.0) / 9.81 # converte a capacidade de Newton para quilogramas força

        f_real_limite = f_reais * P_ruptura_sistema_N
        desl_m = self.utils.calcular_deslocamento_ptv(
            barras, comprimentos, vetor_areas,
            f_real_limite, f_virtuais, self.E_Pa
        )
        deslocamento_mm = abs(desl_m) * 1000.0

        return P_ruptura_kg, massa_total_g, deslocamento_mm, True

    def fitness(self, cromossomo):
        # Objetivo 1: Maximizar Carga de Ruptura
        # Objetivo 2: Minimizar Peso Total (Massa)
        P_rup, m_total, _, viavel = self.avaliar_indivíduo(cromossomo)
        if not viavel or P_rup <= 0:
            return (0.0, 9999.0)
        return (-P_rup, m_total)

    def dominante(self, a, b):
        # indivíduo 'a' domina o 'b' se ele for melhor
        # ou igual em todos os objetivos e estritamente superior em pelo menos um deles
        return (a[0] <= b[0] and a[1] <= b[1]) and \
               (a[0] <  b[0] or  a[1] <  b[1])

    def fronteira_pareto(self, populacao):
        scores    = [self.fitness(ind) for ind in populacao]
        factiveis = [(i, s) for i, s in enumerate(scores) if s[1] < 9999.0]
        nd = []
        for i, si in factiveis:
            dominado = any(
                self.dominante(sj, si)
                for j, sj in factiveis if j != i
            )
            if not dominado:
                nd.append(i)
        return nd, scores

    def torneio_pareto(self, populacao, scores):
        i1, i2 = random.sample(range(len(populacao)), 2)
        s1, s2  = scores[i1], scores[i2]
        if self.dominante(s1, s2):
            return populacao[i1].copy()
        elif self.dominante(s2, s1):
            return populacao[i2].copy()
        else:
            return populacao[i1].copy() if s1[1] <= s2[1] else populacao[i2].copy()

    def crossover(self, pai, mae):
        if random.random() >= self.p_crossover:
            return pai.copy(), mae.copy()
        ponto = random.randint(1, self.tam_cromossomo - 1)
        f1 = np.concatenate([pai[:ponto], mae[ponto:]])
        f2 = np.concatenate([mae[:ponto], pai[ponto:]])
        return f1, f2

    def mutar(self, individuo):
        for i in range(self.tam_cromossomo):
            if random.random() < self.p_mutacao:
                if i < 4:        # painéis
                    individuo[i] += random.gauss(0, 0.005)
                    individuo[i]  = max(0.03, min(0.114, individuo[i]))
                elif i < 9:      # alturas
                    individuo[i] += random.gauss(0, 0.005)
                    individuo[i]  = max(0.03, min(0.114, individuo[i]))
                elif i < 26:     # seções das barras
                    individuo[i]  = float(random.choice([1, 2, 3]))
                else:            # travamento
                    individuo[i]  = float(random.choice([1, 2, 3]))