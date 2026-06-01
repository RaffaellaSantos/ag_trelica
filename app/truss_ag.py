import numpy as np
import random
from app.utils import Utils


class TrussAG:
    def __init__(
        self,
        forcas_externas: list[list[float]],
        forcas_virtuais: list[list[float]],
        barras_base: int = 4,
        barras_verticais: int = 3,
        barras_total: int = 13,
        E_GPa: float = 200.0,
        densidade: float = 7870.0,
        sigma_escoamento: float = 300.0,
        fator_seguranca: float = 2.0,
        vao_livre_min: float = 8.0,
        tam_populacao: int = 100,
        p_mutacao: float = 0.1,
        p_crossover: float = 0.8,
        num_geracoes: int = 100,
        P1: float = 1e5,
        P2: float = 1e5,
    ):
        self.forcas_externas = forcas_externas
        self.forcas_virtuais = forcas_virtuais
        self.barras_base = barras_base
        self.barras_verticais = barras_verticais
        self.barras_total = barras_total
        self.E_GPa = E_GPa
        self.densidade = densidade
        self.sigma_escoamento = sigma_escoamento
        self.fator_seguranca = fator_seguranca
        self.sigma_admissivel = sigma_escoamento / fator_seguranca
        self.vao_livre_min = vao_livre_min
        self.tam_populacao = tam_populacao
        self.p_mutacao = p_mutacao
        self.p_crossover = p_crossover
        self.num_geracoes = num_geracoes
        self.P1 = P1
        self.P2 = P2
        self.utils = Utils()
        self.comprimentos_disponiveis = [1.0, 2.0, 3.0]
        self.areas_disponiveis = [3e-4, 4e-4, 5e-4]
        self.tam_cromossomo = self.barras_base + self.barras_verticais + self.barras_total
        self.populacao = [self.gerar_cromossomo() for _ in range(self.tam_populacao)]
        self.historico_nuvem = []
        self.historico_fronteira_plot = []
        self.historico_melhor_cromossomo = []
        self.historico_melhor_obj = []

    def iniciar(self):
        for gen in range(self.num_geracoes):
            objetivos = [self.fitness(ind) for ind in self.populacao]

            nuvem = []
            for ind in self.populacao:
                try:
                    massa, delta, factivel = self.avaliar_objetivos(ind)
                    nuvem.append([massa, delta, 1.0 if factivel else 0.0])
                except Exception:
                    pass
            self.historico_nuvem.append(nuvem)

            fronteira_plot = self.fronteira_factivel_real(self.populacao)
            self.historico_fronteira_plot.append(fronteira_plot)

            melhor_cromossomo = self.melhor_solucao(self.populacao)
            try:
                massa_b, delta_b = self.massa_e_deslocamento(melhor_cromossomo)
            except Exception:
                massa_b, delta_b = 0.0, 0.0
            self.historico_melhor_cromossomo.append(melhor_cromossomo.copy())
            self.historico_melhor_obj.append([massa_b, delta_b])

            print(f"Geracao {gen + 1} | Pontos na Fronteira factivel: {len(fronteira_plot)} | Melhor m: {massa_b:.2f} kg | delta_C: {delta_b:.3f} mm")

            frentes, ranks = self.ordenar_nao_dominado(objetivos)
            crowding = [0.0] * len(self.populacao)
            for frente in frentes:
                dist = self.distancia_aglomeracao(frente, objetivos)
                for idx in frente:
                    crowding[idx] = dist[idx]

            filhos = []
            while len(filhos) < self.tam_populacao:
                pai = self.torneio(ranks, crowding)
                mae = self.torneio(ranks, crowding)
                primogenito, ultimogenito = self.crossover(pai, mae)
                self.mutar(primogenito)
                self.mutar(ultimogenito)
                filhos.append(primogenito)
                if len(filhos) < self.tam_populacao:
                    filhos.append(ultimogenito)

            obj_filhos = [self.fitness(ind) for ind in filhos]
            combinada = self.populacao + filhos
            obj_comb = objetivos + obj_filhos

            frentes_c, _ = self.ordenar_nao_dominado(obj_comb)
            nova_populacao = []
            for frente in frentes_c:
                if len(nova_populacao) + len(frente) <= self.tam_populacao:
                    nova_populacao.extend(combinada[idx] for idx in frente)
                else:
                    dist = self.distancia_aglomeracao(frente, obj_comb)
                    ordenado = sorted(frente, key=lambda idx: dist[idx], reverse=True)
                    faltam = self.tam_populacao - len(nova_populacao)
                    nova_populacao.extend(combinada[idx] for idx in ordenado[:faltam])
                    break
            self.populacao = [ind.copy() for ind in nova_populacao]

        objetivos_finais = [self.fitness(ind) for ind in self.populacao]
        fronteira_final, _ = self.obter_fronteira_pareto(self.populacao, objetivos_finais)

        return (
            fronteira_final,
            self.historico_melhor_cromossomo,
            self.historico_nuvem,
            self.historico_melhor_obj,
            self.historico_fronteira_plot,
        )

    def decodificar_cromossomo(self, cromossomo: np.ndarray):
        l_bases = cromossomo[0:self.barras_base]
        l_verticais = cromossomo[self.barras_base:self.barras_base + self.barras_verticais]
        vetor_areas = cromossomo[self.barras_base + self.barras_verticais:]

        return l_bases, l_verticais, vetor_areas

    def calcular_deslocamento(self, cromossomo: np.ndarray):
        l_bases, l_verticais, vetor_areas = self.decodificar_cromossomo(cromossomo)

        nos, barras, comprimentos, angulos = self.utils.calcular_comprimentos_howe(l_bases, l_verticais)
        f_real, reacoes_reais, f_virtual = self.utils.calcular_reacoes_e_forcas(nos, barras, angulos, self.forcas_externas, self.forcas_virtuais)
        resultados, deslocamento_total = self.utils.calcular_deslocamento_virtual(barras, comprimentos, vetor_areas, f_real, f_virtual, self.E_GPa)

        return f_real, resultados, deslocamento_total, comprimentos, vetor_areas

    def massa_e_deslocamento(self, cromossomo: np.ndarray):
        f_real, _, deslocamento_total, comprimentos, vetor_areas = self.calcular_deslocamento(cromossomo)
        massa = self.densidade * sum(L * A for L, A in zip(comprimentos, vetor_areas))
        return massa, abs(deslocamento_total)

    def avaliar_objetivos(self, cromossomo: np.ndarray):
        f_real, _, deslocamento_total, comprimentos, vetor_areas = self.calcular_deslocamento(cromossomo)
        massa = self.densidade * sum(L * A for L, A in zip(comprimentos, vetor_areas))
        delta = abs(deslocamento_total)
        penalidade = self.calcular_penalidades(cromossomo, f_real, vetor_areas)
        return massa, delta, penalidade <= 1e-9

    def calcular_penalidades(self, cromossomo: np.ndarray, f_real, vetor_areas):
        tensoes_MPa = [abs(N) / A / 1000.0 for N, A in zip(f_real, vetor_areas)]
        viol_tensao = sum(max(0.0, t - self.sigma_admissivel) for t in tensoes_MPa)

        l_bases, _, _ = self.decodificar_cromossomo(cromossomo)
        vao = float(sum(l_bases))
        viol_vao = max(0.0, self.vao_livre_min - vao)

        return self.P1 * viol_tensao + self.P2 * viol_vao

    def fitness(self, cromossomo: np.ndarray):
        try:
            f_real, _, deslocamento_total, comprimentos, vetor_areas = self.calcular_deslocamento(cromossomo)
        except Exception:
            return 1e12, 1e12

        massa = self.densidade * sum(L * A for L, A in zip(comprimentos, vetor_areas))
        delta_c = abs(deslocamento_total)

        penalidade = self.calcular_penalidades(cromossomo, f_real, vetor_areas)

        f1 = massa + penalidade
        f2 = delta_c + penalidade

        return f1, f2

    def avaliar_completo(self, cromossomo: np.ndarray):
        f_real, resultados, deslocamento_total, comprimentos, vetor_areas = self.calcular_deslocamento(cromossomo)
        massa = self.densidade * sum(L * A for L, A in zip(comprimentos, vetor_areas))
        tensoes_MPa = [abs(N) / A / 1000.0 for N, A in zip(f_real, vetor_areas)]
        l_bases, l_verticais, _ = self.decodificar_cromossomo(cromossomo)

        return {
            'deslocamento_C_mm': abs(deslocamento_total),
            'massa_kg': massa,
            'areas_m2': [float(a) for a in vetor_areas],
            'comprimentos_m': [float(c) for c in comprimentos],
            'forcas_kN': [float(n) for n in f_real],
            'tensoes_MPa': tensoes_MPa,
            'vao_livre_m': float(sum(l_bases)),
            'sigma_admissivel_MPa': self.sigma_admissivel,
        }

    def fronteira_factivel_real(self, populacao):
        pontos = []
        for ind in populacao:
            try:
                massa, delta, factivel = self.avaliar_objetivos(ind)
                if factivel:
                    pontos.append((massa, delta))
            except Exception:
                continue
        nao_dominados = []
        for i, (mi, di) in enumerate(pontos):
            dominado = False
            for j, (mj, dj) in enumerate(pontos):
                if i != j and mj <= mi and dj <= di and (mj < mi or dj < di):
                    dominado = True
                    break
            if not dominado:
                nao_dominados.append((mi, di))
        nao_dominados = sorted(set(nao_dominados))
        return [[m, d] for m, d in nao_dominados]

    def melhor_solucao(self, candidatos):
        melhor = None
        melhor_score = np.inf
        for ind in candidatos:
            try:
                massa, delta = self.massa_e_deslocamento(ind)
                f_real, _, _, _, vetor_areas = self.calcular_deslocamento(ind)
                penal = self.calcular_penalidades(ind, f_real, vetor_areas)
                score = 0.1 * massa + delta + penal
                if score < melhor_score:
                    melhor_score = score
                    melhor = ind
            except Exception:
                continue
        if melhor is None:
            melhor = candidatos[0]
        return melhor

    def torneio(self, ranks, crowding):
        a, b = random.sample(range(len(self.populacao)), 2)
        if ranks[a] < ranks[b]:
            vencedor = a
        elif ranks[b] < ranks[a]:
            vencedor = b
        else:
            vencedor = a if crowding[a] >= crowding[b] else b
        return self.populacao[vencedor].copy()

    def dominante(self, obj_a, obj_b):
        return (obj_a[0] <= obj_b[0] and obj_a[1] <= obj_b[1]) and (obj_a[0] < obj_b[0] or obj_a[1] < obj_b[1])

    def ordenar_nao_dominado(self, objetivos):
        n = len(objetivos)
        dominados_por = [[] for _ in range(n)]
        contador_dominacao = [0] * n
        rank = [0] * n
        frentes = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self.dominante(objetivos[p], objetivos[q]):
                    dominados_por[p].append(q)
                elif self.dominante(objetivos[q], objetivos[p]):
                    contador_dominacao[p] += 1
            if contador_dominacao[p] == 0:
                rank[p] = 0
                frentes[0].append(p)

        i = 0
        while frentes[i]:
            proxima = []
            for p in frentes[i]:
                for q in dominados_por[p]:
                    contador_dominacao[q] -= 1
                    if contador_dominacao[q] == 0:
                        rank[q] = i + 1
                        proxima.append(q)
            i += 1
            frentes.append(proxima)
        frentes.pop()
        return frentes, rank

    def distancia_aglomeracao(self, frente, objetivos):
        dist = {idx: 0.0 for idx in frente}
        tamanho = len(frente)
        if tamanho == 0:
            return dist
        for m in range(2):
            ordenado = sorted(frente, key=lambda idx: objetivos[idx][m])
            dist[ordenado[0]] = float('inf')
            dist[ordenado[-1]] = float('inf')
            f_min = objetivos[ordenado[0]][m]
            f_max = objetivos[ordenado[-1]][m]
            faixa = f_max - f_min
            if faixa == 0:
                continue
            for k in range(1, tamanho - 1):
                dist[ordenado[k]] += (objetivos[ordenado[k + 1]][m] - objetivos[ordenado[k - 1]][m]) / faixa
        return dist

    def obter_fronteira_pareto(self, populacao, avaliacoes):
        fronteira = []
        avaliacoes_fronteira = []
        for i in range(len(populacao)):
            foi_dominado = False
            for j in range(len(populacao)):
                if i != j and self.dominante(avaliacoes[j], avaliacoes[i]):
                    foi_dominado = True
                    break
            if not foi_dominado:
                fronteira.append(populacao[i])
                avaliacoes_fronteira.append(avaliacoes[i])
        return fronteira, avaliacoes_fronteira

    def gerar_cromossomo(self):
        l_bases = [random.choice(self.comprimentos_disponiveis) for _ in range(self.barras_base)]
        l_verticais = [random.choice(self.comprimentos_disponiveis) for _ in range(self.barras_verticais)]
        areas = [random.choice(self.areas_disponiveis) for _ in range(self.barras_total)]

        cromossomo = l_bases + l_verticais + areas
        return np.array(cromossomo, dtype=float)

    def crossover(self, pai: np.ndarray, mae: np.ndarray):
        if np.random.rand() < self.p_crossover:
            ponto = np.random.randint(1, self.tam_cromossomo - 1)
            primogenito = np.concatenate([pai[:ponto], mae[ponto:]])
            ultimogenito = np.concatenate([mae[:ponto], pai[ponto:]])
            return primogenito, ultimogenito
        return pai.copy(), mae.copy()

    def mutar(self, individuo: np.ndarray):
        for i in range(self.tam_cromossomo):
            if np.random.rand() < self.p_mutacao:
                if i < self.barras_base + self.barras_verticais:
                    individuo[i] = random.choice(self.comprimentos_disponiveis)
                else:
                    individuo[i] = random.choice(self.areas_disponiveis)