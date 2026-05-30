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
        self.tam_populacao = tam_populacao
        self.p_mutacao = p_mutacao
        self.p_crossover = p_crossover
        self.num_geracoes = num_geracoes
        self.P1 = P1
        self.P2 = P2
        self.utils = Utils()
        self.tam_cromossomo = self.barras_base + self.barras_verticais + self.barras_total
        self.populacao = [self.gerar_cromossomo() for _ in range(self.tam_populacao)]
        self.historico_fronteiras = []
        self.historico_melhor_cromossomo = []

    def iniciar(self):
        for gen in range(self.num_geracoes):
            fitnesses = [self.fitness(individuo) for individuo in self.populacao]

            fronteira, avaliacoes_fronteira = self.obter_fronteira_pareto(self.populacao, fitnesses)
            
            fronteira_real = []
            for ind in fronteira:
                try:
                    f_real, _, desloc_total, comp, v_areas = self.calcular_deslocamento(ind)
                    vol = sum(L * A for L, A in zip(comp, v_areas))
                    fronteira_real.append([vol, abs(desloc_total)])
                except Exception:
                    pass
            self.historico_fronteiras.append(fronteira_real)

            if len(fronteira) > 0:
                melhor_cromossomo = fronteira[0]
                self.historico_melhor_cromossomo.append(melhor_cromossomo.copy())
            else:
                self.historico_melhor_cromossomo.append(self.populacao[0].copy())

            print(f"Geração {gen + 1} | Indivíduos na Fronteira: {len(fronteira)}")
            
            nova_populacao = []
            num_elite = max(1, int(self.tam_populacao * 0.05))
            for elitismo in fronteira[:num_elite]:
                nova_populacao.append(elitismo.copy())

            while len(nova_populacao) < self.tam_populacao:
                pai = self.torneio(fitnesses)
                mae = self.torneio(fitnesses)

                primogenito, ultimogenito = self.crossover(pai, mae)

                self.mutar(primogenito)
                self.mutar(ultimogenito)

                nova_populacao.append(primogenito)

                if len(nova_populacao) < self.tam_populacao:
                    nova_populacao.append(ultimogenito)

            self.populacao = nova_populacao

        avaliacoes_finais = [self.fitness(ind) for ind in self.populacao]
        fronteira_final, _ = self.obter_fronteira_pareto(self.populacao, avaliacoes_finais)

        return fronteira_final, self.historico_melhor_cromossomo, self.historico_fronteiras

    def decodificar_cromossomo(self, cromossomo:np.ndarray):
        l_bases = cromossomo[0:self.barras_base]
        l_verticais = cromossomo[self.barras_base:self.barras_base+self.barras_verticais]
        vetor_areas = cromossomo[self.barras_base+self.barras_verticais:]

        return l_bases, l_verticais, vetor_areas
    
    def calcular_deslocamento(self, cromossomo: np.ndarray):
        l_bases, l_verticais, vetor_areas = self.decodificar_cromossomo(cromossomo)

        nos, barras, comprimentos, angulos = self.utils.calcular_comprimentos_howe(l_bases, l_verticais)
        f_real, reacoes_reais, f_virtual = self.utils.calcular_reacoes_e_forcas(nos, barras, angulos, self.forcas_externas, self.forcas_virtuais)
        resultados, deslocamento_total = self.utils.calcular_deslocamento_virtual(barras, comprimentos, vetor_areas, f_real, f_virtual, self.E_GPa)

        return f_real, resultados, deslocamento_total, comprimentos, vetor_areas
    
    def calcular_penalidades(self, tensoes: list[float], deslocamento_real: float):
        delta_max = 10.0
        T_max = 100.0
        
        penalidade_deslocamento = max(0, abs(deslocamento_real) - delta_max) * self.P1
        soma_penalidade_tensao = sum(max(0, T - T_max)**2 for T in tensoes)
        penalidade_tensao = self.P2 * soma_penalidade_tensao
        
        return penalidade_deslocamento + penalidade_tensao

    def fitness(self, cromossomo: np.ndarray):
        try:
            f_real, _, deslocamento_total, comprimentos, vetor_areas = self.calcular_deslocamento(cromossomo)
        except Exception:
            return 1e9, 1e9

        volume = sum(L * A for L, A in zip(comprimentos, vetor_areas))
        tensoes = [abs(N) / A for N, A in zip(f_real, vetor_areas)]
        tensao_max = max(tensoes) if tensoes else 0.0
        
        vol_ref = 0.01      # m³
        T_ref   = 1e5       # kN/m²
 
        f1_norm = volume / vol_ref
        f2_norm = tensao_max / T_ref
 
        delta_max = 10.0    # mm
        T_max     = 1e5     # kN/m²
 
        pen_desl  = max(0.0, abs(deslocamento_total) - delta_max) / delta_max * self.P1 / vol_ref
        pen_tens  = sum(max(0.0, t - T_max) / T_ref for t in tensoes) * self.P2 / T_ref
 
        penalidade = pen_desl + pen_tens
 
        f1 = f1_norm + penalidade
        f2 = f2_norm + penalidade
 
        return f1, f2

    
    def torneio(self, avaliacoes, num_competidores: int = 3):
        participantes = random.sample(range(self.tam_populacao), num_competidores)
        melhor = participantes[0]
        for i in range(1, num_competidores):
            cand = participantes[i]
            if self.dominante(avaliacoes[cand], avaliacoes[melhor]):
                melhor = cand
        return self.populacao[melhor].copy()

    def dominante(self, obj_a, obj_b):
        return (obj_a[0] <= obj_b[0] and obj_a[1] <= obj_b[1]) and (obj_a[0] < obj_b[0] or obj_a[1] < obj_b[1])

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
        l_bases = [random.randint(1, 4) for _ in range(self.barras_base)]
        l_verticais = [random.randint(1, 5) for _ in range(self.barras_verticais)]
        areas = [np.random.uniform(0.0001, 0.0005) for _ in range(self.barras_total)]
        
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
                if i < self.barras_base:
                    individuo[i] = random.randint(1, 4)
                elif i < self.barras_base + self.barras_verticais:
                    individuo[i] = random.randint(1, 5)
                else:
                    individuo[i] = np.random.uniform(0.0001, 0.0005)