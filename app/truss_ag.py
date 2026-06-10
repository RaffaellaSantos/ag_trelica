import numpy as np
import random
import math
from app.utils import Utils

class Individuo:
    """Estrutura de dados que armazena as características genéticas e físicas da treliça."""

    def __init__(self, p, h, w, n):
        """Inicializa os genes e os parâmetros multiobjetivo."""
        self.p = p
        self.h = h
        self.w = w
        self.n = n
        self.massa_g = float('inf')
        self.tensao_max_mpa = float('inf')
        self.fator_seguranca = 0.0
        self.valido = False
        self.rank = -1
        self.distancia_aglomeracao = 0.0
        self.dominados = []
        self.num_dominantes = 0
        self.dados_extras = {}

class TrussAG:
    """Implementa o Algoritmo Genético Multiobjetivo (Fronteira de Pareto) para Treliças 3D."""

    def __init__(self, tam_populacao=150, num_geracoes=150, carga_projeto_kg=15.0):
        """Configura as propriedades físicas, penalidades e limiares do algoritmo."""
        self.tam_populacao = tam_populacao
        self.num_geracoes = num_geracoes
        self.carga_projeto_kg = carga_projeto_kg
        self.utils = Utils()
        
        # --- ATUALIZADO CONFORME PDF DA ATIVIDADE 9 ---
        self.L_PALITO_CM = 11.4       # Comprimento do palito 
        self.H_SECTION_M = 0.010      # Largura do palito (1,0 cm) 
        self.T_SECTION_M = 0.002      # Espessura do palito (0,2 cm) 
        self.SOBREPOSICAO_CM = 1.0    
        self.RHO = 510.0
        self.E_PA = 3.5e9
        self.SIGMA_T_YIELD = 55e6
        self.SIGMA_C_YIELD = 35e6
        
        # Limites contínuos atualizados para o máximo de 11,4 cm 
        self.P_MIN, self.P_MAX = 3.0, 11.4
        self.H_MIN, self.H_MAX = 3.0, 11.4
        self.W_MIN, self.W_MAX = 4.0, 11.4
        # ----------------------------------------------
        
        self.populacao = [self.gerar_individuo() for _ in range(self.tam_populacao)]
        self.historico_fronteira = []
        self.historico_melhor = []

    def gerar_individuo(self):
        """Gera uma estrutura cromossômica aleatória dentro do domínio válido."""
        p = np.random.uniform(self.P_MIN, self.P_MAX, 4)
        h = np.random.uniform(self.H_MIN, self.H_MAX, 5)
        w = random.uniform(self.W_MIN, self.W_MAX)
        n = np.random.randint(1, 4, 17)
        return Individuo(p, h, w, n)

    def calcular_fator_seguranca(self, tensao_atuante_pa, eh_tracao):
        """Calcula o quão segura é a treliça para aguentar a tensão atual."""
        if tensao_atuante_pa <= 1e-9:
            return float('inf')
        limite = self.SIGMA_T_YIELD if eh_tracao else self.SIGMA_C_YIELD
        return limite / tensao_atuante_pa

    def calcular_capacidade_compressao(self, n_camadas, length_m):
        """Calcula a resistência à flambagem de Euler para o eixo fraco na orientação vertical."""
        base_m = n_camadas * self.T_SECTION_M
        altura_m = self.H_SECTION_M
        inercia_yy = (altura_m * (base_m ** 3)) / 12.0
        inercia_xx = (base_m * (altura_m ** 3)) / 12.0
        inercia_min = min(inercia_yy, inercia_xx)
        forca_euler = (math.pi ** 2) * self.E_PA * inercia_min / (length_m ** 2)
        area = base_m * altura_m
        forca_esmagamento = self.SIGMA_C_YIELD * area
        return min(forca_euler, forca_esmagamento) / area

    def avaliar(self, ind):
        """Executa a rotina de simulação, determinando a massa e as tensões, validando restrições."""
        p_total = np.sum(ind.p)
        if p_total == 0:
            p_total = 1.0
        ind.p = (ind.p / p_total) * 35.0
        nos, barras, comprimentos_cm, angulos = self.utils.calcular_comprimentos_howe(ind.p, ind.h)
        carga_por_portico = self.carga_projeto_kg / 2.0
        esforcos_n = self.utils.calcular_reacoes_e_forcas(nos, barras, angulos, carga_por_portico)
        comprimentos_fisicos_cm = np.array(comprimentos_cm, dtype=float).copy()
        comprimentos_fisicos_cm[0] += 0.5
        comprimentos_fisicos_cm[3] += 0.5
        palitos_por_portico = 0
        l_util = self.L_PALITO_CM - self.SOBREPOSICAO_CM
        for i in range(17):
            L_cm = comprimentos_fisicos_cm[i]
            if L_cm <= self.L_PALITO_CM:
                k = 1
            else:
                k = math.ceil((L_cm - self.L_PALITO_CM) / l_util) + 1
            palitos_por_portico += k * ind.n[i]
        palitos_travamento = 1 if ind.w <= self.L_PALITO_CM else math.ceil((ind.w - self.L_PALITO_CM) / l_util) + 1
        total_travamentos = 10 * palitos_travamento
        total_palitos = 2 * palitos_por_portico + total_travamentos
        vol_palito = (self.L_PALITO_CM / 100.0) * self.H_SECTION_M * self.T_SECTION_M
        massa_g = total_palitos * (self.RHO * vol_palito) * 1000.0
        ind.massa_g = massa_g
        ind.dados_extras = {
            'comprimentos_cm': comprimentos_fisicos_cm,
            'total_palitos': total_palitos,
            'esforcos_n': esforcos_n
        }
        if massa_g >= 600.0 or total_palitos > 150:
            ind.valido = False
            ind.massa_g = massa_g + 10000.0
            ind.tensao_max_mpa = 10000.0
            ind.fator_seguranca = 0.0
            return
        piores_tensoes = []
        fs_minimo = float('inf')
        for i, forca in enumerate(esforcos_n):
            area = ind.n[i] * self.T_SECTION_M * self.H_SECTION_M
            tensao_pa = abs(forca) / area
            comprimento_m = comprimentos_cm[i] / 100.0
            if forca > 1e-9:
                piores_tensoes.append(tensao_pa)
                fs = self.calcular_fator_seguranca(tensao_pa, True)
            elif forca < -1e-9:
                limite_comp_pa = self.calcular_capacidade_compressao(ind.n[i], comprimento_m)
                tensao_efetiva_pa = tensao_pa * (self.SIGMA_C_YIELD / limite_comp_pa)
                piores_tensoes.append(tensao_efetiva_pa)
                fs = limite_comp_pa / tensao_pa if tensao_pa > 0 else float('inf')
            else:
                piores_tensoes.append(0.0)
                fs = float('inf')
            if fs < fs_minimo:
                fs_minimo = fs
        ind.tensao_max_mpa = max(piores_tensoes) / 1e6
        ind.fator_seguranca = fs_minimo
        ind.valido = (fs_minimo >= 1.0)
        if not ind.valido:
            ind.tensao_max_mpa += 1000.0

    def classificar_pareto(self, populacao):
        """Aplica o processo de ordenação não-dominada do NSGA-II."""
        fronteiras = [[]]
        for p in populacao:
            p.dominados = []
            p.num_dominantes = 0
            for q in populacao:
                condicao_dominacao = (p.massa_g <= q.massa_g and p.tensao_max_mpa <= q.tensao_max_mpa) and \
                                     (p.massa_g < q.massa_g or p.tensao_max_mpa < q.tensao_max_mpa)
                condicao_dominada = (q.massa_g <= p.massa_g and q.tensao_max_mpa <= p.tensao_max_mpa) and \
                                    (q.massa_g < p.massa_g or q.tensao_max_mpa < p.tensao_max_mpa)
                if condicao_dominacao:
                    p.dominados.append(q)
                elif condicao_dominada:
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

    def cruzar_e_mutar(self, p1, p2):
        """Aplica crossover BLX-alpha contínuo e uniforme discreto, seguido de mutação gaussiana."""
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
            f.p = np.clip(f.p, self.P_MIN, self.P_MAX)
            f.h = np.clip(f.h, self.H_MIN, self.H_MAX)
            f.w = np.clip(f.w, self.W_MIN, self.W_MAX)
        return filhos[0], filhos[1]

    def iniciar(self):
        """Laço principal evolutivo gerenciando as gerações da Fronteira Pareto."""
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
                if len(front) <= para_proxima:
                    self.populacao.extend(front)
                    para_proxima -= len(front)
                else:
                    front_ordenada = sorted(front, key=lambda x: x.massa_g)
                    self.populacao.extend(front_ordenada[:para_proxima])
                    break
            front_atual = [[p.massa_g, p.tensao_max_mpa] for p in fronteiras[0] if p.valido]
            self.historico_fronteira.append(front_atual)
            
            validos = [p for p in self.populacao if p.valido]
            front_zero_validos = [p for p in fronteiras[0] if p.valido]
            
            # --- LÓGICA DO COTOVELO (KNEE POINT) ---
            if front_zero_validos:
                # 1. Encontra os extremos da fronteira para normalização
                min_massa = min(p.massa_g for p in front_zero_validos)
                max_massa = max(p.massa_g for p in front_zero_validos)
                min_tensao = min(p.tensao_max_mpa for p in front_zero_validos)
                max_tensao = max(p.tensao_max_mpa for p in front_zero_validos)

                # 2. Calcula a distância normalizada até o Ponto Ideal (Mínima Massa, Mínima Tensão)
                def dist_ideal(ind):
                    nm = (ind.massa_g - min_massa) / (max_massa - min_massa) if max_massa > min_massa else 0.0
                    nt = (ind.tensao_max_mpa - min_tensao) / (max_tensao - min_tensao) if max_tensao > min_tensao else 0.0
                    return (nm ** 2) + (nt ** 2)

                # 3. O melhor indivíduo é o que tem a menor distância (o cotovelo da curva)
                melhor_fator = min(front_zero_validos, key=dist_ideal)
                
            elif validos:
                # Fallback caso não haja válidos na primeira fronteira
                melhor_fator = max(validos, key=lambda x: (x.fator_seguranca * self.carga_projeto_kg) / x.massa_g)
            else:
                # Fallback se não houver treliças válidas (escolhe a menos pior)
                melhor_fator = min(self.populacao, key=lambda x: x.massa_g * x.tensao_max_mpa)
                
            self.historico_melhor.append(melhor_fator)
            
        return self.historico_melhor[-1], self.historico_fronteira, self.historico_melhor