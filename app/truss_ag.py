import numpy as np
import random
import math
from app.utils import Utils

class TrussAG:
    def __init__(self, tam_populacao=150, num_geracoes=200, fator_seguranca=2.0):
        self.tam_populacao = tam_populacao
        self.num_geracoes = num_geracoes
        self.utils = Utils()
        
        # Propriedades do Abaixador de Língua
        self.STICK_LENGTH_CM = 14.0
        self.SOBREPOSICAO_CM = 1.0
        
        # Orientação "De Cutelo" (Na Vertical)
        self.H_SECTION = 0.014  # Altura no eixo Y (1.4 cm)
        self.T_SECTION = 0.002  # Espessura no eixo Z (0.2 cm) por palito
        
        self.RHO = 510.0 # kg/m³ da madeira
        self.E_GPa = 3.5 # Módulo de Elasticidade (Bétula/Madeira)
        
        # Tensões Limites e Fator de Segurança
        self.FS = fator_seguranca
        self.SIGMA_T_ALLOW = 55e6 / self.FS # Tensão admissível Tração
        self.SIGMA_C_ALLOW = 35e6 / self.FS # Tensão admissível Compressão
        
        # Limites Geométricos do AG
        self.P_MIN, self.P_MAX = 3.0, 14.0
        self.H_MIN, self.H_MAX = 3.0, 14.0
        self.W_MIN, self.W_MAX = 4.0, 14.0
        
        self.populacao = [self.gerar_cromossomo() for _ in range(self.tam_populacao)]

    def reparar_paineis(self, p):
        """Garante que a soma dos painéis horizontais (vão livre) seja exatos 40 cm"""
        p = np.clip(p, self.P_MIN, self.P_MAX)
        diff = 40.0 - np.sum(p)
        for _ in range(10):
            if abs(diff) < 1e-9: break
            p += diff / 4.0
            p = np.clip(p, self.P_MIN, self.P_MAX)
            diff = 40.0 - np.sum(p)
        return p

    def gerar_cromossomo(self):
        p = self.reparar_paineis(np.random.uniform(self.P_MIN, self.P_MAX, 4))
        h = np.random.uniform(self.H_MIN, self.H_MAX, 5)
        w = random.uniform(self.W_MIN, self.W_MAX)
        n = np.random.randint(1, 4, 17) # 1, 2 ou 3 palitos por barra
        return {'p': p, 'h': h, 'w': w, 'n': n, 'fitness': 0.0}

    def calcular_palitos_barra(self, L_cm, n_camadas):
        """
        Calcula palitos necessários descontando a sobreposição de 1 cm nas emendas.
        """
        if L_cm <= self.STICK_LENGTH_CM:
            k_longitudinal = 1
        else:
            comprimento_util_extra = self.STICK_LENGTH_CM - self.SOBREPOSICAO_CM
            k_longitudinal = math.ceil((L_cm - self.STICK_LENGTH_CM) / comprimento_util_extra) + 1
            
        return k_longitudinal * n_camadas

    def calcular_capacidade_barra(self, force_unit, n_camadas, length_m):
        """Calcula a carga limite teórica considerando a orientação na VERTICAL e a colagem."""
        # Seção composta (palitos colados funcionam como uma única viga)
        espessura_total = n_camadas * self.T_SECTION
        area = self.H_SECTION * espessura_total
        
        if force_unit > 0.0:
            return self.SIGMA_T_ALLOW * area # Falha por Tração
        
        # Flambagem de Euler (Compressão)
        # Como estão colados na vertical, a flambagem ocorrerá no eixo mais fraco (eixo Z, para o lado).
        # A inércia de uma seção retangular é (base * altura^3) / 12
        # Eixo fraco: base é 1.4 cm (H_SECTION), altura contra flambagem é a espessura total.
        inercia_minima = (self.H_SECTION * (espessura_total ** 3)) / 12.0
        
        euler = (math.pi ** 2) * (self.E_GPa * 1e9) * inercia_minima / (length_m ** 2)
        
        return min(self.SIGMA_C_ALLOW * area, euler)

    def avaliar_fitness(self, ind):
        ind['p'] = self.reparar_paineis(ind['p'])
        nos, barras, comprimentos_m, angulos = self.utils.calcular_comprimentos_howe(ind['p'], ind['h'])
        
        axial_unit = self.utils.calcular_reacoes_e_forcas(nos, barras, angulos)
        if axial_unit is None:
            return 0.0 
            
        comprimentos_cm = np.array(comprimentos_m) * 100.0
        comprimentos_fisicos_cm = comprimentos_cm.copy()
        
        # Adiciona a sobra de 1.5 cm nas barras extremas do banzo inferior
        comprimentos_fisicos_cm[0] += 1.5
        comprimentos_fisicos_cm[3] += 1.5
        
        total_palitos_portico = 0
        for i in range(17):
            n_camadas = ind['n'][i]
            L_cm = comprimentos_fisicos_cm[i]
            total_palitos_portico += self.calcular_palitos_barra(L_cm, n_camadas)
            
        w_cm = ind['w']
        palitos_por_travamento = self.calcular_palitos_barra(w_cm, 1)
        total_palitos_travamentos = 10 * palitos_por_travamento
        
        total_palitos_3d = 2 * total_palitos_portico + total_palitos_travamentos
        
        vol_um_palito_m3 = (self.STICK_LENGTH_CM / 100.0) * self.H_SECTION * self.T_SECTION
        massa_um_palito_kg = self.RHO * vol_um_palito_m3
        massa_g = total_palitos_3d * massa_um_palito_kg * 1000.0
        
        if massa_g >= 600.0 or total_palitos_3d > 150:
            return 0.001

        rupturas = []
        for i, coeff in enumerate(axial_unit):
            if abs(coeff) > 1e-9:
                capacidade = self.calcular_capacidade_barra(coeff, ind['n'][i], comprimentos_m[i])
                rupturas.append(capacidade / abs(coeff))
                
        carga_max_portico = min(rupturas)
        carga_max_3d_kg = (2.0 * carga_max_portico) / 9.81
        
        ind['fitness'] = carga_max_3d_kg / massa_g
        ind['carga_kg'] = carga_max_3d_kg
        ind['massa_g'] = massa_g
        ind['palitos_total'] = total_palitos_3d
        return ind['fitness']

    def cruzar(self, p1, p2):
        c1, c2 = self.gerar_cromossomo(), self.gerar_cromossomo()
        alpha = 0.35
        c1['p'] = self.reparar_paineis(p1['p'] * alpha + p2['p'] * (1 - alpha))
        c2['p'] = self.reparar_paineis(p2['p'] * alpha + p1['p'] * (1 - alpha))
        c1['h'] = p1['h'] * alpha + p2['h'] * (1 - alpha)
        c2['h'] = p2['h'] * alpha + p1['h'] * (1 - alpha)
        c1['w'] = (p1['w'] + p2['w']) / 2.0
        c2['w'] = (p1['w'] + p2['w']) / 2.0
        
        mask = np.random.rand(17) > 0.5
        c1['n'] = np.where(mask, p1['n'], p2['n'])
        c2['n'] = np.where(mask, p2['n'], p1['n'])
        return c1, c2

    def iniciar(self):
        for gen in range(self.num_geracoes):
            for ind in self.populacao:
                self.avaliar_fitness(ind)
                
            self.populacao.sort(key=lambda x: x['fitness'], reverse=True)
            
            nova_geracao = self.populacao[:20] 
            while len(nova_geracao) < self.tam_populacao:
                p1, p2 = random.sample(self.populacao[:50], 2)
                f1, f2 = self.cruzar(p1, p2)
                nova_geracao.extend([f1, f2])
            
            self.populacao = nova_geracao[:self.tam_populacao]
            
        return self.populacao[0]