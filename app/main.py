from app.truss_ag import TrussAG
from app.visualizer import Visualizer
import numpy as np
import math
import sys
import os
from datetime import datetime


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def main():
    populacao = 100
    geracoes = 90
    carga_kg = 15.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"relatorio_{timestamp}.txt"

    with open(nome_arquivo, "w", encoding="utf-8") as log:
        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, log)

        try:
            ag = TrussAG(tam_populacao=populacao, num_geracoes=geracoes, carga_projeto_kg=carga_kg)
            melhor_trelica, historico_fronteira, historico_melhor, historico_pop = ag.iniciar()

            print("="*60)
            print("RELATORIO DA MELHOR TRELICA 3D (HOWE)")
            print("="*60)
            carga_ruptura_kg = melhor_trelica.fator_seguranca * carga_kg
            print(f"Massa Calculada        : {melhor_trelica.massa_g:.2f} g")
            print(f"Fator de Seguranca     : {melhor_trelica.fator_seguranca:.3f}")
            print(f"Peso Maximo Suportado  : {carga_ruptura_kg:.2f} kg  ({carga_ruptura_kg * 9.81:.1f} N)")
            print(f"Eficiencia (Fit Real)  : {melhor_trelica.eficiencia:.4f} kg/g")
            print(f"Tensao Max. Atuante    : {melhor_trelica.tensao_max_mpa:.2f} MPa")
            print(f"Deslocamento (PTV)     : {melhor_trelica.dados_extras['deslocamento_mm']:.2f} mm")
            print(f"Palitos Utilizados     : {melhor_trelica.dados_extras['total_palitos']} unidades")
            print(f"Largura 3D (w)         : {melhor_trelica.w:.2f} cm")
            print(f"Paineis (p1..p4) cm    : {np.round(melhor_trelica.p, 2)} (Comprimento total: {np.sum(melhor_trelica.p):.2f} cm)")
            print(f"Alturas (h1..h5) cm    : {np.round(melhor_trelica.h, 2)}")
            print(f"Camadas p/ Barra (n)   : {melhor_trelica.n}")

            print("\nDetalhamento das Barras:")
            print(f"  {'Barra':<6} {'Geometrico':>12} {'Corte':>10} {'Camadas':>9} {'Pecas/camada':>14}")
            node_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            bar_conns = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9),
                         (0, 5), (6, 1), (7, 2), (8, 3), (4, 9), (5, 1), (6, 2), (8, 2), (9, 3)]

            L_PALITO = 14.0
            SOBREP = 4.0
            l_util = L_PALITO - SOBREP

            for idx, (i, j) in enumerate(bar_conns):
                name = f"{node_names[i]}-{node_names[j]}"
                geom = melhor_trelica.dados_extras['comprimentos_geometricos_cm'][idx]
                corte = melhor_trelica.dados_extras['comprimentos_cm'][idx]
                camadas = melhor_trelica.n[idx]
                if corte <= L_PALITO:
                    k = 1
                    obs = ""
                else:
                    k = math.ceil((corte - L_PALITO) / l_util) + 1
                    obs = " *** EMENDA ***"
                print(f"  {name:<6} {geom:>9.2f} cm {corte:>7.2f} cm {camadas:>9} {k:>14}{obs}")

            print(f"\nRelatorio salvo em: {os.path.abspath(nome_arquivo)}")

            vis = Visualizer()
            vis.gerar_animacao(historico_melhor, historico_fronteira, historico_pop, geracoes, "best_truss_3d.gif")
            todas_pops = [pt for pop in historico_pop for pt in pop]
            vis.gerar_imagem_otima(melhor_trelica, historico_fronteira[-1], todas_pops, geracoes, "best_truss_final.png")
        finally:
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()