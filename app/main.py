from app.truss_ag import TrussAG
from app.visualizer import Visualizer
import numpy as np

def main():
    """Inicia a análise, coordena o Algoritmo Genético e a exportação de dados."""
    populacao = 100
    geracoes = 80
    carga_kg = 15.0
    ag = TrussAG(tam_populacao=populacao, num_geracoes=geracoes, carga_projeto_kg=carga_kg)
    melhor_trelica, historico_fronteira, historico_melhor = ag.iniciar()
    print("="*60)
    print("RELATÓRIO DA MELHOR TRELIÇA 3D (Abaixador de Língua)")
    print("="*60)
    print(f"Massa Calculada        : {melhor_trelica.massa_g:.2f} g")
    print(f"Fator de Segurança     : {melhor_trelica.fator_seguranca:.3f}")
    print(f"Tensão Máx. Atuante    : {melhor_trelica.tensao_max_mpa:.2f} MPa")
    print(f"Palitos Utilizados     : {melhor_trelica.dados_extras['total_palitos']} unidades")
    print(f"Largura 3D (w)         : {melhor_trelica.w:.2f} cm")
    print(f"Painéis (p1..p4) cm    : {np.round(melhor_trelica.p, 2)}")
    print(f"Alturas (h1..h5) cm    : {np.round(melhor_trelica.h, 2)}")
    print(f"Camadas p/ Barra (n)   : {melhor_trelica.n}")
    vis = Visualizer()
    print("\nSalvando animação evolutiva em: best_truss_3d.gif")
    vis.gerar_animacao(historico_melhor, historico_fronteira, geracoes, "best_truss_3d.gif")
    print("Salvando imagem estrutural ótima em: best_truss_final.png")
    vis.gerar_imagem_otima(melhor_trelica, historico_fronteira[-1], geracoes, "best_truss_final.png")

if __name__ == "__main__":
    main()