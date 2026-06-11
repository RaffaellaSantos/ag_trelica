from app.truss_ag import TrussAG
from app.visualizer import Visualizer
import numpy as np

def main():
    populacao = 100
    geracoes = 90
    carga_kg = 15.0
    ag = TrussAG(tam_populacao=populacao, num_geracoes=geracoes, carga_projeto_kg=carga_kg)
    melhor_trelica, historico_fronteira, historico_melhor, historico_pop = ag.iniciar()
    
    print("="*60)
    print("RELATORIO DA MELHOR TRELICA 3D (HOWE)")
    print("="*60)
    print(f"Massa Calculada        : {melhor_trelica.massa_g:.2f} g")
    print(f"Fator de Seguranca     : {melhor_trelica.fator_seguranca:.3f}")
    print(f"Carga de Ruptura       : {melhor_trelica.fator_seguranca * carga_kg:.2f} kg")
    print(f"Eficiencia (Fit Real)  : {melhor_trelica.eficiencia:.4f} kg/g")
    print(f"Tensao Max. Atuante    : {melhor_trelica.tensao_max_mpa:.2f} MPa")
    print(f"Deslocamento (PTV)     : {melhor_trelica.dados_extras['deslocamento_mm']:.2f} mm")
    print(f"Palitos Utilizados     : {melhor_trelica.dados_extras['total_palitos']} unidades")
    print(f"Largura 3D (w)         : {melhor_trelica.w:.2f} cm")
    print(f"Paineis (p1..p4) cm    : {np.round(melhor_trelica.p, 2)} (Total: {np.sum(melhor_trelica.p):.2f} cm)")
    print(f"Alturas (h1..h5) cm    : {np.round(melhor_trelica.h, 2)}")
    print(f"Camadas p/ Barra (n)   : {melhor_trelica.n}")
    
    print("\nDetalhamento das Barras:")
    node_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    bar_conns = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), 
                 (0, 5), (6, 1), (7, 2), (8, 3), (4, 9), (5, 1), (6, 2), (8, 2), (9, 3)]
    
    for idx, (i, j) in enumerate(bar_conns):
        name = f"{node_names[i]}-{node_names[j]}"
        length = melhor_trelica.dados_extras['comprimentos_cm'][idx]
        sticks = melhor_trelica.n[idx]
        print(f"{name}: {length:.2f} cm ({sticks} palito{'s' if sticks > 1 else ''})")
        
    vis = Visualizer()
    vis.gerar_animacao(historico_melhor, historico_fronteira, historico_pop, geracoes, "best_truss_3d.gif")
    todas_pops = [pt for pop in historico_pop for pt in pop]
    vis.gerar_imagem_otima(melhor_trelica, historico_fronteira[-1], todas_pops, geracoes, "best_truss_final.png")

if __name__ == "__main__":
    main()