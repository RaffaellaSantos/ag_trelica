from app.truss_ag import TrussAG
import numpy as np

def main():
    # Inicializando com Fator de Segurança 2.0 
    ag = TrussAG(tam_populacao=150, num_geracoes=200, fator_seguranca=2.0)
    
    print("Iniciando Otimização...")
    melhor_trelica = ag.iniciar()
    
    print("\n" + "="*60)
    print("RELATÓRIO DA MELHOR TRELIÇA 3D (Abaixador de Língua)")
    print("="*60)
    print(f"Eficiência (Fitness)   : {melhor_trelica['fitness']:.5f} kg/g")
    print(f"Carga Suportada (FS=2) : {melhor_trelica['carga_kg']:.2f} kg")
    print(f"Massa Calculada        : {melhor_trelica['massa_g']:.2f} g")
    print(f"Palitos Utilizados     : {melhor_trelica['palitos_total']} unidades")
    print(f"Largura 3D (w)         : {melhor_trelica['w']:.2f} cm")
    
    print("\n--- GEOMETRIA ---")
    print(f"Painéis (p1..p4) cm    : {np.round(melhor_trelica['p'], 2)}")
    print(f"Alturas (h1..h5) cm    : {np.round(melhor_trelica['h'], 2)}")
    print(f"Camadas p/ Barra (n)   : {melhor_trelica['n']}")
    
    print("\n--- DETALHES DE CONSTRUÇÃO ---")
    print("* O vão longitudinal para os cálculos de apoio foi mantido em 40 cm.")
    print("* O banzo inferior tem um comprimento físico total de 43 cm.")
    print("* O algoritmo alocou palitos prevendo exatamente 1 cm de sobreposição por emenda.")

if __name__ == "__main__":
    main()