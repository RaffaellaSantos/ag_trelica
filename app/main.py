from app.truss_ag import TrussAG
from app.visualizer import animar_evolucao

def main():
    forcas_externas = [
        (0, 0.0, -10.0),
        (5, 0.0, -10.0),
        (6, 0.0, -20.0),
        (7, 0.0, -20.0),
        (4, 0.0, -10.0),
    ]
    forcas_virtuais = [
        (2, 0.0, -1.0)
    ]

    truss_ag = TrussAG(
        forcas_externas, 
        forcas_virtuais, 
        barras_base=4,
        barras_verticais=3,
        num_geracoes=200,
        tam_populacao=200,
        P1=10000.0, 
        P2=50000.0
    )

    fronteira_final, historico_cromossomos, historico_fronteiras = truss_ag.iniciar()

    print("\nEvolução Concluída.")
    print(f"Total de soluções ótimas encontradas na Fronteira de Pareto Final: {len(fronteira_final)}")

    animar_evolucao(
        historico_cromossomos, 
        historico_fronteiras, 
        truss_ag.num_geracoes,
        barras_base=truss_ag.barras_base,
        barras_verticais=truss_ag.barras_verticais
    )

if __name__ == "__main__":
    main()