from app.truss_ag import TrussAG
from app.visualizer import animar_evolucao, gerar_imagem_final


def main():
    forcas_externas = [
        (0, 0.0, 0.0),
        (5, 0.0, -70.0),
        (6, 0.0, -10.0),
        (7, 0.0, -40.0),
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
        E_GPa=200.0,
        densidade=7870.0,
        sigma_escoamento=300.0,
        fator_seguranca=2.0,
        vao_livre_min=8.0,
        num_geracoes=100,
        tam_populacao=100,
        P1=1.0e4,
        P2=5.0e4,
    )

    fronteira_final, historico_cromossomos, historico_nuvem, historico_melhor_obj, historico_fronteira_plot = truss_ag.iniciar()

    melhor = truss_ag.melhor_solucao(fronteira_final)
    rel = truss_ag.avaliar_completo(melhor)

    print("\nEvolucao Concluida.")
    print(f"Total de solucoes otimas na Fronteira de Pareto Final: {len(fronteira_final)}")

    animar_evolucao(
        historico_cromossomos,
        historico_nuvem,
        historico_fronteira_plot,
        historico_melhor_obj,
        truss_ag.num_geracoes,
        forcas_externas,
        barras_base=truss_ag.barras_base,
        barras_verticais=truss_ag.barras_verticais,
        gif_path="best_truss.gif",
    )

    nuvem_total = [ponto for geracao in historico_nuvem for ponto in geracao]

    gerar_imagem_final(
        melhor,
        nuvem_total,
        historico_fronteira_plot[-1],
        [rel['massa_kg'], rel['deslocamento_C_mm']],
        truss_ag.num_geracoes,
        truss_ag.num_geracoes,
        forcas_externas,
        barras_base=truss_ag.barras_base,
        barras_verticais=truss_ag.barras_verticais,
        png_path="best_truss_final.png",
    )

    nomes_barras = [
        "1  (A-B)", "2  (B-C)", "3  (C-D)", "4  (D-E)",
        "5  (A-F)", "6  (B-F)", "7  (F-C)", "8  (C-H)",
        "9  (C-G)", "10 (D-G)", "11 (G-E)", "12 (F-H)", "13 (H-G)",
    ]

    print("\n" + "=" * 60)
    print("RESULTADOS DA TRELICA OTIMA")
    print("=" * 60)
    print(f"2.1) Deslocamento no ponto C ........: {rel['deslocamento_C_mm']:.3f} mm")
    print(f"2.2) Massa total da trelica .........: {rel['massa_kg']:.2f} kg")
    print(f"     Vao livre .......................: {rel['vao_livre_m']:.2f} m")
    print(f"     Tensao admissivel (sigmaE/Sg) ...: {rel['sigma_admissivel_MPa']:.1f} MPa")

    print("\n2.3) Areas das secoes transversais (m^2):")
    for n, a in zip(nomes_barras, rel['areas_m2']):
        print(f"     Barra {n:<10}: {a:.2e} m^2")

    print("\n2.4) Comprimentos das barras (m):")
    for n, c in zip(nomes_barras, rel['comprimentos_m']):
        print(f"     Barra {n:<10}: {c:.3f} m")

    print("\n2.5) Forcas axiais das barras (kN):")
    for n, fx in zip(nomes_barras, rel['forcas_kN']):
        estado = "Tracao  " if fx >= 0 else "Compress"
        print(f"     Barra {n:<10}: {fx:+8.2f} kN  ({estado})")

    print("\n2.6) Tensoes axiais das barras (MPa):")
    for n, t in zip(nomes_barras, rel['tensoes_MPa']):
        ok = "OK" if t <= rel['sigma_admissivel_MPa'] + 1e-6 else "VIOLA"
        print(f"     Barra {n:<10}: {t:8.2f} MPa  [{ok}]")
    print("=" * 60)


if __name__ == "__main__":
    main()