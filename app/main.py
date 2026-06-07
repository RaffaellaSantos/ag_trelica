import numpy as np
import random
from app.truss_ag import TrussPalitoAG
from app.visualizer import animar_evolucao, animar_pareto, gerar_imagem_final


def main():
    print("=" * 65)
    print("Inicializando o Algoritmo Genético...")
    ag = TrussPalitoAG(
        tam_populacao=150,
        num_geracoes=200,
        p_mutacao=0.20,
        p_crossover=0.85,
    )

    historico_cromossomos = []
    historico_nuvem       = []
    historico_pareto      = []
    forcas_reais          = [(2, 0.0, -100.0)]

    # População inicial
    populacao = []
    while len(populacao) < ag.tam_populacao:
        ind = ag.gerar_cromossomo()
        if ag.fitness(ind)[1] < 9999.0:
            populacao.append(ind)
        elif len(populacao) < ag.tam_populacao * 0.1:
            populacao.append(ind)

    melhor_global = populacao[0].copy()
    melhor_P_rup  = ag.avaliar_indivíduo(melhor_global)[0]

    for gen in range(ag.num_geracoes):
        scores = [ag.fitness(ind) for ind in populacao]

        for i, ind in enumerate(populacao):
            P_rup = -scores[i][0]
            if P_rup > melhor_P_rup:
                melhor_P_rup  = P_rup
                melhor_global = ind.copy()

        nd_idx, _ = ag.fronteira_pareto(populacao)
        pareto_pts = []
        for i in nd_idx:
            P_r, m_g, _, _ = ag.avaliar_indivíduo(populacao[i])
            pareto_pts.append([P_r, m_g])

        nuvem = []
        for ind in populacao:
            P_r, m_g, _, viav = ag.avaliar_indivíduo(ind)
            if viav:
                nuvem.append([P_r, m_g, 1.0])
            else:
                nuvem.append([0.0, 0.0, 0.0])

        historico_cromossomos.append(melhor_global.copy())
        historico_nuvem.append(nuvem)
        historico_pareto.append(pareto_pts)

        P_rup, m_total, desl, _ = ag.avaliar_indivíduo(melhor_global)
        _, _, secoes, n_trav    = ag.decodificar(melhor_global)
        print(f"Geração {gen:03d} | Fitness: {P_rup/m_total:.4f} kg/g | Suporta: {P_rup:.2f} kg | Massa: {m_total:.1f} g | Trav: {n_trav} palito(s) | ")

        nova_pop = [melhor_global.copy()]
        while len(nova_pop) < ag.tam_populacao:
            pai = ag.torneio_pareto(populacao, scores)
            mae = ag.torneio_pareto(populacao, scores)
            f1, f2 = ag.crossover(pai, mae)
            ag.mutar(f1)
            ag.mutar(f2)
            nova_pop.append(f1)
            if len(nova_pop) < ag.tam_populacao:
                nova_pop.append(f2)
        populacao = nova_pop

    # Resultados finais
    P_rup, m_total, delta_C, _ = ag.avaliar_indivíduo(melhor_global)
    l_bases, l_vert, secoes, n_trav = ag.decodificar(melhor_global)

    print("\n" + "=" * 65)
    print(" FICHA TÉCNICA DA TRELIÇA ÓTIMA")
    print("=" * 65)
    print(f" Carga de Ruptura Prevista    : {P_rup:.2f} kg")
    print(f" Massa Total da Estrutura     : {m_total:.2f} g")
    print(f" Deslocamento Vertical (Nó C) : {delta_C:.3f} mm")
    print(f" Fitness (Carga/Massa)        : {P_rup/m_total:.4f} kg/g")
    print(f" Travamentos entre faces      : {n_trav} palito(s) por nó")
    print(f" Total de palitos             : {sum(secoes)*2 + 10*n_trav}")
    print(f" Vão livre                    : {sum(l_bases)*100:.1f} cm")
    print(f" Painéis (cm)                 : {[round(v*100,2) for v in l_bases]}")
    print(f" Alturas (cm)                 : {[round(v*100,2) for v in l_vert]}")
    print("=" * 65)

    animar_evolucao(
        historico_cromossomos=historico_cromossomos,
        num_geracoes=ag.num_geracoes,
        forcas_externas=forcas_reais,
        gif_path="best_truss.gif",
    )
    animar_pareto(
        historico_nuvem=historico_nuvem,
        historico_pareto=historico_pareto,
        num_geracoes=ag.num_geracoes,
        gif_path="pareto_evolution.gif",
    )
    gerar_imagem_final(
        cromossomo=melhor_global,
        gen=ag.num_geracoes,
        forcas_externas=forcas_reais,
        png_path="best_truss_final.png",
    )
    print("[Sucesso] Arquivos gerados!")


if __name__ == "__main__":
    main()