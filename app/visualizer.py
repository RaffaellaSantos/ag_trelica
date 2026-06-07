import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from app.utils import Utils

# Cores conforme a imagem do professor
COR_1_PALITO  = "#1f4ef5"   # Azul   — 1 palito  (menor área)
COR_2_PALITOS = "#16a34a"   # Verde  — 2 palitos
COR_3_PALITOS = "#ef2b2b"   # Vermelho — 3 palitos (maior área)
PRETO  = "#000000"
CINZA  = "#aaaaaa"
ROXO   = "#7e22ce"
ROSA   = "#fca3a3"
BRANCO = "#ffffff"


def _cor_barra(n):
    if n <= 1:
        return COR_1_PALITO, 2.0
    elif n == 2:
        return COR_2_PALITOS, 3.5
    else:
        return COR_3_PALITOS, 5.0


def _desenhar_quadro_simples(ax, cromossomo, gen, forcas_externas, melhor_obj=None):
    utils = Utils()

    l_bases     = cromossomo[0:4].copy()
    l_verticais = cromossomo[4:9].copy()
    soma_p = sum(l_bases)
    if soma_p > 0:
        l_bases = (l_bases / soma_p) * 0.40
    else:
        l_bases = np.array([0.10, 0.10, 0.10, 0.10])

    nos, barras, _, _ = utils.calcular_comprimentos_howe_3d(l_bases, l_verticais)
    nos_cm = nos * 100.0

    ax.clear()
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    # Limites com margem para as setas de força
    x_max = nos_cm[:, 0].max()
    y_max = nos_cm[:, 1].max()
    ax.set_xlim(-5.0, x_max + 5.0)
    ax.set_ylim(-5.0, y_max + 7.0)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Barras coloridas por número de palitos
    for k, (i, j) in enumerate(barras):
        n = int(round(cromossomo[9 + k]))
        cor, lw = _cor_barra(n)
        ax.plot(
            [nos_cm[i, 0], nos_cm[j, 0]],
            [nos_cm[i, 1], nos_cm[j, 1]],
            color=cor, lw=lw, solid_capstyle="round", zorder=2,
        )

    # Nós
    ax.scatter(nos_cm[:, 0], nos_cm[:, 1], s=45, color=PRETO, zorder=4)

    # Rótulos A-E nos nós inferiores
    for idx, rot in enumerate(["A", "B", "C", "D", "E"]):
        ax.text(nos_cm[idx, 0], nos_cm[idx, 1] - 1.2, rot,
                ha="center", va="top", fontsize=11,
                color=PRETO, weight="bold", zorder=5)

    # Apoios
    for idx, rolico in [(0, False), (4, True)]:
        x, y = nos_cm[idx]
        ax.plot([x - 1.5, x, x + 1.5], [y - 1.8, y, y - 1.8],
                color=PRETO, lw=1.3, zorder=5)
        ax.plot([x - 1.8, x + 1.8], [y - 1.8, y - 1.8],
                color=PRETO, lw=1.3, zorder=5)
        if rolico:
            ax.plot([x - 1.8, x + 1.8], [y - 2.4, y - 2.4],
                    color=PRETO, lw=1.3, zorder=5)

    # Setas de força em cada nó com valor em N
    nos_com_forca = [(no, fx, fy) for no, fx, fy in forcas_externas if abs(fy) > 0 or abs(fx) > 0]
    if nos_com_forca:
        y_topo_seta = y_max + 5.5  # altura da cauda da seta
        for no_idx, fx, fy in nos_com_forca:
            if no_idx >= len(nos_cm):
                continue
            x, y = nos_cm[no_idx]
            # Seta apontando para baixo no nó
            ax.annotate(
                "",
                xy=(x, y),
                xytext=(x, y_topo_seta),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=PRETO,
                    lw=1.8,
                ),
                zorder=5,
            )
            # Valor da força acima da seta
            mag = abs(fy)
            unidade = "N" if mag < 1000 else "kN"
            valor   = mag if mag < 1000 else mag / 1000
            ax.text(x, y_topo_seta + 0.4, f"{valor:.0f} {unidade}",
                    ha="center", va="bottom", fontsize=9,
                    color=PRETO, weight="bold", zorder=5)

    # Legenda de seções
    handles = [
        Line2D([0], [0], color=COR_1_PALITO,  lw=2,   label=r"$A_1$ = 1 palito  (0,20 cm²)"),
        Line2D([0], [0], color=COR_2_PALITOS, lw=3.5, label=r"$A_2$ = 2 palitos (0,40 cm²)"),
        Line2D([0], [0], color=COR_3_PALITOS, lw=5,   label=r"$A_3$ = 3 palitos (0,60 cm²)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9.5, handlelength=2.0)

    if melhor_obj:
        P_rup, m_g = melhor_obj
        txt = (f"$\\downarrow\\ \\delta$ (node C): {P_rup:.2f} mm\n\n"
               f"m: {m_g:.1f} g")
        ax.text(0.52, 0.45, txt,
                transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color=PRETO,
                bbox=dict(boxstyle="round,pad=0.5", fc=BRANCO, ec=PRETO, lw=1.0),
                zorder=7)

    # Título
    ax.set_title(
        f"Genetic_Algorithm _Optimization _Truss (Gen : {gen:03d})",
        fontsize=12, color=PRETO, fontweight="bold", pad=10,
        bbox=dict(boxstyle="round,pad=0.4", fc=BRANCO, ec=PRETO, lw=1.0),
    )


def animar_evolucao(historico_cromossomos, num_geracoes, forcas_externas, historico_melhor_obj=None, gif_path="best_truss.gif"):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")

    def update(frame):
        obj = None
        if historico_melhor_obj and frame < len(historico_melhor_obj):
            obj = historico_melhor_obj[frame]
        _desenhar_quadro_simples(ax, historico_cromossomos[frame], frame, forcas_externas, obj)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(historico_cromossomos), interval=60, repeat=False, blit=False)
    print(f"Salvando animação estrutural em '{gif_path}'...")
    ani.save(gif_path, writer="pillow", fps=15, dpi=100)
    plt.close(fig)
    print(f"GIF salvo: {gif_path}")


def animar_pareto(historico_nuvem, historico_pareto, num_geracoes, gif_path="pareto_evolution.gif"):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")

    def update(frame):
        ax.clear()
        ax.set_facecolor("white")

        # Nuvem acumulada
        for g in range(frame):
            dados = [p for p in historico_nuvem[g] if p[2] == 1.0 and p[1] > 0]
            if dados:
                arr = np.array(dados)
                ax.scatter(arr[:, 1], arr[:, 0], color="#e0e0e0", s=8, alpha=0.25, zorder=1)

        # Nuvem atual
        dados_atual = [p for p in historico_nuvem[frame] if p[2] == 1.0 and p[1] > 0]
        if dados_atual:
            arr = np.array(dados_atual)
            ax.scatter(arr[:, 1], arr[:, 0], color=CINZA, s=16, alpha=0.85, label="Viável", zorder=2)

        # Fronteira de Pareto
        if frame < len(historico_pareto) and historico_pareto[frame]:
            fp = np.array(historico_pareto[frame])
            fp_sorted = fp[fp[:, 1].argsort()]
            ax.plot(fp_sorted[:, 1], fp_sorted[:, 0], color=ROXO, lw=1.8, zorder=4)
            ax.scatter(fp_sorted[:, 1], fp_sorted[:, 0], color=ROXO, s=28, edgecolors=BRANCO, linewidths=0.5, label="Fronteira Pareto", zorder=5)

        # Limites dinâmicos
        todos = []
        for g in range(frame + 1):
            todos += [p for p in historico_nuvem[g] if p[2] == 1.0 and p[1] > 0]
        if todos:
            arr    = np.array(todos)
            m_min, m_max = arr[:, 1].min(), arr[:, 1].max()
            p_min, p_max = arr[:, 0].min(), arr[:, 0].max()
            dm = max((m_max - m_min) * 0.1, 1.0)
            dp = max((p_max - p_min) * 0.1, 0.5)
            ax.set_xlim(max(0, m_min - dm), m_max + dm)
            ax.set_ylim(max(0, p_min - dp), p_max + dp)

        ax.set_xlabel("Massa da estrutura (g)", fontsize=10)
        ax.set_ylabel("Carga de ruptura (kg)", fontsize=10)
        ax.set_title(f"Fronteira de Pareto — Geração {frame+1:03d}/{num_geracoes}", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(historico_nuvem), interval=60, repeat=False, blit=False)
    print(f"Salvando evolução de Pareto em '{gif_path}'...")
    ani.save(gif_path, writer="pillow", fps=15, dpi=100)
    plt.close(fig)
    print(f"GIF salvo: {gif_path}")


def gerar_imagem_final(cromossomo, gen, forcas_externas, melhor_obj=None, png_path="best_truss_final.png"):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    _desenhar_quadro_simples(ax, cromossomo, gen, forcas_externas, melhor_obj)
    fig.savefig(png_path, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"Imagem final salva: {png_path}")
    plt.close(fig)