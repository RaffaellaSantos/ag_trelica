import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from app.utils import Utils


AZUL  = "#1f4ef5"
VERDE = "#16a34a"
VERM  = "#ef2b2b"
PRETO = "#000000"
CINZA = "#b8b8b8"
ROXO  = "#7e22ce"
ROSA  = "#f3b0b0"


def _area_color(area: float) -> str:
    idx = int(round(area / 1e-4))
    if idx <= 3:
        return AZUL
    elif idx == 4:
        return VERDE
    else:
        return VERM


def _desenhar_apoio(ax, x, y, rolico=False):
    h = 0.42
    b = 0.30
    ax.plot([x - b, x, x + b], [y - h, y, y - h], color=PRETO, lw=1.3, zorder=6)
    ax.plot([x - b - 0.05, x + b + 0.05], [y - h, y - h], color=PRETO, lw=1.3, zorder=6)
    for dx in np.linspace(-b, b, 6):
        ax.plot([x + dx, x + dx - 0.12], [y - h, y - h - 0.16], color=PRETO, lw=0.8, zorder=6)
    if rolico:
        ax.plot([x - b - 0.05, x + b + 0.05], [y - h - 0.20, y - h - 0.20], color=PRETO, lw=1.3, zorder=6)


def _desenhar_quadro(ax, ax_p, cromossomo, nuvem, fronteira_plot, melhor_obj, gen, num_geracoes,
                     forcas_externas, barras_base, barras_verticais):
    utils = Utils()
    lb = cromossomo[:barras_base]
    lv = cromossomo[barras_base:barras_base + barras_verticais]
    areas = cromossomo[barras_base + barras_verticais:]

    nos, barras, comprimentos, _ = utils.calcular_comprimentos_howe(lb, lv)

    ax.clear()
    ax.set_facecolor("white")
    x_max = float(max(nos[:, 0]))
    ax.set_xlim(-1.0, x_max + 1.2)
    ax.set_ylim(-1.8, 8.4)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, x_max + 1, 2))
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.tick_params(colors=PRETO, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PRETO)

    for k, (i, j) in enumerate(barras):
        ax.plot(
            [nos[i, 0], nos[j, 0]],
            [nos[i, 1], nos[j, 1]],
            color=_area_color(areas[k]),
            lw=2.7,
            solid_capstyle="round",
            zorder=2,
        )

    ax.scatter(nos[:, 0], nos[:, 1], s=34, color=PRETO, zorder=4)

    rotulos = ["A", "B", "C", "D", "E"]
    for idx in range(5):
        ax.text(nos[idx, 0], nos[idx, 1] - 0.55, rotulos[idx],
                ha="center", va="top", fontsize=11, color=PRETO, zorder=5)

    _desenhar_apoio(ax, nos[0, 0], nos[0, 1], rolico=False)
    _desenhar_apoio(ax, nos[4, 0], nos[4, 1], rolico=True)

    forcas_aplicadas = [(no, fx, fy) for (no, fx, fy) in forcas_externas if abs(fy) > 0 or abs(fx) > 0]
    if forcas_aplicadas:
        y_tail = max(nos[no][1] for (no, fx, fy) in forcas_aplicadas) + 1.2
        for (no, fx, fy) in forcas_aplicadas:
            x, y = nos[no]
            mag = abs(fy)
            ax.annotate(
                "", xy=(x, y), xytext=(x, y_tail),
                arrowprops=dict(arrowstyle="-|>", color=PRETO, lw=1.7),
                zorder=5,
            )
            ax.text(x, y_tail + 0.10, f"{mag:.0f} kN", ha="center", va="bottom",
                    fontsize=10, color=PRETO, zorder=5)

    handles = [
        Line2D([0], [0], color=AZUL,  lw=3, label=r"$A_1 = 3.\times10^{-4}\ m^2$"),
        Line2D([0], [0], color=VERDE, lw=3, label=r"$A_2 = 4.\times10^{-4}\ m^2$"),
        Line2D([0], [0], color=VERM,  lw=3, label=r"$A_3 = 5.\times10^{-4}\ m^2$"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10, frameon=False,
              handlelength=1.8, bbox_to_anchor=(1.0, 1.0))

    ax.text(0.66, 0.74, "OPTIMAL TRUSS", transform=ax.transAxes,
            color=VERM, fontsize=15, fontweight="bold", ha="center", zorder=5)

    massa_b, delta_b = (melhor_obj if melhor_obj else (0.0, 0.0))
    ax.text(
        0.60, 0.40,
        f"$\\downarrow\\ \\delta$ (node C): {delta_b:.2f} mm\n\nm: {massa_b:.1f} kg",
        transform=ax.transAxes, ha="center", va="center", fontsize=11, color=PRETO,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=PRETO, lw=1.0), zorder=7,
    )

    ax.set_title(
        f"Genetic_Algorithm _Optimization _Truss (Gen : {gen})",
        fontsize=13, color=PRETO, fontweight="bold", pad=12,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PRETO, lw=1.0),
    )

    ax.text(0.5, 0.965, "Prof. Rubel; rcruzneto@uea.edu.br", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color=CINZA, zorder=5)

    ax_p.clear()
    ax_p.set_facecolor("white")
    if nuvem:
        arr = np.array(nuvem)
        if arr.shape[1] >= 3:
            fact = arr[:, 2] > 0.5
            infact = ~fact
            if infact.any():
                ax_p.scatter(arr[infact, 0], arr[infact, 1], s=10, color=ROSA,
                             alpha=0.30, label="Infactivel", zorder=1)
            if fact.any():
                ax_p.scatter(arr[fact, 0], arr[fact, 1], s=14, color=CINZA,
                             alpha=0.85, label="Indiv.", zorder=2)
        else:
            ax_p.scatter(arr[:, 0], arr[:, 1], s=14, color=CINZA, alpha=0.85,
                         label="Indiv.", zorder=2)
    if fronteira_plot:
        fp = np.array(fronteira_plot)
        ax_p.plot(fp[:, 0], fp[:, 1], color=ROXO, lw=1.6, zorder=4)
        ax_p.scatter(fp[:, 0], fp[:, 1], s=24, color=ROXO, edgecolors="white",
                     linewidths=0.5, label="Fronteira Pareto", zorder=5)
    if melhor_obj:
        ax_p.scatter([melhor_obj[0]], [melhor_obj[1]], s=55, color=AZUL,
                     edgecolors="white", linewidths=0.6, label="Fittest Indiv.", zorder=6)

    referencia = []
    if nuvem:
        arr_ref = np.array(nuvem)
        if arr_ref.shape[1] >= 3:
            factiveis = arr_ref[arr_ref[:, 2] > 0.5][:, :2]
            if len(factiveis):
                referencia.append(factiveis)
    if fronteira_plot:
        referencia.append(np.array(fronteira_plot))
    if referencia:
        ref = np.vstack(referencia)
        x_mn, x_mx = ref[:, 0].min(), ref[:, 0].max()
        y_mn, y_mx = ref[:, 1].min(), ref[:, 1].max()
        dx = (x_mx - x_mn) or 1.0
        dy = (y_mx - y_mn) or 1.0
        ax_p.set_xlim(x_mn - 0.06 * dx, x_mx + 0.10 * dx)
        ax_p.set_ylim(max(0.0, y_mn - 0.18 * dy), y_mx + 0.25 * dy)

    ax_p.set_xlabel("Structural truss mass (kg)", fontsize=8, color=PRETO)
    ax_p.set_ylabel("Deformation of node C (mm)", fontsize=8, color=PRETO)
    ax_p.tick_params(colors=PRETO, labelsize=7)
    ax_p.legend(fontsize=6.2, loc="upper right", framealpha=0.9,
                markerscale=1.0, handletextpad=0.3, ncol=2,
                bbox_to_anchor=(1.0, 1.10))
    for spine in ax_p.spines.values():
        spine.set_color(PRETO)


def animar_evolucao(
    historico_cromossomos,
    historico_nuvem,
    historico_fronteira_plot,
    historico_melhor_obj,
    num_geracoes: int,
    forcas_externas,
    barras_base: int = 4,
    barras_verticais: int = 3,
    gif_path: str = "best_truss.gif",
):
    fig = plt.figure(figsize=(10, 9), facecolor="white")
    ax = fig.add_axes([0.07, 0.07, 0.9, 0.84])
    ax_p = fig.add_axes([0.165, 0.655, 0.285, 0.205])

    def update(frame):
        nuvem_acumulada = []
        for g in range(min(frame + 1, len(historico_nuvem))):
            nuvem_acumulada.extend(historico_nuvem[g])
        fronteira_plot = historico_fronteira_plot[frame] if frame < len(historico_fronteira_plot) else []
        _desenhar_quadro(
            ax, ax_p,
            historico_cromossomos[frame],
            nuvem_acumulada,
            fronteira_plot,
            historico_melhor_obj[frame] if frame < len(historico_melhor_obj) else None,
            frame + 1, num_geracoes, forcas_externas, barras_base, barras_verticais,
        )
        return []

    ani = animation.FuncAnimation(
        fig, update, frames=len(historico_cromossomos),
        interval=120, repeat=False, blit=False,
    )

    print(f"Salvando animacao em '{gif_path}'...")
    ani.save(gif_path, writer="pillow", fps=8, dpi=90)
    print(f"GIF salvo: {gif_path}")
    plt.close(fig)


def gerar_imagem_final(
    cromossomo,
    nuvem,
    fronteira_plot,
    melhor_obj,
    gen,
    num_geracoes,
    forcas_externas,
    barras_base: int = 4,
    barras_verticais: int = 3,
    png_path: str = "best_truss_final.png",
):
    fig = plt.figure(figsize=(10, 9), facecolor="white")
    ax = fig.add_axes([0.07, 0.07, 0.9, 0.84])
    ax_p = fig.add_axes([0.165, 0.655, 0.285, 0.205])

    _desenhar_quadro(
        ax, ax_p, cromossomo, nuvem, fronteira_plot, melhor_obj,
        gen, num_geracoes, forcas_externas, barras_base, barras_verticais,
    )

    fig.savefig(png_path, dpi=130, facecolor="white")
    print(f"Imagem final salva: {png_path}")
    plt.close(fig)