"""
Visualizador para a treliça Howe de palitos (Atividade 9).
Layout: treliça à esquerda | scatter massa × carga à direita (subplot separado).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# ── Paleta ───────────────────────────────────────────────────────────────────
AZUL    = "#1f4ef5"
VERDE   = "#16a34a"
VERM    = "#ef2b2b"
PRETO   = "#111111"
CINZA   = "#9ca3af"
ROXO    = "#7e22ce"
ROSA    = "#fca5a5"
LARANJA = "#f97316"   # exclusivo para carga/nó C

# ── Topologia ────────────────────────────────────────────────────────────────
CONNECTIVITY = [
    ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"),
    ("F", "G"), ("G", "H"), ("H", "I"), ("I", "J"),
    ("A", "F"), ("G", "B"), ("H", "C"), ("I", "D"), ("E", "J"),
    ("F", "B"), ("G", "C"), ("I", "C"), ("J", "D"),
]
LOWER_NODES = {"A", "B", "C", "D", "E"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bar_color(n: int) -> str:
    return (AZUL, VERDE, VERM)[min(max(int(n), 1), 3) - 1]


def _bar_lw(n: int) -> float:
    return 1.1 + int(n) * 1.0


def _coords_cm(p_cm, h_cm) -> dict:
    x = [0.0, p_cm[0], p_cm[0]+p_cm[1], p_cm[0]+p_cm[1]+p_cm[2], 40.0]
    return {
        "A": (x[0], 0.0), "B": (x[1], 0.0), "C": (x[2], 0.0),
        "D": (x[3], 0.0), "E": (x[4], 0.0),
        "F": (x[0], h_cm[0]), "G": (x[1], h_cm[1]),
        "H": (x[2], h_cm[2]), "I": (x[3], h_cm[3]),
        "J": (x[4], h_cm[4]),
    }


def _draw_support(ax, x, y, roller=False):
    h, b = 0.9, 0.55
    ax.plot([x-b, x, x+b], [y-h, y, y-h], color=PRETO, lw=1.1, zorder=6)
    ax.plot([x-b-0.1, x+b+0.1], [y-h, y-h], color=PRETO, lw=1.1, zorder=6)
    for dx in np.linspace(-b, b, 5):
        ax.plot([x+dx, x+dx-0.18], [y-h, y-h-0.22], color=PRETO, lw=0.7, zorder=6)
    if roller:
        ax.plot([x-b-0.1, x+b+0.1], [y-h-0.30, y-h-0.30], color=PRETO, lw=1.1, zorder=6)


def pareto_front(pts):
    """Subconjunto não-dominado de [(mass_g, load_kg)]: minimizar massa, maximizar carga."""
    if not pts:
        return []
    result = []
    for i, (mi, li) in enumerate(pts):
        dominated = any(
            j != i and mj <= mi and lj >= li and (mj < mi or lj > li)
            for j, (mj, lj) in enumerate(pts)
        )
        if not dominated:
            result.append((mi, li))
    return sorted(result)


# ── Subplots ─────────────────────────────────────────────────────────────────

def _draw_truss(ax, ind, gen, total_gens):
    """Desenha a treliça no eixo ax."""
    ax.clear()
    p_cm  = ind.p_cm
    h_cm  = ind.h_cm
    n_arr = ind.n
    d     = ind.data
    nodes = _coords_cm(p_cm, h_cm)
    max_h = float(max(h_cm))

    ax.set_facecolor("white")
    ax.set_xlim(-3.0, 43.5)
    ax.set_ylim(-2.5, max_h + 5.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (cm)", fontsize=9, color=PRETO)
    ax.set_ylabel("y  (cm)", fontsize=9, color=PRETO)
    ax.tick_params(colors=PRETO, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(PRETO)

    # Barras
    for k, (na, nb) in enumerate(CONNECTIVITY):
        xa, ya = nodes[na]
        xb, yb = nodes[nb]
        ax.plot([xa, xb], [ya, yb],
                color=_bar_color(n_arr[k]), lw=_bar_lw(n_arr[k]),
                solid_capstyle="round", zorder=2)
        xm, ym = (xa+xb)/2, (ya+yb)/2
        ax.text(xm, ym + 0.22, str(int(n_arr[k])),
                ha="center", va="bottom", fontsize=6, color=PRETO, zorder=5)

    # Nós
    for name, (xn, yn) in nodes.items():
        color = LARANJA if name == "C" else PRETO
        ax.scatter([xn], [yn], s=28, color=color, zorder=4)
        dy = -0.75 if name in LOWER_NODES else 0.55
        va = "top" if name in LOWER_NODES else "bottom"
        ax.text(xn, yn + dy, name, ha="center", va=va,
                fontsize=9, color=color, fontweight="bold", zorder=5)

    # Apoios
    _draw_support(ax, *nodes["A"], roller=False)
    _draw_support(ax, *nodes["E"], roller=True)

    # Seta de carga em C
    xc, yc = nodes["C"]
    ax.annotate("", xy=(xc, yc), xytext=(xc, yc + 2.8),
                arrowprops=dict(arrowstyle="-|>", color=LARANJA, lw=2.0), zorder=5)
    ax.text(xc, yc + 3.0, "P", ha="center", va="bottom",
            fontsize=11, color=LARANJA, fontweight="bold", zorder=5)

    # Legenda de seção
    handles = [
        Line2D([0], [0], color=AZUL,  lw=3, label="1 palito"),
        Line2D([0], [0], color=VERDE, lw=3, label="2 palitos"),
        Line2D([0], [0], color=VERM,  lw=3, label="3 palitos"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              frameon=True, framealpha=0.9, edgecolor=PRETO, handlelength=1.8)

    # Título
    ax.set_title(
        f"Treliça Howe — Palitos  |  Geração {gen}/{total_gens}",
        fontsize=12, color=PRETO, fontweight="bold", pad=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PRETO, lw=0.8),
    )

    # Caixa de resultados
    load_kg  = d.get("theoretical_load_kg", 0.0)
    m_g      = d.get("mass_g", 0.0)
    delta_mm = d.get("delta_c_at_rupture_mm", 0.0)
    crit     = d.get("critical_bar", "—")
    fit_str  = f"{ind.fitness:.5f}" if ind.fitness > 0 else "inviável"
    info = (
        f"Fitness : {fit_str} kg/g\n"
        f"Carga   : {load_kg:.3f} kg\n"
        f"Massa   : {m_g:.2f} g\n"
        f"Δ_C     : {delta_mm:.2f} mm\n"
        f"Crítica : {crit}"
    )
    ax.text(0.98, 0.98, info, transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=PRETO, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=PRETO, lw=0.8),
            zorder=7)


def _draw_scatter(ax, cloud_pts, pareto_pts, best_obj, gen, total_gens):
    """Desenha scatter massa × carga no eixo ax."""
    ax.clear()
    ax.set_facecolor("white")
    ax.tick_params(colors=PRETO, labelsize=8)
    ax.set_xlabel("Massa total  (g)", fontsize=9, color=PRETO)
    ax.set_ylabel("Carga de ruptura  (kg)", fontsize=9, color=PRETO)
    ax.set_title("Espaço de busca", fontsize=10,
                 color=PRETO, fontweight="bold", pad=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color=CINZA)
    for sp in ax.spines.values():
        sp.set_color(PRETO)

    if cloud_pts:
        arr = np.asarray(cloud_pts, dtype=float)
        inf_m = arr[:, 2] < 0.5
        fea_m = ~inf_m
        if inf_m.any():
            ax.scatter(arr[inf_m, 0], arr[inf_m, 1],
                       s=9, color=ROSA, alpha=0.25, label="Inviável", zorder=1)
        if fea_m.any():
            ax.scatter(arr[fea_m, 0], arr[fea_m, 1],
                       s=12, color=CINZA, alpha=0.65, label="Viável", zorder=2)

    if pareto_pts:
        fp = np.asarray(pareto_pts, dtype=float)
        fp = fp[fp[:, 0].argsort()]
        ax.plot(fp[:, 0], fp[:, 1], color=ROXO, lw=1.6, zorder=4)
        ax.scatter(fp[:, 0], fp[:, 1], s=28, color=ROXO,
                   edgecolors="white", linewidths=0.5,
                   label="Front. Pareto", zorder=5)

    if best_obj:
        ax.scatter([best_obj[0]], [best_obj[1]], s=70, color=AZUL,
                   edgecolors="white", linewidths=0.8,
                   label="Melhor indiv.", zorder=6)

    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9,
              markerscale=1.1, handletextpad=0.4,
              edgecolor=PRETO)

    # Texto de geração
    ax.text(0.03, 0.03, f"Gen {gen}/{total_gens}",
            transform=ax.transAxes, fontsize=8, color=PRETO,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PRETO, lw=0.6))


# ── API pública ───────────────────────────────────────────────────────────────

def animate_evolution(
    history_best,
    history_cloud,
    history_pareto,
    history_best_obj,
    total_gens: int,
    gif_path: str = "best_truss.gif",
    max_frames: int = 50,
):
    """GIF da evolução com treliça (esq.) e scatter (dir.) em subplots separados."""
    n = len(history_best)
    indices = (
        list(range(n)) if n <= max_frames
        else [round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
    )

    # Nuvem acumulada (últimas 15 gerações para não sobrecarregar)
    cloud_acc: list = []
    cloud_upto: list = []
    for gen_cloud in history_cloud:
        cloud_acc.extend(gen_cloud)
        cloud_upto.append(cloud_acc[-15 * 250:])

    fig = plt.figure(figsize=(14, 6), facecolor="white")
    ax_t = fig.add_axes([0.04, 0.10, 0.56, 0.83])   # treliça: esquerda
    ax_s = fig.add_axes([0.67, 0.11, 0.29, 0.80])   # scatter: direita

    def update(frame_idx):
        i = indices[frame_idx]
        _draw_truss(ax_t, history_best[i], i + 1, total_gens)
        _draw_scatter(ax_s, cloud_upto[i],
                      history_pareto[i], history_best_obj[i],
                      i + 1, total_gens)
        return []

    ani = animation.FuncAnimation(
        fig, update, frames=len(indices),
        interval=140, repeat=False, blit=False,
    )

    print(f"Salvando animação em '{gif_path}' ({len(indices)} quadros)...")
    ani.save(gif_path, writer="pillow", fps=7, dpi=90)
    print(f"GIF salvo: {gif_path}")
    plt.close(fig)


def save_final_image(
    ind,
    cloud_pts,
    pareto_pts,
    best_obj,
    total_gens: int,
    png_path: str = "best_truss_final.png",
):
    """PNG final com treliça à esquerda e scatter à direita."""
    fig = plt.figure(figsize=(14, 6), facecolor="white")
    ax_t = fig.add_axes([0.04, 0.10, 0.56, 0.83])
    ax_s = fig.add_axes([0.67, 0.11, 0.29, 0.80])
    _draw_truss(ax_t, ind, total_gens, total_gens)
    _draw_scatter(ax_s, cloud_pts, pareto_pts, best_obj, total_gens, total_gens)
    fig.savefig(png_path, dpi=130, facecolor="white")
    print(f"Imagem final salva: {png_path}")
    plt.close(fig)
