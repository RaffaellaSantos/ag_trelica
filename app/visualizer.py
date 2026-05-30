import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from app.utils import Utils


# ── Paleta ────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#21262d"
TEXT_COL = "#e6edf3"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
PURPLE   = "#bc8cff"
RED_C    = "#ff7b72"
MUTED    = "#8b949e"


def _bar_color(length: float) -> str:
    if length <= 1.5:
        return ACCENT
    elif length <= 2.5:
        return GREEN
    else:
        return RED_C


def animar_evolucao(
    historico_cromossomos,
    historico_fronteiras,
    num_geracoes: int,
    barras_base: int = 4,
    barras_verticais: int = 3,
):
    utils = Utils()
    num_barras = 13  # fixo para a treliça Howe

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[1.6, 1],
        height_ratios=[2.2, 1],
        hspace=0.38,
        wspace=0.32,
        left=0.05, right=0.97,
        top=0.93,  bottom=0.07,
    )

    ax_truss  = fig.add_subplot(gs[:, 0])
    ax_pareto = fig.add_subplot(gs[0, 1])
    ax_info   = fig.add_subplot(gs[1, 1])

    for ax in (ax_truss, ax_pareto, ax_info):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    # ── Configuração estática do ax_truss ─────────────────────────────────────
    # Determina limites globais varrendo todos os frames
    all_x, all_y = [], []
    for crom in historico_cromossomos:
        lb = crom[:barras_base]
        lv = crom[barras_base:barras_base + barras_verticais]
        nos, _, _, _ = utils.calcular_comprimentos_howe(lb, lv)
        all_x.extend(nos[:, 0])
        all_y.extend(nos[:, 1])

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pad_x = (x_max - x_min) * 0.12 or 0.5
    pad_y = (y_max - y_min) * 0.35 or 0.5

    ax_truss.set_xlim(x_min - pad_x, x_max + pad_x)
    ax_truss.set_ylim(y_min - pad_y * 1.5, y_max + pad_y)
    ax_truss.set_aspect('equal')
    ax_truss.set_xlabel("x (m)", color=MUTED, fontsize=9)
    ax_truss.set_ylabel("y (m)", color=MUTED, fontsize=9)
    ax_truss.tick_params(colors=MUTED)
    ax_truss.grid(True, color=GRID_COL, linestyle='--', linewidth=0.6, alpha=0.8)

    len_legend = [
        mpatches.Patch(color=ACCENT, label="L ≤ 1.5 m"),
        mpatches.Patch(color=GREEN,  label="1.5 < L ≤ 2.5 m"),
        mpatches.Patch(color=RED_C,  label="L > 2.5 m"),
    ]
    ax_truss.legend(
        handles=len_legend,
        loc='upper right',
        fontsize=8,
        framealpha=0.25,
        facecolor=PANEL_BG,
        edgecolor=GRID_COL,
        labelcolor=TEXT_COL,
    )

    title_truss = ax_truss.set_title(
        f"Treliça Howe — Geração 1 / {num_geracoes}",
        fontsize=13, color=TEXT_COL, fontweight='bold', pad=10,
    )

    # ── Objetos mutáveis da treliça (criados uma vez) ─────────────────────────
    bar_lines   = [ax_truss.plot([], [], color=ACCENT, linewidth=2,
                                 solid_capstyle='round', zorder=2)[0]
                   for _ in range(num_barras)]
    bar_labels  = [ax_truss.text(0, 0, '', fontsize=6.5, color=TEXT_COL,
                                 ha='center', va='bottom',
                                 fontfamily='monospace', zorder=5)
                   for _ in range(num_barras)]
    node_scatter = ax_truss.scatter([], [], s=60, color=TEXT_COL,
                                    zorder=3, edgecolors=BG, linewidths=1.2)
    node_labels  = [ax_truss.text(0, 0, str(k), fontsize=7, color=MUTED,
                                  ha='center', va='top', fontfamily='monospace')
                    for k in range(8)]  # 8 nós fixos

    vol_text = ax_truss.text(
        0.02, 0.02, '',
        fontsize=8, color=ACCENT, ha='left', va='bottom',
        fontfamily='monospace', transform=ax_truss.transAxes, zorder=6,
    )

    # ── Configuração estática do ax_pareto ────────────────────────────────────
    ax_pareto.set_facecolor(PANEL_BG)
    ax_pareto.set_title("Fronteira de Pareto", fontsize=10, color=TEXT_COL,
                         fontweight='bold', pad=6)
    ax_pareto.set_xlabel("Volume (m³)", color=MUTED, fontsize=8)
    ax_pareto.set_ylabel("Tensão Máx. (kN/m²)", color=MUTED, fontsize=8)
    ax_pareto.tick_params(colors=MUTED, labelsize=7)
    ax_pareto.grid(True, color=GRID_COL, linestyle='--', linewidth=0.5, alpha=0.7)

    # Objetos mutáveis do Pareto
    scatter_hist  = ax_pareto.scatter([], [], s=12, color=MUTED, alpha=0.2, zorder=1)
    pareto_line,  = ax_pareto.plot([], [], color=PURPLE, linewidth=1.2, zorder=2)
    scatter_front = ax_pareto.scatter([], [], s=30, color=PURPLE,
                                      edgecolors=BG, linewidths=0.8, zorder=3)

    # ── Configuração estática do ax_info ──────────────────────────────────────
    ax_info.axis('off')
    ax_info.set_title("Áreas Transversais (m²)", fontsize=10, color=TEXT_COL,
                       fontweight='bold', pad=6)

    cols       = 3
    rows_need  = int(np.ceil(num_barras / cols))
    col_w      = 1.0 / cols
    row_h      = 0.85 / max(rows_need, 1)

    # Cria caixas e textos de área uma única vez
    area_rects = []
    area_texts = []
    for k in range(num_barras):
        col = k % cols
        row = k // cols
        cx  = 0.05 + col * col_w
        cy  = 0.88 - row * row_h

        rect = mpatches.FancyBboxPatch(
            (cx, cy - row_h * 0.6), col_w * 0.9, row_h * 0.7,
            boxstyle="round,pad=0.01",
            linewidth=0.5, edgecolor=GRID_COL,
            facecolor=plt.cm.YlOrRd(0.5), alpha=0.7,
            transform=ax_info.transAxes, clip_on=False,
        )
        ax_info.add_patch(rect)
        area_rects.append(rect)

        txt = ax_info.text(
            cx + col_w * 0.45, cy - row_h * 0.25,
            f"B{k+1:02d}\n—",
            fontsize=6.2, color=TEXT_COL, ha='center', va='center',
            fontfamily='monospace', transform=ax_info.transAxes,
        )
        area_texts.append(txt)

    vol_info_text = ax_info.text(
        0.5, -0.04, '',
        fontsize=8, color=ACCENT, ha='center', va='bottom',
        transform=ax_info.transAxes, fontfamily='monospace',
    )

    # ── Estado acumulado do Pareto ─────────────────────────────────────────────
    acc_vols = []
    acc_tens = []

    # ── Função de update — só modifica dados, não recria artistas ─────────────
    def update(frame: int):
        cromossomo = historico_cromossomos[frame]
        l_bases = cromossomo[:barras_base]
        l_verts = cromossomo[barras_base:barras_base + barras_verticais]
        areas   = cromossomo[barras_base + barras_verticais:]

        nos, barras, comprimentos, _ = utils.calcular_comprimentos_howe(l_bases, l_verts)

        a_min   = areas.min()
        a_max   = areas.max()
        a_range = a_max - a_min if a_max > a_min else 1e-9

        # — Barras —
        for idx, (i, j) in enumerate(barras):
            x = [nos[i, 0], nos[j, 0]]
            y = [nos[i, 1], nos[j, 1]]
            L   = comprimentos[idx]
            lw  = 1.5 + 6.0 * (areas[idx] - a_min) / a_range
            bar_lines[idx].set_data(x, y)
            bar_lines[idx].set_color(_bar_color(L))
            bar_lines[idx].set_linewidth(lw)

            mx = (nos[i, 0] + nos[j, 0]) / 2
            my = (nos[i, 1] + nos[j, 1]) / 2
            bar_labels[idx].set_position((mx, my + 0.08))
            bar_labels[idx].set_text(f"{L:.2f}m")

        # — Nós —
        node_scatter.set_offsets(nos)
        for k, (nx, ny) in enumerate(nos):
            node_labels[k].set_position((nx, ny - 0.18))

        # — Título e volume —
        title_truss.set_text(f"Treliça Howe — Geração {frame + 1} / {num_geracoes}")
        vol_cur = sum(L * A for L, A in zip(comprimentos, areas))
        vol_text.set_text(f"Vol: {vol_cur:.5f} m³")

        # — Pareto —
        pontos = historico_fronteiras[frame]
        if pontos:
            vs = [p[0] for p in pontos]
            ts = [p[1] for p in pontos]
            acc_vols.extend(vs)
            acc_tens.extend(ts)

            sorted_pts = sorted(zip(vs, ts))
            svs = [p[0] for p in sorted_pts]
            sts = [p[1] for p in sorted_pts]
            pareto_line.set_data(svs, sts)
            scatter_front.set_offsets(np.c_[svs, sts])

        if acc_vols:
            scatter_hist.set_offsets(np.c_[acc_vols, acc_tens])
            # Reajusta limites do Pareto dinamicamente
            ax_pareto.set_xlim(min(acc_vols) * 0.95, max(acc_vols) * 1.05)
            ax_pareto.set_ylim(min(acc_tens) * 0.95, max(acc_tens) * 1.05)

        # — Áreas —
        for k, A in enumerate(areas):
            norm_a = (A - a_min) / a_range if a_range > 1e-12 else 0.5
            area_rects[k].set_facecolor(plt.cm.YlOrRd(0.25 + 0.65 * norm_a))
            area_texts[k].set_text(f"B{k+1:02d}\n{A*1e4:.2f}×10⁻⁴")

        vol_info_text.set_text(f"Volume atual: {vol_cur:.5f} m³")

        return (bar_lines + bar_labels + [node_scatter] + node_labels
                + [title_truss, vol_text, pareto_line, scatter_front,
                   scatter_hist, vol_info_text] + area_rects + area_texts)

    # ── Animação ──────────────────────────────────────────────────────────────
    ani = animation.FuncAnimation(
        fig, update,
        frames=len(historico_cromossomos),
        interval=120,
        repeat=False,
        blit=True,   # só redesenha os artistas retornados por update()
    )

    fig.suptitle(
        "Otimização Multiobjetivo de Treliça Howe — NSGA-II",
        fontsize=14, color=TEXT_COL, fontweight='bold', y=0.98,
    )

    # ── Salvar GIF ────────────────────────────────────────────────────────────
    gif_path = "best_truss.gif"
    print(f"Salvando animação em '{gif_path}'… (isso pode levar alguns instantes)")
    ani.save(gif_path, writer="pillow", fps=8, dpi=100)
    print(f"GIF salvo com sucesso: {gif_path}")

    plt.show()