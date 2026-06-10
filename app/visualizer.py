import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from app.utils import Utils

class Visualizer:
    """Ferramenta para desenhar a geometria bidimensional e a evolução multiobjetivo em Pareto."""

    def __init__(self):
        """Define configurações visuais, paleta de cor e o manipulador geométrico."""
        self.utils = Utils()
        self.AZUL = "#1f4ef5"
        self.VERDE = "#16a34a"
        self.VERM = "#ef2b2b"
        self.PRETO = "#111111"
        self.CINZA = "#9ca3af"
        self.ROXO = "#7e22ce"
        self.LARANJA = "#f97316"
        
        self.CONNECTIVITY = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8), (8, 9),
            (0, 5), (6, 1), (7, 2), (8, 3), (4, 9),
            (5, 1), (6, 2), (8, 2), (9, 3)
        ]
        self.NODE_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        self.LOWER_NODES = {0, 1, 2, 3, 4}

    def _cor_barra(self, n):
        """Retorna a cor baseada no número de camadas de palitos."""
        return (self.AZUL, self.VERDE, self.VERM)[min(max(int(n), 1), 3) - 1]

    def _espessura_barra(self, n):
        """Retorna a espessura visual da linha baseada no número de camadas."""
        return 1.1 + int(n) * 1.0

    def _desenhar_apoio(self, ax, x, y, rolete=False):
        """Desenha a representação gráfica dos apoios fixos e móveis."""
        h, b = 0.9, 0.55
        ax.plot([x - b, x, x + b], [y - h, y, y - h], color=self.PRETO, lw=1.1, zorder=6)
        ax.plot([x - b - 0.1, x + b + 0.1], [y - h, y - h], color=self.PRETO, lw=1.1, zorder=6)
        for dx in np.linspace(-b, b, 5):
            ax.plot([x + dx, x + dx - 0.18], [y - h, y - h - 0.22], color=self.PRETO, lw=0.7, zorder=6)
        if rolete:
            ax.plot([x - b - 0.1, x + b + 0.1], [y - h - 0.30, y - h - 0.30], color=self.PRETO, lw=1.1, zorder=6)

    def _desenhar_trelica(self, ax, ind, gen, total_gens):
        """Desenha a geometria 2D da treliça, incluindo nós, barras e o relatório textual."""
        ax.clear()
        nos, barras, _, _ = self.utils.calcular_comprimentos_howe(ind.p, ind.h)
        
        ax.set_facecolor("white")
        ax.set_xlim(-3.0, 43.5)
        ax.set_ylim(-3.5, float(max(ind.h)) + 5.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x  (cm)", fontsize=9, color=self.PRETO)
        ax.set_ylabel("y  (cm)", fontsize=9, color=self.PRETO)
        ax.tick_params(colors=self.PRETO, labelsize=8)
        
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(self.PRETO)

        for k, (i, j) in enumerate(self.CONNECTIVITY):
            xa, ya = nos[i]
            xb, yb = nos[j]
            ax.plot([xa, xb], [ya, yb],
                    color=self._cor_barra(ind.n[k]), lw=self._espessura_barra(ind.n[k]),
                    solid_capstyle="round", zorder=2)
            xm, ym = (xa + xb) / 2, (ya + yb) / 2
            ax.text(xm, ym + 0.22, str(int(ind.n[k])),
                    ha="center", va="bottom", fontsize=6, color=self.PRETO, zorder=5)

        for idx, (xn, yn) in enumerate(nos):
            name = self.NODE_NAMES[idx]
            color = self.LARANJA if name == "C" else self.PRETO
            ax.scatter([xn], [yn], s=28, color=color, zorder=4)
            dy = -0.75 if idx in self.LOWER_NODES else 0.55
            va = "top" if idx in self.LOWER_NODES else "bottom"
            ax.text(xn, yn + dy, name, ha="center", va=va,
                    fontsize=9, color=color, fontweight="bold", zorder=5)

        self._desenhar_apoio(ax, nos[0][0], nos[0][1], rolete=False)
        self._desenhar_apoio(ax, nos[4][0], nos[4][1], rolete=True)

        xc, yc = nos[2]
        ax.annotate("", xy=(xc, yc - 2.0), xytext=(xc, yc),
                    arrowprops=dict(arrowstyle="-|>", color=self.LARANJA, lw=2.0), zorder=5)
        ax.text(xc, yc - 2.4, "P", ha="center", va="top",
                fontsize=11, color=self.LARANJA, fontweight="bold", zorder=5)

        handles = [
            Line2D([0], [0], color=self.AZUL,  lw=3, label="1 palito"),
            Line2D([0], [0], color=self.VERDE, lw=3, label="2 palitos"),
            Line2D([0], [0], color=self.VERM,  lw=3, label="3 palitos"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8,
                  frameon=True, framealpha=0.9, edgecolor=self.PRETO, handlelength=1.8)

        ax.set_title(
            f"Treliça Howe — Palitos  |  Geração {gen}/{total_gens}",
            fontsize=12, color=self.PRETO, fontweight="bold", pad=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=self.PRETO, lw=0.8),
        )

        info = (
            f"Fator S.: {ind.fator_seguranca:.3f}\n"
            f"Massa   : {ind.massa_g:.2f} g\n"
            f"Tens.Máx: {ind.tensao_max_mpa:.2f} MPa\n"
            f"Larg. (w): {ind.w:.2f} cm\n"
        )
        ax.text(0.98, 0.98, info, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=self.PRETO, family="monospace",
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=self.PRETO, lw=0.8),
                zorder=7)

    def _desenhar_scatter(self, ax, pareto_pts, best_obj, gen, total_gens):
        """Desenha o espaço de busca mapeando Massa Total versus Tensão Máxima Atuante."""
        ax.clear()
        ax.set_facecolor("white")
        ax.tick_params(colors=self.PRETO, labelsize=8)
        ax.set_xlabel("Massa total  (g)", fontsize=9, color=self.PRETO)
        ax.set_ylabel("Tensão Máx. Atuante (MPa)", fontsize=9, color=self.PRETO)
        ax.set_title("Espaço de busca", fontsize=10, color=self.PRETO, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color=self.CINZA)
        
        for sp in ax.spines.values():
            sp.set_color(self.PRETO)

        if pareto_pts:
            fp = np.asarray(pareto_pts, dtype=float)
            if len(fp) > 0:
                fp = fp[fp[:, 0].argsort()]
                ax.plot(fp[:, 0], fp[:, 1], color=self.ROXO, lw=1.6, zorder=4)
                ax.scatter(fp[:, 0], fp[:, 1], s=28, color=self.ROXO,
                           edgecolors="white", linewidths=0.5,
                           label="Front. Pareto", zorder=5)

        if best_obj:
            ax.scatter([best_obj[0]], [best_obj[1]], s=70, color=self.AZUL,
                       edgecolors="white", linewidths=0.8,
                       label="Melhor indiv.", zorder=6)

        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9,
                  markerscale=1.1, handletextpad=0.4, edgecolor=self.PRETO)

        ax.text(0.03, 0.03, f"Gen {gen}/{total_gens}",
                transform=ax.transAxes, fontsize=8, color=self.PRETO,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=self.PRETO, lw=0.6))

    def gerar_animacao(self, historico_melhor, historico_fronteira, total_gens, caminho_gif):
        """Renderiza e salva a evolução do algoritmo genético em um arquivo GIF dual-plot."""
        fig = plt.figure(figsize=(14, 6), facecolor="white")
        ax_t = fig.add_axes([0.04, 0.10, 0.56, 0.83])
        ax_s = fig.add_axes([0.67, 0.11, 0.29, 0.80])

        def update(frame_idx):
            self._desenhar_trelica(ax_t, historico_melhor[frame_idx], frame_idx + 1, total_gens)
            best_obj = [historico_melhor[frame_idx].massa_g, historico_melhor[frame_idx].tensao_max_mpa]
            self._desenhar_scatter(ax_s, historico_fronteira[frame_idx], best_obj, frame_idx + 1, total_gens)
            return []

        ani = animation.FuncAnimation(fig, update, frames=total_gens, interval=140, repeat=False, blit=False)
        ani.save(caminho_gif, writer="pillow", fps=7, dpi=90)
        plt.close(fig)

    def gerar_imagem_otima(self, melhor_ind, fronteira_final, total_gens, caminho_png):
        """Renderiza e salva a figura estática final da melhor treliça e da fronteira de Pareto."""
        fig = plt.figure(figsize=(14, 6), facecolor="white")
        ax_t = fig.add_axes([0.04, 0.10, 0.56, 0.83])
        ax_s = fig.add_axes([0.67, 0.11, 0.29, 0.80])
        
        self._desenhar_trelica(ax_t, melhor_ind, total_gens, total_gens)
        best_obj = [melhor_ind.massa_g, melhor_ind.tensao_max_mpa]
        self._desenhar_scatter(ax_s, fronteira_final, best_obj, total_gens, total_gens)
        
        fig.savefig(caminho_png, dpi=130, facecolor="white")
        plt.close(fig)