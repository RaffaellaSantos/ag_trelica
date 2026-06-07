"""
AG para treliça Howe 3D — dois pórticos planos paralelos + travamentos laterais.
(Atividade 9 — versão 3D fisicamente construível)

Cromossomo (27 genes):
  Bloco 1 — contínuo (10 genes):
    p1..p4  ∈ [3, 11.4] cm  largura de cada painel  (Σ = 40 cm)
    h1..h5  ∈ [3, 11.4] cm  altura de cada vertical
    w       ∈ [2,  6.0] cm  largura entre os dois pórticos

  Bloco 2 — discreto (17 genes):
    n1..n17 ∈ {1, 2, 3}     palitos colados por barra (seção de cada pórtico)

Modelo estrutural:
  • Cada pórtico plano é analisado isoladamente (2D) com carga P/2.
  • O pórtico falha quando P/2 = carga_ruptura_frame → P_3D = 2 × P_frame.
  • Deslocamento Δ_C idem ao 2D (força por pórtico = P_3D/2 = P_frame_ruptura).
  • 10 travamentos (A–A', B–B', …, J–J') com 1 palito cada, comprimento w.
  • Massa total = 2 × massa_pórtico + massa_travamentos.
  • R5 (Σni ≤ 150): conta 2 × Σni_pórtico + 10 travamentos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import numpy as np


# ── Constantes ────────────────────────────────────────────────────────────────
G  = 9.80665
SPAN_CM                  = 40.0
PANEL_MIN_CM, PANEL_MAX_CM = 3.0, 11.4
HEIGHT_MIN_CM, HEIGHT_MAX_CM = 3.0, 11.4
WIDTH_MIN_CM,  WIDTH_MAX_CM  = 2.0, 6.0   # largura entre pórticos
STICK_LENGTH_CM          = 11.4
MAX_STICKS               = 150
MAX_MASS_G               = 600.0
N_BRACE                  = 1               # palitos por travamento (fixo)
N_BRACE_BARS             = 10             # nós: A–A' … J–J'

E            = 3_500e6          # Pa
RHO          = 510.0            # kg/m³
SIGMA_T_ALLOW = 55e6 / 2.0     # Pa  (tração,     Sg = 2)
SIGMA_C_ALLOW = 35e6 / 2.0     # Pa  (compressão, Sg = 2)
B = 0.010                       # m
T = 0.002                       # m

LOAD_NODE        = "C"
DIAGONAL_INDICES = [13, 14, 15, 16]

BAR_NAMES = [
    "A-B", "B-C", "C-D", "D-E",
    "F-G", "G-H", "H-I", "I-J",
    "A-F", "G-B", "H-C", "I-D", "E-J",
    "F-B", "G-C", "I-C", "J-D",
]

CONNECTIVITY = [
    ("A","B"),("B","C"),("C","D"),("D","E"),
    ("F","G"),("G","H"),("H","I"),("I","J"),
    ("A","F"),("G","B"),("H","C"),("I","D"),("E","J"),
    ("F","B"),("G","C"),("I","C"),("J","D"),
]

_BAR_TYPE = {
    "A-B":"Banzo inferior","B-C":"Banzo inferior",
    "C-D":"Banzo inferior","D-E":"Banzo inferior",
    "F-G":"Banzo superior","G-H":"Banzo superior",
    "H-I":"Banzo superior","I-J":"Banzo superior",
    "A-F":"Vertical ext.", "G-B":"Vertical int.",
    "H-C":"Vertical central","I-D":"Vertical int.",
    "E-J":"Vertical ext.",
    "F-B":"Diagonal","G-C":"Diagonal",
    "I-C":"Diagonal","J-D":"Diagonal",
}


# ── Cromossomo ────────────────────────────────────────────────────────────────
@dataclass
class Individual:
    p_cm: np.ndarray      # (4,)
    h_cm: np.ndarray      # (5,)
    w_cm: float           # largura entre pórticos
    n:    np.ndarray      # (17,) ∈ {1,2,3}
    fitness:  float = 0.0
    feasible: bool  = False
    data: dict = field(default_factory=dict)


# ── Geometria e seção ─────────────────────────────────────────────────────────
def section_area(n_sticks: int) -> float:
    return B * n_sticks * T


def repair_panels(p_cm: np.ndarray) -> np.ndarray:
    p = np.clip(np.array(p_cm, dtype=float), PANEL_MIN_CM, PANEL_MAX_CM)
    for _ in range(100):
        diff = SPAN_CM - float(np.sum(p))
        if abs(diff) < 1e-9:
            break
        free = p < PANEL_MAX_CM - 1e-12 if diff > 0 else p > PANEL_MIN_CM + 1e-12
        if not np.any(free):
            break
        p[free] += diff / np.sum(free)
        p = np.clip(p, PANEL_MIN_CM, PANEL_MAX_CM)
    return p


def node_coords(p_cm: np.ndarray, h_cm: np.ndarray) -> dict[str, np.ndarray]:
    p = p_cm / 100.0
    h = h_cm / 100.0
    x = [0.0, p[0], p[0]+p[1], p[0]+p[1]+p[2], SPAN_CM/100.0]
    return {
        "A": np.array([x[0],0.0]), "B": np.array([x[1],0.0]),
        "C": np.array([x[2],0.0]), "D": np.array([x[3],0.0]),
        "E": np.array([x[4],0.0]),
        "F": np.array([x[0],h[0]]), "G": np.array([x[1],h[1]]),
        "H": np.array([x[2],h[2]]), "I": np.array([x[3],h[3]]),
        "J": np.array([x[4],h[4]]),
    }


# ── Análise estrutural (pórtico plano, carga unitária) ───────────────────────
def solve_axial_forces(nodes: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Equilíbrio de nós 2D sob 1 N em C. Retorna (forças_unitárias, comprimentos)."""
    n_bars = len(CONNECTIVITY)
    names  = list(nodes)
    idx    = {nm: i for i, nm in enumerate(names)}
    n_nodes = len(names)

    lengths = np.empty(n_bars)
    cos_sin = np.empty((n_bars, 2))
    for e, (na, nb) in enumerate(CONNECTIVITY):
        d = nodes[nb] - nodes[na]
        L = float(math.hypot(d[0], d[1]))
        if L < 1e-12:
            raise ValueError(f"Barra {BAR_NAMES[e]}: comprimento nulo.")
        lengths[e] = L
        cos_sin[e] = d[0]/L, d[1]/L

    A_eq = np.zeros((2*n_nodes, n_bars+3))
    b_eq = np.zeros(2*n_nodes)

    for e, (na, nb) in enumerate(CONNECTIVITY):
        cx, cy = cos_sin[e]
        ia, ib = idx[na], idx[nb]
        A_eq[2*ia,   e] += cx;  A_eq[2*ia+1, e] += cy
        A_eq[2*ib,   e] -= cx;  A_eq[2*ib+1, e] -= cy

    A_eq[2*idx["A"],     n_bars]   = 1.0
    A_eq[2*idx["A"]+1,   n_bars+1] = 1.0
    A_eq[2*idx["E"]+1,   n_bars+2] = 1.0
    b_eq[2*idx[LOAD_NODE]+1]       = 1.0

    try:
        x = np.linalg.solve(A_eq, b_eq)
    except np.linalg.LinAlgError:
        raise ValueError("Sistema singular: geometria instável.")
    return x[:n_bars], lengths


def ptv_displacement_unit(axial_unit, lengths, areas) -> float:
    return float(np.sum(axial_unit**2 * lengths / (E * areas)))


# ── Massa e palitos — modelo 3D ───────────────────────────────────────────────
def mass_g_3d(lengths_m: np.ndarray, sections: np.ndarray, w_m: float) -> float:
    """2 pórticos idênticos + 10 travamentos de comprimento w."""
    frame_vol  = sum(section_area(int(n)) * float(L)
                     for L, n in zip(lengths_m, sections))
    brace_vol  = N_BRACE_BARS * section_area(N_BRACE) * w_m
    return RHO * (2.0 * frame_vol + brace_vol) * 1000.0


def stick_count_3d(sections: np.ndarray) -> int:
    """R5: 2 × Σni_pórtico + 10 travamentos (N_BRACE=1 cada)."""
    return 2 * int(np.sum(sections)) + N_BRACE_BARS * N_BRACE


def bar_capacity(unit_axial_n: float, n_sticks: int) -> float:
    area = section_area(n_sticks)
    return SIGMA_T_ALLOW * area if unit_axial_n > 0.0 else SIGMA_C_ALLOW * area


# ── Avaliação ─────────────────────────────────────────────────────────────────
def evaluate(ind: Individual) -> Individual:
    ind.p_cm = repair_panels(ind.p_cm)
    ind.h_cm = np.clip(ind.h_cm, HEIGHT_MIN_CM, HEIGHT_MAX_CM)
    ind.w_cm = float(np.clip(ind.w_cm, WIDTH_MIN_CM, WIDTH_MAX_CM))
    ind.n    = np.clip(np.rint(ind.n), 1, 3).astype(int)

    areas = np.array([section_area(int(x)) for x in ind.n])

    try:
        axial_unit, lengths_m = solve_axial_forces(
            node_coords(ind.p_cm, ind.h_cm))
    except ValueError as err:
        ind.fitness, ind.feasible = 0.0, False
        ind.data = {"penalty_reasons": [str(err)]}
        return ind

    lengths_cm = lengths_m * 100.0
    w_m        = ind.w_cm / 100.0
    reasons    = []

    if np.any(lengths_cm[DIAGONAL_INDICES] > STICK_LENGTH_CM + 1e-9):
        reasons.append("R4: diagonal > 11,4 cm")

    total_sticks = stick_count_3d(ind.n)
    if total_sticks > MAX_STICKS:
        reasons.append(f"R5: {total_sticks} palitos > 150")

    total_mass = mass_g_3d(lengths_m, ind.n, w_m)
    if total_mass >= MAX_MASS_G:
        reasons.append(f"R6: massa {total_mass:.1f} g ≥ 600 g")

    # Ruptura por pórtico; carga 3D = 2 × ruptura do pórtico (P/2 por pórtico)
    rupture = np.full(len(CONNECTIVITY), np.inf)
    for i, coeff in enumerate(axial_unit):
        if abs(coeff) < 1e-12:
            continue
        rupture[i] = bar_capacity(float(coeff), int(ind.n[i])) / abs(coeff)

    load_n_frame = float(np.min(rupture))   # ruptura de 1 pórtico sob P/2
    load_n_3d    = 2.0 * load_n_frame       # carga total da estrutura 3D
    load_kg_3d   = load_n_3d / G

    feasible = not reasons
    ind.fitness  = (load_kg_3d / total_mass) if feasible and total_mass > 0 else 0.0
    ind.feasible = feasible

    stresses_mpa = (axial_unit * load_n_frame / areas) / 1e6
    delta_unit   = ptv_displacement_unit(axial_unit, lengths_m, areas)
    critical     = int(np.argmin(rupture))

    ind.data = {
        "penalty_reasons":        reasons,
        "axial_unit_n":           axial_unit,
        "lengths_cm":             lengths_cm,
        "w_cm":                   ind.w_cm,
        "mass_g":                 total_mass,
        "stick_count":            total_sticks,
        "theoretical_load_n":     load_n_3d,
        "theoretical_load_kg":    load_kg_3d,
        "load_n_per_frame":       load_n_frame,
        "critical_bar":           BAR_NAMES[critical],
        "stresses_at_rupture_mpa":stresses_mpa,
        "delta_c_at_rupture_mm":  delta_unit * load_n_frame * 1000.0,
    }
    return ind


# ── AG ────────────────────────────────────────────────────────────────────────
def _pareto_front(pts):
    result = []
    for i, (mi, li) in enumerate(pts):
        if not any(j != i and mj <= mi and lj >= li and (mj < mi or lj > li)
                   for j, (mj, lj) in enumerate(pts)):
            result.append((mi, li))
    return [[m, l] for m, l in sorted(result)]


def random_individual() -> Individual:
    p = repair_panels(np.random.uniform(PANEL_MIN_CM, PANEL_MAX_CM, 4))
    h = np.random.uniform(HEIGHT_MIN_CM, HEIGHT_MAX_CM, 5)
    w = random.uniform(WIDTH_MIN_CM, WIDTH_MAX_CM)
    n = np.random.randint(1, 4, 17)
    return evaluate(Individual(p, h, w, n))


def tournament(population, k):
    return max(random.sample(population, k), key=lambda x: x.fitness)


def crossover(p1: Individual, p2: Individual,
              alpha: float = 0.35) -> tuple[Individual, Individual]:
    def blx(a, b):
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        d = hi - lo
        return (np.random.uniform(lo-alpha*d, hi+alpha*d),
                np.random.uniform(lo-alpha*d, hi+alpha*d))

    pa, pb = blx(p1.p_cm, p2.p_cm)
    ha, hb = blx(p1.h_cm, p2.h_cm)

    # BLX para w (escalar)
    lo_w = min(p1.w_cm, p2.w_cm); hi_w = max(p1.w_cm, p2.w_cm)
    dw   = (hi_w - lo_w) * alpha
    wa   = random.uniform(lo_w - dw, hi_w + dw)
    wb   = random.uniform(lo_w - dw, hi_w + dw)

    mask = np.random.rand(17) < 0.5
    na   = np.where(mask, p1.n, p2.n)
    nb   = np.where(mask, p2.n, p1.n)

    c1 = Individual(repair_panels(pa),
                    np.clip(ha, HEIGHT_MIN_CM, HEIGHT_MAX_CM),
                    float(np.clip(wa, WIDTH_MIN_CM, WIDTH_MAX_CM)),
                    na.astype(int))
    c2 = Individual(repair_panels(pb),
                    np.clip(hb, HEIGHT_MIN_CM, HEIGHT_MAX_CM),
                    float(np.clip(wb, WIDTH_MIN_CM, WIDTH_MAX_CM)),
                    nb.astype(int))
    return c1, c2


def mutate(ind: Individual,
           p_cont: float = 0.20, p_disc: float = 0.08) -> Individual:
    p, h, n = ind.p_cm.copy(), ind.h_cm.copy(), ind.n.copy()
    w = ind.w_cm

    for i in range(4):
        if random.random() < p_cont:
            p[i] += random.gauss(0.0, 0.80)
    for i in range(5):
        if random.random() < p_cont:
            h[i] += random.gauss(0.0, 0.60)
    if random.random() < p_cont:
        w += random.gauss(0.0, 0.40)
    for i in range(17):
        if random.random() < p_disc:
            choices = [1, 2, 3]
            choices.remove(int(n[i]))
            n[i] = random.choice(choices)

    ind.p_cm = repair_panels(p)
    ind.h_cm = np.clip(h, HEIGHT_MIN_CM, HEIGHT_MAX_CM)
    ind.w_cm = float(np.clip(w, WIDTH_MIN_CM, WIDTH_MAX_CM))
    ind.n    = n.astype(int)
    return ind


def run_ga(
    population_size: int = 250,
    generations:     int = 400,
    crossover_rate:  float = 0.90,
    elitism:         int = 8,
    tournament_k:    int = 4,
    seed:            int | None = 42,
) -> tuple[Individual, dict]:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    population = [random_individual() for _ in range(population_size)]
    best       = max(population, key=lambda x: x.fitness)

    h_best, h_cloud, h_pareto, h_best_obj = [], [], [], []

    for gen in range(1, generations + 1):
        population.sort(key=lambda x: x.fitness, reverse=True)
        new_pop = population[:elitism]

        while len(new_pop) < population_size:
            p1, p2 = tournament(population, tournament_k), tournament(population, tournament_k)
            c1, c2 = crossover(p1, p2) if random.random() < crossover_rate else (
                Individual(p1.p_cm.copy(), p1.h_cm.copy(), p1.w_cm, p1.n.copy()),
                Individual(p2.p_cm.copy(), p2.h_cm.copy(), p2.w_cm, p2.n.copy()),
            )
            new_pop.append(evaluate(mutate(c1)))
            new_pop.append(evaluate(mutate(c2)))

        population = new_pop[:population_size]
        gen_best   = max(population, key=lambda x: x.fitness)
        if gen_best.fitness > best.fitness:
            best = gen_best

        cloud     = []
        feas_pts  = []
        for ind in population:
            if ind.data:
                m = float(ind.data.get("mass_g", 0.0))
                l = float(ind.data.get("theoretical_load_kg", 0.0))
                f = 1.0 if ind.feasible else 0.0
                cloud.append([m, l, f])
                if ind.feasible:
                    feas_pts.append((m, l))

        h_best.append(best)
        h_cloud.append(cloud)
        h_pareto.append(_pareto_front(feas_pts))
        h_best_obj.append([float(best.data.get("mass_g", 0.0)),
                           float(best.data.get("theoretical_load_kg", 0.0))])

        if gen == 1 or gen % 25 == 0:
            viaveis = sum(1 for x in population if x.feasible)
            print(f"Geração {gen:4d} | fitness = {best.fitness:.6f} "
                  f"| carga = {best.data['theoretical_load_kg']:.3f} kg "
                  f"| massa = {best.data['mass_g']:.2f} g "
                  f"| w = {best.w_cm:.2f} cm "
                  f"| viáveis = {viaveis}/{population_size}")

    return best, {"best": h_best, "cloud": h_cloud,
                  "pareto": h_pareto, "best_obj": h_best_obj}


# ── Relatórios ────────────────────────────────────────────────────────────────
def print_report(best: Individual) -> None:
    d = best.data
    print("\n" + "="*72)
    print("MELHOR TRELIÇA 3D ENCONTRADA")
    print("="*72)
    print(f"\nPainéis p1..p4 : {np.round(best.p_cm,3)}  Σ={sum(best.p_cm):.1f} cm")
    print(f"Alturas h1..h5 : {np.round(best.h_cm,3)}")
    print(f"Largura w      : {best.w_cm:.3f} cm")
    print(f"\n{'Barra':>5} | {'n':>2} | {'L(cm)':>7}")
    print("-"*22)
    for bar, n, L in zip(BAR_NAMES, best.n, d["lengths_cm"]):
        print(f"{bar:>5} | {int(n):>2} | {L:7.3f}")
    print(f"\nFitness 3D      : {best.fitness:.6f} kg/g")
    print(f"Carga 3D        : {d['theoretical_load_kg']:.4f} kg")
    print(f"Carga por pórtico: {d['load_n_per_frame']/G:.4f} kg")
    print(f"Massa total 3D  : {d['mass_g']:.4f} g")
    print(f"Palitos (R5)    : {d['stick_count']}")
    print(f"Barra crítica   : {d['critical_bar']}")
    print(f"Δ_C (ruptura)   : {d['delta_c_at_rupture_mm']:.4f} mm")
    print(f"Viável          : {best.feasible}")


def print_report_builder(best: Individual) -> None:
    """Relatório 3D completo para os construtores."""
    d          = best.data
    p, h       = best.p_cm, best.h_cm
    w_cm       = best.w_cm
    n_arr      = best.n
    lengths_cm = d["lengths_cm"]
    load_n_3d  = d["theoretical_load_n"]
    load_kg_3d = d["theoretical_load_kg"]
    load_n_fr  = d["load_n_per_frame"]
    m_g        = d["mass_g"]
    stresses   = d["stresses_at_rupture_mpa"]
    axial_unit = d["axial_unit_n"]
    axial_fr   = axial_unit * load_n_fr   # forças em cada pórtico na ruptura
    delta_mm   = d["delta_c_at_rupture_mm"]
    crit       = d["critical_bar"]

    W   = 78
    SEP = "═" * W
    sep = "─" * W

    x_nós = [0.0, p[0], p[0]+p[1], p[0]+p[1]+p[2], 40.0]
    nós = {
        "A":(x_nós[0],0.0),"B":(x_nós[1],0.0),"C":(x_nós[2],0.0),
        "D":(x_nós[3],0.0),"E":(x_nós[4],0.0),
        "F":(x_nós[0],h[0]),"G":(x_nós[1],h[1]),"H":(x_nós[2],h[2]),
        "I":(x_nós[3],h[3]),"J":(x_nós[4],h[4]),
    }

    # ── inventário físico ─────────────────────────────────────────────────────
    inv = []
    phys_frame = 0
    for bar, ni, L in zip(BAR_NAMES, n_arr, lengths_cm):
        segs = math.ceil(L / 11.4)
        phys = int(ni) * segs
        phys_frame += phys
        inv.append({"bar":bar,"tipo":_BAR_TYPE.get(bar,""),
                    "L":L,"n":int(ni),"segs":segs,"phys":phys,
                    "area_cm2":1.0*int(ni)*0.2})

    # travamentos: de 1 palito cortam-se várias peças de w_cm
    cuts_per_stick = max(1, int(11.4 / w_cm))
    brace_sticks   = math.ceil(N_BRACE_BARS / cuts_per_stick)
    total_phys     = 2 * phys_frame + brace_sticks

    print(f"\n{SEP}")
    print("  RELATÓRIO DE CONSTRUÇÃO — TRELIÇA HOWE 3D DE PALITOS".center(W))
    print("  Atividade 9  |  EST/UEA  |  Algoritmos de Otimização".center(W))
    print(SEP)

    print(f"\n  MATERIAL")
    print(f"  {'Madeira':28s}: Bétula (palito de picolé)")
    print(f"  {'E':28s}: 3 500 MPa   ρ = 510 kg/m³")
    print(f"  {'σt adm (Sg=2)':28s}: 27.5 MPa   σc adm: 17.5 MPa")
    print(f"  {'Palito':28s}: 11.4 × 1.0 × 0.2 cm")

    print(f"\n{sep}")
    print("  1. GEOMETRIA")
    print(sep)
    print(f"\n  Painéis (p1…p4): {' | '.join(f'{v:.3f} cm' for v in p)}   Σ = {sum(p):.1f} cm")
    print(f"  Alturas (h1…h5): {' | '.join(f'{v:.3f} cm' for v in h)}")
    print(f"  Largura (w)    : {w_cm:.3f} cm  (distância entre pórticos)")
    print(f"\n  {'Nó':>3}  {'x (cm)':>8}  {'y (cm)':>8}  Nível")
    print(f"  {'---':>3}  {'--------':>8}  {'--------':>8}  --------")
    for nm,(xn,yn) in nós.items():
        print(f"  {nm:>3}  {xn:8.3f}  {yn:8.3f}  "
              f"{'Banzo inferior' if yn==0 else 'Banzo superior'}")
    print(f"\n  Cada pórtico é replicado com deslocamento z = {w_cm:.3f} cm.")
    print(f"  Nós espelhados: A'…J'  (mesmas x,y — z = {w_cm:.3f} cm)")

    print(f"\n{sep}")
    print("  2. INVENTÁRIO DE BARRAS  (por pórtico — replicar × 2)")
    print(sep)
    print(f"\n  {'Barra':>5}  {'Tipo':18}  {'L (cm)':>7}  {'n':>2}  {'A(cm²)':>6}  {'Palitos/pórtico':>15}")
    print(f"  {'─'*5}  {'─'*18}  {'─'*7}  {'─'*2}  {'─'*6}  {'─'*15}")
    for r in inv:
        print(f"  {r['bar']:>5}  {r['tipo']:18}  {r['L']:7.3f}  "
              f"{r['n']:>2}  {r['area_cm2']:6.4f}  {r['phys']:>15}")
    print(f"\n  Subtotal por pórtico      : {phys_frame} palitos")
    print(f"  2 pórticos                : {2*phys_frame} palitos")
    print(f"  Travamentos (10 × w={w_cm:.1f}cm): {brace_sticks} palitos físicos "
          f"(cortar {cuts_per_stick} peças de {w_cm:.1f} cm por palito)")
    print(f"  TOTAL FÍSICO              : {total_phys} palitos")
    print(f"  Recomenda-se comprar      : {int(total_phys*1.15)+1} palitos (+15 % folga)")
    print(f"\n  R5 (Σni ≤ 150): 2×{int(sum(n_arr))} + 10 = {stick_count_3d(n_arr)}  "
          f"{'[OK]' if stick_count_3d(n_arr)<=150 else '[VIOLA]'}")

    print(f"\n{sep}")
    print(f"  3. ESFORÇOS NA RUPTURA  "
          f"(P_total = {load_kg_3d:.4f} kg  |  P/pórtico = {load_n_fr/G:.4f} kg)")
    print(sep)
    print(f"\n  {'Barra':>5}  {'Tipo':18}  {'N (N)':>9}  {'σ (MPa)':>8}  "
          f"{'σadm':>6}  {'FS':>5}  Estado")
    print(f"  {'─'*5}  {'─'*18}  {'─'*9}  {'─'*8}  {'─'*6}  {'─'*5}  ─────────")
    for bar, ni, nr, st in zip(BAR_NAMES, n_arr, axial_fr, stresses):
        if abs(nr) < 1e-9:
            estado="zero      "; σadm=27.5; fs_s="  —  "
        elif nr > 0:
            estado="tração    "; σadm=27.5
            fs_s=f"{σadm/abs(st):.2f}" if abs(st)>1e-9 else "  ∞  "
        else:
            estado="compressão"; σadm=17.5
            fs_s=f"{σadm/abs(st):.2f}" if abs(st)>1e-9 else "  ∞  "
        ok="OK" if abs(st)<=σadm+1e-6 else "VIOLA"
        print(f"  {bar:>5}  {_BAR_TYPE.get(bar,''):18}  {nr:+9.2f}  "
              f"{st:+8.3f}  {σadm:>6.1f}  {fs_s:>5}  {estado} [{ok}]")

    print(f"\n{sep}")
    print("  4. RESULTADOS GLOBAIS")
    print(sep)
    print(f"\n  {'Massa total (2 pórticos + travamentos)':42s}: {m_g:.4f} g")
    print(f"  {'Carga de ruptura total (3D)':42s}: {load_kg_3d:.4f} kg")
    print(f"  {'Carga por pórtico na ruptura':42s}: {load_n_fr/G:.4f} kg")
    print(f"  {'Fitness 3D (carga/massa)':42s}: {best.fitness:.6f} kg/g")
    print(f"  {'Deslocamento Δ_C na ruptura':42s}: {delta_mm:.4f} mm")
    print(f"  {'Barra crítica':42s}: {crit}  ({_BAR_TYPE.get(crit,'')})")
    print(f"  {'Largura entre pórticos (w)':42s}: {w_cm:.3f} cm")
    print(f"  {'Palitos totais (físicos estimados)':42s}: {total_phys}")
    print(f"  {'Solução viável':42s}: {'Sim ✓' if best.feasible else 'Não ✗'}")

    print(f"\n{sep}")
    print("  5. GUIA DE CORTE E MONTAGEM 3D")
    print(sep)
    a_cortar = [r for r in inv if r["L"] < 11.35]
    print(f"\n  CORTES NECESSÁRIOS (por pórtico — repetir para o 2º pórtico):")
    for r in a_cortar:
        sobra = 11.4 - r["L"]
        print(f"    {r['bar']:>5}  {r['L']:.3f} cm  →  retirar {sobra:.3f} cm  "
              f"({r['n']} camada(s))")
    print(f"\n  TRAVAMENTOS (10 peças de {w_cm:.2f} cm cada):")
    print(f"    De 1 palito de 11.4 cm cortam-se {cuts_per_stick} travamentos.")
    print(f"    Total de palitos para travamentos: {brace_sticks}")
    print(f"""
  SEQUÊNCIA DE MONTAGEM:
    1. Trace 2 gabaritos idênticos em papel milimetrado (um por pórtico).
    2. Monte o PÓRTICO 1 completo sobre o gabarito.
    3. Monte o PÓRTICO 2 idêntico sobre o segundo gabarito.
    4. Aguarde cura completa dos dois pórticos (≥ 30 min).
    5. Posicione os pórticos lado a lado com {w_cm:.1f} cm de separação.
    6. Cole os 10 TRAVAMENTOS conectando nós correspondentes:
         Banzo inferior : A─A'  B─B'  C─C'  D─D'  E─E'
         Banzo superior : F─F'  G─G'  H─H'  I─I'  J─J'
    7. Use esquadro para garantir 90° entre os pórticos e os travamentos.

  PONTOS CRÍTICOS:
    • Nó C e C' (centro do banzo inferior) = ponto de carga.
      Pendurar o balde no ponto médio entre C e C'.
    • Vão livre entre apoios: 40 cm.
    • Pese a estrutura completa antes do ensaio (previsto: {m_g:.1f} g).

  PREVISÃO DO ENSAIO:
    Carga de ruptura: {load_kg_3d:.2f} kg   |   Eficiência: {best.fitness:.4f} kg/g
""")
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    melhor, historico = run_ga()
    print_report(melhor)
    print_report_builder(melhor)

    try:
        from visualizer_howe import animate_evolution, save_final_image
        cloud_total = [pt for gen in historico["cloud"] for pt in gen]
        animate_evolution(historico["best"], historico["cloud"],
                          historico["pareto"], historico["best_obj"],
                          total_gens=400, gif_path="best_truss_3d.gif")
        save_final_image(melhor, cloud_total,
                         historico["pareto"][-1], historico["best_obj"][-1],
                         total_gens=400, png_path="best_truss_3d_final.png")
    except Exception as exc:
        print(f"\n[Visualizador] {exc}")
