"""
Algoritmo Genético para otimização de treliça Howe de 4 painéis com palitos.
(Atividade 9 — extensão da treliça de aço da Atividade 8)

Cromossomo (26 genes):
- p1..p4 (contínuo): larguras dos painéis, em cm, com soma reparada para 40 cm
- h1..h5 (contínuo): alturas das verticais, em cm
- n1..n17 (discreto): palitos colados por barra, em {1, 2, 3}

Análise estrutural (treliça plana):
- Forças axiais: equilíbrio de nós com carga unitária no nó C (P = 1)
- Deslocamento em C: Princípio dos Trabalhos Virtuais (PTV)
- Como o sistema é linear, a força real em cada barra é P · n_i, onde n_i é a
  força da carga unitária; por isso uma única análise resolve forças e PTV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import sys
from pathlib import Path

import numpy as np

# Reaproveitamento da Atividade 8 — equilíbrio de nós já implementado em app/utils.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.utils import Utils as _AppUtils


# =========================
# Constantes do problema (PDF)
# =========================

G = 9.80665
SPAN_CM = 40.0
PANEL_MIN_CM, PANEL_MAX_CM = 3.0, 11.4
HEIGHT_MIN_CM, HEIGHT_MAX_CM = 3.0, 11.4
STICK_LENGTH_CM = 11.4
MAX_STICKS = 150
MAX_MASS_G = 600.0

# Coeficientes de penalidade aditiva — mesmo padrão da Atividade 8
P_R4 = 10.0    # (kg/g) por cm de diagonal além de 11,4 cm
P_R5 = 0.010   # (kg/g) por palito além de 150
P_R6 = 0.005   # (kg/g) por grama além de 600 g

E = 3_500e6                 # Pa
RHO = 510.0                 # kg/m³
SIGMA_T_ALLOW = 55e6 / 2.0  # Pa (tração, Sg = 2)
SIGMA_C_ALLOW = 35e6 / 2.0  # Pa (compressão, Sg = 2)
B = 0.010                   # m, largura do palito
T = 0.002                   # m, espessura de 1 palito
LOAD_NODE = "C"
DIAGONAL_INDICES = [13, 14, 15, 16]  # F-B, G-C, I-C, J-D (restrição R4)

BAR_NAMES = [
    "A-B", "B-C", "C-D", "D-E",        # 4 banzo inferior
    "F-G", "G-H", "H-I", "I-J",        # 4 banzo superior
    "A-F", "G-B", "H-C", "I-D", "E-J", # 5 verticais
    "F-B", "G-C", "I-C", "J-D",        # 4 diagonais Howe
]

CONNECTIVITY = [
    ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"),
    ("F", "G"), ("G", "H"), ("H", "I"), ("I", "J"),
    ("A", "F"), ("G", "B"), ("H", "C"), ("I", "D"), ("E", "J"),
    ("F", "B"), ("G", "C"), ("I", "C"), ("J", "D"),
]


@dataclass
class Individual:
    p_cm: np.ndarray            # (4,) larguras dos painéis
    h_cm: np.ndarray            # (5,) alturas das verticais
    n: np.ndarray               # (17,) palitos por barra, em {1, 2, 3}
    fitness: float = 0.0
    objectives: tuple = field(default_factory=lambda: (1e12, 1e12))  # (massa+pen, -carga+pen)
    feasible: bool = False
    data: dict = field(default_factory=dict)


# =========================
# Geometria e propriedades
# =========================

def section_area(n_sticks: int) -> float:
    """Área da seção com n palitos colados face a face, em m²."""
    return B * (n_sticks * T)


def repair_panels(p_cm: np.ndarray) -> np.ndarray:
    """Ajusta p1..p4 para 3 <= pi <= 11,4 cm e soma = 40 cm (R1, R3)."""
    p = np.clip(np.array(p_cm, dtype=float), PANEL_MIN_CM, PANEL_MAX_CM)
    for _ in range(100):
        diff = SPAN_CM - float(np.sum(p))
        if abs(diff) < 1e-9:
            break
        if diff > 0:
            free = p < PANEL_MAX_CM - 1e-12
        else:
            free = p > PANEL_MIN_CM + 1e-12
        if not np.any(free):
            break
        p[free] += diff / np.sum(free)
        p = np.clip(p, PANEL_MIN_CM, PANEL_MAX_CM)
    return p


def node_coords(p_cm: np.ndarray, h_cm: np.ndarray) -> dict[str, np.ndarray]:
    """Coordenadas (x, y) dos 10 nós, em metros."""
    p = p_cm / 100.0
    h = h_cm / 100.0
    x = [0.0, p[0], p[0] + p[1], p[0] + p[1] + p[2], SPAN_CM / 100.0]
    return {
        "A": np.array([x[0], 0.0]), "B": np.array([x[1], 0.0]),
        "C": np.array([x[2], 0.0]), "D": np.array([x[3], 0.0]),
        "E": np.array([x[4], 0.0]),
        "F": np.array([x[0], h[0]]), "G": np.array([x[1], h[1]]),
        "H": np.array([x[2], h[2]]), "I": np.array([x[3], h[3]]),
        "J": np.array([x[4], h[4]]),
    }


# =========================
# Análise: equilíbrio de nós + PTV
# =========================

def solve_axial_forces(nodes: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Equilíbrio de nós sob carga unitária (1 N) para baixo em C.
    Reaproveitado de app/utils.py (Atividade 8): Utils.calcular_reacoes_e_forcas.
    Apoios: A (pino) em índice 0 e E (rolete) em índice 4 — coincidem com
    o hardcode da Atividade 8 (A[0], A[1], A[2*4+1]).
    Retorna (forças axiais unitárias, comprimentos), ambos por barra.
    """
    names = list(nodes)
    idx   = {nm: i for i, nm in enumerate(names)}

    nos_arr    = np.array([nodes[nm] for nm in names])
    barras_arr = np.array([[idx[na], idx[nb]] for na, nb in CONNECTIVITY])

    lengths = np.empty(len(CONNECTIVITY))
    angulos = []
    for e, (na, nb) in enumerate(CONNECTIVITY):
        d = nodes[nb] - nodes[na]
        L = float(math.hypot(d[0], d[1]))
        if L < 1e-12:
            raise ValueError(f"Barra {BAR_NAMES[e]}: comprimento nulo.")
        lengths[e] = L
        angulos.append((d[0] / L, d[1] / L))

    # Carga unitária (1 N) para baixo no nó de carga
    carga_unit = [(idx[LOAD_NODE], 0.0, -1.0)]

    try:
        axial_unit, _, _ = _AppUtils().calcular_reacoes_e_forcas(
            nos_arr, barras_arr, angulos,
            forcas_externas=carga_unit,
            forcas_virtuais=carga_unit,
        )
    except np.linalg.LinAlgError:
        raise ValueError("Sistema singular: geometria instável.")

    return axial_unit, lengths


def ptv_displacement_unit(axial_unit: np.ndarray, lengths: np.ndarray, areas: np.ndarray) -> float:
    """
    Deslocamento vertical em C sob carga unitária, pelo PTV:
    Δ_C = Σ (n_i · n_i · L_i) / (E · A_i).
    Para carga P, o deslocamento real é P · Δ_C (sistema linear).
    """
    return float(np.sum(axial_unit ** 2 * lengths / (E * areas)))


# =========================
# Capacidade, massa e restrições
# =========================

def stick_count(sections: np.ndarray) -> int:
    """Total de palitos Σni conforme R5 do enunciado."""
    return int(np.sum(sections))


def mass_g(lengths_m: np.ndarray, sections: np.ndarray) -> float:
    volume = sum(section_area(int(n)) * float(L) for L, n in zip(lengths_m, sections))
    return RHO * volume * 1000.0


def calcular_penalidades(lengths_cm: np.ndarray, total_sticks: int, total_mass: float) -> float:
    """Penalidade aditiva proporcional à violação — mesmo padrão da Atividade 8."""
    viol_r4 = float(np.sum(np.maximum(0.0, lengths_cm[DIAGONAL_INDICES] - STICK_LENGTH_CM)))
    viol_r5 = max(0, total_sticks - MAX_STICKS)
    viol_r6 = max(0.0, total_mass - MAX_MASS_G)
    return P_R4 * viol_r4 + P_R5 * viol_r5 + P_R6 * viol_r6


def bar_capacity(unit_axial_n: float, n_sticks: int, length_m: float) -> float:
    """
    Capacidade da barra (N).
    Tração    : σt_adm × A
    Compressão: min(σc_adm × A,  Euler)
      Euler = n × π²·E·I_palito / L²  — palitos flambam em paralelo (∝ n),
      pois a cola PVA não garante seção monolítica sob flambagem.
    """
    area          = section_area(n_sticks)
    if unit_axial_n > 0.0:
        return SIGMA_T_ALLOW * area
    inertia_single = B * T ** 3 / 12.0
    euler          = n_sticks * math.pi ** 2 * E * inertia_single / length_m ** 2
    return min(SIGMA_C_ALLOW * area, euler)


def evaluate(ind: Individual) -> Individual:
    """Avalia restrições R1–R8 e calcula o fitness (carga de ruptura / massa)."""
    # R1, R2, R3 garantidas pelo reparo/clipagem do cromossomo.
    ind.p_cm = repair_panels(ind.p_cm)
    ind.h_cm = np.clip(ind.h_cm, HEIGHT_MIN_CM, HEIGHT_MAX_CM)
    ind.n = np.clip(np.rint(ind.n), 1, 3).astype(int)

    areas = np.array([section_area(int(x)) for x in ind.n])

    try:
        axial_unit, lengths_m = solve_axial_forces(node_coords(ind.p_cm, ind.h_cm))
    except ValueError as error:
        ind.fitness, ind.feasible = 0.0, False
        ind.data = {"penalty_reasons": [str(error)]}
        return ind

    lengths_cm = lengths_m * 100.0
    reasons = []

    # R4: diagonais devem caber em um palito inteiro (sem emenda).
    if np.any(lengths_cm[DIAGONAL_INDICES] > STICK_LENGTH_CM + 1e-9):
        reasons.append("R4: diagonal maior que 11,4 cm")

    total_sticks = stick_count(ind.n)
    if total_sticks > MAX_STICKS:
        reasons.append("R5: mais de 150 palitos")

    total_mass = mass_g(lengths_m, ind.n)
    if total_mass >= MAX_MASS_G:
        reasons.append("R6: massa >= 600 g")

    # Carga de ruptura teórica = min_i (capacidade_i / |força unitária_i|).
    rupture = np.full(len(CONNECTIVITY), np.inf)
    for i, coeff in enumerate(axial_unit):
        if abs(coeff) < 1e-12:
            continue
        cap = bar_capacity(float(coeff), int(ind.n[i]), float(lengths_m[i]))
        rupture[i] = cap / abs(coeff)

    load_n = float(np.min(rupture))
    load_kg = load_n / G

    penalidade = calcular_penalidades(lengths_cm, total_sticks, total_mass)
    feasible = (penalidade == 0.0)
    ind.fitness = (load_kg / total_mass) if feasible and total_mass > 0 else 0.0
    ind.objectives = (total_mass + penalidade, -load_kg + penalidade)
    ind.feasible = feasible

    stresses_mpa = (axial_unit * load_n / areas) / 1e6
    delta_unit = ptv_displacement_unit(axial_unit, lengths_m, areas)
    critical = int(np.argmin(rupture))
    ind.data = {
        "penalty_reasons": reasons,
        "axial_unit_n": axial_unit,
        "lengths_cm": lengths_cm,
        "mass_g": total_mass,
        "stick_count": total_sticks,
        "theoretical_load_n": load_n,
        "theoretical_load_kg": load_kg,
        "critical_bar": BAR_NAMES[critical],
        "stresses_at_rupture_mpa": stresses_mpa,
        "delta_c_at_rupture_mm": delta_unit * load_n * 1000.0,
    }
    return ind


# =========================
# Algoritmo Genético
# =========================

def _pareto_front(pts: list[tuple[float, float]]) -> list[list[float]]:
    """
    Dado [(mass_g, load_kg), ...] de indivíduos viáveis, retorna o subconjunto
    não-dominado — minimizar massa, maximizar carga — como [[m, l], ...].
    """
    result = []
    for i, (mi, li) in enumerate(pts):
        dominated = any(
            j != i and mj <= mi and lj >= li and (mj < mi or lj > li)
            for j, (mj, lj) in enumerate(pts)
        )
        if not dominated:
            result.append((mi, li))
    return [[m, l] for m, l in sorted(result)]


def _dominates(obj_a: tuple, obj_b: tuple) -> bool:
    """obj_a domina obj_b se for ≤ em todos e < em pelo menos um (ambos minimizados)."""
    return (obj_a[0] <= obj_b[0] and obj_a[1] <= obj_b[1] and
            (obj_a[0] < obj_b[0] or obj_a[1] < obj_b[1]))


def _non_dominated_sort(population: list[Individual]) -> tuple[list, list]:
    n = len(population)
    dominated_by = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts: list[list[int]] = [[]]
    rank = [0] * n

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(population[p].objectives, population[q].objectives):
                dominated_by[p].append(q)
            elif _dominates(population[q].objectives, population[p].objectives):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts, rank


def _crowding_distance(front: list[int], population: list[Individual]) -> dict:
    dist: dict[int, float] = {idx: 0.0 for idx in front}
    size = len(front)
    if size <= 2:
        for idx in front:
            dist[idx] = float('inf')
        return dist
    for m in range(2):
        ordered = sorted(front, key=lambda idx: population[idx].objectives[m])
        dist[ordered[0]] = dist[ordered[-1]] = float('inf')
        f_min = population[ordered[0]].objectives[m]
        f_max = population[ordered[-1]].objectives[m]
        span = f_max - f_min
        if span == 0:
            continue
        for k in range(1, size - 1):
            dist[ordered[k]] += (
                population[ordered[k + 1]].objectives[m] -
                population[ordered[k - 1]].objectives[m]
            ) / span
    return dist


def nsga_tournament(population: list[Individual], rank: list, crowding: dict) -> Individual:
    a, b = random.sample(range(len(population)), 2)
    if rank[a] < rank[b]:
        return population[a]
    elif rank[b] < rank[a]:
        return population[b]
    return population[a] if crowding.get(a, 0.0) >= crowding.get(b, 0.0) else population[b]


def random_individual() -> Individual:
    p = repair_panels(np.random.uniform(PANEL_MIN_CM, PANEL_MAX_CM, size=4))
    h = np.random.uniform(HEIGHT_MIN_CM, HEIGHT_MAX_CM, size=5)
    n = np.random.randint(1, 4, size=17)
    return evaluate(Individual(p, h, n))


def crossover(p1: Individual, p2: Individual, alpha: float = 0.35) -> tuple[Individual, Individual]:
    """BLX-alpha para genes contínuos (p, h) e cruzamento uniforme para n."""
    def blx(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        diff = hi - lo
        c1 = np.random.uniform(lo - alpha * diff, hi + alpha * diff)
        c2 = np.random.uniform(lo - alpha * diff, hi + alpha * diff)
        return c1, c2

    pa, pb = blx(p1.p_cm, p2.p_cm)
    ha, hb = blx(p1.h_cm, p2.h_cm)
    mask = np.random.rand(17) < 0.5
    na = np.where(mask, p1.n, p2.n)
    nb = np.where(mask, p2.n, p1.n)

    c1 = Individual(repair_panels(pa), np.clip(ha, HEIGHT_MIN_CM, HEIGHT_MAX_CM), na.astype(int))
    c2 = Individual(repair_panels(pb), np.clip(hb, HEIGHT_MIN_CM, HEIGHT_MAX_CM), nb.astype(int))
    return c1, c2


def mutate(ind: Individual, p_mut_cont: float = 0.20, p_mut_disc: float = 0.08) -> Individual:
    p, h, n = ind.p_cm.copy(), ind.h_cm.copy(), ind.n.copy()

    for i in range(4):
        if random.random() < p_mut_cont:
            p[i] += random.gauss(0.0, 0.80)
    for i in range(5):
        if random.random() < p_mut_cont:
            h[i] += random.gauss(0.0, 0.60)
    for i in range(17):
        if random.random() < p_mut_disc:
            choices = [1, 2, 3]
            choices.remove(int(n[i]))
            n[i] = random.choice(choices)

    ind.p_cm = repair_panels(p)
    ind.h_cm = np.clip(h, HEIGHT_MIN_CM, HEIGHT_MAX_CM)
    ind.n = n.astype(int)
    return ind


def run_ga(
    population_size: int = 250,
    generations: int = 400,
    crossover_rate: float = 0.90,
    seed: int | None = 42,
) -> tuple[Individual, dict]:
    """NSGA-II bi-objetivo: minimizar massa, maximizar carga. Retorna (melhor, histórico)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    population = [random_individual() for _ in range(population_size)]
    feas = [x for x in population if x.feasible]
    best = max(feas if feas else population, key=lambda x: x.fitness)

    history_best:     list[Individual]        = []
    history_cloud:    list[list[list[float]]] = []
    history_pareto:   list[list[list[float]]] = []
    history_best_obj: list[list[float]]       = []

    for gen in range(1, generations + 1):
        fronts, rank = _non_dominated_sort(population)
        crowding: dict[int, float] = {}
        for front in fronts:
            crowding.update(_crowding_distance(front, population))

        offspring: list[Individual] = []
        while len(offspring) < population_size:
            p1 = nsga_tournament(population, rank, crowding)
            p2 = nsga_tournament(population, rank, crowding)
            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1 = Individual(p1.p_cm.copy(), p1.h_cm.copy(), p1.n.copy())
                c2 = Individual(p2.p_cm.copy(), p2.h_cm.copy(), p2.n.copy())
            offspring.append(evaluate(mutate(c1)))
            offspring.append(evaluate(mutate(c2)))

        combined = population + offspring
        fronts_c, _ = _non_dominated_sort(combined)
        crowding_c: dict[int, float] = {}
        for front in fronts_c:
            crowding_c.update(_crowding_distance(front, combined))

        next_pop: list[Individual] = []
        for front in fronts_c:
            if len(next_pop) + len(front) <= population_size:
                next_pop.extend(combined[idx] for idx in front)
            else:
                remaining = population_size - len(next_pop)
                ordered = sorted(front, key=lambda idx: crowding_c.get(idx, 0.0), reverse=True)
                next_pop.extend(combined[idx] for idx in ordered[:remaining])
                break
        population = next_pop

        feas = [x for x in population if x.feasible]
        gen_best = max(feas if feas else population, key=lambda x: x.fitness)
        if gen_best.fitness > best.fitness:
            best = gen_best

        cloud: list[list[float]] = []
        feas_pts: list[tuple[float, float]] = []
        for ind in population:
            if ind.data:
                m = float(ind.data.get("mass_g", 0.0))
                l = float(ind.data.get("theoretical_load_kg", 0.0))
                f = 1.0 if ind.feasible else 0.0
                cloud.append([m, l, f])
                if ind.feasible:
                    feas_pts.append((m, l))

        history_best.append(best)
        history_cloud.append(cloud)
        history_pareto.append(_pareto_front(feas_pts))
        history_best_obj.append([
            float(best.data.get("mass_g", 0.0)),
            float(best.data.get("theoretical_load_kg", 0.0)),
        ])

        if gen == 1 or gen % 25 == 0:
            viaveis = sum(1 for x in population if x.feasible)
            print(
                f"Geração {gen:4d} | fitness = {best.fitness:.6f} "
                f"| carga = {best.data['theoretical_load_kg']:.3f} kg "
                f"| massa = {best.data['mass_g']:.2f} g "
                f"| viáveis = {viaveis}/{population_size}"
            )

    historico = {
        "best":     history_best,
        "cloud":    history_cloud,
        "pareto":   history_pareto,
        "best_obj": history_best_obj,
    }
    return best, historico


# =========================
# Relatório
# =========================

def print_report(best: Individual) -> None:
    d = best.data
    print("\n" + "=" * 70)
    print("MELHOR TRELIÇA ENCONTRADA")
    print("=" * 70)

    print(f"\nPainéis p1..p4 (cm): {np.round(best.p_cm, 3)}  soma = {np.sum(best.p_cm):.1f} cm")
    print(f"Alturas h1..h5 (cm): {np.round(best.h_cm, 3)}")

    print("\nSeções e comprimentos por barra:")
    print(f"{'Barra':>5s} | {'Palitos':>7s} | {'L (cm)':>7s}")
    print("-" * 27)
    for bar, n, L in zip(BAR_NAMES, best.n, d["lengths_cm"]):
        print(f"{bar:>5s} | {int(n):>7d} | {L:7.3f}")

    print("\nResultados principais:")
    print(f"  Fitness teórico:        {best.fitness:.6f} kg/g")
    print(f"  Carga de ruptura:       {d['theoretical_load_kg']:.4f} kg")
    print(f"  Massa total:            {d['mass_g']:.4f} g")
    print(f"  Palitos estimados:      {d['stick_count']}")
    print(f"  Barra crítica:          {d['critical_bar']}")
    print(f"  Δ_C na ruptura:         {d['delta_c_at_rupture_mm']:.4f} mm")
    print(f"  Viável:                 {best.feasible}")
    if not best.feasible:
        for reason in d["penalty_reasons"]:
            print("   -", reason)

    print("\nForças e tensões na carga de ruptura:")
    axial = d["axial_unit_n"] * d["theoretical_load_n"]
    print(f"{'Barra':>5s} | {'N (N)':>10s} | {'σ (MPa)':>9s} | {'Estado':>11s}")
    print("-" * 46)
    for bar, ax, st in zip(BAR_NAMES, axial, d["stresses_at_rupture_mpa"]):
        estado = "tração" if ax > 1e-9 else "compressão" if ax < -1e-9 else "zero"
        print(f"{bar:>5s} | {ax:10.3f} | {st:9.3f} | {estado:>11s}")


_BAR_TYPE = {
    "A-B": "Banzo inferior",  "B-C": "Banzo inferior",
    "C-D": "Banzo inferior",  "D-E": "Banzo inferior",
    "F-G": "Banzo superior",  "G-H": "Banzo superior",
    "H-I": "Banzo superior",  "I-J": "Banzo superior",
    "A-F": "Vertical ext.",   "G-B": "Vertical int.",
    "H-C": "Vertical central","I-D": "Vertical int.",
    "E-J": "Vertical ext.",
    "F-B": "Diagonal",        "G-C": "Diagonal",
    "I-C": "Diagonal",        "J-D": "Diagonal",
}

_SIGMA_T_ADM_MPA = 27.5
_SIGMA_C_ADM_MPA = 17.5
_STICK_LEN_CM    = 11.4
_B_CM, _T_CM     = 1.0, 0.2


def print_report_builder(best: Individual) -> None:
    """Relatório completo para os construtores da treliça."""
    import math as _math

    d          = best.data
    p          = best.p_cm
    h          = best.h_cm
    n_arr      = best.n
    lengths_cm = d["lengths_cm"]
    axial_unit = d["axial_unit_n"]
    load_n     = d["theoretical_load_n"]
    load_kg    = d["theoretical_load_kg"]
    m_g        = d["mass_g"]
    stresses   = d["stresses_at_rupture_mpa"]
    delta_mm   = d["delta_c_at_rupture_mm"]
    crit       = d["critical_bar"]

    axial_real = axial_unit * load_n

    W   = 76
    SEP = "═" * W
    sep = "─" * W

    # ── Nós ──────────────────────────────────────────────────────────────────
    x_nós = [0.0, p[0], p[0]+p[1], p[0]+p[1]+p[2], 40.0]
    nós = {
        "A": (x_nós[0], 0.0), "B": (x_nós[1], 0.0),
        "C": (x_nós[2], 0.0), "D": (x_nós[3], 0.0), "E": (x_nós[4], 0.0),
        "F": (x_nós[0], h[0]), "G": (x_nós[1], h[1]),
        "H": (x_nós[2], h[2]), "I": (x_nós[3], h[3]), "J": (x_nós[4], h[4]),
    }

    # ── Inventário físico ─────────────────────────────────────────────────────
    inv = []
    total_phys = 0
    for bar, ni, L in zip(BAR_NAMES, n_arr, lengths_cm):
        segs = _math.ceil(L / _STICK_LEN_CM)
        phys = int(ni) * segs
        total_phys += phys
        inv.append({"bar": bar, "tipo": _BAR_TYPE.get(bar, ""),
                    "L": L, "n": int(ni), "segs": segs, "phys": phys,
                    "area_cm2": _B_CM * int(ni) * _T_CM})

    print(f"\n{SEP}")
    print("  RELATÓRIO DE CONSTRUÇÃO — TRELIÇA HOWE DE PALITOS".center(W))
    print(SEP)

    # ── Geometria ─────────────────────────────────────────────────────────────
    print(f"\n  Painéis (p1…p4) : {' | '.join(f'{v:.3f}' for v in p)} cm   Σ = {sum(p):.1f} cm")
    print(f"  Alturas (h1…h5) : {' | '.join(f'{v:.3f}' for v in h)} cm")
    print(f"\n  {'Nó':>3}  {'x (cm)':>8}  {'y (cm)':>8}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}")
    for nm, (xn, yn) in nós.items():
        print(f"  {nm:>3}  {xn:8.3f}  {yn:8.3f}")

    # ── Inventário ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  {'Barra':>5}  {'Tipo':18}  {'L (cm)':>7}  {'n':>2}  {'A (cm²)':>7}  {'Palitos':>7}")
    print(f"  {'─'*5}  {'─'*18}  {'─'*7}  {'─'*2}  {'─'*7}  {'─'*7}")
    for row in inv:
        print(f"  {row['bar']:>5}  {row['tipo']:18}  {row['L']:7.3f}  "
              f"{row['n']:>2}  {row['area_cm2']:7.4f}  {row['phys']:>7}")
    print(f"\n  Palitos físicos: {total_phys}  "
          f"(comprar {int(total_phys * 1.15) + 1} com folga de 15 %)")

    # ── Esforços ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  ESFORÇOS  —  P ruptura = {load_kg:.4f} kg  ({load_n:.2f} N)")
    print(f"\n  {'Barra':>5}  {'N (N)':>9}  {'σ (MPa)':>8}  {'FS':>5}  Estado")
    print(f"  {'─'*5}  {'─'*9}  {'─'*8}  {'─'*5}  ─────────")
    for bar, nr, st in zip(BAR_NAMES, axial_real, stresses):
        if abs(nr) < 1e-9:
            estado = "zero      "; σadm = _SIGMA_T_ADM_MPA; fs_s = "  —  "
        elif nr > 0:
            estado = "tração    "; σadm = _SIGMA_T_ADM_MPA
            fs_s = f"{σadm/abs(st):.2f}" if abs(st) > 1e-9 else "  ∞  "
        else:
            estado = "compressão"; σadm = _SIGMA_C_ADM_MPA
            fs_s = f"{σadm/abs(st):.2f}" if abs(st) > 1e-9 else "  ∞  "
        ok = "" if abs(st) <= σadm + 1e-6 else " ← VIOLA"
        print(f"  {bar:>5}  {nr:+9.2f}  {st:+8.3f}  {fs_s:>5}  {estado}{ok}")

    # ── Resultados ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    sticks_gene = int(sum(n_arr))
    print(f"  Carga de ruptura : {load_kg:.4f} kg")
    print(f"  Massa total      : {m_g:.4f} g")
    print(f"  Fitness          : {best.fitness:.6f} kg/g")
    print(f"  Δ_C na ruptura   : {delta_mm:.4f} mm")
    print(f"  Barra crítica    : {crit}  ({_BAR_TYPE.get(crit,'')})")
    print(f"  Palitos R5       : {sticks_gene}/150  {'[OK]' if sticks_gene<=150 else '[VIOLA]'}")
    print(f"  Viável           : {'Sim' if best.feasible else 'Não'}")
    if not best.feasible:
        for r in d.get("penalty_reasons", []):
            print(f"    ✗ {r}")

    # ── Cortes ────────────────────────────────────────────────────────────────
    a_cortar = [r for r in inv if r["L"] < _STICK_LEN_CM - 0.05]
    if a_cortar:
        print(f"\n{sep}")
        print(f"  CORTES  (palito inteiro = {_STICK_LEN_CM} cm)")
        print(f"  {'Barra':>5}  {'L (cm)':>7}  {'retirar':>8}  Camadas")
        for r in a_cortar:
            print(f"  {r['bar']:>5}  {r['L']:7.3f}  {_STICK_LEN_CM-r['L']:8.3f} cm"
                  f"  {r['n']} camada(s)")
    print(f"\n{SEP}")


if __name__ == "__main__":
    melhor, historico = run_ga()
    print_report(melhor)
    print_report_builder(melhor)

    try:
        from visualizer_howe import animate_evolution, save_final_image
        cloud_total = [pt for gen in historico["cloud"] for pt in gen]
        animate_evolution(
            historico["best"],
            historico["cloud"],
            historico["pareto"],
            historico["best_obj"],
            total_gens=400,
            gif_path="best_truss.gif",
        )
        save_final_image(
            melhor,
            cloud_total,
            historico["pareto"][-1],
            historico["best_obj"][-1],
            total_gens=400,
            png_path="best_truss_final.png",
        )
    except Exception as exc:
        print(f"\n[Visualizador] {exc}")
