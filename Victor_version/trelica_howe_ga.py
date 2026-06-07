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

import numpy as np


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

E = 3_500e6                 # Pa
RHO = 510.0                 # kg/m³
SIGMA_T_ALLOW = 55e6 / 2.0  # Pa (tração, Sg = 2)
SIGMA_C_ALLOW = 35e6 / 2.0  # Pa (compressão, Sg = 2)
B = 0.010                   # m, largura do palito
T = 0.002                   # m, espessura de 1 palito
K_BUCKLING = 1.0            # barras biarticuladas

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

def solve_axial_forces(nodes: dict[str, np.ndarray], areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Equilíbrio de nós (treliça plana 2D) sob carga unitária (1 N) para baixo em C.
    Apoios: A (pino, x e y) e E (rolete, y). Convenção: positivo = tração.
    Retorna (forças axiais unitárias, comprimentos das barras), ambos por barra.
    """
    names = list(nodes)
    idx = {name: i for i, name in enumerate(names)}
    ndof = 2 * len(names)

    K = np.zeros((ndof, ndof))
    lengths = np.empty(len(CONNECTIVITY))
    cos_sin = np.empty((len(CONNECTIVITY), 2))

    for e, (na, nb) in enumerate(CONNECTIVITY):
        d = nodes[nb] - nodes[na]
        L = float(math.hypot(d[0], d[1]))
        c, s = d[0] / L, d[1] / L
        lengths[e] = L
        cos_sin[e] = (c, s)

        ke = (E * areas[e] / L) * np.array([
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ])
        dofs = [2 * idx[na], 2 * idx[na] + 1, 2 * idx[nb], 2 * idx[nb] + 1]
        K[np.ix_(dofs, dofs)] += ke

    force = np.zeros(ndof)
    force[2 * idx[LOAD_NODE] + 1] = -1.0  # 1 N para baixo no nó C

    fixed = {2 * idx["A"], 2 * idx["A"] + 1, 2 * idx["E"] + 1}
    free = [i for i in range(ndof) if i not in fixed]

    try:
        u_free = np.linalg.solve(K[np.ix_(free, free)], force[free])
    except np.linalg.LinAlgError:
        raise ValueError("Matriz de rigidez singular: geometria instável.")

    u = np.zeros(ndof)
    u[free] = u_free

    axial = np.empty(len(CONNECTIVITY))
    for e, (na, nb) in enumerate(CONNECTIVITY):
        c, s = cos_sin[e]
        ia, ib = idx[na], idx[nb]
        delta = c * (u[2 * ib] - u[2 * ia]) + s * (u[2 * ib + 1] - u[2 * ia + 1])
        axial[e] = E * areas[e] / lengths[e] * delta

    return axial, lengths


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

def stick_count(lengths_m: np.ndarray, sections: np.ndarray) -> int:
    """Palitos totais, contando emendas para barras maiores que 11,4 cm."""
    stick_len_m = STICK_LENGTH_CM / 100.0
    return sum(int(n) * math.ceil(L / stick_len_m) for L, n in zip(lengths_m, sections))


def mass_g(lengths_m: np.ndarray, sections: np.ndarray) -> float:
    volume = sum(section_area(int(n)) * float(L) for L, n in zip(lengths_m, sections))
    return RHO * volume * 1000.0


def bar_capacity(unit_axial_n: float, n_sticks: int, length_m: float) -> float:
    """
    Capacidade da barra (N). Tração: σt·A. Compressão: menor entre escoamento
    e flambagem de Euler. Palitos colados flambam em paralelo (capacidade ∝ n),
    pois a cola não garante seção monolítica.
    """
    area = section_area(n_sticks)
    if unit_axial_n > 0.0:
        return SIGMA_T_ALLOW * area
    inertia_single = B * T ** 3 / 12.0
    euler = n_sticks * (math.pi ** 2) * E * inertia_single / ((K_BUCKLING * length_m) ** 2)
    return min(SIGMA_C_ALLOW * area, euler)


def evaluate(ind: Individual) -> Individual:
    """Avalia restrições R1–R8 e calcula o fitness (carga de ruptura / massa)."""
    # R1, R2, R3 garantidas pelo reparo/clipagem do cromossomo.
    ind.p_cm = repair_panels(ind.p_cm)
    ind.h_cm = np.clip(ind.h_cm, HEIGHT_MIN_CM, HEIGHT_MAX_CM)
    ind.n = np.clip(np.rint(ind.n), 1, 3).astype(int)

    areas = np.array([section_area(int(x)) for x in ind.n])

    try:
        axial_unit, lengths_m = solve_axial_forces(node_coords(ind.p_cm, ind.h_cm), areas)
    except ValueError as error:
        ind.fitness, ind.feasible = 0.0, False
        ind.data = {"penalty_reasons": [str(error)]}
        return ind

    lengths_cm = lengths_m * 100.0
    reasons = []

    # R4: diagonais devem caber em um palito inteiro (sem emenda).
    if np.any(lengths_cm[DIAGONAL_INDICES] > STICK_LENGTH_CM + 1e-9):
        reasons.append("R4: diagonal maior que 11,4 cm")

    total_sticks = stick_count(lengths_m, ind.n)
    if total_sticks > MAX_STICKS:
        reasons.append("R5: mais de 150 palitos")

    total_mass = mass_g(lengths_m, ind.n)
    if total_mass >= MAX_MASS_G:
        reasons.append("R6: massa >= 600 g")

    # Carga de ruptura teórica = min_i (capacidade_i / |força unitária_i|).
    rupture = np.full(len(CONNECTIVITY), np.inf)
    capacities = np.full(len(CONNECTIVITY), np.inf)
    for i, coeff in enumerate(axial_unit):
        if abs(coeff) < 1e-12:
            continue
        cap = bar_capacity(float(coeff), int(ind.n[i]), float(lengths_m[i]))
        capacities[i] = cap
        rupture[i] = cap / abs(coeff)

    load_n = float(np.min(rupture))
    load_kg = load_n / G

    # R7/R8: tensão na carga de ruptura dentro do admissível.
    stresses_mpa = (axial_unit * load_n / areas) / 1e6
    if np.any(stresses_mpa > SIGMA_T_ALLOW / 1e6 + 1e-6):
        reasons.append("R7: tração acima do admissível")
    if np.any(stresses_mpa < -SIGMA_C_ALLOW / 1e6 - 1e-6):
        reasons.append("R8: compressão acima do admissível")

    feasible = not reasons
    ind.fitness = (load_kg / total_mass) if feasible and total_mass > 0 else 0.0
    ind.feasible = feasible

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

def random_individual() -> Individual:
    p = repair_panels(np.random.uniform(PANEL_MIN_CM, PANEL_MAX_CM, size=4))
    h = np.random.uniform(HEIGHT_MIN_CM, HEIGHT_MAX_CM, size=5)
    n = np.random.randint(1, 4, size=17)
    return evaluate(Individual(p, h, n))


def tournament(population: list[Individual], k: int) -> Individual:
    return max(random.sample(population, k), key=lambda x: x.fitness)


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
    elitism: int = 8,
    tournament_k: int = 4,
    seed: int | None = 42,
) -> Individual:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    population = [random_individual() for _ in range(population_size)]
    best = max(population, key=lambda x: x.fitness)

    for gen in range(1, generations + 1):
        population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = population[:elitism]

        while len(new_population) < population_size:
            p1 = tournament(population, tournament_k)
            p2 = tournament(population, tournament_k)
            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1 = Individual(p1.p_cm.copy(), p1.h_cm.copy(), p1.n.copy())
                c2 = Individual(p2.p_cm.copy(), p2.h_cm.copy(), p2.n.copy())
            new_population.append(evaluate(mutate(c1)))
            new_population.append(evaluate(mutate(c2)))

        population = new_population[:population_size]
        gen_best = max(population, key=lambda x: x.fitness)
        if gen_best.fitness > best.fitness:
            best = gen_best

        if gen == 1 or gen % 25 == 0:
            viaveis = sum(1 for x in population if x.feasible)
            print(
                f"Geração {gen:4d} | fitness = {best.fitness:.6f} "
                f"| carga = {best.data['theoretical_load_kg']:.3f} kg "
                f"| massa = {best.data['mass_g']:.2f} g "
                f"| viáveis = {viaveis}/{population_size}"
            )

    return best


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


if __name__ == "__main__":
    melhor = run_ga()
    print_report(melhor)
