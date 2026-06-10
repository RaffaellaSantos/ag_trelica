"""
export_ag_data.py
-----------------
Roda o AG e gera ag_data.json + ag_data.js na mesma pasta (animacao/).

Execute a partir de qualquer lugar:
    python animacao/export_ag_data.py
    # ou, de dentro de animacao/:
    python export_ag_data.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path

# Localiza Victor_version/ independente do cwd
_HERE        = Path(__file__).resolve().parent          # .../animacao/
_VICTOR_DIR  = _HERE.parent / "Victor_version"          # .../Victor_version/
sys.path.insert(0, str(_VICTOR_DIR))

from trelica_howe_3d import run_ga, CONNECTIVITY, node_coords, Individual  # type: ignore

_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
_IDX   = {n: i for i, n in enumerate(_NAMES)}


def _ind_to_3d(ind: Individual) -> dict:
    """Converte o melhor indivíduo de uma geração em dados 3D para o viewer."""
    p_cm     = ind.p_cm
    h_cm     = ind.h_cm
    w_cm     = float(ind.w_cm)
    n_sticks = [int(x) for x in ind.n]

    coords2d = node_coords(p_cm, h_cm)   # {nome: array([x_m, y_m])}

    # 20 nós 3D: pórtico 1 (z=0) depois pórtico 2 (z=w)
    nodes_3d: list[list[float]] = []
    for name in _NAMES:
        xy = coords2d[name] * 100.0          # m → cm
        nodes_3d.append([float(xy[0]), float(xy[1]), 0.0])
    for name in _NAMES:
        xy = coords2d[name] * 100.0
        nodes_3d.append([float(xy[0]), float(xy[1]), w_cm])

    members: list[list[int]] = []
    areas:   list[int]       = []
    types:   list[str]       = []

    # Pórtico 1
    for (na, nb), n in zip(CONNECTIVITY, n_sticks):
        members.append([_IDX[na], _IDX[nb]])
        areas.append(n); types.append("frame")

    # Pórtico 2 (offset +10)
    for (na, nb), n in zip(CONNECTIVITY, n_sticks):
        members.append([_IDX[na] + 10, _IDX[nb] + 10])
        areas.append(n); types.append("frame")

    # Travamentos laterais A–A', …, J–J'
    for i in range(10):
        members.append([i, i + 10])
        areas.append(1); types.append("brace")

    return {"nodes": nodes_3d, "members": members,
            "areas": areas, "types": types}


def main(generations: int = 400, population: int = 250) -> None:
    # Saída sempre na mesma pasta do script (animacao/)
    out_json = _HERE / "ag_data.json"
    out_js   = _HERE / "ag_data.js"

    print(f"Iniciando AG 3D: {generations} gerações, {population} indivíduos …")

    best, history = run_ga(
        population_size=population,
        generations=generations,
        seed=42,
    )

    all_gens: list[dict] = []
    for g, ind in enumerate(history["best"]):
        data3d   = _ind_to_3d(ind)
        best_obj = history["best_obj"][g]   # [mass_g, load_kg]

        all_gens.append({
            "gen":      g + 1,
            **data3d,
            "mass_g":   float(best_obj[0]),
            "load_kg":  float(best_obj[1]),
            "fitness":  float(ind.fitness),
            "feasible": bool(ind.feasible),
            "w_cm":     float(ind.w_cm),
        })

    # ag_data.json (para serve.py / run_viewer.bat)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_gens, f, indent=2)

    # ag_data.js (para abrir trelica_3d.html direto, sem servidor)
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("// Gerado por export_ag_data.py — usado por trelica_3d.html\n")
        f.write("window.AG_DATA = ")
        json.dump(all_gens, f)
        f.write(";\n")

    print(f"\n✓  Exportados em animacao/:")
    print(f"   ag_data.json  (para servidor HTTP / run_viewer.bat)")
    print(f"   ag_data.js    (para abrir trelica_3d.html direto no browser)")
    print(f"\n   Gerações salvas : {len(all_gens)}")
    print(f"   Melhor fitness  : {best.fitness:.6f} kg/g")
    print(f"   Carga de ruptura: {best.data['theoretical_load_kg']:.3f} kg")
    print(f"   Massa total     : {best.data['mass_g']:.2f} g")
    print(f"   Largura (w)     : {best.w_cm:.2f} cm")
    print(f"\nAbra animacao/trelica_3d.html no browser para ver a animação.")


if __name__ == "__main__":
    main()
