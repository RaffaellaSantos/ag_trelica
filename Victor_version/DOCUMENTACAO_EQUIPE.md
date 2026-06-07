# Comparação dos Três Algoritmos — Atividade 9
**Uso interno da equipe**

---

## Visão Rápida

| | `trelica_howe_ga.py` | `trelica_howe_ga_corrigido.py` | `trelica_howe_3d.py` |
|---|---|---|---|
| **Nome** | Original | 2D Corrigido | 3D Construível |
| **Genes** | 26 | 26 | 27 (+w) |
| **Análise estrutural** | Matriz de rigidez (FEM) | Equilíbrio de nós (app/utils) | Equilíbrio de nós (app/utils) |
| **Euler buckling** | n × I_single ✓ | n × I_single ✓ | n × I_single ✓ |
| **Contagem palitos (R5)** | Σ nᵢ × ⌈L/11.4⌉ ✗ | Σ nᵢ ✓ | 2·Σ nᵢ + 10 ✓ |
| **Estrutura física** | 1 pórtico (cai de lado) | 1 pórtico (cai de lado) | 2 pórticos + travamentos ✓ |
| **run_ga retorna** | `Individual` | `(Individual, histórico)` | `(Individual, histórico)` |
| **GIF / PNG** | Não | Sim | Sim |
| **Usar para** | Comparação / histórico | Apresentação ao professor | Construção física |

---

## O que estava errado no original e por que corrigimos

### 1. Método de análise estrutural

**Original:** Matriz de rigidez (método FEM de deslocamentos)
```python
K = (E·A/L) · [matriz de cossenos]   # precisa das áreas A
u = K⁻¹ · F                          # resolve deslocamentos
N = (E·A/L) · δ                      # back-calcula forças
```
Para treliça isostática, as forças axiais **não dependem das áreas**. Usar FEM aqui é matematicamente correto mas desnecessariamente acoplado às seções.

**Corrigido:** Equilíbrio de nós (método dos nós) — reaproveitado de `app/utils.py`:
```python
A_eq · [F₁..F₁₇, Ax, Ay, Ey]ᵀ = b   # 20×20, só geometria
```
Resolve diretamente as forças sem precisar das áreas. É o mesmo método da Atividade 8.

---

### 2. Contagem de palitos (R5)

**Original:** `Σ nᵢ × ⌈Lᵢ/11.4⌉` — contava emendas para barras longas.

**Problema:** O PDF define R5 como `Σ nᵢ ≤ 150` onde nᵢ são os genes de seção. A contagem com emendas não está no enunciado e penalizava soluções desnecessariamente.

**Corrigido:** `Σ nᵢ` — soma direta dos genes de seção.

---

### 3. Verificações R7/R8 redundantes

**Original:** Após calcular a carga de ruptura, verificava se as tensões excediam o admissível. Por construção, `P_rup = min(cap/|n̄|)` garante que nenhuma barra supera o admissível — as verificações nunca disparavam.

**Corrigido:** Removidas.

---

### 4. Por que o 3D aceita mais carga

O 3D tem **dois pórticos em paralelo**. Cada um suporta P/2:
```
P_3D = 2 × P_frame
massa_3D = 2 × massa_frame + massa_travamentos
Fitness_3D ≈ Fitness_2D   (carga e massa ambos ≈ dobram)
```

O 3D não é "mais eficiente" — tem o mesmo fitness. A vantagem é ser **fisicamente construível** (fica em pé sozinho).

---

### 5. Euler buckling — por que palitos em paralelo e não monolítico

O PDF (seção 3) dá `I = b·(n·t)³/12` para calcular **deslocamento** (PTV). Para **flambagem**, a questão é diferente:

| Abordagem | Fórmula | Capacidade para n=3 (L=10cm) |
|---|---|---|
| Monolítico (n³·I₁) | I = b·(n·t)³/12 | 621 N → ~32 kg |
| Paralelo (n·I₁) ← **usamos** | I_eff = n × I_palito | 69 N → ~3.5 kg |
| Realidade | entre os dois | variável |

A cola PVA não garante cisalhamento suficiente para comportamento monolítico sob flambagem. Usar `n·I₁` é conservador e dá valores no intervalo **3–12 kg** esperado pelo PDF.

Sem Euler algum, chegávamos a 100–200 kg (completamente irrealista). Sem o ajuste paralelo→monolítico, chegávamos a 50–120 kg (ainda muito alto).

---

## Evolução do código — linha do tempo

```
trelica_howe_ga.py  (original)
 ├─ FEM (matriz de rigidez) — correto mas desnecessariamente complexo
 ├─ Euler n×I_single ✓
 ├─ stick_count com emendas ✗
 ├─ R7/R8 redundantes ✗
 └─ sem histórico / visualização
        ↓ corrigido
trelica_howe_ga_corrigido.py
 ├─ Equilíbrio de nós (app/utils.py) ✓
 ├─ Euler n×I_single ✓
 ├─ stick_count = Σnᵢ ✓
 ├─ R7/R8 removidos ✓
 ├─ run_ga retorna histórico → GIF + PNG ✓
 └─ 1 pórtico (não fica em pé)
        ↓ extensão
trelica_howe_3d.py
 ├─ tudo do corrigido ✓
 ├─ gene w (largura entre pórticos) ✓
 ├─ P_3D = 2 × P_frame ✓
 ├─ massa_3D = 2 × frame + travamentos ✓
 └─ fisicamente construível ✓
```

---

## Parâmetros do AG (todos os arquivos)

| Parâmetro | Valor |
|---|---|
| População | 250 |
| Gerações | 400 |
| Cruzamento (BLX-α) | α = 0.35, prob = 90% |
| Mutação contínua | Gaussiana, σ ≈ 0.6–0.8 cm, prob = 20%/gene |
| Mutação discreta | Troca aleatória em {1,2,3}, prob = 8%/gene |
| Torneio | k = 4 |
| Elitismo | 8 indivíduos |
| Seed padrão | 42 |

---

## Qual arquivo usar

| Situação | Arquivo |
|---|---|
| Arguição com o professor | `trelica_howe_ga_corrigido.py` ou `trelica_howe_3d.py` |
| Construção física da treliça | `trelica_howe_3d.py` |
| Comparação / entender o histórico | `trelica_howe_ga.py` |
| Ver GIF da evolução | `trelica_howe_ga_corrigido.py` (roda mais rápido) |

---

## Valores de referência (geometria uniforme p=10, h=5 cm)

| n | 2D (carga) | 3D (carga) | Barra crítica | Observação |
|---|---|---|---|---|
| 1 | ~1.2 kg | ~2.4 kg | H-I | Euler domina |
| 2 | ~2.4 kg | ~4.7 kg | H-I | Euler domina |
| 3 | ~3.5 kg | ~7.0 kg | H-I | Euler domina |

O AG com 400 gerações encontra geometria ótima (h mais alto, distribuição de seções melhor) e deve chegar a valores maiores dentro de faixas realistas.
