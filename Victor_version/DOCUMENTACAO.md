# Documentação — Otimização de Treliça Howe com AG
**Atividade 9 · EST/UEA · Algoritmos de Otimização**

---

## Visão Geral dos Três Arquivos

| | `trelica_howe_ga.py` | `trelica_howe_ga_corrigido.py` | `trelica_howe_3d.py` |
|---|---|---|---|
| **Status** | Versão original | Versão corrigida (2D) | Versão 3D construível |
| **Genes** | 26 | 26 | 27 |
| **Análise estrutural** | Matriz de rigidez (FEM) | Equilíbrio de nós (sistema linear) | Equilíbrio de nós (sistema linear) |
| **Flambagem de Euler** | Sim | Não | Não |
| **Contagem de palitos** | `Σ nᵢ × ⌈Lᵢ/11.4⌉` | `Σ nᵢ` | `2×Σ nᵢ + 10` |
| **Estrutura física** | 1 pórtico (não fica em pé) | 1 pórtico (não fica em pé) | 2 pórticos + travamentos |
| **Saídas** | Relatório no terminal | Terminal + GIF + PNG | Terminal + GIF + PNG |
| **Retorno do `run_ga`** | `Individual` | `(Individual, histórico)` | `(Individual, histórico)` |

---

## O Problema

**MINLP** — Otimização Mista Inteira Não-Linear (Mixed-Integer Non-Linear Programming).

- Variáveis **contínuas**: geometria da treliça
- Variáveis **discretas**: número de palitos por seção
- Objetivo: **maximizar** eficiência estrutural = carga de ruptura / massa total

### Estrutura da Treliça Howe

```
F────G────H────I────J     ← banzo superior (5 nós)
|  ╲ |  ╲ |  ╱ |  ╱ |
|   ╲|   ╲|  ╱ |╱   |
A────B────C────D────E     ← banzo inferior (5 nós)
↑                   ↑
apoio pino A     apoio rolete E
              ↑
           balde (P)
```

- **17 barras** no total
- **10 nós** (A–E no banzo inferior, F–J no banzo superior)
- **Vão livre**: 40 cm (p1 + p2 + p3 + p4 = 40 cm)
- **Carga**: concentrada no nó C (centro do banzo inferior)

| Grupo | Barras | Qtd |
|---|---|---|
| Banzo inferior | A-B, B-C, C-D, D-E | 4 |
| Banzo superior | F-G, G-H, H-I, I-J | 4 |
| Verticais externas | A-F, E-J | 2 |
| Verticais internas | G-B, H-C, I-D | 3 |
| Diagonais | F-B, G-C, I-C, J-D | 4 |

### Material — Palito de Picolé (Bétula)

| Propriedade | Valor |
|---|---|
| Módulo de elasticidade E | 3 500 MPa |
| Massa específica ρ | 510 kg/m³ |
| Resistência à tração σt | 55 MPa → admissível **27,5 MPa** (Sg = 2) |
| Resistência à compressão σc | 35 MPa → admissível **17,5 MPa** (Sg = 2) |
| Comprimento do palito | 11,4 cm |
| Seção | 1,0 cm × 0,2 cm |

### Seções Disponíveis

| n palitos | Área A | Observação |
|---|---|---|
| 1 | 0,20 cm² | palito simples |
| 2 | 0,40 cm² | dois palitos colados face a face |
| 3 | 0,60 cm² | três palitos colados face a face |

---

## 1. `trelica_howe_ga.py` — Versão Original

### Cromossomo — 26 genes

```
┌─────────────────────────────┬───────────────────────────────────────────────┐
│   BLOCO 1 — Contínuo (9)    │            BLOCO 2 — Discreto (17)            │
├──────────┬──────────────────┼────────────────────────┬──────────────────────┤
│ p1 p2 p3 p4 │ h1 h2 h3 h4 h5 │ n1 n2 n3 n4 n5 n6 n7 n8 │ n9..n13  │ n14..n17│
│ painéis  │    alturas       │  banzo inf + sup       │ verticais │  diag   │
└──────────┴──────────────────┴────────────────────────┴──────────────────────┘
   [3,11.4] cm    [3,11.4] cm           {1, 2, 3} palitos
```

- `p1..p4`: largura de cada painel (soma = 40 cm, restrição R3)
- `h1..h5`: altura das verticais A-F, G-B, H-C, I-D, E-J
- `n1..n4`: seção das barras do banzo inferior
- `n5..n8`: seção das barras do banzo superior
- `n9..n13`: seção das verticais
- `n14..n17`: seção das diagonais

### Análise Estrutural — Matriz de Rigidez (FEM)

Monta a **matriz de rigidez global** K (20×20) usando as áreas das seções:

```
kₑ = (E·A/L) · [cossenos diretores]
```

Resolve para deslocamentos → calcula forças axiais por deformação:

```
Nᵢ = (E·Aᵢ/Lᵢ) · δᵢ
```

> **Problema**: precisa das áreas como entrada. Para treliça isostática, as forças axiais **não dependem das áreas** — a análise fica matematicamente correta mas desnecessariamente acoplada às seções.

### Capacidade das Barras — Com Euler

```python
# Tração:
cap = σt_adm × A

# Compressão (menor entre resistência e flambagem):
I_palito = b·t³/12        # inércia de 1 palito
F_euler = n × π²·E·I / L²  # n palitos independentes (sem seção monolítica)
cap = min(σc_adm × A,  F_euler)
```

> **Nota**: assume que os palitos flambam independentemente (cola não garante seção sólida). Isso é conservador mas não está no enunciado do PDF.

### Contagem de Palitos — Com Emendas

```python
stick_count = Σ nᵢ × ⌈Lᵢ / 11.4⌉
```

Conta emendas para barras maiores que 11,4 cm. Ex.: barra de 15 cm com n=2 → 2 × 2 = 4 palitos.

> **Diverge do PDF** que define R5 como simplesmente `Σ nᵢ ≤ 150`.

### Fitness

```
Fitness = (Carga de Ruptura Teórica [kg]) / (Massa Total [g])   se viável
        = 0                                                        se inviável
```

**Carga de ruptura**: menor carga P que provoca falha em qualquer barra:

```
P_rup_i = cap_i / |n̄ᵢ|
P_rup   = min(P_rup_i)   para todas as barras
```

Onde `n̄ᵢ` é a força axial sob carga unitária (P = 1 N).

### Restrições

| # | Restrição | Critério |
|---|---|---|
| R1 | Comprimento horizontal | 3 ≤ pᵢ ≤ 11,4 cm |
| R2 | Comprimento vertical | 3 ≤ hᵢ ≤ 11,4 cm |
| R3 | Vão total | Σpᵢ = 40 cm |
| R4 | Diagonal física | √(pᵢ² + hᵢ²) ≤ 11,4 cm |
| R5 | Limite de palitos | Σ nᵢ×⌈Lᵢ/11.4⌉ ≤ 150 |
| R6 | Massa total | m < 600 g |
| R7 | Tensão de tração | σ ≤ 27,5 MPa |
| R8 | Tensão de compressão | \|σ\| ≤ 17,5 MPa |

> **R7 e R8** são verificados após calcular a carga de ruptura — por construção, nunca são violados. São código morto.

### Operadores Genéticos

| Operador | Bloco contínuo (p, h) | Bloco discreto (n) |
|---|---|---|
| **Cruzamento** | BLX-α (α = 0,35) | Uniforme (máscara binária 50/50) |
| **Mutação** | Gaussiana σ=0,8 (p), σ=0,6 (h), prob=20% | Substituição aleatória em {1,2,3}, prob=8% |
| **Seleção** | Torneio k=4 | — |
| **Elitismo** | 8 melhores passam direto | — |

### Problemas Identificados

1. ❌ Análise por matriz de rigidez depende desnecessariamente das áreas
2. ❌ Flambagem de Euler não está no enunciado do PDF
3. ❌ `stick_count` usa fórmula diferente do PDF (com emendas)
4. ❌ R7 e R8 redundantes (nunca disparam)
5. ❌ `run_ga` não coleta histórico → sem GIF/PNG
6. ❌ Não constrói fisicamente uma estrutura 3D

---

## 2. `trelica_howe_ga_corrigido.py` — Versão Corrigida (2D)

### O que mudou em relação ao original

| Aspecto | Original | Corrigido |
|---|---|---|
| Análise estrutural | Matriz de rigidez (FEM) | **Equilíbrio de nós** (sistema 20×20) |
| Flambagem de Euler | Presente | **Removida** |
| `stick_count` | `Σ nᵢ×⌈L/11.4⌉` | **`Σ nᵢ`** (conforme PDF) |
| R7/R8 | Verificados (redundantes) | **Removidos** |
| `run_ga` | Retorna `Individual` | **Retorna `(Individual, histórico)`** |
| Visualização | Nenhuma | **GIF + PNG + scatter Pareto** |

### Cromossomo — 26 genes (igual ao original)

```
┌─────────────────────────────┬───────────────────────────────────────────────┐
│   BLOCO 1 — Contínuo (9)    │            BLOCO 2 — Discreto (17)            │
├─────────────┬───────────────┼────────────────────────┬──────────────────────┤
│ p1 p2 p3 p4 │ h1 h2 h3 h4 h5│ n1..n4  n5..n8         │ n9..n13  │ n14..n17 │
│  painéis    │   alturas     │ banzo inf  banzo sup   │ verticais│  diag    │
└─────────────┴───────────────┴────────────────────────┴──────────────────────┘
   [3,11.4] cm    [3,11.4] cm           {1, 2, 3} palitos
```

### Análise Estrutural — Equilíbrio de Nós

Monta diretamente o **sistema de equilíbrio de forças**. Para cada nó: ΣFx = 0 e ΣFy = 0.

**Sistema 20×20**:

```
A_eq · [F₁ … F₁₇,  Ax,  Ay,  Ey]ᵀ  =  b_eq
        ───────────  ─────────────
        17 barras    3 reações de apoio
```

Contribuição de cada barra `e` conectando nós `a` e `b`:

```
linha 2a   += cxₑ · Fₑ     (ΣFx no nó a)
linha 2a+1 += cyₑ · Fₑ     (ΣFy no nó a)
linha 2b   -= cxₑ · Fₑ     (ΣFx no nó b)
linha 2b+1 -= cyₑ · Fₑ     (ΣFy no nó b)
```

Onde `(cxₑ, cyₑ)` = cossenos diretores da barra (vetor unitário de a → b).

Reações: pino em A (Ax, Ay) e rolete em E (Ey), cada uma adicionada como coluna extra.

Lado direito: `b[2·índice(C)+1] = 1,0` (1 N para baixo no nó C).

> **Vantagem**: não precisa das áreas. Para treliça isostática, forças axiais são puramente geométricas.

### Capacidade das Barras — Sem Euler

```python
# Tração:
cap = σt_adm × A = 27.5 MPa × (n × 1,0cm × 0,2cm)

# Compressão:
cap = σc_adm × A = 17.5 MPa × (n × 1,0cm × 0,2cm)
```

### PTV — Deslocamento em C

```
Δ_C = Σ (n̄ᵢ² · Lᵢ) / (E · Aᵢ)   [m/N]
```

`n̄ᵢ` = força axial unitária (carga = 1 N). Para carga real P:
`Δ_C_real = P · Δ_C_unit`

Na carga de ruptura: `Δ_C = load_N × Δ_C_unit × 1000 mm`

### Fitness

```
Fitness = Carga_ruptura [kg] / Massa_total [g]   se viável
        = 0                                       se inviável
```

**Penalidade multiplicativa**: qualquer violação de R1–R6 zera o fitness.

### Histórico e Visualização

`run_ga` retorna `(best, historico)` onde:

```python
historico = {
    "best":     [Individual por geração],   # melhor acumulado
    "cloud":    [[(mass, load, feasible)]],  # todos os indivíduos por geração
    "pareto":   [[[mass, load]]],            # fronteira Pareto por geração
    "best_obj": [[mass, load]],              # objetivo do melhor por geração
}
```

**Fronteira Pareto** do scatter: minimizar massa × maximizar carga de ruptura.

Arquivos gerados:
- `best_truss.gif` — animação da evolução (≤ 50 quadros amostrados)
- `best_truss_final.png` — imagem final com treliça + scatter

### Limitação Principal

> A estrutura é **matematicamente correta** mas **não fica em pé sozinha** — um pórtico plano
> precisa de apoio lateral para não tombar. Para uso real em laboratório, construir dois
> pórticos + travamentos (ver `trelica_howe_3d.py`).

---

## 3. `trelica_howe_3d.py` — Versão 3D Fisicamente Construível

### Motivação

Um único pórtico 2D tomba lateralmente. A versão 3D replica o pórtico e os conecta:

```
Vista de topo (z):
     F────────────F'
    /|            /|
   / |           / |
  A──┼──────────A' |     ← pórtico 1 (z=0) e pórtico 2 (z=w)
  |  G──────────|──G'
  | /            | /
  |/             |/
  B──────────────B'
  
  ←──── w cm ────→
```

Os 10 travamentos conectam nós correspondentes:
`A–A', B–B', C–C', D–D', E–E', F–F', G–G', H–H', I–I', J–J'`

### Cromossomo — 27 genes

```
┌─────────────────────────────────────┬──────────────────────────────────────┐
│      BLOCO 1 — Contínuo (10)        │       BLOCO 2 — Discreto (17)        │
├───────────────┬─────────────────┬───┼──────────────────────────────────────┤
│ p1  p2  p3  p4│ h1  h2  h3  h4 h5  │  w  │ n1..n4  n5..n8  n9..n13  n14..n17│
│   painéis     │    alturas      │   │larg.│ banzo_inf banzo_sup vert.   diag │
└───────────────┴─────────────────┴───┴─────┴──────────────────────────────── ┘
   [3,11.4] cm     [3,11.4] cm  [2,6] cm         {1, 2, 3} palitos
```

O gene `w` é o único gene novo em relação à versão 2D.

### Modelo Estrutural 3D

Cada pórtico é analisado **isoladamente em 2D**. Sob carga total P:
- Cada pórtico recebe **P/2** (simetria)
- O pórtico falha quando `P/2 = P_frame_ruptura`
- Logo: **P_3D = 2 × P_frame_ruptura**

```
┌────────────────────────────────────────────────────┐
│  Análise 2D (1 pórtico, carga P/2)                 │
│                                                    │
│  Forças unitárias: n̄ᵢ  (sob 1N em C)              │
│  Ruptura do pórtico: P_frame = min(capᵢ / |n̄ᵢ|)   │
│                                                    │
│  ↓ Escalar para estrutura 3D:                      │
│                                                    │
│  P_3D   = 2 × P_frame                             │
│  massa  = 2 × massa_pórtico + massa_travamentos    │
│  Δ_C    = P_frame × Δ_unit × 1000  (igual ao 2D)  │
└────────────────────────────────────────────────────┘
```

### Massa Total 3D

```python
# Volume de um pórtico:
V_frame = Σ section_area(nᵢ) × Lᵢ

# Volume dos 10 travamentos (n=1 palito cada, comprimento w):
V_brace = 10 × (B × T) × w_m

# Massa total:
massa_3D = ρ × (2 × V_frame + V_brace) × 1000  [gramas]
```

### Contagem de Palitos R5

```
Σni_3D = 2 × Σni_pórtico  +  10 × 1  ≤   150
```

Valores típicos:
- Pórtico com n mínimo (tudo 1): Σni = 17 → total = 2×17+10 = **44**
- Pórtico com n máximo (tudo 3): Σni = 51 → total = 2×51+10 = **112 < 150 ✓**

### Fitness 3D

```
Fitness = P_3D [kg] / massa_3D [g]
        = (2 × P_frame [kg]) / (2 × massa_frame + massa_brace [g])
```

> O fitness 3D é **ligeiramente menor** que o 2D porque o denominador cresce mais
> que o numerador (travamentos adicionam massa sem adicionar carga).

### Comparação 2D × 3D (mesma geometria, p=10, h=5, n tipico)

| Grandeza | 2D | 3D |
|---|---|---|
| Massa | ~19 g | ~42 g |
| Carga de ruptura | ~36 kg | ~71 kg |
| Fitness | ~1,84 kg/g | ~1,71 kg/g |
| Palitos R5 | ~21 | ~52 |
| Fica em pé sozinha | **Não** | **Sim** |

### Gene `w` — Largura entre Pórticos

- Intervalo: [2, 6] cm
- Cruzamento: BLX-α escalar (mesmo α = 0,35 dos outros genes contínuos)
- Mutação: Gaussiana σ = 0,4 cm, prob = 20%
- Impacto: aumentar w aumenta a massa dos travamentos (penaliza fitness)
- O AG tende a minimizar w para reduzir massa

### Travamentos — Detalhes de Construção

- 10 travamentos de comprimento `w` cm cada
- Seção fixa: 1 palito (não é gene — irrelevante estruturalmente)
- De um palito de 11,4 cm cortam-se `⌊11.4/w⌋` travamentos
- Palitos físicos necessários: `⌈10 / ⌊11.4/w⌋⌉`

Exemplo com w = 3 cm: `⌊11.4/3⌋ = 3` peças por palito → `⌈10/3⌉ = 4` palitos físicos.

---

## Fluxo de Execução (versões corrigida e 3D)

```
1. Gera população inicial aleatória
        ↓
2. Para cada indivíduo → evaluate():
   ├─ repair_panels()      (garante Σp = 40 cm, R1, R3)
   ├─ clip h, w            (R2)
   ├─ solve_axial_forces() (equilíbrio de nós, 20×20)
   ├─ Verifica R4, R5, R6  (diagonais, palitos, massa)
   ├─ bar_capacity()       (σt ou σc × A, por barra)
   ├─ Carga de ruptura     (min capᵢ / |n̄ᵢ|)
   └─ fitness = carga/massa (0 se inviável)
        ↓
3. Seleção por torneio (k=4)
4. Cruzamento BLX-α (contínuo) + uniforme (discreto)  [90% dos pares]
5. Mutação gaussiana (contínuo, p=20%) + troca (discreto, p=8%)
6. Elitismo: 8 melhores preservados sem mutação
7. Repete por 400 gerações
        ↓
8. Retorna melhor indivíduo + histórico
9. Gera relatório no terminal
10. Gera best_truss.gif e best_truss_final.png
```

---

## Restrições — Comparação entre Versões

| # | Restrição | Original | Corrigido | 3D |
|---|---|---|---|---|
| R1 | 3 ≤ pᵢ ≤ 11,4 cm | ✓ | ✓ | ✓ |
| R2 | 3 ≤ hᵢ ≤ 11,4 cm | ✓ | ✓ | ✓ |
| R3 | Σpᵢ = 40 cm | ✓ | ✓ | ✓ |
| R4 | diagonal ≤ 11,4 cm | ✓ | ✓ | ✓ |
| R5 | palitos ≤ 150 | `Σ nᵢ⌈L/11.4⌉` | `Σ nᵢ` | `2Σnᵢ+10` |
| R6 | massa < 600 g | 1 pórtico | 1 pórtico | **2 pórticos + trav.** |
| R7 | σ_tração ≤ 27,5 MPa | redundante | removida | removida |
| R8 | σ_compressão ≤ 17,5 MPa | redundante | removida | removida |
| — | w ∈ [2,6] cm | — | — | ✓ (novo) |

---

## Parâmetros do AG

| Parâmetro | Valor padrão |
|---|---|
| Tamanho da população | 250 |
| Gerações | 400 |
| Taxa de cruzamento | 90% |
| Elitismo | 8 |
| Torneio k | 4 |
| α BLX (contínuo) | 0,35 |
| σ mutação p (painéis) | 0,80 cm |
| σ mutação h (alturas) | 0,60 cm |
| σ mutação w (largura, só 3D) | 0,40 cm |
| Prob. mutação contínua | 20% por gene |
| Prob. mutação discreta | 8% por gene |
| Seed padrão | 42 |

---

## Como Executar

```bash
cd Victor_version/

# Versão original (sem visualização):
python3 trelica_howe_ga.py

# Versão 2D corrigida (com GIF e PNG):
python3 trelica_howe_ga_corrigido.py

# Versão 3D construível (com GIF e PNG):
python3 trelica_howe_3d.py
```

**Arquivos gerados:**

| Arquivo | Versão |
|---|---|
| `best_truss.gif` | 2D corrigida |
| `best_truss_final.png` | 2D corrigida |
| `best_truss_3d.gif` | 3D |
| `best_truss_3d_final.png` | 3D |

---

## Qual Usar?

| Situação | Arquivo recomendado |
|---|---|
| Entender o código original / comparar | `trelica_howe_ga.py` |
| Apresentação do AG (arguição) | `trelica_howe_ga_corrigido.py` |
| Construção física do protótipo | `trelica_howe_3d.py` |
