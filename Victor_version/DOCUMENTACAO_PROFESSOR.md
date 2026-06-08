# Algoritmo Genético para Otimização de Treliça Howe de Palitos
**Atividade 9 — EST/UEA · Algoritmos de Otimização**
**Prof. Rubelmar Azevedo Cruz Neto**

---

## 1. Formulação do Problema

O problema é classificado como **MINLP** (Mixed-Integer Non-Linear Programming): variáveis contínuas para a geometria e discretas para as seções transversais.

**Formulação bi-objetivo** (NSGA-II):

```
Minimizar  f1(x) = massa_total [g]  +  penalidade(x)
Minimizar  f2(x) = −carga_ruptura [kg]  +  penalidade(x)
```

Minimizar `f2` equivale a maximizar a carga de ruptura. A formulação bi-objetivo expõe a fronteira Pareto entre leveza e resistência, permitindo ao projetista escolher o ponto de compromisso adequado.

---

## 2. Estrutura da Treliça — Howe 4 Painéis

```
F────G────H────I────J     ← banzo superior
|  ╲ |  ╲ |  ╱ |  ╱ |
A────B────C────D────E     ← banzo inferior
              ↑
           Carga P
           (balde)
```

| Grupo | Barras | Quantidade |
|---|---|---|
| Banzo inferior | A-B, B-C, C-D, D-E | 4 |
| Banzo superior | F-G, G-H, H-I, I-J | 4 |
| Verticais | A-F, G-B, H-C, I-D, E-J | 5 |
| Diagonais (Howe) | F-B, G-C, I-C, J-D | 4 |
| **Total** | | **17 barras, 10 nós** |

- **Vão livre:** 40 cm · **Carga concentrada:** nó C
- Apoio A = pino (x, y) · Apoio E = rolete (y)

---

## 3. Cromossomo

### Versão 2D — 26 genes

```
┌─────────────────────────────────┬────────────────────────────────────┐
│     BLOCO 1 — Contínuo (9)      │      BLOCO 2 — Discreto (17)       │
├─────────────┬───────────────────┼────────────────────────────────────┤
│  p1 p2 p3 p4 │  h1 h2 h3 h4 h5 │  n1…n4   n5…n8   n9…n13  n14…n17 │
│   painéis    │     alturas      │  b.inf   b.sup   vert.    diag.   │
└─────────────┴───────────────────┴────────────────────────────────────┘
   [3, 11.4] cm    [3, 11.4] cm            {1, 2, 3} palitos
```

### Versão 3D — 27 genes (adiciona gene `w`)

```
┌────────────────────────────────────┬──────────────────────────────────┐
│      BLOCO 1 — Contínuo (10)       │      BLOCO 2 — Discreto (17)     │
├──────────────┬────────────────┬────┼──────────────────────────────────┤
│  p1 p2 p3 p4  │  h1 h2 h3 h4 h5  │  w │        n1…n17               │
└──────────────┴────────────────┴────┴──────────────────────────────────┘
                                  [3, 11.4] cm
```

`w` = largura entre os dois pórticos paralelos (gene novo).

**Restrição de reparo:** `p1+p2+p3+p4 = 40 cm` é garantida por um algoritmo iterativo de redistribuição após qualquer operação genética. R1 e R2 são garantidas por clipagem imediata.

---

## 4. Análise Estrutural — Equilíbrio de Nós

**Reaproveitado da Atividade 8** — `app/utils.py`: `Utils.calcular_reacoes_e_forcas`.

Para cada nó `k`, escreve-se ΣFx = 0 e ΣFy = 0. O sistema 20×20 tem como incógnitas as 17 forças axiais + 3 reações de apoio (Ax, Ay, Ey):

```
A_eq · [F₁ … F₁₇,  Ax,  Ay,  Ey]ᵀ  =  b
```

- Carga aplicada: **1 N para baixo no nó C** (análise unitária)
- A linearidade permite obter a carga de ruptura sem nova resolução do sistema

**Deslocamento em C** pelo Princípio dos Trabalhos Virtuais:

```
Δ_C = P · Σ (n̄ᵢ² · Lᵢ) / (E · Aᵢ)
```

onde `n̄ᵢ` são as forças axiais sob carga unitária.

---

## 5. Capacidade das Barras e Carga de Ruptura

### Tração
```
cap = σt_adm · A = 27.5 MPa × A      (Sg = 2)
```

### Compressão — mínimo entre resistência e flambagem de Euler
```
cap = min(σc_adm · A,  F_euler)

F_euler = n · π² · E · I_palito / L²    (palitos em paralelo, ∝ n)
I_palito = b · t³ / 12                  (1 palito individual)
```

> **Por que palitos em paralelo?** A cola PVA não garante transmissão de cisalhamento suficiente para comportamento monolítico sob flambagem. Cada palito flamba independentemente — premissa conservadora e fisicamente justificada. Sem essa restrição os resultados chegam a 50–120 kg (irrealistas); com ela, 3–12 kg (condizente com o enunciado).

### Carga de Ruptura Teórica
```
P_rup_i = cap_i / |n̄ᵢ|     (para cada barra i)
P_rup   = min(P_rup_i)       (barra que falha primeiro)
```

### Versão 3D
Cada pórtico recebe P/2. A estrutura falha quando P/2 = P_frame_ruptura:
```
P_3D = 2 × P_frame
massa_3D = 2 × massa_pórtico + 10 × section_area(1) × w
```

---

## 6. Restrições e Penalidade

| # | Restrição | Critério | Tratamento |
|---|---|---|---|
| R1 | Comprimento painel | 3 ≤ pᵢ ≤ 11.4 cm | Reparo/clipagem |
| R2 | Altura vertical | 3 ≤ hᵢ ≤ 11.4 cm | Clipagem |
| R3 | Vão total | Σpᵢ = 40 cm | Reparo iterativo |
| R4 | Diagonal física | √(pᵢ²+hᵢ²) ≤ 11.4 cm | Penalidade aditiva |
| R5 | Total de palitos | Σnᵢ ≤ 150 (2D) · 2·Σnᵢ+10 ≤ 150 (3D) | Penalidade aditiva |
| R6 | Massa total | m < 600 g | Penalidade aditiva |

**Penalidade aditiva** — mesmo padrão da Atividade 8:
```
penalidade = P_R4 × Σ max(0, Lᵢ_diagonal − 11.4)   [P_R4 = 10.0]
           + P_R5 × max(0, Σnᵢ − 150)               [P_R5 = 0.010]
           + P_R6 × max(0, massa_g − 600)            [P_R6 = 0.005]
```

A penalidade é adicionada a ambos os objetivos. Indivíduos viáveis têm penalidade = 0 e competem pela fronteira Pareto diretamente.

---

## 7. Algoritmo Genético — NSGA-II

### Parâmetros

| Parâmetro | Valor |
|---|---|
| Tamanho da população | 250 |
| Gerações | 400 |
| Taxa de cruzamento | 90% |
| Seed padrão | 42 |

### Seleção — Torneio Binário NSGA-II
Sorteia 2 indivíduos:
- Ganha o de **menor rank de Pareto**
- Empate: ganha o de **maior distância de aglomeração** (mais isolado na frente)

### Cruzamento — BLX-α (α = 0.35) para genes contínuos
```
lo = min(p1, p2),  hi = max(p1, p2),  d = hi − lo
c₁ = U[lo − 0.35·d,  hi + 0.35·d]
c₂ = U[lo − 0.35·d,  hi + 0.35·d]
```
Permite explorar além do intervalo dos pais, aumentando diversidade.

Para genes discretos: **cruzamento uniforme** (máscara binária 50/50).

### Mutação
| Tipo | Operação | Probabilidade |
|---|---|---|
| Contínua (p, h) | Gaussiana σ = 0.80 cm (p) / 0.60 cm (h) | 20% por gene |
| Contínua (w) | Gaussiana σ = 0.40 cm | 20% |
| Discreta (n) | Substituição aleatória em {1, 2, 3} | 8% por gene |

### Seleção da Próxima Geração (elitismo implícito)
1. Combina pais (250) + filhos (250) → 500 indivíduos
2. Classifica todos em frentes de Pareto (`_non_dominated_sort`)
3. Preenche a próxima geração com a frente 1 inteira, depois a frente 2, etc.
4. Se uma frente não cabe inteira: ordena por distância de aglomeração e pega os mais isolados

O elitismo é **implícito**: a frente 1 da geração atual sempre sobrevive para a próxima.

---

## 8. Saídas do Algoritmo

1. **Terminal** — log a cada 25 gerações + relatório técnico + relatório de construção
2. **`best_truss.gif`** / **`best_truss_3d.gif`** — animação da evolução (400 frames)
3. **`best_truss_final.png`** / **`best_truss_3d_final.png`** — treliça ótima + scatter Pareto

O scatter mostra a **fronteira Pareto** (minimizar massa, maximizar carga) acumulada ao longo da evolução.

---

## 9. Valores Esperados

Com base nas propriedades do material e Euler para palitos em paralelo:

| Configuração | Carga 2D | Carga 3D |
|---|---|---|
| h=5 cm, n=1 | ~1.2 kg | ~2.4 kg |
| h=5 cm, n=2 | ~2.4 kg | ~4.7 kg |
| h=5 cm, n=3 | ~3.5 kg | ~7.0 kg |
| Otimizado (AG) | 5–15 kg | 10–25 kg |

---

## 10. Como Executar

```bash
cd Victor_version/

# Versão 2D (um pórtico):
python3 trelica_howe_ga_corrigido.py

# Versão 3D (dois pórticos + travamentos):
python3 trelica_howe_3d.py
```
