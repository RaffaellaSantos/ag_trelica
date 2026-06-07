# Algoritmo Genético para Otimização de Treliça Howe de Palitos
**Atividade 9 — EST/UEA · Algoritmos de Otimização**
**Prof. Rubelmar Azevedo Cruz Neto**

---

## 1. Formulação do Problema

O problema é classificado como **MINLP** (Mixed-Integer Non-Linear Programming): variáveis contínuas para a geometria e discretas para as seções transversais.

**Objetivo:** maximizar a eficiência estrutural

```
Fitness(x) = Carga de Ruptura Teórica [kg] / Massa Total [g]  ×  P(x)
```

onde `P(x) = 1` se todas as restrições forem satisfeitas, `P(x) = 0` caso contrário (penalidade multiplicativa).

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
| Verticais externas | A-F, E-J | 2 |
| Verticais internas | G-B, H-C, I-D | 3 |
| Diagonais | F-B, G-C, I-C, J-D | 4 |
| **Total** | | **17 barras** |

- **10 nós**, **vão livre 40 cm**, **carga concentrada no nó C**
- Apoio A = pino (x e y fixos) · Apoio E = rolete (só y fixo)

---

## 3. Cromossomo

### Versão 2D — 26 genes

```
┌──────────────────────────┬─────────────────────────────────────┐
│   BLOCO 1 — Contínuo (9) │       BLOCO 2 — Discreto (17)       │
├──────────┬───────────────┼──────────────────────────────────────┤
│ p1 p2 p3 p4 │ h1 h2 h3 h4 h5 │ n1…n4   n5…n8  n9…n13  n14…n17 │
│  painéis    │    alturas      │ b.inf  b.sup  vert.   diag.   │
└──────────┴───────────────┴──────────────────────────────────────┘
  [3, 11.4] cm   [3, 11.4] cm          {1, 2, 3} palitos
```

### Versão 3D — 27 genes (adiciona gene `w`)

```
┌──────────────────────────────┬──────────────────────────────────┐
│    BLOCO 1 — Contínuo (10)   │      BLOCO 2 — Discreto (17)     │
├──────────┬───────────────┬───┼──────────────────────────────────┤
│ p1 p2 p3 p4 │ h1..h5 │ w  │         n1…n17                   │
└──────────┴───────────────┴───┴──────────────────────────────────┘
                        [2,6] cm
```

`w` = largura entre os dois pórticos paralelos (gene novo).

**Restrição de reparo:** `p1+p2+p3+p4 = 40 cm` é garantida por um algoritmo iterativo de redistribuição após qualquer operação genética.

---

## 4. Análise Estrutural — Equilíbrio de Nós

Reaproveitado de **`app/utils.py` (Atividade 8)**: `Utils.calcular_reacoes_e_forcas`.

Para cada nó `k`, escreve-se ΣFx = 0 e ΣFy = 0. O sistema resultante 20×20 tem como incógnitas as 17 forças axiais + 3 reações de apoio (Ax, Ay, Ey):

```
A_eq · [F₁ … F₁₇,  Ax,  Ay,  Ey]ᵀ  =  b
```

- Carga aplicada: **1 N para baixo no nó C** (análise unitária)
- A linearidade do sistema permite obter a carga de ruptura sem resolver novamente

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

> **Por que palitos em paralelo?** A cola PVA não garante transmissão de cisalhamento suficiente entre os palitos para comportamento monolítico sob flambagem. Cada palito flamba independentemente — premissa conservadora e fisicamente justificada.

### Carga de Ruptura Teórica
```
P_rup_i = cap_i / |n̄ᵢ|     (para cada barra i)
P_rup   = min(P_rup_i)       (barra que falha primeiro)
```

### Versão 3D
Cada pórtico recebe P/2. A estrutura falha quando P/2 = P_frame_ruptura:
```
P_3D = 2 × P_frame
massa_3D = 2 × massa_pórtico + massa_travamentos
```

---

## 6. Restrições

| # | Restrição | Critério |
|---|---|---|
| R1 | Comprimento painel | 3 ≤ pᵢ ≤ 11.4 cm |
| R2 | Altura vertical | 3 ≤ hᵢ ≤ 11.4 cm |
| R3 | Vão total | Σpᵢ = 40 cm |
| R4 | Diagonal física | √(pᵢ² + hᵢ²) ≤ 11.4 cm |
| R5 | Total de palitos | Σnᵢ ≤ 150 (2D) · 2·Σnᵢ+10 ≤ 150 (3D) |
| R6 | Massa total | m < 600 g |
| R7 | Tensão de tração | σ ≤ 27.5 MPa |
| R8 | Tensão de compressão | \|σ\| ≤ 17.5 MPa |

**Penalidade:** multiplicativa — qualquer violação zera o fitness.

---

## 7. Algoritmo Genético

### Parâmetros padrão

| Parâmetro | Valor |
|---|---|
| Tamanho da população | 250 |
| Gerações | 400 |
| Taxa de cruzamento | 90% |
| Elitismo | 8 melhores preservados |
| Torneio | k = 4 |

### Cruzamento — BLX-α (α = 0.35) para genes contínuos
```
c₁ = U[lo − α·d,  hi + α·d]
c₂ = U[lo − α·d,  hi + α·d]

onde lo = min(p1, p2),  hi = max(p1, p2),  d = hi − lo
```
Permite explorar além do intervalo dos pais, aumentando diversidade.

Para genes discretos: **cruzamento uniforme** (máscara binária 50/50).

### Mutação
| Tipo | Operação | Probabilidade |
|---|---|---|
| Contínua (p, h, w) | Perturbação gaussiana σ ≈ 0.6–0.8 cm | 20% por gene |
| Discreta (n) | Substituição aleatória em {1, 2, 3} | 8% por gene |

### Seleção e elitismo
- **Torneio de tamanho k=4**: sorteia 4 indivíduos, retorna o melhor
- **Elitismo**: os 8 melhores passam direto para a próxima geração sem mutação

---

## 8. Saídas do Algoritmo

1. **Terminal** — log a cada 25 gerações + relatório completo ao final
2. **`best_truss.gif`** — animação com todas as 400 gerações
3. **`best_truss_final.png`** — imagem final: treliça ótima + scatter massa × carga

O scatter mostra a **fronteira Pareto** (minimizar massa, maximizar carga) acumulada ao longo da evolução.

---

## 9. Valores Esperados

Com base nas propriedades do material e na fórmula de Euler para palitos em paralelo:

| Configuração | Carga 2D | Carga 3D |
|---|---|---|
| h=5 cm, n=1 | ~1.2 kg | ~2.4 kg |
| h=5 cm, n=2 | ~2.4 kg | ~4.7 kg |
| h=5 cm, n=3 | ~3.5 kg | ~7.0 kg |
| Otimizado (AG) | 5–15 kg | 10–25 kg |

Os valores estão alinhados com o intervalo de **3–12 kg** indicado no PDF para geometrias típicas.

---

## 10. Como Executar

```bash
cd Victor_version/

# Versão 2D (um pórtico):
python3 trelica_howe_ga_corrigido.py

# Versão 3D (dois pórticos + travamentos):
python3 trelica_howe_3d.py
```
