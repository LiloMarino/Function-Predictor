# Function Predictor (NEAT)

## Visão Geral

Este projeto explora o uso do algoritmo NEAT (*NeuroEvolution of Augmenting Topologies*) para evoluir redes neurais capazes de prever e extrapolar sequências numéricas geradas por funções matemáticas desconhecidas.

O foco do experimento não é descobrir explicitamente a fórmula da função, mas sim aprender padrões estruturais a partir de amostras e prever os próximos valores da sequência.

A proposta investiga até que ponto uma rede evoluída consegue generalizar comportamentos matemáticos observando apenas valores anteriores.

---

## Problema

O sistema recebe como entrada uma sequência de valores:

```text
[y₁, y₂, y₃, ..., yₙ]
```

e deve prever:

```text
yₙ₊₁
```

A função geradora da sequência é desconhecida pela rede.

O problema é tratado como extrapolação de sequência determinística, e não como regressão simbólica ou identificação explícita de fórmulas matemáticas.

---

## Famílias de Funções

O modelo será exposto a diferentes famílias de funções, incluindo:

* Polinomiais

  * Linear
  * Quadrática
  * Cúbica
  * Quartica
* Exponenciais
* Logarítmicas
* Trigonométricas

  * Seno
  * Cosseno

Cada família poderá conter diversas variações de parâmetros, escalas e deslocamentos.

---

## Objetivo do Experimento

O objetivo principal é avaliar a capacidade do NEAT de:

* Aprender padrões matemáticos a partir de amostras
* Generalizar entre diferentes funções da mesma família
* Extrapolar valores fora da janela observada
* Evoluir topologias neurais eficientes para previsão de sequência

O fitness das redes é calculado com base no erro entre os valores previstos e os valores reais em pontos futuros da sequência.

---

## Escopo

O projeto investiga conceitos relacionados a:

* Neuroevolução
* Extrapolação de sequência
* Generalização de padrões
* Inferência implícita de comportamento matemático
* Aprendizado baseado em observação parcial

Embora seja possível explorar futuramente a identificação da função geradora, esse não é o foco principal do experimento.

---

## Ideia Central

Dada apenas uma sequência de números, a rede deve ser capaz de inferir o comportamento subjacente da série e continuar sua progressão sem acesso direto à fórmula original.

Exemplo:

```text
Entrada: [1, 4, 9, 16, 25]
Saída esperada: 36
```

O desafio está em aprender o padrão da sequência, e não memorizar funções específicas.
