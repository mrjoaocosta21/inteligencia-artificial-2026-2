# Investigação e correções

## Pergunta investigada

Por que remover 85% dos idosos não piorava o recall médio da regressão logística,
embora o terceiro pilar do seminário peça explicar falhas na identificação de positivos?

O efeito exigido pelo enunciado não é uma consequência automática da remoção para
qualquer modelo ou sorteio. A investigação encontrou prejuízo por falsos positivos
na logística e no KNN, e aumento médio de falsos negativos na árvore complementar.
Não foi encontrado erro na fórmula do recall nem vazamento de `num` para as entradas.

## Controles preservados

- Base Cleveland original: hash e fonte registrados em `resultados/metadados.json`.
- Corte `age > 60`, alvo `num > 0`, limpeza e divisão estratificada originais.
- Remoção uniforme e aleatória dentro dos idosos, independentemente do alvo.
- 55 idosos no treino original, 8 em B; não idosos mantidos integralmente.
- Teste intacto e compartilhado entre cenários da mesma semente.
- Pré-processamento ajustado somente no treino, sem escolher limiar no teste.
- Todas as sementes 0–49 e todos os modelos comparados foram registrados.

A/B/C da regressão logística reproduziram exatamente as taxas do CSV anterior
à alteração (diferença máxima 0 ao ler os dois CSVs). A árvore e o KNN foram
acrescentados após observar a divergência original e são análises exploratórias,
com parâmetros fixos e sem busca de hiperparâmetros orientada pela queda no teste.

## 1. Retirada da idade

`python -m src.diagnostico_idade` agora existe e gera uma comparação pareada:

| Logística, idosos | Recall A | Recall B | Especificidade A | Especificidade B |
|---|---:|---:|---:|---:|
| Com idade | 0,851 | 0,866 | 0,771 | 0,678 |
| Sem idade | 0,860 | 0,869 | 0,762 | 0,691 |

A piora da especificidade permanece ao remover `age`. A idade explícita não é
necessária para esse efeito nesta configuração. Isso não identifica variáveis proxy
nem comprova um mecanismo causal específico envolvendo outra característica.
As afirmações anteriores sobre esse experimento agora têm código e resultados.

## 2. Dependência do modelo

| Modelo, idosos | Recall A | Recall B | FN médio A | FN médio B | Recall B: pior/igual/melhor |
|---|---:|---:|---:|---:|---|
| Logística | 0,851 | 0,866 | 1,94 | 1,74 | 4 / 34 / 12 |
| Árvore: profundidade 3, folha mínima 5 | 0,752 | 0,726 | 3,22 | 3,56 | 18 / 18 / 14 |
| KNN: 5 vizinhos | 0,866 | 0,872 | 1,74 | 1,66 | 8 / 30 / 12 |

Na árvore, a mudança média B−A no recall é −0,026, com desvio 0,143 entre divisões.
A grande variação impede apresentar a direção média como resultado universal ou
estatisticamente confirmado. Os quantis dos deltas em `sensibilidade_modelos_diferencas.csv`
são descrições dos sorteios, não intervalos de confiança.

O modelo muda a forma de combinar as características. Uma regressão logística
compartilha coeficientes entre pacientes; uma árvore aprende cortes e decisões por folha.
Ao retirar observações, os cortes e a composição das folhas podem mudar. Esse mecanismo
é verificável nos caminhos exportados; não significa que a árvore seja clinicamente melhor.

## 3. Exemplo auditável na semente original 42

Para idosos, na árvore:

| Cenário | TN | FP | FN | TP | Recall |
|---|---:|---:|---:|---:|---:|
| A | 5 | 6 | 2 | 11 | 11/13 = 0,846 |
| B | 7 | 4 | 5 | 8 | 8/13 = 0,615 |
| C | 9 | 2 | 3 | 10 | 10/13 = 0,769 |
| D | 6 | 5 | 4 | 9 | 9/13 = 0,692 |

Os novos falsos negativos de B são as linhas originais **2, 162 e 194** da base.
A linha 2 é de um paciente de 67 anos com doença (`target=1`):

- A: `ca > 0,5`, indicador `thal=3` presente e indicador `cp=3` ausente.
  A folha tem 22 registros e proporção de doentes 0,727; a previsão é positiva.
- B: `ca > 0,5`, indicador `thal=3` presente e `trestbps > 123`.
  A folha tem 13 registros e proporção de doentes 0,308; a previsão é negativa.

As folhas têm condições diferentes: não são o mesmo conjunto apenas com menos linhas.
A remoção modifica a árvore e a decisão para um paciente que continua no teste.
`arvore_casos_semente_42.csv` registra todos os 24 idosos, não apenas os três novos erros,
com caminhos e probabilidades de A/B/C/D. Os limiares numéricos do CSV estão nas unidades
originais; os arquivos de árvore completos usam valores padronizados, explicitamente rotulados.
Essas proporções de folhas não são probabilidades calibradas de risco clínico.

## 4. Separação do efeito da divisão e da remoção

Mantendo treino/teste em 42 e variando apenas a remoção na logística:

- Recall dos idosos: 1,000 em A e média 0,995 em B.
- Especificidade dos idosos: 0,636 em A e média 0,538 em B.
- Entre os 8 idosos mantidos no treino B, há de 1 a 8 doentes, conforme a remoção.
- A probabilidade média prevista para idosos sem doença passa de 0,461 a 0,510.

Esse diagnóstico mostra que a piora de especificidade também ocorre ao manter o
teste fixo; não depende apenas da troca dos pacientes de teste entre repetições.
A composição instável dos poucos idosos restantes é um fator plausível a investigar,
mas esta análise não isola todos os mecanismos de mudança dos parâmetros.

## 5. Mitigação avaliada

C conserva a tentativa de reponderação original. Pesos corrigem a influência total
do grupo no objetivo do classificador, mas não recuperam exemplos diversos excluídos.
O scaler continua sem pesos. C não foi renomeado como correção bem-sucedida.

D simula coleta adicional: sorteia metade dos idosos excluídos do treino e os repõe,
sem duplicar registros e sem usar pacientes do teste. Passa de 8 para 32 idosos.
A fração de reposição é fixa, não escolhida pelas métricas de teste.

| Idosos | B | C | D |
|---|---:|---:|---:|
| Especificidade, logística | 0,678 | 0,705 | 0,740 |
| Acurácia, logística | 0,780 | 0,779 | 0,803 |
| Recall, logística | 0,866 | 0,842 | 0,857 |
| Recall, árvore | 0,726 | 0,706 | 0,745 |

D recupera parte do desempenho agregado, com trocas entre métricas e variação.
Na árvore da semente 42, C ajuda mais o recall que D; nas médias, ocorre o contrário.
Isso ilustra por que uma única divisão não deve fundamentar a escolha da mitigação.
Coleta real mais representativa continua sendo uma proposta, cuja eficácia precisaria
de validação em outra amostra. KNN não aceita `sample_weight`, por isso não tem C.

## Correções entregues e validação

- Módulo de ablação da idade implementado; investigação de modelos e remoção isolada adicionada.
- Identificador de linha preservado; CSVs agora registram matrizes, denominadores,
  composição dos treinos e previsões por paciente.
- Gráficos agregados, matrizes e caminhos de decisão gerados.
- Gráfico A/B/C/D específico da árvore e comando único `python -m src.apresentacao`
  adicionados para a demonstração dos quatro pilares.
- Reposição parcial avaliada e reponderação descrita como tentativa com resultados mistos.
- Discussão ética e roteiro dos quatro pilares adicionados; conclusões ajustadas à evidência.
- Tabelas Markdown corrigidas; `oldpeak` documentado como real e 13 entradas distinguidas do alvo.
- Hash, versões, parâmetros e sementes registrados; versões principais disponibilizadas para reprodução.
- Oito testes passam, cobrindo métricas conhecidas, denominadores ausentes, entradas inválidas,
  definição de grupos, preservação do teste, integridade da reposição, pré-processamento e exportações.

## Correspondência com a professora

O código atende ao filtro solicitado e oferece um exemplo concreto de positivos não
identificados na árvore, acompanhado de análise agregada e mitigação. A regressão
logística mantém sua conclusão original, agora explicitamente distinguida desse mecanismo.
Permanece uma questão de interpretação do enunciado: se a professora exige que o
resultado principal tenha queda consistente de recall, a evidência exploratória atual
precisa ser apresentada com essa limitação. Não se afirma que a remoção sempre reduz recall.
