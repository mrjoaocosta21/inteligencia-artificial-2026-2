# Viés por sub-representação em doença cardíaca

Seminário de Inteligência Artificial 2026.2 — Cenário 3: Diagnóstico e Saúde Pública.

## Objetivo e correspondência com o enunciado

Avaliar o efeito de remover 85% dos pacientes com `age > 60` somente do treino.
O enunciado pede explicar falhas na identificação de casos positivos no grupo minoritário.
A regressão logística original apresenta principalmente aumento de falsos positivos;
uma árvore de decisão complementar apresenta aumento médio de falsos negativos.
Ambos os resultados são apresentados, com sua variabilidade, sem escolher sementes
ou ajustar modelos para obter queda no teste.

Veja o [relatório da investigação](RELATORIO-INVESTIGACAO.md) e o
[roteiro dos quatro pilares](roteiro-apresentacao.md). A aceitação de falsos positivos
como atendimento do terceiro pilar ainda não foi esclarecida com a professora.

## Base e preparação

[Heart Disease — UCI](https://archive.ics.uci.edu/dataset/45/heart+disease), arquivo Cleveland.

- 303 registros, 13 características e `num` (14 colunas originais).
- 6 registros com `?` em `ca` ou `thal` são excluídos; restam 297, incluindo 79 idosos.
  Os 6 excluídos têm idade até 60; a limpeza altera ligeiramente a composição da base.
- `target = (num > 0)`: ausência versus presença de doença, não previsão de doença futura.
- `idoso = (age > 60)`: regra do exercício, não definição clínica ou legal universal.
- `paciente_id` é o número da linha no arquivo original, não um identificador pessoal.
- Se o arquivo local não existir, `src.dados` o baixa da UCI.

A divisão 70%/30% ocorre antes da remoção e é estratificada por `(idoso, target)`.
O teste tem 90 pacientes: 24 idosos (13 doentes, 11 sem doença) e 66 não idosos.
Cada treino original tem 55 idosos; a remoção mantém 8 e exclui 47
(85,45%, por arredondamento para registros inteiros), sem excluir não idosos.

## Execução

Execute os comandos dentro de `seminario-vies-cardio/`, usando Python 3.10 ou superior:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.dados
python -m src.grupos
python -m src.divisao
python -m src.desequilibrio
python -m src.modelo
python -m src.diagnostico_idade
python -m src.mitigacao
python -m src.investigacao
python -m src.apresentacao       # demonstração única dos quatro pilares
python -m unittest discover -s tests -v
```

Os módulos refazem suas próprias etapas anteriores; não dependem da ordem de
execução para carregar os dados. As três últimas análises geram os artefatos em
`resultados/`. Use `python -m src.<nome>` em vez de `python src/<nome>.py`.

As versões principais usadas na validação estão em `requirements-reproducao.txt`
e `resultados/metadados.json` (Python 3.13.14). Para reproduzir esse ambiente,
use `python -m pip install -r requirements-reproducao.txt` em um ambiente compatível.
`requirements.txt` mantém os mínimos suportados; outras versões podem mudar os resultados.

## Modelos e comparação

A regressão logística continua sendo a referência original (`C=1`, `lbfgs`).
Características numéricas são padronizadas, categorias recebem one-hot e variáveis
binárias passam diretamente. `num`, `target`, `idoso` e `paciente_id` não entram no modelo.
Todo o pré-processamento é ajustado exclusivamente no respectivo treino.
Usa-se `predict()` padrão, sem escolher limiar no teste; as probabilidades não são
validadas como estimativas de risco clínico.

- **A — treino completo:** referência sem a remoção artificial; não implica ausência de outros vieses.
- **B — sub-representação:** remove 85% dos idosos do treino.
- **C — reponderação:** usa B, com peso `55/8` nos idosos, apenas no classificador.
  O StandardScaler continua sem pesos, preservando o experimento original.
- **D — reposição parcial:** acrescenta a B metade dos 47 idosos excluídos do treino,
  sorteados sem consultar o teste. Restam 32 idosos. É simulação de coleta adicional.

As sementes 0 a 49 variam a divisão e a remoção. A/B/C/D compartilham exatamente
o mesmo teste dentro de cada repetição, mas os pacientes do teste mudam entre sementes.
As repetições reutilizam pacientes: o desvio padrão descreve variação entre sorteios,
não um intervalo de confiança de 50 amostras independentes.

Médias nos idosos para a regressão logística:

| Métrica | A | B | C | D |
|---|---:|---:|---:|---:|
| Recall | 0,851 | 0,866 | 0,842 | 0,857 |
| Especificidade | 0,771 | 0,678 | 0,705 | 0,740 |
| Acurácia | 0,814 | 0,780 | 0,779 | 0,803 |
| Falsos negativos por repetição | 1,94 | 1,74 | 2,06 | 1,86 |
| Falsos positivos por repetição | 2,52 | 3,54 | 3,24 | 2,86 |

## Investigação da divergência

- **Ablação da idade:** a especificidade dos idosos cai de 0,762 para 0,691 sem `age`.
  Retirar a idade não elimina esse prejuízo; isso não comprova que outra variável seja proxy.
- **Sensibilidade ao modelo:** uma árvore com profundidade 3 e pelo menos 5 casos por folha
  apresenta recall dos idosos de 0,752 em A e 0,726 em B. A queda ocorre em 18/50
  repetições, melhora em 14 e empata em 18: a direção média existe, mas é instável.
  O KNN de 5 vizinhos passa de 0,866 para 0,872, sem queda média de recall.
- **Demonstração fixa em 42:** na árvore, os falsos negativos dos idosos passam de 2 para 5
  e o recall de 11/13 para 8/13. Na logística, ambos os cenários têm zero falsos negativos.
  A semente 42 foi mantida como referência original; não é estimativa do efeito médio.
- **Remoção isolada:** mantendo a divisão em 42 e variando somente a remoção, a logística
  apresenta especificidade média de 0,636 em A e 0,538 em B. Os 8 idosos mantidos
  incluem de 1 a 8 doentes, mostrando a variabilidade da composição do treino.

## Mitigação: resultados e limites

C melhora um pouco a especificidade média da logística, mas piora recall e não
recupera acurácia. Na árvore, C também não recupera o recall médio de B.
Reponderação altera a influência de exemplos existentes; não recupera diversidade perdida.

D recupera parcialmente a especificidade e a acurácia média da logística.
Na árvore, recupera parte do recall: 0,726 em B para 0,745 em D.
Isso apoia investigar coleta mais representativa, mas não prova eficácia em outro hospital
nem estabelece uma estratégia universalmente superior.

## Artefatos e auditoria

| Arquivo em `resultados/` | Conteúdo |
|---|---|
| `comparacao_cenarios.csv` | A/B/C/D da logística: taxas, TN/FP/FN/TP, denominadores e composição dos treinos |
| `predicoes_cenarios.csv` | Alvo, previsão e probabilidade por linha original, semente e cenário |
| `comparacao_cenarios.png`, `matrizes_semente_42.png` | Comparação agregada e matrizes da logística |
| `resumo_cenarios.csv`, `diferencas_pareadas.csv` | Médias, desvios e mudanças dentro de cada divisão |
| `diagnostico_idade*.csv` | Logística com/sem idade e diferenças B−A |
| `sensibilidade_modelos*.csv`, `sensibilidade_modelos.png` | Todos os modelos e diferenças pareadas; KNN não aceita C |
| `comparacao_cenarios_arvore.csv`, `comparacao_cenarios_arvore.png` | A/B/C/D da árvore nas 50 repetições |
| `resumo_cenarios_arvore.csv` | Médias e desvios da árvore usados na apresentação |
| `predicoes_modelos.csv`, `erros_pareados.csv` | A/B por paciente, incluindo novos FN e FP |
| `remocao_teste_fixo*.csv` | Divisão 42, variando somente a remoção |
| `matrizes_arvore_semente_42.png`, `arvore_casos_semente_42.csv` | Matrizes e caminhos de decisão de todos os 24 idosos |
| `arvore_[A-D]_semente_42.txt` | Árvores com limiares numéricos padronizados; o CSV usa unidades originais |
| `metadados.json` | Hash da base, versões, sementes, parâmetros e regras do experimento |

## Ética e limitações

Falsos negativos podem atrasar a identificação de doença se a saída for usada para
decidir encaminhamento. Falsos positivos podem gerar preocupação e avaliações
adicionais. A importância de cada erro depende do fluxo assistencial; melhorar uma
métrica pode piorar outra, como observado na reponderação.

A coleta desigual pode fazer grupos terem desempenho diferente mesmo sem uma regra
explícita que os exclua. Por isso é necessário avaliar os erros por grupo, ampliar a
representação e prever revisão humana e monitoramento caso um sistema seja desenvolvido
para uso real. Retirar idade das entradas, por si só, não assegura equidade.

Este é um experimento educacional: base histórica pequena, uma instituição, apenas
13 idosos doentes no teste e remoção aleatória simplificada. Não mede eventos futuros,
não testa um fluxo clínico e não permite generalização para hospitais atuais.
A árvore complementar e o KNN são análises exploratórias. O resultado agregado
não demonstra uma queda universal de recall por sub-representação.

Referência da base: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.
Documentação técnica: [prevenção de vazamento](https://scikit-learn.org/stable/common_pitfalls.html),
[árvore de decisão](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
e [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
