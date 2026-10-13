# Roteiro dos quatro pilares

Para executar toda a demonstração em um único comando:

```sh
python -m src.apresentacao
```

O comando imprime as contagens, a matriz da árvore na semente 42, o resultado
agregado das 50 repetições e as mitigações. Também atualiza os gráficos da árvore.

## Abertura: caso e pergunta

“Simulamos um hospital cuja coleta preventiva inclui poucos idosos. Usamos a base
pública Cleveland da UCI para avaliar como a retirada de 85% dos idosos do treino
muda os erros de classificação de doença cardíaca nesse grupo.”

O alvo representa presença de doença no registro, não risco de doença futura.
O experimento é educacional; não avalia um produto em atendimento real.

## Pilar 1 — Explicar o código e discutir ética/técnica

Mostre os módulos em sequência:

1. `dados.py`: lê 14 colunas, reconhece `?`, exclui 6 linhas e cria `target = num > 0`.
   Há 13 entradas e um alvo. O número da linha é guardado apenas para auditoria.
2. `grupos.py`: define `idoso = age > 60`, conforme o exercício.
3. `divisao.py`: divide antes de remover e estratifica por idade e doença.
4. `modelo.py`: padroniza números e codifica categorias somente a partir do treino.
   `num`, `target`, `idoso` e o identificador não entram nas características.

Explique os modelos: logística aprende uma combinação dos atributos; árvore aprende
regras e decisões nas folhas; KNN usa os cinco vizinhos mais próximos após transformação.
Os dois últimos são comparações exploratórias, sem procura de parâmetros para produzir queda.

Discussão ética a apresentar:

- Um doente classificado como sem doença é falso negativo. Em um fluxo que use a
  previsão para encaminhar, isso poderia atrasar sua identificação.
- Um paciente sem doença classificado como doente é falso positivo. Isso pode gerar
  preocupação e avaliações adicionais; esse também é um prejuízo.
- Coleta desigual pode produzir desigualdade nos erros. Acurácia geral pode ocultar
  resultados diferentes por idade; por isso medimos os grupos separadamente.
- Retirar idade das entradas não garante equidade. A ablação confirmou que a perda
  de especificidade permanece; não identificamos proxies específicos.
- Melhorar uma métrica pode piorar outra. Pesos, coleta, limiar e supervisão humana
  exigiriam avaliação conforme o fluxo real; neste trabalho, limiares não são ajustados.
- A base é histórica, pequena e de uma instituição. Não garante representatividade
  de um hospital atual. As probabilidades não foram calibradas para risco clínico.

## Pilar 2 — Mostrar exatamente a criação do desequilíbrio

Abra `src/desequilibrio.py` e mostre:

```python
n_manter = len(idosos) - round(len(idosos) * fracao)
idosos_mantidos = idosos.sample(n=n_manter, random_state=semente)
treino_desbalanceado = pd.concat([outros, idosos_mantidos])
```

Execute `python -m src.desequilibrio`:

| Conjunto | Total | Idosos | Não idosos |
|---|---:|---:|---:|
| Treino original | 207 | 55 | 152 |
| Treino com remoção | 160 | 8 | 152 |
| Teste preservado | 90 | 24 | 66 |

“A remoção é aleatória apenas entre idosos do treino, sem selecionar pela doença.
Excluímos 47 de 55, aproximadamente 85%. O teste preserva as proporções da base original.”

## Pilar 3 — Explicar o mecanismo e mostrar positivos não identificados

Defina as quatro células da matriz:

| Alvo real / previsão | Sem doença | Com doença |
|---|---|---|
| Sem doença | TN: acerto | FP: falso positivo |
| Com doença | FN: falso negativo | TP: acerto |

`Recall = TP / (TP + FN)` e `especificidade = TN / (TN + FP)`.
Cada teste contém 13 idosos doentes e 11 idosos sem doença.

Primeiro mostre `resultados/sensibilidade_modelos.png`:

“Na logística, o recall médio não caiu: 0,851 em A e 0,866 em B.
O prejuízo apareceu na especificidade, de 0,771 para 0,678.
No KNN também não houve queda média de recall. Na árvore, o recall passou
de 0,752 para 0,726. Portanto, o tipo de erro depende do modelo.”

Depois mostre `resultados/matrizes_arvore_semente_42.png`:

“Mantivemos a semente 42 do projeto original. Na árvore, A identifica 11 dos 13
idosos doentes; B identifica 8. Os falsos negativos passam de 2 para 5.”

Abra `resultados/arvore_casos_semente_42.csv` e use a linha técnica 2:

“Este paciente tem 67 anos e doença registrada. Na árvore A, chega a uma folha
com proporção de doentes 0,727 e é identificado. Após a remoção, os cortes mudam:
ele chega a outra folha, com proporção 0,308, e passa a ser classificado como sem doença.
O paciente do teste é o mesmo; mudou o aprendizado a partir do treino.”

As outras duas novas falhas são as linhas 162 e 194. Os caminhos numéricos do CSV
estão nas unidades originais. As folhas A/B têm condições diferentes, não são a mesma
folha com menos pacientes. As proporções de folhas não são risco clínico calibrado.

Apresente a ressalva imediatamente: “Nas 50 divisões, o recall da árvore piorou em 18,
empatou em 18 e melhorou em 14. Esse exemplo explica o mecanismo, mas não demonstra
que a remoção sempre cause queda. As repetições reutilizam os mesmos pacientes.”

## Pilar 4 — Propor e avaliar mitigação

Mostre `src/mitigacao.py`:

- C: peso `55/8` para os 8 idosos restantes, somente no classificador.
- D: acrescenta metade dos idosos excluídos do treino; são 32 idosos no total.
  É simulação de coleta adicional, sem duplicações e sem trazer pacientes do teste.

Mostre `resultados/comparacao_cenarios_arvore.png`. Ele compara A/B/C/D na árvore
pela média das 50 repetições; as barras de erro mostram o desvio padrão entre
sorteios, não intervalo de confiança. Depois use as matrizes da semente 42 como
exemplo concreto:

“Na logística, C melhora um pouco a especificidade, mas reduz recall e não recupera
acurácia. D recupera parcialmente especificidade e acurácia. Na árvore, o recall médio
é 0,726 em B, 0,706 em C e 0,745 em D. A reponderação não recupera a diversidade
dos registros removidos; a reposição parcial teve resultados agregados melhores nessas métricas.”

“Na divisão 42 da árvore, C ajuda mais o recall que D, enquanto nas médias ocorre
o contrário. Por isso avaliamos várias divisões. Esses resultados apoiam investigar
coleta representativa e acompanhamento dos erros por grupo, sem garantir eficácia
clínica ou superioridade universal de uma estratégia.”

## Encerramento e pergunta que pode surgir

“Cumprimos a remoção solicitada, preservamos o teste e mostramos como decisões e
erros mudam. A árvore ilustra doentes idosos não identificados; a logística revela
outro prejuízo, por falsos positivos. Mantemos ambos os resultados e suas limitações.”

Se perguntarem por que o modelo original não perdeu recall: o enunciado orienta o
experimento, mas não garante o resultado de qualquer classificador. A análise encontrou
dependência do modelo e do sorteio; não encontrou erro de cálculo que justificasse
alterar artificialmente a conclusão da logística. A aceitação do prejuízo por falsos
positivos no terceiro pilar ainda precisa ser esclarecida com a professora.
