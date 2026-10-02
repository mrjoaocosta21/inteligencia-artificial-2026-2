# Viés por sub-representação em triagem de doença cardíaca

Seminário de Inteligência Artificial 2026.2 - Cenário 3: Diagnóstico e Saúde Pública.

## Objetivo

Mostrar como a sub-representação de pacientes idosos (> 60 anos) no treino afeta
as previsões de um modelo de triagem de doença cardíaca para esse grupo.
Para isso, 85% dos idosos são removidos do treino (nunca do teste) e o desempenho
é comparado em três cenários.

## Base de dados

Heart Disease, arquivo Cleveland (UCI Machine Learning Repository).
https://archive.ics.uci.edu/dataset/45/heart+disease

- 303 pacientes e 14 variáveis; 6 pacientes com valores ausentes foram removidos (297 restam).
- Alvo: `num` (0 a 4) transformado em binário: 0 = sem doença, 1 = com doença.
- O arquivo é baixado automaticamente para `data/` na primeira execução.

## Como rodar

    pip install -r requirements.txt

Cada passo pode ser executado separadamente, a partir da pasta do projeto:

    python -m src.dados               # 1. carrega e prepara os dados
    python -m src.grupos              # 2. cria o grupo "idoso" (age > 60)
    python -m src.divisao             # 3. separa treino e teste (estratificado)
    python -m src.desequilibrio       # 4. remove 85% dos idosos do treino
    python -m src.modelo              # 5. Regressão Logística, métricas por grupo
    python -m src.diagnostico_idade   #    extra: o efeito depende da variável age?
    python -m src.mitigacao           # 6. mitigação e comparação (gera resultados/)

Use sempre `python -m src.<nome>` (e não `python src/<nome>.py`).

## Estrutura

    data/         base de dados
    src/          código, um arquivo por etapa
    resultados/   gráfico e tabela gerados pelo passo 6
    requirements.txt
    README.md

## Cenários comparados

- **A) sem viés:** treino completo.
- **B) com viés:** treino sem 85% dos idosos (8 idosos restam).
- **C) mitigado:** treino de B, com peso maior para os idosos restantes.

Todos avaliados no mesmo teste (24 idosos, 66 não idosos), com 50 repetições
(sementes 0 a 49). Valores abaixo: média nos idosos.

| Métrica | A) sem viés | B) com viés | C) mitigado |
|---|---|---|---|
| Recall | 0,851 | 0,866 | 0,842 |
| Especificidade | 0,771 | 0,678 | 0,705 |
| Acurácia | 0,814 | 0,780 | 0,779 |

## Principais conclusões

- O viés aparece como **mais falsos positivos** nos idosos (saudáveis marcados como doentes),
  e não como doentes não detectados: o recall dos idosos não caiu.
- Retirar a variável `age` não eliminou o efeito; outras variáveis podem atuar como
  substitutas da idade (hipótese, não testada diretamente).
- A reponderação melhorou a especificidade dos idosos só um pouco e de forma inconsistente.
  A correção mais eficaz é coletar mais dados de idosos.

## Limitações

- Base pequena: o teste tem só 24 idosos, e o treino com viés tem 8. As métricas oscilam
  bastante entre sorteios (por isso as 50 repetições).
- Uma única instituição (Cleveland); os resultados podem não valer para outras populações.
- A remoção é aleatória, o que é uma simplificação de como a sub-representação ocorre na prática.