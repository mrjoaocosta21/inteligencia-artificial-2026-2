"""Ablação de age: compara A/B com e sem idade nas mesmas 50 divisões.

O grupo de avaliação continua sendo definido pela idade original. Retirar age
das entradas não retira idosos da avaliação nem testa diretamente variáveis proxy.
"""

from pathlib import Path

import pandas as pd

from src.dados import carregar_dados
from src.desequilibrio import remover_idosos
from src.divisao import dividir_treino_teste
from src.grupos import criar_grupo_idoso
from src.modelo import N_REPETICOES, metricas_por_grupo, treinar_e_avaliar
from src.relatorios import diferencas_pareadas

PASTA_SAIDA = Path(__file__).resolve().parent.parent / "resultados"


def investigar(df: pd.DataFrame, n_repeticoes: int = N_REPETICOES) -> pd.DataFrame:
    linhas = []
    for semente in range(n_repeticoes):
        treino, teste = dividir_treino_teste(df, semente)
        desb = remover_idosos(treino, semente=semente)
        for incluir_idade in (True, False):
            for cenario, dados in (("A", treino), ("B", desb)):
                res = treinar_e_avaliar(dados, teste, incluir_idade=incluir_idade)
                for grupo in metricas_por_grupo(res):
                    p = res["predicoes"]
                    p = p[p.idoso == (1 if grupo["grupo"] == "idoso" else 0)]
                    linhas.append({"semente": semente, "cenario": cenario,
                                   "configuracao": "com_idade" if incluir_idade else "sem_idade",
                                   **grupo,
                                   "prob_media_doentes": p.loc[p.target == 1, "prob_doenca"].mean(),
                                   "prob_media_saudaveis": p.loc[p.target == 0, "prob_doenca"].mean()})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    r = investigar(criar_grupo_idoso(carregar_dados()))
    PASTA_SAIDA.mkdir(exist_ok=True)
    r.to_csv(PASTA_SAIDA / "diagnostico_idade.csv", index=False)
    d = diferencas_pareadas(r, [("A", "B")], estratos=("configuracao", "grupo"))
    d.to_csv(PASTA_SAIDA / "diagnostico_idade_diferencas.csv", index=False)
    print(r.groupby(["configuracao", "grupo", "cenario"])[["recall", "especif", "fn", "fp"]]
          .mean().round(3).to_string())
    print("\nA ablação avalia age; não demonstra que outra variável seja proxy de idade.")
