"""Demonstração única dos quatro pilares exigidos no seminário.

Execute, a partir da pasta do projeto:
    python -m src.apresentacao

O comando usa a árvore de decisão no pilar 3 porque nela a sub-representação
gera mais falsos negativos entre idosos. A regressão logística continua no
projeto como controle: nela, o prejuízo principal aparece nos falsos positivos.
"""

from pathlib import Path

import pandas as pd

from src.dados import carregar_dados
from src.desequilibrio import FRACAO_REMOVIDA, remover_idosos
from src.divisao import dividir_treino_teste
from src.grupos import criar_grupo_idoso
from src.mitigacao import CENARIOS, coletar, grafico, matrizes, rodar_cenarios
from src.modelo import MODELOS, N_REPETICOES, SEMENTE_PRINCIPAL

PASTA_SAIDA = Path(__file__).resolve().parent.parent / "resultados"


def tabela_grupo(resultados: dict, grupo: str = "idoso") -> pd.DataFrame:
    """Resume as matrizes da semente principal para leitura durante a apresentação."""
    linhas = []
    for cenario, res in resultados.items():
        linhas.append({
            "cenário": cenario,
            "descrição": CENARIOS[cenario].split(") ", 1)[1],
            "TN": res[f"tn_{grupo}"],
            "FP": res[f"fp_{grupo}"],
            "FN": res[f"fn_{grupo}"],
            "TP": res[f"tp_{grupo}"],
            "recall": res[f"recall_{grupo}"],
            "especificidade": res[f"especif_{grupo}"],
        })
    return pd.DataFrame(linhas).set_index("cenário")


def resumo_agregado(r: pd.DataFrame, grupo: str = "idoso") -> pd.DataFrame:
    """Média e variação dos cenários da árvore nas repetições completas."""
    sub = r[r.grupo == grupo]
    resumo = sub.groupby("cenario")[["recall", "especif", "acc", "fn", "fp"]].agg(
        ["mean", "std"]
    )
    resumo.columns = [f"{metrica}_{estatistica}" for metrica, estatistica in resumo.columns]
    return resumo.reindex(CENARIOS)


def executar(n_repeticoes: int = N_REPETICOES) -> None:
    """Executa e imprime a demonstração, além de gerar os dois gráficos principais."""
    df = criar_grupo_idoso(carregar_dados())
    treino, teste = dividir_treino_teste(df, SEMENTE_PRINCIPAL)
    treino_desb = remover_idosos(treino, semente=SEMENTE_PRINCIPAL)

    print("\nPILAR 1 — BASE E MODELO")
    print(f"Base limpa: {len(df)} pacientes; idosos (> 60): {int(df.idoso.sum())}.")
    print(f"Modelo central da demonstração: {MODELOS['arvore']}.")
    print("O teste permanece separado; o pré-processamento é ajustado somente no treino.")

    print("\nPILAR 2 — DESEQUILÍBRIO ARTIFICIAL")
    print("Código: src/desequilibrio.py, função remover_idosos().")
    print(f"Idosos no treino: {int(treino.idoso.sum())} -> {int(treino_desb.idoso.sum())} ")
    print(f"Remoção efetiva: {1 - treino_desb.idoso.sum() / treino.idoso.sum():.1%} ")
    print(f"Não idosos preservados: {int((treino.idoso == 0).sum())}; teste intacto: {len(teste)} pacientes.")

    resultados = rodar_cenarios(df, SEMENTE_PRINCIPAL, tipo="arvore")
    principal = tabela_grupo(resultados)
    print("\nPILAR 3 — FALHAS NOS CASOS POSITIVOS (IDOSOS, SEMENTE 42)")
    print(principal.to_string(formatters={
        "recall": lambda valor: f"{valor:.1%}",
        "especificidade": lambda valor: f"{valor:.1%}",
    }))
    print(f"Em A há {principal.loc['A', 'FN']} falsos negativos; em B há {principal.loc['B', 'FN']}.")
    print("O exemplo explica o mecanismo; as 50 repetições abaixo mostram sua instabilidade.")

    agregado = coletar(df, n_repeticoes=n_repeticoes, tipo="arvore")
    resumo = resumo_agregado(agregado)
    print(f"\nRESULTADO AGREGADO — ÁRVORE, {n_repeticoes} REPETIÇÕES")
    print(resumo.to_string(formatters={
        coluna: (lambda valor: f"{valor:.3f}") for coluna in resumo.columns
    }))

    print("\nPILAR 4 — MITIGAÇÃO")
    print("C repondera os 8 idosos restantes; D repõe metade dos idosos excluídos do treino.")
    print(f"Recall médio dos idosos: B={resumo.loc['B', 'recall_mean']:.3f}, "
          f"C={resumo.loc['C', 'recall_mean']:.3f}, D={resumo.loc['D', 'recall_mean']:.3f}.")
    print("D recupera parte do recall médio; C ajuda na semente 42, mas não na média.")

    PASTA_SAIDA.mkdir(exist_ok=True)
    agregado.to_csv(PASTA_SAIDA / "comparacao_cenarios_arvore.csv", index=False)
    resumo.reset_index().to_csv(PASTA_SAIDA / "resumo_cenarios_arvore.csv", index=False)
    grafico(agregado, PASTA_SAIDA / "comparacao_cenarios_arvore.png",
            titulo_modelo=MODELOS["arvore"])
    matrizes(agregado, PASTA_SAIDA / "matrizes_arvore_semente_42.png",
             titulo_modelo=MODELOS["arvore"])
    print(f"\nGráficos gerados em: {PASTA_SAIDA}")
    print("Limite: base pequena e histórica; o resultado não prova queda universal de recall.")


if __name__ == "__main__":
    executar()
