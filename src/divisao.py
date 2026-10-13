"""
PASSO 3 - Separar treino e teste (ANTES de remover qualquer idoso)

Por que a ordem importa:
  - O TESTE preserva as proporções da base original, sem remover idosos.
    Isso não garante representatividade da população de um hospital real.
  - A remoção de 85% dos idosos vale SÓ para o TREINO (passo 4).
  Se removêssemos antes de dividir, o teste também perderia idosos e a
  avaliação deixaria de mostrar o prejuízo que o seminário quer demonstrar.

Estratificação:
  Usamos stratify com a combinação (idoso, target), assim treino e teste mantêm
  a mesma proporção de idosos E a mesma proporção de doentes dentro de cada grupo.

Uso nos próximos passos:
    from src.divisao import dividir_treino_teste
    treino, teste = dividir_treino_teste(df)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.dados import carregar_dados
from src.grupos import criar_grupo_idoso

SEMENTE = 42        # fixa o sorteio: o resultado é o mesmo toda vez que rodar
TAMANHO_TESTE = 0.3  # 30% para teste, 70% para treino


def dividir_treino_teste(df: pd.DataFrame, semente: int = SEMENTE):
    """Retorna (treino, teste), ambos como DataFrames completos."""
    # Rótulo que junta os dois critérios: ex. "1_0" = idoso sem doença
    estrato = df["idoso"].astype(str) + "_" + df["target"].astype(str)

    treino, teste = train_test_split(
        df,
        test_size=TAMANHO_TESTE,
        random_state=semente,
        stratify=estrato,
    )
    return treino.reset_index(drop=True), teste.reset_index(drop=True)


def contagem(df: pd.DataFrame, nome: str) -> None:
    """Mostra quantos pacientes de cada grupo existem em um conjunto."""
    total = len(df)
    idosos = int(df["idoso"].sum())
    doentes_idosos = int(df.loc[df["idoso"] == 1, "target"].sum())
    doentes_outros = int(df.loc[df["idoso"] == 0, "target"].sum())
    print(f"{nome:<7} total={total:>3} | idosos={idosos:>2} "
          f"({idosos / total:.1%}) | doentes entre idosos={doentes_idosos:>2} "
          f"| doentes entre não idosos={doentes_outros:>2}")


if __name__ == "__main__":
    df = criar_grupo_idoso(carregar_dados())
    treino, teste = dividir_treino_teste(df)

    contagem(df, "Base")
    contagem(treino, "Treino")
    contagem(teste, "Teste")
