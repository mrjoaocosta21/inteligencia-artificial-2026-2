"""
PASSO 2 - Criar o grupo "idoso" e conhecer o tamanho de cada grupo

Regra (segue o enunciado): idoso = age > 60.

Ainda NÃO removemos ninguém. Aqui só medimos a situação ANTES do desequilíbrio,
para termos um ponto de comparação na apresentação.

Uso nos próximos passos:
    from src.grupos import criar_grupo_idoso
    df = criar_grupo_idoso(carregar_dados())
"""

import pandas as pd

from src.dados import carregar_dados

LIMITE_IDADE = 60  # idoso = age > 60


def criar_grupo_idoso(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a coluna 'idoso' (1 = idoso, 0 = não idoso)."""
    df = df.copy()  # não altera o DataFrame original
    df["idoso"] = (df["age"] > LIMITE_IDADE).astype(int)
    return df


def resumo_por_grupo(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela com tamanho de cada grupo e proporção de doentes em cada um."""
    resumo = df.groupby("idoso").agg(
        pacientes=("target", "size"),
        com_doenca=("target", "sum"),
        prevalencia=("target", "mean"),  # % de doentes dentro do grupo
        idade_media=("age", "mean"),
    )
    resumo["% da base"] = resumo["pacientes"] / len(df)
    resumo.index = resumo.index.map({0: "não idoso (<= 60)", 1: "idoso (> 60)"})
    return resumo.round(3)


if __name__ == "__main__":
    df = criar_grupo_idoso(carregar_dados())

    print(f"Regra: idoso = age > {LIMITE_IDADE}\n")
    print(resumo_por_grupo(df).to_string())

    # Quantos idosos sobrariam se removêssemos 85% (preview do passo 4)
    n_idosos = int(df["idoso"].sum())
    print(f"\nTotal de idosos na base inteira: {n_idosos}")
    print(f"85% de {n_idosos} = {round(n_idosos * 0.85)} (será aplicado só no TREINO, no passo 4)")