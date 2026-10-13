"""
PASSO 1 - Obter e preparar os dados (UCI Heart Disease, base Cleveland)

Este módulo será reaproveitado pelos próximos passos:
    from src.dados import carregar_dados
    df = carregar_dados()

O que ele faz:
  1. Baixa a base pública da UCI (se ainda não estiver na pasta data/)
  2. Dá nome às 14 colunas
  3. Trata os valores ausentes ('?')
  4. Cria o alvo binário 'target' (0 = sem doença, 1 = com doença)

Ainda NÃO cria o grupo "idoso" nem remove ninguém (passos 2 e 4).
"""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

# Pasta raiz do projeto (sobe duas pastas a partir de src/dados.py)
RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO = RAIZ / "data" / "processed_cleveland.data"

# Fonte pública original (repositório UCI)
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")

# O arquivo NÃO traz cabeçalho: nomes na ordem da tabela de variáveis.
COLUNAS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def baixar_dados() -> None:
    """Baixa o arquivo da UCI para data/ caso ele ainda não exista."""
    if ARQUIVO.exists():
        return
    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando base de {URL} ...")
    urlretrieve(URL, ARQUIVO)
    print(f"Salvo em {ARQUIVO}")


def carregar_dados(remover_ausentes: bool = True) -> pd.DataFrame:
    """Retorna o DataFrame pronto para os próximos passos."""
    baixar_dados()

    # na_values="?" faz o pandas tratar '?' como ausente (NaN);
    # sem isso, 'ca' e 'thal' virariam texto.
    df = pd.read_csv(ARQUIVO, header=None, names=COLUNAS, na_values="?")
    # Identificador técnico: número da linha no arquivo, NÃO identidade do paciente.
    # Preservado após filtros para auditar treino/teste e rastrear os erros.
    df["paciente_id"] = df.index + 1

    if remover_ausentes:
        # Só 6 de 303 linhas têm ausentes (4 em 'ca', 2 em 'thal').
        df = df.dropna().reset_index(drop=True)

    # num: 0 = sem doença; 1-4 = com doença (graus de gravidade).
    # O enunciado só pede presença x ausência.
    df["target"] = (df["num"] > 0).astype(int)
    return df


if __name__ == "__main__":
    bruto = pd.read_csv(ARQUIVO if ARQUIVO.exists() else (baixar_dados() or ARQUIVO),
                        header=None, names=COLUNAS, na_values="?")
    print(f"Linhas e colunas brutas: {bruto.shape}")
    aus = bruto.isna().sum()
    print("\nAusentes por coluna:\n", aus[aus > 0])

    df = carregar_dados()
    print(f"\nApós remover ausentes: {len(df)} pacientes")
    print("\nAlvo binário (0 = sem doença, 1 = com doença):")
    print(df["target"].value_counts().sort_index())
    print("\nResumo da idade:")
    print(df["age"].describe())
