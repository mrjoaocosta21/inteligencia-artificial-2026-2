"""
PASSO 4 - Criar o desequilíbrio populacional artificial  (PILAR 2 DO SEMINÁRIO)

Enunciado: remover 85% dos pacientes idosos (> 60 anos) do TREINO.

Pontos que precisam ficar claros na apresentação:
  - A remoção acontece SÓ no treino. O teste não é tocado.
  - A escolha dos pacientes removidos é aleatória (com semente fixa).
  - Isso simula um hospital em que idosos quase não aparecem na coleta preventiva.

Uso nos próximos passos:
    from src.desequilibrio import remover_idosos
    treino_desbalanceado = remover_idosos(treino)
"""

import pandas as pd

from src.dados import carregar_dados
from src.divisao import contagem, dividir_treino_teste
from src.grupos import criar_grupo_idoso

FRACAO_REMOVIDA = 0.85  # 85% dos idosos saem do treino
SEMENTE = 42


def remover_idosos(treino: pd.DataFrame, fracao: float = FRACAO_REMOVIDA,
                   semente: int = SEMENTE) -> pd.DataFrame:
    """Retorna uma cópia do treino com `fracao` dos idosos removida."""

    if not 0 <= fracao <= 1:
        raise ValueError("A fração removida deve estar entre 0 e 1.")

    idosos = treino[treino["idoso"] == 1]
    outros = treino[treino["idoso"] == 0]

    # ==============================================================
    # >>> AQUI É CRIADO O DESEQUILÍBRIO POPULACIONAL ARTIFICIAL <<<
    # Sorteamos quais idosos PERMANECEM (15%) e descartamos o resto.
    # Os não idosos ficam intactos.
    # ==============================================================
    n_manter = len(idosos) - round(len(idosos) * fracao)
    idosos_mantidos = idosos.sample(n=n_manter, random_state=semente)

    treino_desbalanceado = pd.concat([outros, idosos_mantidos])
    # embaralha para não deixar os idosos todos no fim da tabela
    return treino_desbalanceado.sample(frac=1, random_state=semente).reset_index(drop=True)


if __name__ == "__main__":
    df = criar_grupo_idoso(carregar_dados())
    treino, teste = dividir_treino_teste(df)
    treino_desb = remover_idosos(treino)

    contagem(treino, "Treino ORIGINAL")
    contagem(treino_desb, "Treino SEM 85% DOS IDOSOS")
    contagem(teste, "Teste (intacto)")

    ant = int(treino["idoso"].sum())
    dep = int(treino_desb["idoso"].sum())
    print(f"\nIdosos no treino: {ant} -> {dep}  (removidos: {ant - dep})")
    print(f"Idosos viram {dep / len(treino_desb):.1%} do treino "
          f"(antes: {ant / len(treino):.1%}), mas continuam {teste['idoso'].mean():.1%} do teste.")
