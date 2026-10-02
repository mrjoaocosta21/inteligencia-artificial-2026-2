"""
PASSO 6 - Mitigação e comparação COM e SEM o viés  (PILAR 4)

Três cenários, todos avaliados no MESMO teste (intacto):
  A) SEM viés   : treino completo (referência)
  B) COM viés   : treino sem 85% dos idosos (o desequilíbrio do enunciado)
  C) MITIGADO   : mesmo treino de B, mas com PESOS maiores para os idosos que sobraram

Mitigação usada - reponderação:
  Os 8 idosos que restaram passam a valer, juntos, o mesmo que os 55 originais.
  Peso de cada idoso = (idosos antes da remoção) / (idosos que sobraram) ~ 6,9
  Os não idosos continuam com peso 1.
  Não inventa pacientes novos: só muda o quanto cada um "conta" no treino.

Saídas (pasta resultados/):
  comparacao_cenarios.csv  -> números de cada repetição
  comparacao_cenarios.png  -> gráfico para a apresentação
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # gera o arquivo sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dados import carregar_dados
from src.desequilibrio import remover_idosos
from src.divisao import dividir_treino_teste
from src.grupos import criar_grupo_idoso
from src.modelo import N_REPETICOES, treinar_e_avaliar

PASTA_SAIDA = Path(__file__).resolve().parent.parent / "resultados"
CENARIOS = {"A": "A) sem viés", "B": "B) com viés", "C": "C) mitigado"}
METRICAS = {"recall": "Recall", "especif": "Especificidade", "acc": "Acurácia"}


def pesos_para_idosos(treino_original: pd.DataFrame, treino_desb: pd.DataFrame) -> np.ndarray:
    """Peso 1 para não idosos; para idosos, devolve o 'valor' que tinham antes da remoção."""
    peso_idoso = treino_original["idoso"].sum() / treino_desb["idoso"].sum()
    return np.where(treino_desb["idoso"] == 1, peso_idoso, 1.0)


def rodar_tres_cenarios(df: pd.DataFrame, semente: int) -> dict:
    treino, teste = dividir_treino_teste(df, semente=semente)
    treino_desb = remover_idosos(treino, semente=semente)
    return {
        "A": treinar_e_avaliar(treino, teste),
        "B": treinar_e_avaliar(treino_desb, teste),
        "C": treinar_e_avaliar(treino_desb, teste,
                               sample_weight=pesos_para_idosos(treino, treino_desb)),
    }


def coletar(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for s in range(N_REPETICOES):
        for cen, res in rodar_tres_cenarios(df, s).items():
            for grupo in ("idoso", "nao_idoso"):
                linhas.append({"semente": s, "cenario": cen, "grupo": grupo,
                               **{m: res[f"{m}_{grupo}"] for m in METRICAS}})
    return pd.DataFrame(linhas)


def grafico(r: pd.DataFrame, caminho: Path) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    cores = {"A": "#4C9F70", "B": "#D1495B", "C": "#3A6EA5"}
    largura = 0.26
    for ax, (grupo, titulo) in zip(eixos, [("idoso", "Pacientes IDOSOS (> 60)"),
                                           ("nao_idoso", "Pacientes NÃO idosos (<= 60)")]):
        for i, cen in enumerate(CENARIOS):
            sub = r[(r.grupo == grupo) & (r.cenario == cen)]
            medias = [sub[m].mean() for m in METRICAS]
            erros = [sub[m].std() for m in METRICAS]
            ax.bar(np.arange(3) + (i - 1) * largura, medias, largura, yerr=erros,
                   capsize=3, color=cores[cen], label=CENARIOS[cen])
        ax.set_xticks(range(3))
        ax.set_xticklabels(METRICAS.values())
        ax.set_title(titulo)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    eixos[0].set_ylabel(f"Média de {N_REPETICOES} repetições (barra = desvio padrão)")
    handles, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(handles, rotulos, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(caminho, dpi=150)


if __name__ == "__main__":
    df = criar_grupo_idoso(carregar_dados())
    r = coletar(df)

    PASTA_SAIDA.mkdir(exist_ok=True)
    r.to_csv(PASTA_SAIDA / "comparacao_cenarios.csv", index=False)
    grafico(r, PASTA_SAIDA / "comparacao_cenarios.png")

    for grupo, nome in [("idoso", "IDOSOS"), ("nao_idoso", "NÃO IDOSOS")]:
        print(f"\n=== {nome} (média ± desvio, {N_REPETICOES} repetições) ===")
        tab = {}
        for cen, rotulo in CENARIOS.items():
            sub = r[(r.grupo == grupo) & (r.cenario == cen)]
            tab[rotulo] = {nm: f"{sub[m].mean():.3f} ± {sub[m].std():.3f}"
                           for m, nm in METRICAS.items()}
        print(pd.DataFrame(tab).to_string())

    # Em quantas repetições a mitigação melhorou a especificidade dos idosos?
    esp = r[r.grupo == "idoso"].pivot(index="semente", columns="cenario", values="especif")
    print(f"\nEspecificidade dos idosos: C > B em {(esp.C > esp.B).sum()} de {N_REPETICOES} "
          f"repetições; C < B em {(esp.C < esp.B).sum()}; igual em {(esp.C == esp.B).sum()}.")
    rec = r[r.grupo == "idoso"].pivot(index="semente", columns="cenario", values="recall")
    print(f"Recall dos idosos:         C >= B em {(rec.C >= rec.B).sum()} de {N_REPETICOES} repetições.")
    print(f"\nArquivos salvos em {PASTA_SAIDA}")