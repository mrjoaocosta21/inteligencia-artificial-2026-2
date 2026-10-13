"""
PASSO 6 - Mitigação e comparação COM e SEM o viés  (PILAR 4)

Quatro cenários, avaliados no mesmo teste DENTRO de cada repetição:
  A) REFERÊNCIA : treino completo (sem a remoção artificial)
  B) COM viés   : treino sem 85% dos idosos (o desequilíbrio do enunciado)
  C) PESOS      : mesmo treino de B, mas com PESOS maiores para os idosos que sobraram
  D) REPOSIÇÃO  : B + metade dos idosos excluídos, simulando recuperar dados no treino

Mitigação usada - reponderação:
  Os 8 idosos que restaram passam a valer, juntos, o mesmo que os 55 originais.
  Peso de cada idoso = (idosos antes da remoção) / (idosos que sobraram) ~ 6,9
  Os não idosos continuam com peso 1.
  Não inventa pacientes novos: só muda o quanto cada um "conta" no treino.

Saídas (pasta resultados/):
  comparacao_cenarios.csv  -> taxas, matrizes e composição dos treinos
  predicoes_cenarios.csv   -> previsões por paciente, semente e cenário
  comparacao_cenarios.png  -> gráfico para a apresentação
  matrizes_semente_42.png  -> erros da demonstração principal
  resumo_cenarios.csv, diferencas_pareadas.csv, metadados.json
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
from src.modelo import N_REPETICOES, SEMENTE_PRINCIPAL, metricas_por_grupo, treinar_e_avaliar
from src.relatorios import diferencas_pareadas, salvar_metadados

PASTA_SAIDA = Path(__file__).resolve().parent.parent / "resultados"
CENARIOS = {"A": "A) treino completo", "B": "B) sub-representação",
            "C": "C) reponderação", "D": "D) reposição parcial"}
METRICAS = {"recall": "Recall", "especif": "Especificidade", "acc": "Acurácia"}


def pesos_para_idosos(treino_original: pd.DataFrame, treino_desb: pd.DataFrame) -> np.ndarray:
    """Peso 1 para não idosos; para idosos, devolve o 'valor' que tinham antes da remoção."""
    n_original, n_restante = treino_original["idoso"].sum(), treino_desb["idoso"].sum()
    if n_original and not n_restante:
        raise ValueError("Reponderação não recupera um grupo sem nenhum registro no treino.")
    peso_idoso = n_original / n_restante if n_restante else 1.0
    return np.where(treino_desb["idoso"] == 1, peso_idoso, 1.0)


def repor_idosos(treino_original: pd.DataFrame, treino_desb: pd.DataFrame,
                 semente: int, fracao: float = 0.5) -> pd.DataFrame:
    """Simula coleta adicional, sorteando metade dos excluídos do TREINO.

    A fração é fixada antes da avaliação. Nenhum paciente do teste é usado.
    Trata-se de simulação: não é coleta real nem garantia de eficácia clínica.
    """
    if not 0 <= fracao <= 1:
        raise ValueError("A fração reposta deve estar entre 0 e 1.")
    excluidos = treino_original[(treino_original["idoso"] == 1)
                               & ~treino_original["paciente_id"].isin(treino_desb["paciente_id"])]
    adicionais = excluidos.sample(n=round(len(excluidos) * fracao), random_state=semente)
    return pd.concat([treino_desb, adicionais]).sample(frac=1, random_state=semente).reset_index(drop=True)


def rodar_cenarios(df: pd.DataFrame, semente: int, tipo: str = "logistica") -> dict:
    treino, teste = dividir_treino_teste(df, semente=semente)
    treino_desb = remover_idosos(treino, semente=semente)
    treinos = {"A": treino, "B": treino_desb, "C": treino_desb,
               "D": repor_idosos(treino, treino_desb, semente)}
    resultados = {}
    for cenario, dados in treinos.items():
        if tipo == "knn" and cenario == "C":
            continue  # KNN não implementa sample_weight; não simular pesos por duplicação.
        pesos = pesos_para_idosos(treino, treino_desb) if cenario == "C" else None
        res = treinar_e_avaliar(dados, teste, sample_weight=pesos, tipo=tipo)
        res["composicao"] = {"treino_n": len(dados), "treino_idosos": int(dados.idoso.sum()),
                             "treino_idosos_doentes": int(dados.loc[dados.idoso == 1, "target"].sum()),
                             "treino_nao_idosos": int((dados.idoso == 0).sum()),
                             "treino_nao_idosos_doentes": int(dados.loc[dados.idoso == 0, "target"].sum())}
        resultados[cenario] = res
    return resultados


def coletar_resultados(df: pd.DataFrame, n_repeticoes: int = N_REPETICOES,
                       tipo: str = "logistica") -> tuple:
    linhas, predicoes = [], []
    for s in range(n_repeticoes):
        for cen, res in rodar_cenarios(df, s, tipo=tipo).items():
            for grupo in metricas_por_grupo(res):
                linhas.append({"semente": s, "cenario": cen, **res["composicao"], **grupo})
            predicoes.append(res["predicoes"].assign(semente=s, cenario=cen))
    return pd.DataFrame(linhas), pd.concat(predicoes, ignore_index=True)


def coletar(df: pd.DataFrame, n_repeticoes: int = N_REPETICOES,
            tipo: str = "logistica") -> pd.DataFrame:
    return coletar_resultados(df, n_repeticoes, tipo=tipo)[0]


def grafico(r: pd.DataFrame, caminho: Path, titulo_modelo: str = "Regressão logística") -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    cores = {"A": "#4C9F70", "B": "#D1495B", "C": "#3A6EA5", "D": "#9467BD"}
    largura = 0.20
    for ax, (grupo, titulo) in zip(eixos, [("idoso", "Pacientes IDOSOS (> 60)"),
                                           ("nao_idoso", "Pacientes NÃO idosos (<= 60)")]):
        for i, cen in enumerate(CENARIOS):
            sub = r[(r.grupo == grupo) & (r.cenario == cen)]
            medias = [sub[m].mean() for m in METRICAS]
            erros = [sub[m].std() for m in METRICAS]
            ax.bar(np.arange(3) + (i - 1.5) * largura, medias, largura, yerr=erros,
                   capsize=3, color=cores[cen], label=CENARIOS[cen])
        ax.set_xticks(range(3))
        ax.set_xticklabels(METRICAS.values())
        ax.set_title(titulo)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    eixos[0].set_ylabel("Desempenho médio")
    fig.suptitle(f"{titulo_modelo} — quatro cenários, {r.semente.nunique()} repetições",
                 fontsize=14, fontweight="bold")
    handles, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(handles, rotulos, loc="lower center", bbox_to_anchor=(0.5, 0.05),
               ncol=2, frameon=False)
    fig.text(0.5, 0.02, "Barras de erro = desvio padrão entre sorteios; não são intervalos de confiança.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.18, 1, 0.94))
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def matrizes(r: pd.DataFrame, caminho: Path, semente: int = SEMENTE_PRINCIPAL,
             titulo_modelo: str = "Regressão logística") -> None:
    """Mostra os erros com denominadores, inclusive quando o efeito não aparece."""
    fig, eixos = plt.subplots(2, len(CENARIOS), figsize=(14, 6))
    for linha, grupo in enumerate(("idoso", "nao_idoso")):
        for coluna, (cenario, titulo) in enumerate(CENARIOS.items()):
            sub = r[(r.semente == semente) & (r.grupo == grupo) & (r.cenario == cenario)].iloc[0]
            matriz = np.array([[sub.tn, sub.fp], [sub.fn, sub.tp]], dtype=int)
            ax = eixos[linha, coluna]
            ax.imshow(matriz, cmap="Blues", vmin=0, vmax=r[r.semente == semente].n.max())
            for (i, j), valor in np.ndenumerate(matriz):
                ax.text(j, i, str(valor), ha="center", va="center", fontsize=15)
            ax.set_xticks([0, 1], ["Sem\ndoença", "Com\ndoença"])
            ax.set_yticks([0, 1], ["Sem doença", "Com doença"])
            ax.set_xlabel("Previsão")
            ax.set_ylabel("Alvo real")
            ax.set_title(f"{titulo}\n{'Idosos' if linha == 0 else 'Não idosos'}")
    fig.suptitle(f"{titulo_modelo} — matrizes de confusão, semente {semente}")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = criar_grupo_idoso(carregar_dados())
    r, predicoes = coletar_resultados(df)

    PASTA_SAIDA.mkdir(exist_ok=True)
    r.to_csv(PASTA_SAIDA / "comparacao_cenarios.csv", index=False)
    predicoes.to_csv(PASTA_SAIDA / "predicoes_cenarios.csv", index=False)
    resumo = r.groupby(["grupo", "cenario"])[list(METRICAS) + ["fn", "fp"]].agg(["mean", "std"])
    resumo.columns = [f"{metrica}_{estatistica}" for metrica, estatistica in resumo.columns]
    resumo.reset_index().to_csv(PASTA_SAIDA / "resumo_cenarios.csv", index=False)
    diferencas_pareadas(r, [("A", "B"), ("B", "C"), ("B", "D")]).to_csv(
        PASTA_SAIDA / "diferencas_pareadas.csv", index=False)
    grafico(r, PASTA_SAIDA / "comparacao_cenarios.png")
    matrizes(r, PASTA_SAIDA / "matrizes_semente_42.png")
    salvar_metadados(PASTA_SAIDA / "metadados.json", df)

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
