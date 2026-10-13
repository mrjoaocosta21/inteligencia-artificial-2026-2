"""Sensibilidade ao modelo e à remoção, sem seleção de sementes ou parâmetros.

Modelos e parâmetros fixos: logística, árvore curta e KNN com 5 vizinhos.
Relata todos os resultados. Não seleciona o modelo pela queda no teste.
Uma segunda análise fixa treino/teste em 42 e varia apenas a remoção.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import export_text

from src.dados import carregar_dados
from src.desequilibrio import remover_idosos
from src.divisao import dividir_treino_teste
from src.grupos import criar_grupo_idoso
from src.mitigacao import grafico, matrizes, rodar_cenarios
from src.modelo import (
    FEATURES, MODELOS, NUMERICAS, N_REPETICOES, SEMENTE_PRINCIPAL,
    metricas_por_grupo, treinar_e_avaliar,
)
from src.relatorios import diferencas_pareadas

PASTA_SAIDA = Path(__file__).resolve().parent.parent / "resultados"


def sensibilidade_modelos(df: pd.DataFrame, n_repeticoes: int = N_REPETICOES) -> tuple:
    linhas, predicoes = [], []
    for semente in range(n_repeticoes):
        for tipo in MODELOS:
            for cenario, res in rodar_cenarios(df, semente, tipo).items():
                for grupo in metricas_por_grupo(res):
                    linhas.append({"semente": semente, "modelo": tipo, "cenario": cenario,
                                   **res["composicao"], **grupo})
                if cenario in ("A", "B"):
                    predicoes.append(res["predicoes"].assign(semente=semente, modelo=tipo, cenario=cenario))
    return pd.DataFrame(linhas), pd.concat(predicoes, ignore_index=True)


def remocao_teste_fixo(df: pd.DataFrame, n_repeticoes: int = N_REPETICOES) -> pd.DataFrame:
    treino, teste = dividir_treino_teste(df, SEMENTE_PRINCIPAL)
    referencia = treinar_e_avaliar(treino, teste)
    linhas = []
    for semente in range(n_repeticoes):
        desb = remover_idosos(treino, semente=semente)
        for cenario, res in (("A", referencia), ("B", treinar_e_avaliar(desb, teste))):
            dados = treino if cenario == "A" else desb
            for grupo in metricas_por_grupo(res):
                p = res["predicoes"]
                p = p[p.idoso == (1 if grupo["grupo"] == "idoso" else 0)]
                linhas.append({"semente": semente, "semente_divisao": SEMENTE_PRINCIPAL,
                               "cenario": cenario, **grupo, "treino_idosos": int(dados.idoso.sum()),
                               "treino_idosos_doentes": int(dados.loc[dados.idoso == 1, "target"].sum()),
                               "prob_media_doentes": p.loc[p.target == 1, "prob_doenca"].mean(),
                               "prob_media_saudaveis": p.loc[p.target == 0, "prob_doenca"].mean()})
    return pd.DataFrame(linhas)


def erros_pareados(predicoes: pd.DataFrame) -> pd.DataFrame:
    chaves = ["semente", "modelo", "paciente_id", "age", "idoso", "target"]
    a = predicoes[predicoes.cenario == "A"].drop(columns="cenario")
    b = predicoes[predicoes.cenario == "B"].drop(columns="cenario")
    r = a.merge(b, on=chaves, suffixes=("_A", "_B"), validate="one_to_one")
    r["novo_falso_negativo"] = (r.target == 1) & (r.predicao_A == 1) & (r.predicao_B == 0)
    r["novo_falso_positivo"] = (r.target == 0) & (r.predicao_A == 0) & (r.predicao_B == 1)
    return r


def grafico_sensibilidade(r: pd.DataFrame, caminho: Path) -> None:
    import matplotlib.pyplot as plt  # backend Agg definido ao importar mitigacao

    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    modelos = [modelo for modelo in MODELOS if modelo in r.modelo.unique()]
    for ax, metrica, titulo in zip(
        eixos, ("recall", "especif"), ("Recall dos idosos", "Especificidade dos idosos")
    ):
        for i, (cenario, cor, rotulo) in enumerate((("A", "#4C9F70", "Treino completo"),
                                                    ("B", "#D1495B", "Sub-representação"))):
            grupos = [r[(r.modelo == modelo) & (r.cenario == cenario) & (r.grupo == "idoso")][metrica]
                      for modelo in modelos]
            ax.bar(np.arange(len(modelos)) + (i - 0.5) * 0.35, [g.mean() for g in grupos],
                   0.35, yerr=[g.std() for g in grupos], capsize=3, color=cor, label=rotulo)
        nomes = {"logistica": "Logística", "arvore": "Árvore", "knn": "KNN"}
        ax.set_xticks(range(len(modelos)), [nomes[modelo] for modelo in modelos])
        ax.set_title(titulo)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    eixos[0].set_ylabel("Média entre sorteios; barras = desvio padrão")
    eixos[0].legend(loc="lower left")
    fig.suptitle(f"Sensibilidade ao modelo — {r.semente.nunique()} repetições, análise exploratória")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def caminho_arvore(modelo, paciente: pd.DataFrame) -> tuple:
    """Explica um caminho; converte limiares numéricos de z-score às unidades originais."""
    pre, clf = modelo.named_steps["pre"], modelo.named_steps["clf"]
    x = pre.transform(paciente[FEATURES])
    valores = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    nomes = pre.get_feature_names_out()
    scaler = pre.named_transformers_["num"]
    nos = clf.decision_path(x).indices
    passos = []
    for no in nos:
        feature = clf.tree_.feature[no]
        if feature < 0:
            continue
        nome, limiar, valor = nomes[feature], clf.tree_.threshold[no], valores[feature]
        # Usa o ramo efetivamente percorrido, incluindo a conversão interna para float32.
        operador = "<=" if clf.tree_.children_left[no] in nos else ">"
        if nome.startswith("num__"):
            coluna = nome.removeprefix("num__")
            indice = NUMERICAS.index(coluna)
            limiar = limiar * scaler.scale_[indice] + scaler.mean_[indice]
            valor = paciente[coluna].iloc[0]
            nome = coluna
        passos.append(f"{nome} ({valor:.3f}) {operador} {limiar:.3f}")
    folha = int(clf.apply(x)[0])
    return " -> ".join(passos), int(clf.tree_.n_node_samples[folha])


def explicar_arvore(df: pd.DataFrame, pasta: Path) -> pd.DataFrame:
    """Registra todos os idosos da semente principal, inclusive acertos e erros antigos."""
    resultados = rodar_cenarios(df, SEMENTE_PRINCIPAL, tipo="arvore")
    _, teste = dividir_treino_teste(df, SEMENTE_PRINCIPAL)
    linhas = []
    for cenario, res in resultados.items():
        modelo = res["modelo"]
        texto = export_text(modelo.named_steps["clf"],
                            feature_names=list(modelo.named_steps["pre"].get_feature_names_out()),
                            show_weights=True, decimals=3)
        (pasta / f"arvore_{cenario}_semente_42.txt").write_text(
            "ATENÇÃO: limiares num__ neste arquivo são padronizados (z-score).\n"
            "No CSV de caminhos, os limiares numéricos estão nas unidades originais.\n" + texto,
            encoding="utf-8")
    for _, paciente in teste[teste.idoso == 1].sort_values("paciente_id").iterrows():
        linha = {"semente": SEMENTE_PRINCIPAL, "paciente_id": int(paciente.paciente_id),
                 "age": paciente.age, "target": int(paciente.target)}
        for cenario, res in resultados.items():
            pred = res["predicoes"].set_index("paciente_id").loc[paciente.paciente_id]
            caminho, n_folha = caminho_arvore(res["modelo"], paciente.to_frame().T)
            linha.update({f"predicao_{cenario}": int(pred.predicao),
                          f"prob_doenca_{cenario}": pred.prob_doenca,
                          f"caminho_{cenario}": caminho, f"casos_folha_{cenario}": n_folha})
        linha["novo_falso_negativo"] = (
            linha["target"] == 1 and linha["predicao_A"] == 1 and linha["predicao_B"] == 0
        )
        linhas.append(linha)
    r = pd.DataFrame(linhas)
    r.to_csv(pasta / "arvore_casos_semente_42.csv", index=False)
    return r


if __name__ == "__main__":
    df = criar_grupo_idoso(carregar_dados())
    PASTA_SAIDA.mkdir(exist_ok=True)
    r, predicoes = sensibilidade_modelos(df)
    r.to_csv(PASTA_SAIDA / "sensibilidade_modelos.csv", index=False)
    grafico(r[r.modelo == "arvore"], PASTA_SAIDA / "comparacao_cenarios_arvore.png",
            titulo_modelo=MODELOS["arvore"])
    grafico_sensibilidade(r, PASTA_SAIDA / "sensibilidade_modelos.png")
    matrizes(r[r.modelo == "arvore"], PASTA_SAIDA / "matrizes_arvore_semente_42.png",
              titulo_modelo=MODELOS["arvore"])
    explicar_arvore(df, PASTA_SAIDA)
    diferencas_pareadas(r, [("A", "B"), ("B", "C"), ("B", "D")], ("modelo", "grupo")).to_csv(
        PASTA_SAIDA / "sensibilidade_modelos_diferencas.csv", index=False)
    predicoes.to_csv(PASTA_SAIDA / "predicoes_modelos.csv", index=False)
    erros_pareados(predicoes).to_csv(PASTA_SAIDA / "erros_pareados.csv", index=False)
    fixo = remocao_teste_fixo(df)
    fixo.to_csv(PASTA_SAIDA / "remocao_teste_fixo.csv", index=False)
    diferencas_pareadas(fixo, [("A", "B")]).to_csv(
        PASTA_SAIDA / "remocao_teste_fixo_diferencas.csv", index=False)
    print(r.groupby(["modelo", "grupo", "cenario"])[["recall", "especif", "fn", "fp"]]
          .mean().round(3).to_string())
    print("\nTeste fixo em 42, variando somente a remoção (logística):")
    print(fixo.groupby(["grupo", "cenario"])[["recall", "especif", "fn", "fp"]]
          .mean().round(3).to_string())
    print("\nOs modelos são comparações exploratórias; não são seleção pelo resultado do teste.")
