"""
PASSO 5 - Treinar a Regressão Logística e medir o desempenho POR GRUPO  (PILAR 3)

Compara dois cenários, sempre avaliados no MESMO teste (intacto):
  A) Treino completo            -> referência (sem viés artificial)
  B) Treino sem 85% dos idosos  -> o desequilíbrio do enunciado

Métrica central: RECALL = dos pacientes que TÊM doença, quantos o modelo acertou.
Em triagem, recall baixo significa doentes mandados para casa (falso negativo).

Como há só 8 idosos no treino B, um único sorteio é frágil. Por isso o script
também repete tudo com várias sementes e mostra média e variação.

Uso nos próximos passos:
    from src.modelo import treinar_e_avaliar
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dados import carregar_dados
from src.desequilibrio import remover_idosos
from src.divisao import dividir_treino_teste
from src.grupos import criar_grupo_idoso

NUMERICAS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAS = ["cp", "restecg", "slope", "thal"]   # mais de 2 valores -> one-hot
BINARIAS = ["sex", "fbs", "exang"]                 # já são 0/1
FEATURES = NUMERICAS + CATEGORICAS + BINARIAS

SEMENTE_PRINCIPAL = 42
N_REPETICOES = 50


def montar_modelo() -> Pipeline:
    """Pré-processamento + Regressão Logística num só objeto."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("bin", "passthrough", BINARIAS),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def treinar_e_avaliar(treino: pd.DataFrame, teste: pd.DataFrame,
                      sample_weight=None) -> dict:
    """Treina no `treino` e devolve métricas por grupo no `teste`."""
    modelo = montar_modelo()
    modelo.fit(treino[FEATURES], treino["target"], clf__sample_weight=sample_weight)
    pred = modelo.predict(teste[FEATURES])

    res = {"recall_geral": recall_score(teste["target"], pred)}
    for nome, valor in [("idoso", 1), ("nao_idoso", 0)]:
        m = teste["idoso"] == valor
        y, p = teste.loc[m, "target"], pred[m.to_numpy()]
        res[f"recall_{nome}"] = recall_score(y, p)
        res[f"fn_{nome}"] = int(((y == 1) & (p == 0)).sum())   # doentes não detectados
        res[f"pos_{nome}"] = int((y == 1).sum())               # doentes existentes
        tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum())                  # saudáveis marcados como doentes
        res[f"fp_{nome}"] = fp
        res[f"especif_{nome}"] = tn / (tn + fp)                # saudáveis corretamente liberados
        res[f"acc_{nome}"] = float((y == p).mean())
    res["modelo"] = modelo
    return res


def rodar_cenarios(df: pd.DataFrame, semente: int):
    """Roda A (treino completo) e B (sem 85% dos idosos) numa mesma divisão."""
    treino, teste = dividir_treino_teste(df, semente=semente)
    a = treinar_e_avaliar(treino, teste)
    b = treinar_e_avaliar(remover_idosos(treino, semente=semente), teste)
    return a, b


if __name__ == "__main__":
    df = criar_grupo_idoso(carregar_dados())

    # ---- 1) Um sorteio só (semente 42), para ler os números com calma ----
    a, b = rodar_cenarios(df, SEMENTE_PRINCIPAL)
    print(f"=== Semente {SEMENTE_PRINCIPAL} (teste: idosos={a['pos_idoso']} doentes; "
          f"não idosos={a['pos_nao_idoso']} doentes) ===")
    tabela = pd.DataFrame({
        "A) treino completo": [a["recall_idoso"], a["recall_nao_idoso"], a["recall_geral"],
                               a["fn_idoso"], a["fn_nao_idoso"]],
        "B) sem 85% idosos": [b["recall_idoso"], b["recall_nao_idoso"], b["recall_geral"],
                              b["fn_idoso"], b["fn_nao_idoso"]],
    }, index=["recall idosos", "recall não idosos", "recall geral",
              "doentes idosos não detectados", "doentes não idosos não detectados"])
    print(tabela.round(3).to_string())

    # ---- 2) Repetir com várias sementes: média e variação ----
    linhas = []
    for s in range(N_REPETICOES):
        ra, rb = rodar_cenarios(df, s)
        linhas.append({"A_idoso": ra["recall_idoso"], "B_idoso": rb["recall_idoso"],
                       "A_nao": ra["recall_nao_idoso"], "B_nao": rb["recall_nao_idoso"],
                       "A_esp_idoso": ra["especif_idoso"], "B_esp_idoso": rb["especif_idoso"],
                       "A_acc_idoso": ra["acc_idoso"], "B_acc_idoso": rb["acc_idoso"],
                       "A_fp_idoso": ra["fp_idoso"], "B_fp_idoso": rb["fp_idoso"]})
    r = pd.DataFrame(linhas)

    print(f"\n=== {N_REPETICOES} repetições (divisão e remoção mudam a cada semente) ===")
    print(f"recall idosos     A: {r.A_idoso.mean():.3f} ± {r.A_idoso.std():.3f}   "
          f"B: {r.B_idoso.mean():.3f} ± {r.B_idoso.std():.3f}")
    print(f"recall não idosos A: {r.A_nao.mean():.3f} ± {r.A_nao.std():.3f}   "
          f"B: {r.B_nao.mean():.3f} ± {r.B_nao.std():.3f}")
    print(f"Em {(r.B_idoso < r.A_idoso).sum()} de {N_REPETICOES} repetições o recall dos idosos "
          f"piorou em B; em {(r.B_idoso > r.A_idoso).sum()} melhorou; "
          f"em {(r.B_idoso == r.A_idoso).sum()} ficou igual.")

    print("\n--- Outro lado do erro: pacientes idosos SAUDÁVEIS ---")
    print(f"especificidade idosos A: {r.A_esp_idoso.mean():.3f}   B: {r.B_esp_idoso.mean():.3f}   "
          "(saudáveis corretamente liberados)")
    print(f"falsos positivos idosos A: {r.A_fp_idoso.mean():.2f}    B: {r.B_fp_idoso.mean():.2f}    "
          "(média por repetição)")
    print(f"acurácia idosos        A: {r.A_acc_idoso.mean():.3f}   B: {r.B_acc_idoso.mean():.3f}")