"""
PASSO 5 - Treinar a Regressão Logística e medir o desempenho POR GRUPO  (PILAR 3)

Compara dois cenários, sempre avaliados no MESMO teste (intacto):
  A) Treino completo            -> referência (sem viés artificial)
  B) Treino sem 85% dos idosos  -> o desequilíbrio do enunciado

Métrica central: RECALL = dos pacientes que TÊM doença, quantos o modelo acertou.
Recall baixo significa mais doentes classificados como sem doença (falso negativo).
A consequência clínica dependeria de como um hospital usasse a classificação.

Como há só 8 idosos no treino B, um único sorteio é frágil. Por isso o script
também repete tudo com várias sementes e mostra média e variação.

Uso nos próximos passos:
    from src.modelo import treinar_e_avaliar
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
METRICAS_GRUPO = ("recall", "especif", "acc", "tn", "fp", "fn", "tp", "pos", "neg", "n")
MODELOS = {"logistica": "Regressão logística", "arvore": "Árvore (profundidade 3)",
           "knn": "KNN (5 vizinhos)"}


def montar_modelo(incluir_idade: bool = True, tipo: str = "logistica",
                  semente: int = SEMENTE_PRINCIPAL) -> Pipeline:
    """Pré-processamento ajustado no treino e modelo com parâmetros fixos.

    Árvore e KNN são comparações exploratórias; não substituem a referência.
    Nenhum parâmetro é escolhido pelo desempenho ou pela queda no teste.
    """
    numericas = NUMERICAS if incluir_idade else [c for c in NUMERICAS if c != "age"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("bin", "passthrough", BINARIAS),
    ])
    if tipo == "logistica":
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    elif tipo == "arvore":
        # Árvore curta e folhas com pelo menos 5 casos: comparação interpretável.
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=semente)
    elif tipo == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Modelo desconhecido: {tipo}. Opções: {list(MODELOS)}")
    return Pipeline([("pre", pre), ("clf", clf)])


def avaliar_predicoes(teste: pd.DataFrame, pred) -> dict:
    """Calcula os erros e seus denominadores; taxa sem denominador é NaN."""
    pred = np.asarray(pred)
    if pred.shape != (len(teste),):
        raise ValueError("É necessária exatamente uma previsão por paciente do teste.")
    if not np.isin(pred, [0, 1]).all() or not teste["target"].isin([0, 1]).all():
        raise ValueError("Alvo e previsões devem ser binários (0 ou 1).")
    positivos = teste["target"].to_numpy() == 1
    res = {"recall_geral": float((pred[positivos] == 1).mean()) if positivos.any() else np.nan}
    for nome, valor in [("idoso", 1), ("nao_idoso", 0)]:
        mascara = teste["idoso"].to_numpy() == valor
        y, p = teste.loc[mascara, "target"], pred[mascara]
        if len(y):
            tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, p, labels=[0, 1]).ravel())
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        pos, neg = tp + fn, tn + fp
        metricas = {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "pos": pos, "neg": neg,
                    "n": pos + neg,
                    "recall": tp / pos if pos else np.nan,
                    "especif": tn / neg if neg else np.nan,
                    "acc": (tn + tp) / (pos + neg) if pos + neg else np.nan}
        res.update({f"{m}_{nome}": v for m, v in metricas.items()})
    return res


def metricas_por_grupo(res: dict) -> list[dict]:
    """Linhas auditáveis para os CSVs dos experimentos."""
    return [{"grupo": grupo, **{m: res[f"{m}_{grupo}"] for m in METRICAS_GRUPO}}
            for grupo in ("idoso", "nao_idoso")]


def treinar_e_avaliar(treino: pd.DataFrame, teste: pd.DataFrame,
                      sample_weight=None, incluir_idade: bool = True,
                      tipo: str = "logistica", semente_modelo: int = SEMENTE_PRINCIPAL) -> dict:
    """Treina sem usar o teste no ajuste; pesos atuam apenas no classificador.

    O StandardScaler continua sem pesos, como na implementação original.
    `num`, `target`, `idoso` e `paciente_id` nunca entram como características.
    """
    if treino["target"].nunique() != 2:
        raise ValueError("O treino precisa conter as duas classes do alvo.")
    if sample_weight is not None and tipo == "knn":
        raise ValueError("KNN não aceita reponderação por sample_weight.")
    features = FEATURES if incluir_idade else [c for c in FEATURES if c != "age"]
    modelo = montar_modelo(incluir_idade, tipo, semente_modelo)
    kwargs = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
    modelo.fit(treino[features], treino["target"], **kwargs)
    pred = modelo.predict(teste[features])
    res = avaliar_predicoes(teste, pred)
    predicoes = teste[["paciente_id", "age", "idoso", "target"]].copy()
    predicoes["predicao"] = pred
    predicoes["prob_doenca"] = modelo.predict_proba(teste[features])[:, 1]
    res["predicoes"] = predicoes
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
    if r.B_idoso.mean() >= r.A_idoso.mean():
        print("\nNesta configuração não há queda média do recall dos idosos. "
              "O aumento de falsos positivos é um prejuízo diferente do pedido no pilar 3.")
    print("Veja também: python -m src.diagnostico_idade e python -m src.investigacao.")
