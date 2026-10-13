"""Resumos dos experimentos e informações para reproduzir os resultados."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from src.dados import ARQUIVO, URL
from src.desequilibrio import FRACAO_REMOVIDA
from src.divisao import TAMANHO_TESTE
from src.grupos import LIMITE_IDADE
from src.modelo import N_REPETICOES, SEMENTE_PRINCIPAL


def diferencas_pareadas(r: pd.DataFrame, comparacoes: list[tuple],
                       estratos: tuple = ("grupo",)) -> pd.DataFrame:
    """Mudanças dentro da mesma divisão, sem tratar repetições como novos pacientes.

    Desvio e quantis descrevem variabilidade entre sorteios, não intervalos de
    confiança nem testes de significância. Para fn/fp, maior significa mais erros.
    """
    linhas = []
    for chave, sub in r.groupby(list(estratos)):
        if not isinstance(chave, tuple):
            chave = (chave,)
        identificadores = dict(zip(estratos, chave))
        for metrica in ("recall", "especif", "acc", "fn", "fp"):
            pares = sub.pivot(index="semente", columns="cenario", values=metrica)
            for referencia, novo in comparacoes:
                if referencia not in pares or novo not in pares:
                    continue
                delta = (pares[novo] - pares[referencia]).dropna()
                linhas.append({**identificadores, "comparacao": f"{novo}-{referencia}",
                               "metrica": metrica, "n_pares": len(delta),
                               "delta_medio": delta.mean(), "delta_desvio": delta.std(),
                               "delta_q05": delta.quantile(0.05), "delta_q95": delta.quantile(0.95),
                               "aumentou": int((delta > 0).sum()), "igual": int((delta == 0).sum()),
                               "diminuiu": int((delta < 0).sum())})
    return pd.DataFrame(linhas)


def salvar_metadados(caminho: Path, df: pd.DataFrame,
                     n_repeticoes: int = N_REPETICOES) -> None:
    metadados = {
        "fonte": URL,
        "arquivo_sha256": hashlib.sha256(ARQUIVO.read_bytes()).hexdigest(),
        "paciente_id": "Número da linha no arquivo original; identificador técnico, não pessoal",
        "python": platform.python_version(),
        "versoes": {p: version(p) for p in ("pandas", "numpy", "scikit-learn", "matplotlib")},
        "pacientes_apos_limpeza": len(df), "idosos_apos_limpeza": int(df.idoso.sum()),
        "grupo": f"age > {LIMITE_IDADE}", "fracao_removida": FRACAO_REMOVIDA,
        "fracao_reposta_em_D": 0.5, "fracao_teste": TAMANHO_TESTE,
        "sementes_divisao_e_remocao": list(range(n_repeticoes)),
        "semente_modelo": SEMENTE_PRINCIPAL,
        "modelos": {"logistica": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
                    "arvore": {"max_depth": 3, "min_samples_leaf": 5},
                    "knn": {"n_neighbors": 5, "weights": "uniform"}},
        "decisao": "predict() padrão, sem seleção de limiar no teste",
        "preprocessamento": "Ajustado somente no respectivo treino; sem pesos no StandardScaler",
        "pesos_C": "Aplicados somente ao classificador; KNN não tem cenário C",
        "interpretacao": "Comparações exploratórias; repetições reutilizam pacientes e não são amostras independentes",
    }
    caminho.write_text(json.dumps(metadados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
