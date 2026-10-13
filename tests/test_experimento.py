"""Verificações dos requisitos científicos: isolamento, erros e mitigação."""

import unittest

import numpy as np
import pandas as pd

from src.dados import carregar_dados
from src.desequilibrio import remover_idosos
from src.divisao import dividir_treino_teste
from src.grupos import criar_grupo_idoso
from src.mitigacao import coletar_resultados, pesos_para_idosos, repor_idosos
from src.modelo import NUMERICAS, avaliar_predicoes, treinar_e_avaliar
from src.apresentacao import resumo_agregado, tabela_grupo
from src.mitigacao import rodar_cenarios


class MetricasConhecidas(unittest.TestCase):
    def test_matriz_conhecida_e_denominadores(self):
        teste = pd.DataFrame({"idoso": [1, 1, 1, 1, 0, 0], "target": [0, 0, 1, 1, 0, 1]})
        r = avaliar_predicoes(teste, [0, 1, 0, 1, 0, 1])
        self.assertEqual([r[f"{m}_idoso"] for m in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        self.assertEqual(r["pos_idoso"], 2)
        self.assertEqual(r["neg_idoso"], 2)
        self.assertEqual(r["n_idoso"], 4)
        self.assertEqual(r["recall_idoso"], 0.5)
        self.assertEqual(r["especif_idoso"], 0.5)
        self.assertEqual(r["acc_idoso"], 0.5)
        self.assertEqual(r["recall_nao_idoso"], 1)
        self.assertAlmostEqual(r["recall_geral"], 2 / 3)

    def test_taxa_sem_denominador_nao_vira_desempenho_perfeito(self):
        r = avaliar_predicoes(pd.DataFrame({"idoso": [1, 0], "target": [1, 0]}), [1, 0])
        self.assertTrue(np.isnan(r["especif_idoso"]))
        self.assertTrue(np.isnan(r["recall_nao_idoso"]))
        r = avaliar_predicoes(pd.DataFrame({"idoso": [1, 1], "target": [0, 1]}), [0, 1])
        self.assertEqual(r["n_nao_idoso"], 0)
        self.assertTrue(np.isnan(r["acc_nao_idoso"]))

    def test_previsoes_invalidas_sao_rejeitadas(self):
        teste = pd.DataFrame({"idoso": [1, 0], "target": [1, 0]})
        for previsoes in ([1], [1, 2], [[1], [0]]):
            with self.assertRaises(ValueError):
                avaliar_predicoes(teste, previsoes)


class RequisitosExperimento(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = criar_grupo_idoso(carregar_dados())

    def test_base_e_limite_de_idade(self):
        self.assertEqual(len(self.df), 297)
        self.assertTrue(self.df.paciente_id.is_unique)
        self.assertTrue((self.df.idoso == (self.df.age > 60).astype(int)).all())
        self.assertTrue((self.df.loc[self.df.age == 60, "idoso"] == 0).all())
        self.assertTrue((self.df.target == (self.df.num > 0).astype(int)).all())

    def test_remocao_preserva_teste_e_todos_os_nao_idosos(self):
        for semente in range(50):
            with self.subTest(semente=semente):
                treino, teste = dividir_treino_teste(self.df, semente)
                antes, teste_antes = treino.copy(deep=True), teste.copy(deep=True)
                desb = remover_idosos(treino, semente=semente)
                self.assertTrue(set(treino.paciente_id).isdisjoint(teste.paciente_id))
                self.assertTrue(set(desb.paciente_id).issubset(treino.paciente_id))
                self.assertEqual(set(desb.loc[desb.idoso == 0, "paciente_id"]),
                                 set(treino.loc[treino.idoso == 0, "paciente_id"]))
                self.assertEqual(int(desb.idoso.sum()), int(treino.idoso.sum()) - round(treino.idoso.sum() * 0.85))
                self.assertEqual(int(teste.idoso.sum()), 24)
                pd.testing.assert_frame_equal(treino, antes)
                pd.testing.assert_frame_equal(teste, teste_antes)

    def test_pesos_e_reposicao_so_usam_o_treino(self):
        treino, teste = dividir_treino_teste(self.df)
        desb = remover_idosos(treino)
        pesos = pesos_para_idosos(treino, desb)
        self.assertAlmostEqual(pesos[desb.idoso == 1].sum(), treino.idoso.sum())
        np.testing.assert_array_equal(pesos[desb.idoso == 0], 1)
        reposto = repor_idosos(treino, desb, 42)
        self.assertEqual(len(reposto), len(desb) + round((len(treino) - len(desb)) * 0.5))
        self.assertTrue(reposto.paciente_id.is_unique)
        self.assertTrue(set(reposto.paciente_id).issubset(treino.paciente_id))
        self.assertTrue(set(reposto.paciente_id).isdisjoint(teste.paciente_id))
        self.assertEqual(set(repor_idosos(treino, desb, 42, fracao=1).paciente_id), set(treino.paciente_id))
        with self.assertRaises(ValueError):
            pesos_para_idosos(treino, remover_idosos(treino, fracao=1))
        with self.assertRaises(ValueError):
            remover_idosos(treino, fracao=1.1)

    def test_preprocessamento_usa_apenas_treino_e_ablacao_remove_age(self):
        treino, teste = dividir_treino_teste(self.df)
        teste = teste.copy()
        teste[NUMERICAS] = teste[NUMERICAS] + 10000
        res = treinar_e_avaliar(treino, teste, incluir_idade=False)
        nomes = res["modelo"].named_steps["pre"].get_feature_names_out()
        self.assertNotIn("num__age", nomes)
        self.assertFalse(any(any(proibido in nome for proibido in ("target", "paciente_id", "idoso", "__num"))
                             for nome in nomes))
        colunas = [c for c in NUMERICAS if c != "age"]
        scaler = res["modelo"].named_steps["pre"].named_transformers_["num"]
        np.testing.assert_allclose(scaler.mean_, treino[colunas].mean())

    def test_exportacao_permite_reconstruir_matrizes(self):
        taxas, predicoes = coletar_resultados(self.df, n_repeticoes=2)
        self.assertEqual(len(taxas), 2 * 4 * 2)
        self.assertEqual(len(predicoes), 2 * 4 * 90)
        self.assertFalse(predicoes.duplicated(["semente", "cenario", "paciente_id"]).any())
        for (semente, cenario), p in predicoes.groupby(["semente", "cenario"]):
            r = avaliar_predicoes(p, p.predicao.to_numpy())
            for grupo in ("idoso", "nao_idoso"):
                linha = taxas[(taxas.semente == semente) & (taxas.cenario == cenario)
                              & (taxas.grupo == grupo)].iloc[0]
                for metrica in ("tn", "fp", "fn", "tp", "recall", "especif", "acc"):
                    self.assertEqual(linha[metrica], r[f"{metrica}_{grupo}"])

    def test_resumo_da_apresentacao_reflete_resultados_da_arvore(self):
        resultados = rodar_cenarios(self.df, 42, tipo="arvore")
        tabela = tabela_grupo(resultados)
        self.assertEqual(tabela.loc["A", "FN"], 2)
        self.assertEqual(tabela.loc["B", "FN"], 5)
        self.assertAlmostEqual(tabela.loc["B", "recall"], 8 / 13)

        linhas = []
        for semente in (0, 1):
            for cenario, res in rodar_cenarios(self.df, semente, tipo="arvore").items():
                for grupo in ("idoso", "nao_idoso"):
                    linhas.append({"semente": semente, "cenario": cenario, "grupo": grupo,
                                   "recall": res[f"recall_{grupo}"],
                                   "especif": res[f"especif_{grupo}"],
                                   "acc": res[f"acc_{grupo}"],
                                   "fn": res[f"fn_{grupo}"], "fp": res[f"fp_{grupo}"]})
        resumo = resumo_agregado(pd.DataFrame(linhas))
        self.assertEqual(list(resumo.index), ["A", "B", "C", "D"])
        self.assertTrue(resumo.notna().all().all())


if __name__ == "__main__":
    unittest.main()
