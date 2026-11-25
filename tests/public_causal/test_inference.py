"""Inference tests import exactly the notebook-generated scientific module."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from PublicCausal.notebook_runtime import load_notebook_core


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.c = load_notebook_core(ROOT / "results/models/public_statsbomb_causal_study.ipynb", cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def fixture(self):
        c = self.c
        return c.pd.DataFrame({"T": [1, 1, 0, 0], "Y": [1., 0., 0., -1.],
                              "e": [.2, .4, .2, .4], "m0": [.1, .2, .1, .2],
                              "m1": [.4] * 4, "support": [True] * 4,
                              "original_match_id": ["a", "b", "a", "b"],
                              "decision_row_id": ["s1", "s2", "d1", "d2"], "x": [1., 2., 1., 2.]})

    def test_count_normalization_and_every_row_contribution(self):
        c = self.c
        rows = self.fixture()
        answer = c.att_estimate(rows)
        correction = (.25 * -.1 + (2 / 3) * -1.2) / 2
        expected = .5 - .15 - correction
        self.assertAlmostEqual(answer["att"], expected, places=12)
        self.assertAlmostEqual(answer["rows"].att_contribution.sum(), expected, places=12)
        self.assertAlmostEqual(answer["residual_correction"], correction, places=12)
        self.assertNotAlmostEqual(answer["hajek_aipw_att"], expected, places=5)
        self.assertAlmostEqual(answer["cluster_scores"].score.sum(), 0, places=12)

    def test_unknown_outcomes_and_invalid_propensity(self):
        rows = self.fixture()
        rows.loc[0, "Y"] = float("nan")
        self.assertEqual(self.c.att_estimate(rows)["n_shots"], 1)
        rows.loc[2, "e"] = 1
        with self.assertRaises(ValueError):
            self.c.att_estimate(rows)

    def test_matching_uses_actual_pairs_and_reuse_weights(self):
        c = self.c
        answer = c.att_comparators(self.fixture(), ["x"], k=1)
        self.assertEqual(set(answer["pairs"].shot_id), {"s1", "s2"})
        self.assertEqual(set(answer["pairs"].donor_id), {"d1", "d2"})
        self.assertAlmostEqual(answer["reuse"].matching_weight.sum(), 2)
        matching = answer["summary"].set_index("estimator").loc["NN_matching_ATT", "att"]
        self.assertAlmostEqual(matching, 1.)
        self.assertTrue(answer["summary"].same_treated_ids.all())

    def test_bootstrap_original_groups_and_resume(self):
        c = self.c
        source = self.fixture()
        seen = []
        def fitter(frame, config, frozen_design=None):
            seen.append(set(frame.original_match_id))
            for original, group in frame.groupby("original_match_id"):
                self.assertTrue(group.original_match_id.eq(original).all())
            return {"rows": frame, "diagnostics": {"passed": False}}
        with tempfile.TemporaryDirectory() as directory:
            result = c.cluster_bootstrap(source, {"seed": 14, "bootstrap_max_attempts": 5, "workers": 1}, fitter,
                                         frozen_design={"fixed": True}, n_success=2, checkpoint_dir=directory)
            self.assertEqual(result["n_success"], 2)
            self.assertTrue((result["attempts"].status == "ok").all())
            # Scientifically poor finite draws remain; they are not retried or censored.
            self.assertTrue(all(not record["diagnostics"]["passed"] for record in result["attempts"].to_dict("records")))
            previous = len(seen)
            again = c.cluster_bootstrap(source, {"seed": 14, "bootstrap_max_attempts": 5, "workers": 1}, fitter,
                                        frozen_design={"fixed": True}, n_success=2, checkpoint_dir=directory)
            self.assertEqual(len(seen), previous)
            self.assertEqual(again["attempts"].seed.tolist(), result["attempts"].seed.tolist())

    def test_corrupted_checkpoint_cannot_be_reused(self):
        import json
        c = self.c
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            c._atomic_inference_json(path, [{"att": .2, "status": "ok"}])
            saved = json.loads(path.read_text())
            saved["payload"][0]["att"] = 1.5
            # Tests deliberately corrupt an artifact; maintained source remains notebook-owned.
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "checksum"):
                c._load_inference_checkpoint(path)

    def test_statistic_failure_does_not_erase_primary_draw(self):
        c = self.c
        def fitter(frame, config, frozen_design=None):
            return {"rows": frame, "diagnostics": {"passed": False}}
        def broken_statistic(rows):
            raise ValueError("test tail stage failure")
        result = c.cluster_bootstrap(self.fixture(), {"seed": 8, "workers": 1}, fitter,
                                     n_success=2, stats_fn=broken_statistic)
        self.assertEqual(result["n_success"], 2)
        valid = result["attempts"].loc[result["attempts"].status.eq("ok")]
        self.assertEqual(len(valid), 2)
        self.assertTrue(valid.statistics_status.eq("failed").all())
        structural = result["attempts"].loc[result["attempts"].status.eq("failed")]
        # Two-match resampling can leave one distinct original match. These
        # are recorded structural failures, distinct from endpoint failures.
        self.assertGreater(len(structural), 0)
        for record in structural.to_dict("records"):
            self.assertIn("at least two eligible original matches", record["error"])
            self.assertIn("ValueError", record["error"])
            self.assertTrue(c.pd.isna(record.get("statistics_status")))
        self.assertTrue(result["attempts"].status.isin(["ok", "failed"]).all())

    def test_gamma_one_is_exact_primary_even_with_wrong_distribution(self):
        c = self.c
        rows = self.fixture()
        result = c.dvds_sensitivity(rows, ["x"], gammas=[1], tipping_tolerance=None)
        self.assertAlmostEqual(result["bounds"].att_lower.iloc[0], c.att_estimate(rows)["att"], places=12)
        self.assertAlmostEqual(result["bounds"].att_upper.iloc[0], c.att_estimate(rows)["att"], places=12)

    def test_oracle_lp_ties_and_degenerate_atoms(self):
        c = self.c
        probabilities = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, .5],
                         [.2, .6, .2], [.1, .8, .1], [.25, .25, .5], [.01, .94, .05]]
        for p in probabilities:
            for gamma in [1, 1.25, 1.5, 2, 3, 5, 10]:
                with self.subTest(p=p, gamma=gamma):
                    answer = c.ternary_oracle_bounds(p, gamma)
                    self.assertAlmostEqual(answer["lower"], answer["cvar_lower"], places=10)
                    self.assertAlmostEqual(answer["upper"], answer["cvar_upper"], places=10)
                    self.assertLessEqual(answer["lower"], answer["upper"] + 1e-12)
                    self.assertAlmostEqual(c.np.dot(p, answer["lower_ratios"]), 1, places=10)
                    self.assertAlmostEqual(c.np.dot(p, answer["upper_ratios"]), 1, places=10)
        self.assertEqual(c.ternary_quantile([.25, .5, .25], .25)[0], -1)
        self.assertEqual(c.ternary_quantile([.25, .5, .25], .75)[0], 0)

    def test_dvds_actual_tail_training_provenance(self):
        c = self.c
        rng = c.np.random.default_rng(24)
        n = 120
        rows = c.pd.DataFrame({"x": rng.normal(size=n), "T": [int(i % 4 == 0) for i in range(n)],
                               "Y": rng.choice([-1, 0, 1], n, p=[.05, .8, .15]),
                               "e": .25, "m0": .1, "support": True,
                               "original_match_id": [str(i // 10) for i in range(n)],
                               "outer_fold": [i // 10 % 3 for i in range(n)]})
        result = c.dvds_sensitivity(rows, ["x"], gammas=[1, 2], tipping_tolerance=None)
        self.assertTrue(c.np.isfinite(result["bounds"][["att_lower", "att_upper"]]).all().all())
        for record in result["provenance"].to_dict("records"):
            q, tail, evaluation = map(set, [record["quantile_train_matches"], record["tail_label_train_matches"], record["evaluation_matches"]])
            self.assertTrue(q.isdisjoint(tail) and q.isdisjoint(evaluation) and tail.isdisjoint(evaluation))

    def test_common_domain_excludes_legacy_high_propensity_donor(self):
        c = self.c
        rows = c.pd.DataFrame({"T": [1, 1, 0, 0], "Y": [0., 0., 0., 1.],
                               "e": [.5, .5, .5, .99], "m0": 0.,
                               "support": [True, True, True, False],
                               "original_match_id": ["a", "b", "a", "b"]})
        self.assertEqual(c.att_estimate(rows)["att"], 0.)
        legacy = rows.copy()
        legacy.loc[3, "support"] = True
        self.assertAlmostEqual(c.att_estimate(legacy)["att"], -49.5)

    def test_visibility_truth_conditions_on_selected_treated(self):
        c = self.c
        scenario = next(s for s in c.simulation_scenarios() if s["name"] == "visibility_selection")
        frame = c.simulate_causal_matches(scenario, 14, n_matches=6, decisions_per_match=20)
        original = frame.original_mean_y1 - frame.original_mean_y0
        conditional = frame.true_y1 - frame.true_y0
        self.assertTrue(c.np.allclose(conditional, original * (.30 / .57)))
        expected_e = .57 * frame.original_propensity / (.57 * frame.original_propensity + .82 * (1 - frame.original_propensity))
        self.assertTrue(c.np.allclose(expected_e, frame.true_propensity))
        self.assertFalse(c.np.allclose(conditional, original))
        self.assertEqual(len([s for s in c.simulation_scenarios() if s["family"] == "base24"]), 24)

    def test_simulation_fits_the_actual_notebook_pipeline(self):
        c = self.c
        config = c.scientific_config(dict(outer_folds=3, inner_folds=2, workers=1,
            e_candidates=[{"kind": "logistic", "C": .2}],
            outcome_candidates=[{"kind": "ridge", "alpha": 1}],
            min_shots=5, min_matches=2, min_ess=5, donor_min=2, donor_match_min=2,
            key_smd_max=10, other_smd_max=10, max_weight_share=1, max_match_weight_share=1))
        scenario = c.simulation_scenarios(include_stresses=False)[0]
        with tempfile.TemporaryDirectory() as directory:
            result = c.run_simulation_harness(config, c.fit_study, directory, repetitions=1,
                coverage_cases=False, workers=1, n_matches=12, decisions_per_match=40, scenarios=[scenario])
        self.assertTrue(result["completion"])
        self.assertEqual(result["records"].status.iloc[0], "ok", result["records"].to_dict("records"))
        record = result["records"].iloc[0]
        self.assertLess(record.truth, 0)
        self.assertEqual(record.ci_kind, "match_score_diagnostic_not_production_CI")
        self.assertGreater(record.n_shots, 0)


if __name__ == "__main__":
    unittest.main()
