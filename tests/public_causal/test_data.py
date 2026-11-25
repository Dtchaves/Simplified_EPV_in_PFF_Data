"""Fixture tests: observation, timing, acquisition/protected confirmation.

No live World Cup outcomes are opened. The fixture vocabulary uses fictional
teams/players and tests the production data adapter directly.
"""
import copy
import ast
import io
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PublicCausal import data


def event(identifier, typ, seconds=10, index=2, team=1, period=1, location=None, **extra):
    e = {"id": identifier, "index": index, "period": period,
         "timestamp": f"00:{int(seconds)//60:02d}:{seconds%60:06.3f}",
         "type": {"name": typ}, "team": {"id": team},
         "player": {"id": team*10}, "position": {"name": "Center Forward"}}
    if typ in data.ACTION_TYPES:
        e["location"] = [100, 40] if location is None else location
    e.update(extra)
    return e


def frame(identifier, actor=None, polygon=None, players=None):
    return {"event_uuid": identifier,
            "visible_area": polygon or [70, 20, 120, 20, 120, 60, 70, 60, 70, 20],
            "freeze_frame": [{"actor": True, "teammate": True, "keeper": False, "location": actor or [100, 40]}] + (players or [])}


def match(home_score=0, away_score=0):
    return {"match_id": 1000, "home_team": {"home_team_id": 1},
            "away_team": {"away_team_id": 2}, "home_score": home_score, "away_score": away_score}


def stream(*actions, end=40):
    result = [event("start1", "Half Start", 0, 0), event("start2", "Half Start", 0, 1, team=2)] + list(actions)
    if end is not None:
        result += [event("end1", "Half End", end, 990), event("end2", "Half End", end, 991, team=2)]
    return result


class ObservationTests(unittest.TestCase):
    def test_adapter_exports_scientific_model_feature_contract(self):
        source = Path(__file__).resolve().parents[2]/"results/models/public_statsbomb_causal_study.ipynb"
        if not source.exists():
            self.skipTest("Scientific source notebook has not been assembled")
        # Read the maintained feature declarations without importing scientific
        # dependencies in the stdlib-only adapter tests.
        declarations = {}
        notebook = json.loads(source.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code" or "public-causal-core" not in cell.get("metadata", {}).get("tags", []):
                continue
            text = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            for node in ast.parse(text).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {"NUMERIC_FEATURES", "CATEGORICAL_FEATURES"}:
                            declarations[target.id] = ast.literal_eval(node.value)
        self.assertEqual(set(declarations), {"NUMERIC_FEATURES", "CATEGORICAL_FEATURES"})
        rows, _ = data.build_match_decisions(stream(event("pass", "Pass")), [frame("pass")], match(), "fixture")
        for name in declarations["NUMERIC_FEATURES"] + declarations["CATEGORICAL_FEATURES"]:
            self.assertIn(name, rows[0])
        for name in declarations["NUMERIC_FEATURES"]:
            self.assertIsInstance(rows[0][name], (int, float))
        self.assertTrue(math.isnan(rows[0]["time_since_prev_action"]))

    def test_coordinates_and_goal_opening(self):
        self.assertEqual(data.metric_xy([120, 40]), (52.5, 0))
        self.assertEqual(data.metric_xy([0, 0]), (-52.5, 34))
        self.assertAlmostEqual(data._goal_angle(35, 0), math.degrees(2*math.atan(3.66/17.5)))

    def test_simple_and_invalid_polygon(self):
        polygon = data.visible_polygon(frame("x")["visible_area"])
        self.assertTrue(data.disk_visible((35, 0), polygon, 3))
        self.assertFalse(data.disk_visible((51, 0), polygon, 3))
        with self.assertRaises(ValueError):
            data.visible_polygon([70, 20, 120, 60, 70, 60, 120, 20])
        with self.assertRaises(ValueError):
            data.visible_polygon([70, 20, 80, 20, 90, 20])

    def test_primary_without_keeper_or_full_roster(self):
        players = [{"actor": False, "teammate": False, "keeper": False, "location": [101,40]}]
        rows, audit = data.build_match_decisions(stream(event("pass", "Pass")), [frame("pass", players=players)], match(), "fixture")
        row = rows[0]
        self.assertTrue(row["eligible"], row["drop_reason"])
        self.assertFalse(row["opposing_keeper_visible"])
        self.assertEqual(row["defenders_3m"], 1)
        self.assertAlmostEqual(row["nearest_defender"], .875)
        self.assertEqual(row["second_defender"], 3)

    def test_missing_opponents_are_capped_not_unbounded(self):
        rows, _ = data.build_match_decisions(stream(event("pass", "Pass")), [frame("pass")], match(), "fixture")
        self.assertEqual(rows[0]["nearest_defender"], 3)
        self.assertEqual(rows[0]["defenders_3m"], 0)

    def test_actor_mismatch_and_missing_frame(self):
        actions = stream(event("pass", "Pass"), event("shot", "Shot", 15, 3, shot={"freeze_frame": [{"location": [100,40]}]}))
        rows, _ = data.build_match_decisions(actions, [frame("pass", actor=[102,40])], match(), "fixture")
        self.assertIn("actor_coordinate_mismatch", rows[0]["drop_reason"])
        self.assertIn("missing_360", rows[1]["drop_reason"])

    def test_duplicate_or_unmatched_uuid_fails(self):
        events = stream(event("pass", "Pass"))
        with self.assertRaises(ValueError):
            data.build_match_decisions(events, [frame("pass"), frame("pass")], match(), "fixture")
        with self.assertRaises(ValueError):
            data.build_match_decisions(events, [frame("alien")], match(), "fixture")

    def test_equivalent_continuations_and_ambiguous_treatment(self):
        carry = event("carry", "Carry", index=2, duration=5, carry={"end_location": [120,40]})
        dribble = event("dribble", "Dribble", index=3)
        rows, _ = data.build_match_decisions(stream(carry, dribble), [frame("carry"), frame("dribble")], match(), "fixture")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["eligible"])
        self.assertEqual(rows[0]["source_event_ids"], ["carry", "dribble"])
        rows, _ = data.build_match_decisions(stream(carry, event("shot", "Shot", index=3)), [frame("carry"), frame("shot")], match(), "fixture")
        self.assertIn("ambiguous_treatment_collision", rows[0]["drop_reason"])

    def test_restart_exclusion_does_not_exclude_subsequent_play(self):
        actions = stream(event("throw", "Pass", index=2, pass_={}), event("carry", "Carry", seconds=12, index=3))
        actions[2]["pass"] = {"type": {"name": "Throw-in"}}
        actions[3]["play_pattern"] = {"name": "From Throw In"}
        rows, _ = data.build_match_decisions(actions, [frame("throw"), frame("carry")], match(), "fixture")
        self.assertIn("direct_restart", rows[0]["drop_reason"])
        self.assertTrue(rows[1]["eligible"])

    def test_strict_prior_history_and_no_current_carry_end(self):
        actions = stream(event("earlier", "Pass", 5, 2, location=[90,40]),
                         event("carry", "Carry", 10, 3, duration=999, carry={"end_location": [0,0]}),
                         event("dribble", "Dribble", 10, 4))
        rows, _ = data.build_match_decisions(actions, [frame("carry"), frame("dribble")], match(), "fixture")
        row = rows[1]
        self.assertEqual(row["prior_10s_actions"], 1)
        self.assertEqual(row["time_since_prev_action"], 5)
        self.assertAlmostEqual(row["prior_10s_progress"], 8.75)
        self.assertNotIn("duration", row)
        self.assertNotIn("end_location", row)


class OutcomeTests(unittest.TestCase):
    def test_natural_end_absorbs_without_future_baseline_filter(self):
        rows, _ = data.build_match_decisions(stream(event("pass", "Pass", 38), end=40), [frame("pass")], match(), "fixture")
        self.assertTrue(rows[0]["eligible"])
        self.assertEqual(rows[0]["Y15"], 0)
        self.assertFalse(rows[0]["full_horizon_15"])

    def test_missing_or_abnormal_end_unknown(self):
        for abnormal in [False, True]:
            events = stream(event("pass", "Pass"), end=40 if abnormal else None)
            if abnormal:
                events[-1]["half_end"] = {"early_video_end": True}
            rows, _ = data.build_match_decisions(events, [frame("pass")], match(), "fixture")
            self.assertTrue(math.isnan(rows[0]["Y15"]))
            self.assertIn("unknown_outcome", rows[0]["drop_reason"])

    def test_linked_occurrence_first_goal_signed(self):
        action = event("pass", "Pass", 10, 2)
        goal = event("goal", "Shot", 24, 3, team=2, duration=.1,
                     shot={"outcome": {"name": "Goal"}}, related_events=["conceded"])
        keeper = event("conceded", "Goal Keeper", 26, 4,
                       goalkeeper={"type": {"name": "Goal Conceded"}}, related_events=["goal"])
        rows, _ = data.build_match_decisions(stream(action, goal, keeper), [frame("pass"), frame("goal")], match(away_score=1), "fixture")
        self.assertEqual(rows[0]["Y15"], 0)
        self.assertEqual(rows[0]["Y30"], -1)
        self.assertEqual(rows[0]["Y5"], 0)
        self.assertEqual(rows[0]["Y10"], 0)
        self.assertEqual(rows[0]["Y15_shot_start"], -1)
        self.assertEqual(rows[0]["Y15_duration"], -1)
        self.assertEqual(rows[0]["goal_timing_30"], "linked_concession")

    def test_start_time_sensitivity_preserves_source_event_index(self):
        shot = event("goal", "Shot", 10, 2, duration=2,
                     shot={"outcome": {"name": "Goal"}}, related_events=["keeper"])
        continuation = event("later", "Pass", 10, 3, team=2)
        keeper = event("keeper", "Goal Keeper", 12, 4, team=2,
                       goalkeeper={"type": {"name": "Goal Conceded"}})
        rows, _ = data.build_match_decisions(stream(shot, continuation, keeper), [frame("goal"), frame("later")], match(home_score=1), "fixture")
        later = next(r for r in rows if r["event_id"] == "later")
        self.assertEqual(later["Y15"], -1)
        self.assertEqual(later["Y15_shot_start"], 0)

    def test_own_goal_pair_counts_once(self):
        own_for = event("for", "Own Goal For", 12, 3, team=2, related_events=["against"])
        own_against = event("against", "Own Goal Against", 12, 4, related_events=["for"])
        events = stream(event("pass", "Pass", 10, 2), own_for, own_against)
        rows, audit = data.build_match_decisions(events, [frame("pass")], match(away_score=1), "fixture")
        self.assertEqual(len(audit["goals"]), 1)
        self.assertEqual(rows[0]["Y15"], -1)

    def test_equal_time_order_and_treatment_own_goal(self):
        goal = event("goal", "Shot", 10, 3, duration=0, shot={"outcome": {"name": "Goal"}})
        earlier = event("earlier", "Pass", 10, 2, team=2)
        later = event("later", "Pass", 10, 4, team=2)
        rows, _ = data.build_match_decisions(stream(earlier, goal, later), [frame("earlier"), frame("goal"), frame("later")], match(home_score=1), "fixture")
        by = {r["event_id"]: r for r in rows}
        # Same-time same actor continuation records collapse, hence use the
        # signed-first-goal helper to test the later distinct index directly.
        ledger = data.canonical_goals(stream(goal))
        obs = data.period_observation(stream(goal))
        self.assertEqual(data.signed_first_goal(earlier, ledger, obs, 15)[0], -1)
        self.assertEqual(data.signed_first_goal(later, ledger, obs, 15)[0], 0)
        self.assertEqual(by["goal"]["Y15"], 1)

    def test_windows_do_not_cross_half_and_score_does(self):
        first = event("goal", "Shot", 30, 3, duration=0, shot={"outcome": {"name": "Goal"}})
        second = event("pass2", "Pass", 2, 5, period=2)
        events = stream(event("pass1", "Pass", 38, 4), first, end=40)
        events += [event("start3", "Half Start", 0, 995, period=2), second, event("end3", "Half End", 40, 999, period=2)]
        rows, _ = data.build_match_decisions(events, [frame("pass1"), frame("goal"), frame("pass2")], match(home_score=1), "fixture")
        by = {r["event_id"]: r for r in rows}
        self.assertEqual(by["pass1"]["Y15"], 0)
        self.assertEqual(by["pass2"]["score_difference"], 1)

    def test_penalty_conceded_and_duration_fallback(self):
        shot = event("goal", "Shot", 10, 2, duration=2, shot={"outcome": {"name": "Goal"}}, related_events=["keeper"])
        keeper = event("keeper", "Goal Keeper", 13, 3, team=2, goalkeeper={"type": {"name": "Penalty Conceded"}})
        self.assertEqual(data.canonical_goals(stream(shot, keeper))[0]["time_seconds"], 13)
        self.assertEqual(data.canonical_goals(stream(shot))[0]["time_seconds"], 12)

    def test_score_ledger_contradiction_is_unknown(self):
        rows, audit = data.build_match_decisions(stream(event("pass", "Pass")), [frame("pass")], match(home_score=1), "fixture")
        self.assertFalse(audit["score_reconciled"])
        self.assertTrue(math.isnan(rows[0]["Y15"]))


class AcquisitionTests(unittest.TestCase):
    def test_confirmation_is_guarded_before_network_or_cache(self):
        with tempfile.TemporaryDirectory() as root, patch.object(data, "urlopen") as network:
            with self.assertRaises(PermissionError):
                data.acquire_cohort(root, "confirmation")
            network.assert_not_called()
            self.assertEqual(list(Path(root).iterdir()), [])
            with self.assertRaises(PermissionError):
                data.build_decisions(root, "confirmation")

    def test_confirmation_runtime_lock_integrity_and_required_gates(self):
        from PublicCausal.notebook_runtime import freeze_design_lock, DEFAULT_LOCK_GATES
        with tempfile.TemporaryDirectory() as root:
            lock_path = Path(root)/"design_lock.json"
            payload = {"core_source_hash": "a"*64, "data_manifest_hash": "b"*64,
                       "config_hash": "c"*64, "source_commit": data.SOURCE_COMMIT,
                       "development_gates": {k: True for k in DEFAULT_LOCK_GATES}}
            freeze_design_lock(lock_path, payload)
            self.assertTrue(data.validate_confirmation_lock(lock_path)["confirmation_allowed"])
            with self.assertRaises(PermissionError):
                data.validate_confirmation_lock(payload)
            lock_path.write_text(lock_path.read_text()+" ")
            with self.assertRaises(PermissionError):
                data.validate_confirmation_lock(lock_path)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas required for canonical cache integration")
    def test_canonical_cache_and_raw_corruption(self):
        def response(request, timeout):
            url = request.full_url
            if "/matches/" in url:
                content = [dict(match(), match_id=i) for i in range(51)]
            elif "/events/" in url:
                content = stream(event("pass", "Pass"))
            elif "/three-sixty/" in url:
                content = [frame("pass")]
            elif url.endswith(".json"):
                content = []
            else:
                return io.BytesIO(b"fixture-license")
            return io.BytesIO(json.dumps(content).encode())
        with tempfile.TemporaryDirectory() as root, patch.object(data, "urlopen", side_effect=response):
            data.acquire_cohort(root, "euro2024", limit_matches=1)
            df = data.build_decisions(root, "euro2024")
            self.assertTrue(df.iloc[0].eligible)
            self.assertEqual(data.build_decisions(root, "euro2024").attrs["canonical_key"], df.attrs["canonical_key"])
            path = Path(root)/"raw/data/events/0.json"
            original = path.read_bytes()
            path.write_text("corruption")
            with self.assertRaises(ValueError):
                data.build_decisions(root, "euro2024")
            path.write_bytes(original)
            cache = next((Path(root)/"canonical").glob("*.pkl"))
            cache.write_bytes(b"corrupted canonical cache")
            with self.assertRaises(ValueError):
                data.build_decisions(root, "euro2024")

    def test_partial_pinned_acquisition_resume_and_corruption(self):
        def response(request, timeout):
            url = request.full_url
            self.assertIn(data.SOURCE_COMMIT, url)
            if "/matches/" in url:
                content = [{"match_id": i} for i in range(51)]
            elif url.endswith(".json"):
                content = []
            else:
                return io.BytesIO(b"fixture-license")
            return io.BytesIO(json.dumps(content).encode())
        with tempfile.TemporaryDirectory() as root, patch.object(data, "urlopen", side_effect=response) as network:
            manifest = data.acquire_cohort(root, "euro2024", limit_matches=2)
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(len(manifest["matches"]), 2)
            self.assertEqual(len(manifest["files"]), 10)
            network.reset_mock()
            data.acquire_cohort(root, "euro2024", limit_matches=2)
            network.assert_not_called()
            (Path(root)/"raw/data/events/0.json").write_text("corrupted")
            with self.assertRaises(ValueError):
                data.acquire_cohort(root, "euro2024", limit_matches=2)


if __name__ == "__main__":
    unittest.main()
