"""Pinned public StatsBomb acquisition and observation adapter.

This module has no estimator code. Each candidate row preserves its observation
gate, source UUIDs and outcome timing. Do not fit on rows where ``eligible`` is
false. Confirmation is guarded before even inspecting cached match outcomes.
Coordinates are normalized to 105 x 68 m; snapshots are not tracking data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

SOURCE_COMMIT = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
BASE_URL = f"https://raw.githubusercontent.com/hudl/open-data/{SOURCE_COMMIT}"
ADAPTER_VERSION = "public360-v1"
COHORTS = {"euro2020": (55, 43, 51), "euro2024": (55, 282, 51),
           "worldcup2022": (43, 106, 64)}
ALIASES = {"Euro2020": "euro2020", "Euro2024": "euro2024",
           "WC2022": "worldcup2022", "confirmation": "worldcup2022"}
ACTION_TYPES = {"Shot", "Pass", "Carry", "Dribble"}
RESTART_PASSES = {"Corner", "Free Kick", "Throw-in", "Throw In", "Goal Kick", "Kick Off", "Kickoff"}


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _emit(logger, event, **fields):
    if logger is None:
        return
    if callable(logger):
        logger(event, **fields)
    elif hasattr(logger, "event"):
        logger.event(event, **fields)
    elif hasattr(logger, "log"):
        logger.log(event, **fields)


def validate_confirmation_lock(design_lock):
    """Verify the immutable lock's integrity/schema before opening confirmation.

    The runtime verifies the sealed lock and its mandatory gates. The calling
    study also checks its current scientific source against the frozen hash.
    A bare boolean or unsealed dictionary cannot grant confirmation access.
    """
    if design_lock is None:
        raise PermissionError("World Cup confirmation requires an immutable development design lock")
    if not isinstance(design_lock, (str, Path)):
        raise PermissionError("design_lock must be the immutable runtime lock file path")
    from .notebook_runtime import validate_design_lock
    lock = validate_design_lock(design_lock)
    if lock.get("source_commit", SOURCE_COMMIT) != SOURCE_COMMIT:
        raise PermissionError("Development design lock source revision differs")
    # Runtime checks its sealed sidecar, frozen state and mandatory gates.
    # Keep a convenient acquisition audit hash without modifying the lock.
    lock = {**lock, "lock_sha256": _sha(Path(design_lock).read_bytes())}
    return lock


def _cohorts(cohort, design_lock=None):
    if cohort == "development":
        return ["euro2020", "euro2024"]
    name = ALIASES.get(cohort, str(cohort).lower())
    if name not in COHORTS:
        raise ValueError(f"Unknown cohort {cohort!r}; expected development, euro2020, euro2024, confirmation")
    if name == "worldcup2022":
        validate_confirmation_lock(design_lock)
    return [name]


def _download(root, relative, prior=None, logger=None, retries=3):
    path = root / "raw" / relative
    url = BASE_URL + "/" + quote(relative, safe="/")
    if path.exists():
        payload = path.read_bytes()
        if prior and (_sha(payload) != prior["sha256"] or len(payload) != prior["bytes"]):
            raise ValueError(f"Cached raw file checksum mismatch: {path}")
        # Unmanifested raw files are never silently adopted as pinned source.
        if prior:
            return {**prior, "status": "verified_cache"}
    else:
        payload = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "public360-causal-research/1"})
            with urlopen(request, timeout=90) as response:
                payload = response.read()
            if relative.endswith(".json"):
                json.loads(payload)
            _atomic_bytes(path, payload)
            record = {"path": str(path.relative_to(root)), "source_path": relative,
                      "url": url, "commit": SOURCE_COMMIT, "sha256": _sha(payload),
                      "bytes": len(payload), "status": "downloaded", "acquired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            _emit(logger, "download", **record)
            return record
        except Exception as exc:
            _emit(logger, "download_failure", url=url, attempt=attempt + 1, error=str(exc))
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 4))


def acquire_cohort(data_root, cohort, logger=None, limit_matches=None, design_lock=None):
    """Fetch only pinned metadata, license and the cohort's 3 match file types.

    A limit is explicitly a partial manifest, never a complete study cohort.
    Raw files are shared by the development union, not copied per run.
    Partial progress is committed after each file for interrupted-job resume.
    """
    names = _cohorts(cohort, design_lock)  # Guard BEFORE directory/metadata reads.
    root = Path(data_root)
    manifest_path = root / "manifests" / f"{cohort}.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if old and old.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Existing manifest belongs to another source revision")
    # Reuse records from other cohort manifests only when pinned and verified.
    prior = {}
    for path in sorted((root / "manifests").glob("*.json")):
        candidate = json.loads(path.read_text())
        if candidate.get("source_commit") == SOURCE_COMMIT:
            prior.update({r["source_path"]: r for r in candidate.get("files", [])})
    manifest = {"source_commit": SOURCE_COMMIT, "cohort": cohort, "cohorts": names,
                "adapter_version": ADAPTER_VERSION, "status": "acquiring", "files": [],
                "matches": [], "license": "StatsBomb Public Data User Agreement",
                "license_url": BASE_URL + "/LICENSE.pdf", "limit_matches": limit_matches}
    if "worldcup2022" in names:
        manifest["design_lock_sha256"] = validate_confirmation_lock(design_lock)["lock_sha256"]
        if old and old.get("design_lock_sha256") != manifest["design_lock_sha256"]:
            raise PermissionError("Confirmation cache was acquired under a different design lock")
    records = {}

    def fetch(relative):
        record = _download(root, relative, prior.get(relative), logger)
        records[relative] = record
        manifest["files"] = list(records.values())
        _atomic_bytes(manifest_path, _canonical_bytes(manifest))
        return root / record["path"]

    fetch("LICENSE.pdf")
    fetch("README.md")
    fetch("data/competitions.json")
    selected = []
    for name in names:
        comp, season, expected = COHORTS[name]
        metadata = json.loads(fetch(f"data/matches/{comp}/{season}.json").read_text())
        if len(metadata) != expected or len({m["match_id"] for m in metadata}) != expected:
            raise ValueError(f"Pinned {name} match inventory differs from expected {expected}")
        selected.extend({"match_id": m["match_id"], "cohort": name,
                         "competition_id": comp, "season_id": season} for m in sorted(metadata, key=lambda m: m["match_id"]))
    if limit_matches is not None:
        if not isinstance(limit_matches, int) or limit_matches < 1:
            raise ValueError("limit_matches must be a positive integer")
        # Interleave development years for a balanced deterministic pilot.
        if len(names) > 1:
            groups = [[m for m in selected if m["cohort"] == name] for name in names]
            selected = [m for i in range(max(map(len, groups))) for g in groups for m in g[i:i+1]]
        selected = selected[:limit_matches]
    manifest["matches"] = selected
    for match in selected:
        for kind in ["events", "lineups", "three-sixty"]:
            fetch(f"data/{kind}/{match['match_id']}.json")
    manifest["status"] = "partial" if limit_matches is not None and len(selected) < sum(COHORTS[n][2] for n in names) else "complete"
    manifest["raw_bytes"] = sum(r["bytes"] for r in records.values())
    manifest["data_sha256"] = _sha(_canonical_bytes({r["source_path"]: r["sha256"] for r in records.values()}))
    manifest["manifest_sha256"] = _sha(_canonical_bytes(manifest))
    _atomic_bytes(manifest_path, _canonical_bytes(manifest))
    _emit(logger, "cohort_acquired", cohort=cohort, matches=len(selected), status=manifest["status"], raw_bytes=manifest["raw_bytes"])
    return manifest


def metric_xy(location):
    if not isinstance(location, (list, tuple)) or len(location) < 2:
        raise ValueError("Location needs two finite coordinates")
    x, y = map(float, location[:2])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Nonfinite coordinates")
    return 105 * x / 120 - 52.5, 34 - 68 * y / 80


def time_seconds(timestamp):
    hours, minutes, seconds = map(float, timestamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    t = min(1., max(0., ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length)) if length else 0.
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def visible_polygon(raw):
    """Validate a finite, nondegenerate, simple polygon (closing point optional)."""
    if not isinstance(raw, list) or len(raw) < 6 or len(raw) % 2:
        raise ValueError("Invalid visible polygon")
    points = [metric_xy(raw[i:i + 2]) for i in range(0, len(raw), 2)]
    if points[0] == points[-1]:
        points.pop()
    if len(set(points)) != len(points) or len(points) < 3:
        raise ValueError("Repeated/insufficient polygon vertices")
    area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
    if abs(area2) < 1e-8:
        raise ValueError("Degenerate polygon")
    edges = list(zip(points, points[1:] + points[:1]))
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if j <= i or j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            crossing = (_orient(a, b, c) * _orient(a, b, d) < 0 and
                        _orient(c, d, a) * _orient(c, d, b) < 0)
            touching = min(_segment_distance(a, c, d), _segment_distance(b, c, d),
                           _segment_distance(c, a, b), _segment_distance(d, a, b)) < 1e-9
            if crossing or touching:
                raise ValueError("Self-intersecting polygon")
    return points


def disk_visible(point, polygon, radius=3.):
    inside = False
    distances = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > point[1]) != (b[1] > point[1]):
            crossing = a[0] + (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1])
            if point[0] < crossing:
                inside = not inside
        distances.append(_segment_distance(point, a, b))
    return inside and min(distances) + 1e-9 >= radius


def canonical_goals(events):
    """One ordinary/own score, with source and occurrence timing provenance.

    Own Goal For is authoritative when present. Goal Keeper rows supply time,
    never an extra score. Missing For may be recovered from Against using the
    opponent team inferred from the match's event team vocabulary.
    """
    by_id = {e["id"]: e for e in events}
    teams = {e.get("team", {}).get("id") for e in events} - {None}
    goals, seen = [], set()
    for e in sorted(events, key=lambda e: (e["period"], time_seconds(e["timestamp"]), e["index"])):
        typ = e["type"]["name"]
        shot = e.get("shot", {})
        own = typ in {"Own Goal For", "Own Goal Against"}
        if not own and not (typ == "Shot" and shot.get("outcome", {}).get("name") == "Goal"):
            continue
        linked = [by_id[i] for i in e.get("related_events", []) if i in by_id]
        if typ == "Own Goal Against" and any(q["type"]["name"] == "Own Goal For" for q in linked):
            continue
        identity = frozenset([e["id"]] + [q["id"] for q in linked if q["type"]["name"] in {"Own Goal For", "Own Goal Against"}]) if own else frozenset([e["id"]])
        if identity in seen:
            continue
        seen.add(identity)
        stamp, index, timing = time_seconds(e["timestamp"]), e["index"], "own_goal_event" if own else "shot_start"
        team = e["team"]["id"]
        if typ == "Own Goal Against":
            other = teams - {team}
            if len(other) != 1:
                raise ValueError("Cannot attribute unpaired own goal")
            team = other.pop()
        if not own:
            concession = [q for q in linked if q["period"] == e["period"] and
                          q.get("goalkeeper", {}).get("type", {}).get("name") in {"Goal Conceded", "Penalty Conceded"} and
                          time_seconds(q["timestamp"]) >= stamp]
            if concession:
                occurrence = min(concession, key=lambda q: (time_seconds(q["timestamp"]), q["index"]))
                stamp, index, timing = time_seconds(occurrence["timestamp"]), occurrence["index"], "linked_concession"
            else:
                duration = e.get("duration")
                if isinstance(duration, (float, int)) and math.isfinite(duration) and duration >= 0:
                    stamp += duration
                    timing = "shot_duration"
        goals.append({"event_id": e["id"], "period": e["period"], "time_seconds": stamp,
                      "index": index, "team_id": team, "timing_source": timing,
                      "shot_start_seconds": time_seconds(e["timestamp"]),
                      "shot_start_index": e["index"], "source_ids": sorted(identity)})
    return sorted(goals, key=lambda g: (g["period"], g["time_seconds"], g["index"]))


def period_observation(events):
    observation = {}
    for period in range(1, 5):
        rows = [e for e in events if e["period"] == period]
        if not rows:
            continue
        ends = [e for e in rows if e["type"]["name"] == "Half End"]
        abnormal = any(e.get("half_end", {}).get("early_video_end") or e.get("half_end", {}).get("match_suspended")
                       or e.get("half_end", {}).get("early-video-end") or e.get("half_end", {}).get("match-suspended") for e in ends)
        normal = bool(ends) and not abnormal
        observation[period] = {"complete": normal,
                               "end": max(time_seconds(e["timestamp"]) for e in ends) if ends else None,
                               "end_index": max(e["index"] for e in ends) if ends else None,
                               "reason": "normal_end" if normal else "abnormal_end" if ends else "missing_end"}
    return observation


def signed_first_goal(event, goals, observation, horizon):
    """Natural period end is absorbing; incomplete periods remain unknown."""
    period, stamp, index = event["period"], time_seconds(event["timestamp"]), event["index"]
    obs = observation.get(period, {})
    if not obs.get("complete") or stamp > obs["end"]:
        return math.nan, None, "incomplete_period"
    end = min(stamp + horizon, obs["end"])
    future = [g for g in goals if g["period"] == period and (g["time_seconds"], g["index"]) >= (stamp, index)
              and g["time_seconds"] <= end]
    if future:
        goal = min(future, key=lambda g: (g["time_seconds"], g["index"]))
        return (1. if goal["team_id"] == event["team"]["id"] else -1.), goal["event_id"], goal["timing_source"]
    return 0., None, "period_absorbing" if stamp + horizon > obs["end"] else "complete_window"


def _goal_angle(x, y):
    a = (52.5 - x, 3.66 - y)
    b = (52.5 - x, -3.66 - y)
    return math.degrees(math.atan2(abs(a[0] * b[1] - a[1] * b[0]), a[0] * b[0] + a[1] * b[1]))


def build_match_decisions(events, frames, match, cohort, visibility_radius=3., region="L1"):
    """Pure, fixture-testable adapter. Returns candidates and a match audit.

    History is strictly earlier by native period/time, excluding tied-clock
    current occasion records. Scores use occurrence time and event-index order.
    Current action type enters only T/sensitivity metadata, never history.
    """
    if region not in {"L1", "L2"} or visibility_radius not in {3., 5., 10.}:
        raise ValueError("Prespecified regions L1/L2 and radii 3/5/10 only")
    ids = [e["id"] for e in events]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate event UUIDs")
    frame_ids = [f["event_uuid"] for f in frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Duplicate 360 UUIDs")
    if set(frame_ids) - set(ids):
        raise ValueError("360 UUID does not match an event")
    by_frame = {f["event_uuid"]: f for f in frames}
    ordered = sorted(events, key=lambda e: (e["period"], time_seconds(e["timestamp"]), e["index"]))
    goals, observation = canonical_goals(ordered), period_observation(ordered)
    events_by_id = {e["id"]: e for e in ordered}
    timing_ledgers = {"shot_start": [], "duration": []}
    for goal in goals:
        source = events_by_id[goal["event_id"]]
        start = goal["shot_start_seconds"]
        duration = source.get("duration")
        valid_duration = isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0
        # Own goals keep their recorded occurrence time in both sensitivities.
        delta = duration if valid_duration and source["type"]["name"] == "Shot" else 0.
        for name, value in [("shot_start", start), ("duration", start + delta)]:
            timing_ledgers[name].append({**goal, "time_seconds": value, "index": goal["shot_start_index"],
                                         "timing_source": f"sensitivity_{name}"})
    for goal in goals:
        obs = observation.get(goal["period"], {})
        if obs.get("complete") and goal["time_seconds"] > obs["end"]:
            obs.update(complete=False, reason="goal_after_recorded_period_end")
    home, away = match["home_team"]["home_team_id"], match["away_team"]["away_team_id"]
    expected = {home: match.get("home_score"), away: match.get("away_score")}
    observed = {team: sum(g["team_id"] == team and g["period"] <= 4 for g in goals) for team in [home, away]}
    score_reconciled = all(expected[t] is None or expected[t] == observed[t] for t in expected)
    groups = {}
    for e in ordered:
        if e["type"]["name"] not in ACTION_TYPES:
            continue
        key = (e["period"], e["timestamp"], e["team"]["id"], e.get("player", {}).get("id"))
        # A missing player must not merge unrelated anonymous events.
        if key[-1] is None:
            key += (e["id"],)
        groups.setdefault(key, []).append(e)
    rows = []
    for group in groups.values():
        e = min(group, key=lambda q: q["index"])
        types = {q["type"]["name"] for q in group}
        # Choose a consistent earliest framed record for equivalent controls.
        framed = [q for q in group if q["id"] in by_frame]
        if framed:
            e = min(framed, key=lambda q: q["index"])
        typ, stamp, period, team = e["type"]["name"], time_seconds(e["timestamp"]), e["period"], e["team"]["id"]
        opponent = away if team == home else home
        row = {"match_id": match["match_id"], "original_match_id": match["match_id"],
               "event_id": e["id"], "source_event_ids": sorted(q["id"] for q in group),
               "event_index": e["index"], "team_id": str(team), "opponent_id": str(opponent),
               "player_id": e.get("player", {}).get("id"), "cohort": cohort,
               "T": int("Shot" in types), "action_type": typ, "action_types": sorted(types),
               "contains_carry": "Carry" in types, "period": period, "time_seconds": stamp,
               "position_role": e.get("position", {}).get("name", "Unknown"),
               "eligible": False, "drop_reason": "", "visibility_radius": visibility_radius,
               "Y15": math.nan, "Y30": math.nan, "x": math.nan, "y": math.nan,
               "distance": math.nan, "angle": math.nan, "defenders_3m": math.nan,
               "nearest_defender": math.nan, "second_defender": math.nan}
        reasons = []
        if "Shot" in types and len(types) > 1:
            reasons.append("ambiguous_treatment_collision")
        if len(group) > 1:
            try:
                if any(math.dist(metric_xy(q.get("location")), metric_xy(e.get("location"))) > 1. for q in group):
                    reasons.append("inconsistent_occasion_coordinates")
            except (ValueError, TypeError):
                reasons.append("inconsistent_occasion_coordinates")
        if period not in range(1, 5):
            reasons.append("shootout_or_unknown_period")
        if team not in {home, away}:
            reasons.append("unknown_actor_team")
        if any(q["type"]["name"] == "Shot" and q.get("shot", {}).get("type", {}).get("name") in {"Penalty", "Free Kick", "Corner", "Kick Off"}
               or q["type"]["name"] == "Pass" and q.get("pass", {}).get("type", {}).get("name") in RESTART_PASSES for q in group):
            reasons.append("direct_restart")
        try:
            x, y = metric_xy(e.get("location"))
            row.update(x=x, y=y, distance=math.hypot(52.5 - x, y), angle=_goal_angle(x, y))
            if x < 17.5 or (region == "L2" and not (abs(y) >= 9.16 or x <= 36)):
                reasons.append("outside_region")
        except (ValueError, TypeError):
            reasons.append("invalid_event_coordinates")
        frame = by_frame.get(e["id"])
        if frame is None:
            reasons.append("missing_360")
        else:
            try:
                ff = frame["freeze_frame"]
                actors = [q for q in ff if q.get("actor")]
                if len(actors) != 1 or actors[0].get("teammate") is not True:
                    reasons.append("invalid_actor")
                elif math.dist(metric_xy(actors[0]["location"]), (row["x"], row["y"])) > 1.:
                    reasons.append("actor_coordinate_mismatch")
                locations = [(q, metric_xy(q["location"])) for q in ff]
                if any(not isinstance(q.get("teammate"), bool) for q in ff):
                    raise ValueError("Unknown teammate status")
                polygon = visible_polygon(frame.get("visible_area"))
                for radius in [3, 5, 10]:
                    row[f"visible_disk_{radius}m"] = disk_visible((row["x"], row["y"]), polygon, radius)
                if not row[f"visible_disk_{int(visibility_radius)}m"]:
                    reasons.append("incomplete_local_visibility")
                distances = sorted(math.dist(p, (row["x"], row["y"])) for q, p in locations if not q["teammate"])
                row["defenders_3m"] = sum(d <= 3. + 1e-9 for d in distances)
                row["nearest_defender"] = min(distances[0], 3.) if distances else 3.
                row["second_defender"] = min(distances[1], 3.) if len(distances) > 1 else 3.
                row["visible_attackers"] = sum(q["teammate"] for q in ff)
                row["visible_defenders"] = len(ff) - row["visible_attackers"]
                keepers = [p for q, p in locations if q.get("keeper") and not q["teammate"]]
                row["opposing_keeper_visible"] = bool(keepers)
                row["observed_keeper_x"] = keepers[0][0] if keepers else math.nan
                row["observed_keeper_y"] = keepers[0][1] if keepers else math.nan
                row["observed_teammates_ahead"] = sum(q["teammate"] and not q.get("actor") and p[0] > row["x"] for q, p in locations)
                for radius in [5, 10]:
                    row[f"defenders_{radius}m"] = sum(d <= radius for d in distances) if row[f"visible_disk_{radius}m"] else math.nan
                    row[f"nearest_defender_{radius}m"] = min(distances[0], radius) if distances and row[f"visible_disk_{radius}m"] else float(radius) if row[f"visible_disk_{radius}m"] else math.nan
            except (ValueError, TypeError, KeyError):
                reasons.append("invalid_frame_geometry")
        before = [g for g in goals if (g["period"], g["time_seconds"], g["index"]) < (period, stamp, e["index"])]
        row["score_difference"] = sum(1 if g["team_id"] == team else -1 for g in before)
        history_all, seen_history = [], set()
        for q in ordered:
            if q["type"]["name"] not in ACTION_TYPES or q["team"]["id"] != team:
                continue
            if (q["period"], time_seconds(q["timestamp"])) >= (period, stamp):
                continue
            key = (q["period"], q["timestamp"], q.get("player", {}).get("id"))
            if key in seen_history:
                continue
            seen_history.add(key)
            history_all.append(q)
        history = [q for q in history_all if q["period"] == period]
        recent = [q for q in history if stamp - time_seconds(q["timestamp"]) <= 10.]
        row["history_time_since_prev_action"] = stamp - time_seconds(history[-1]["timestamp"]) if history else math.nan
        row["history_prior_10s_actions"] = len(recent)
        row["history_missing_previous"] = not bool(history)
        row["history_prior_match_actions"] = len(history_all)
        row["history_progress_missing"] = False
        try:
            row["history_prior_10s_progress"] = row["x"] - metric_xy(recent[0]["location"])[0] if recent else 0.
        except (ValueError, TypeError, KeyError):
            row["history_prior_10s_progress"] = math.nan
            row["history_progress_missing"] = True
        for alias in ["time_since_prev_action", "prior_10s_actions", "prior_10s_progress"]:
            row[alias] = row["history_" + alias]
        for horizon in [5, 10, 15, 30]:
            reward, gid, timing = signed_first_goal(e, goals, observation, horizon)
            if not score_reconciled:
                reward, timing = math.nan, "score_ledger_mismatch"
            row[f"Y{horizon}"] = reward
            row[f"goal_event_{horizon}"] = gid
            row[f"goal_timing_{horizon}"] = timing
            obs = observation.get(period, {})
            row[f"full_horizon_{horizon}"] = bool(obs.get("complete") and stamp + horizon <= obs["end"])
        for timing, ledger in timing_ledgers.items():
            reward, gid, provenance = signed_first_goal(e, ledger, observation, 15)
            row[f"Y15_{timing}"] = reward if score_reconciled else math.nan
            row[f"goal_event_15_{timing}"] = gid
            row[f"goal_timing_15_{timing}"] = provenance if score_reconciled else "score_ledger_mismatch"
        if not math.isfinite(row["Y15"]):
            reasons.append("unknown_outcome")
        row["eligible"] = not reasons
        row["drop_reason"] = "|".join(dict.fromkeys(reasons))
        rows.append(row)
    audit = {"match_id": match["match_id"], "source_events": len(events), "frames": len(frames),
             "candidate_rows": len(rows), "eligible_rows": sum(r["eligible"] for r in rows),
             "score_reconciled": score_reconciled, "recorded_score": expected,
             "ledger_score": observed, "goals": goals, "period_observation": observation}
    return rows, audit


def build_decisions(data_root, cohort, logger=None, design_lock=None, visibility_radius=3., region="L1", use_cache=True):
    """Verify every raw checksum, then materialize a hash-keyed canonical table."""
    _cohorts(cohort, design_lock)  # Guard also covers cached confirmation tables.
    import pandas as pd
    root = Path(data_root)
    path = root / "manifests" / f"{cohort}.json"
    if not path.exists():
        raise FileNotFoundError(f"Acquire the cohort first: {path}")
    manifest = json.loads(path.read_text())
    if "worldcup2022" in manifest.get("cohorts", []) and manifest.get("design_lock_sha256") != validate_confirmation_lock(design_lock)["lock_sha256"]:
        raise PermissionError("Confirmation manifest does not match the frozen development lock")
    if manifest.get("source_commit") != SOURCE_COMMIT or manifest.get("status") not in {"complete", "partial"}:
        raise ValueError("Source manifest is unpinned or incomplete acquisition")
    manifest_body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _sha(_canonical_bytes(manifest_body)):
        raise ValueError("Source manifest checksum mismatch")
    records_hash = _sha(_canonical_bytes({r["source_path"]: r["sha256"] for r in manifest["files"]}))
    if records_hash != manifest["data_sha256"]:
        raise ValueError("Source data manifest fingerprint mismatch")
    for record in manifest["files"]:
        payload = (root / record["path"]).read_bytes()
        if _sha(payload) != record["sha256"] or len(payload) != record["bytes"]:
            raise ValueError(f"Raw checksum mismatch: {record['path']}")
    key = _sha(_canonical_bytes({"data": manifest["data_sha256"], "adapter": ADAPTER_VERSION,
                                "adapter_code_sha256": _sha(Path(__file__).read_bytes()),
                                "radius": visibility_radius, "region": region}))
    cache = root / "canonical" / f"{cohort}-{key}.pkl"
    cache_meta = cache.with_suffix(".meta.json")
    if use_cache and cache.exists():
        # Only locally created canonical cache is read, never remote pickle.
        if not cache_meta.exists():
            raise ValueError("Canonical cache is missing checksum metadata")
        meta = json.loads(cache_meta.read_text())
        if meta.get("canonical_key") != key or meta.get("sha256") != _sha(cache.read_bytes()):
            raise ValueError("Canonical cache checksum mismatch")
        frame = pd.read_pickle(cache)
        if frame.attrs.get("canonical_key") != key:
            raise ValueError("Canonical cache fingerprint mismatch")
        return frame
    rows, audits = [], []
    metadata = {}
    for name in manifest["cohorts"]:
        comp, season, _ = COHORTS[name]
        metadata.update({m["match_id"]: m for m in json.loads((root / "raw" / "data" / "matches" / str(comp) / f"{season}.json").read_text())})
    for selected in manifest["matches"]:
        mid = selected["match_id"]
        events = json.loads((root / "raw" / "data" / "events" / f"{mid}.json").read_text())
        frames = json.loads((root / "raw" / "data" / "three-sixty" / f"{mid}.json").read_text())
        match_rows, audit = build_match_decisions(events, frames, metadata[mid], selected["cohort"], visibility_radius, region)
        rows.extend(match_rows)
        audits.append(audit)
        _emit(logger, "match_canonicalized", match_id=mid, candidate_rows=len(match_rows), eligible_rows=audit["eligible_rows"])
    frame = pd.DataFrame(rows)
    frame.attrs.update(canonical_key=key, source_commit=SOURCE_COMMIT, data_sha256=manifest["data_sha256"],
                       adapter_version=ADAPTER_VERSION, cohort=cohort, audits=audits,
                       estimand="signed first goal before horizon or natural period termination")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        frame.to_pickle(tmp)
        os.replace(tmp, cache)
    finally:
        tmp.unlink(missing_ok=True)
    _atomic_bytes(cache.with_suffix(".audit.json"), _canonical_bytes({"canonical_key": key, "matches": audits}))
    _atomic_bytes(cache_meta, _canonical_bytes({"canonical_key": key, "sha256": _sha(cache.read_bytes())}))
    return frame
