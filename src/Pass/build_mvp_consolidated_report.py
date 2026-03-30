from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def _to_rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _build_pp_ps_checkpoint() -> Dict[str, Any]:
    required_files = [
        ROOT / "results/heatmaps/pass_success_checkpoint/pp_example_input_ch1.png",
        ROOT / "results/heatmaps/pass_success_checkpoint/pp_example_surface.png",
        ROOT / "results/heatmaps/pass_selection_checkpoint/ps_example_input_ch1.png",
        ROOT / "results/heatmaps/pass_selection_checkpoint/ps_example_surface.png",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    status = "CHECKPOINT_AUDIT_OK" if not missing else "CHECKPOINT_AUDIT_MISSING_ARTIFACTS"

    return {
        "name": "pp_ps_checkpoint",
        "status": status,
        "artifacts": {
            "pass_success_heatmaps": _to_rel(ROOT / "results/heatmaps/pass_success_checkpoint"),
            "pass_selection_heatmaps": _to_rel(ROOT / "results/heatmaps/pass_selection_checkpoint"),
            "required_files_present": len(required_files) - len(missing),
            "required_files_total": len(required_files),
            "missing_files": missing,
        },
    }


def _build_report() -> Dict[str, Any]:
    reward_summary_path = ROOT / "results/metrics/reward_checkpoint/reward_label_summary.json"
    pe_success_summary_path = ROOT / "results/metrics/pe_success_checkpoint/pe_success_summary.json"
    pe_missed_summary_path = ROOT / "results/metrics/pe_missed_checkpoint/pe_missed_summary.json"
    pe_pressure_summary_path = ROOT / "results/metrics/pe_pressure_lines_checkpoint/pressure_lines_summary.json"
    integration_summary_path = ROOT / "results/metrics/final_integration_checkpoint/integration_summary.json"
    smoke_summary_path = ROOT / "results/metrics/final_smoke_checkpoint/smoke_summary.json"

    for path, label in (
        (reward_summary_path, "reward checkpoint summary"),
        (pe_success_summary_path, "PE-success summary"),
        (pe_missed_summary_path, "PE-missed summary"),
        (pe_pressure_summary_path, "PE pressure-lines summary"),
        (integration_summary_path, "final integration summary"),
        (smoke_summary_path, "final smoke summary"),
    ):
        _require_exists(path, label)

    pp_ps_checkpoint = _build_pp_ps_checkpoint()
    reward_checkpoint = _load_json(reward_summary_path)
    if "status" not in reward_checkpoint:
        reward_checkpoint["status"] = "REWARD_CHECKPOINT_AUDIT_OK"
    pe_success_checkpoint = _load_json(pe_success_summary_path)
    pe_missed_checkpoint = _load_json(pe_missed_summary_path)
    pe_pressure_checkpoint = _load_json(pe_pressure_summary_path)
    integration_checkpoint = _load_json(integration_summary_path)
    smoke_checkpoint = _load_json(smoke_summary_path)

    status_values = [
        pp_ps_checkpoint.get("status", ""),
        str(reward_checkpoint.get("status", "REWARD_CHECKPOINT_AUDIT_OK")),
        str(pe_success_checkpoint.get("status", "")),
        str(pe_missed_checkpoint.get("status", "")),
        str(pe_pressure_checkpoint.get("status", "")),
        str(integration_checkpoint.get("status", "")),
        str(smoke_checkpoint.get("status", "")),
    ]
    overall_status = "MVP_CHECKPOINTS_COMPLETE" if all("OK" in s for s in status_values) else "MVP_CHECKPOINTS_INCOMPLETE"

    return {
        "report_name": "pass_epv_mvp_consolidated_report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "scope": {
            "included": ["PS", "PP", "PE-success", "PE-missed", "shared reward labels", "PP->PE integration", "final smoke regression"],
            "excluded": ["ball drive", "shot", "full calibration workflows"],
        },
        "checkpoint_order": [
            "pp_ps_checkpoint",
            "reward_checkpoint",
            "pe_success_checkpoint",
            "pe_missed_checkpoint",
            "pe_pressure_lines_checkpoint",
            "final_integration_checkpoint",
            "final_smoke_checkpoint",
        ],
        "checkpoints": {
            "pp_ps_checkpoint": pp_ps_checkpoint,
            "reward_checkpoint": reward_checkpoint,
            "pe_success_checkpoint": pe_success_checkpoint,
            "pe_missed_checkpoint": pe_missed_checkpoint,
            "pe_pressure_lines_checkpoint": pe_pressure_checkpoint,
            "final_integration_checkpoint": integration_checkpoint,
            "final_smoke_checkpoint": smoke_checkpoint,
        },
        "highlights": {
            "reward_rows_labeled_total": reward_checkpoint.get("rows_labeled_total"),
            "reward_label_counts": reward_checkpoint.get("label_counts"),
            "pe_success_rows_after_filters": pe_success_checkpoint.get("filter_summary", {}).get("rows_after_filters"),
            "pe_missed_rows_after_filters": pe_missed_checkpoint.get("filter_summary", {}).get("rows_after_filters"),
            "integration_wiring_max_diff": integration_checkpoint.get("wiring_checks", {}),
            "smoke_component_eval_losses": {
                "PP": smoke_checkpoint.get("component_metrics", {}).get("PP", {}).get("eval_loss"),
                "PS": smoke_checkpoint.get("component_metrics", {}).get("PS", {}).get("eval_loss"),
                "PE_SUCCESS": smoke_checkpoint.get("component_metrics", {}).get("PE_SUCCESS", {}).get("eval_loss"),
                "PE_MISSED": smoke_checkpoint.get("component_metrics", {}).get("PE_MISSED", {}).get("eval_loss"),
            },
        },
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    checkpoints = report["checkpoints"]
    smoke = checkpoints["final_smoke_checkpoint"]
    integration = checkpoints["final_integration_checkpoint"]
    reward = checkpoints["reward_checkpoint"]

    lines: List[str] = []
    lines.append("# Pass EPV MVP Consolidated Report")
    lines.append("")
    lines.append(f"- Generated at UTC: {report['generated_at_utc']}")
    lines.append(f"- Overall status: {report['overall_status']}")
    lines.append("")
    lines.append("## Checkpoint Status")
    lines.append("")
    for key in report["checkpoint_order"]:
        status = checkpoints[key].get("status", "UNKNOWN")
        lines.append(f"- {key}: {status}")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    lines.append(f"- Reward labeled rows: {reward.get('rows_labeled_total')}")
    lines.append(f"- Reward label counts: {reward.get('label_counts')}")
    lines.append(
        f"- Integration wiring max diffs: {integration.get('wiring_checks')}"
    )
    lines.append(
        "- Smoke eval losses: "
        + str(
            {
                "PP": smoke.get("component_metrics", {}).get("PP", {}).get("eval_loss"),
                "PS": smoke.get("component_metrics", {}).get("PS", {}).get("eval_loss"),
                "PE_SUCCESS": smoke.get("component_metrics", {}).get("PE_SUCCESS", {}).get("eval_loss"),
                "PE_MISSED": smoke.get("component_metrics", {}).get("PE_MISSED", {}).get("eval_loss"),
            }
        )
    )
    lines.append("")
    lines.append("## Main Artifacts")
    lines.append("")
    lines.append("- results/metrics/reward_checkpoint/reward_label_summary.json")
    lines.append("- results/metrics/pe_success_checkpoint/pe_success_summary.json")
    lines.append("- results/metrics/pe_missed_checkpoint/pe_missed_summary.json")
    lines.append("- results/metrics/pe_pressure_lines_checkpoint/pressure_lines_summary.json")
    lines.append("- results/metrics/final_integration_checkpoint/integration_summary.json")
    lines.append("- results/metrics/final_smoke_checkpoint/smoke_summary.json")
    lines.append("")
    lines.append("## Consolidated Outputs")
    lines.append("")
    lines.append("- results/metrics/mvp_consolidated_report.json")
    lines.append("- results/metrics/mvp_consolidated_report.md")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = _build_report()

    json_output_path = ROOT / "results/metrics/mvp_consolidated_report.json"
    md_output_path = ROOT / "results/metrics/mvp_consolidated_report.md"

    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    with json_output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    with md_output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(_render_markdown(report))

    print("MVP_CONSOLIDATED_REPORT_OK")
    print(f"json={json_output_path}")
    print(f"markdown={md_output_path}")


if __name__ == "__main__":
    main()