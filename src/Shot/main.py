import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Shot.baseline_xg import build_baseline_xg_dataset, load_baseline_xg, train_baseline_xg
    from Shot.data import build_shot_epv_dataset
    from Shot.trainer import ShotTrainer, test_shot_model
else:
    from .baseline_xg import build_baseline_xg_dataset, load_baseline_xg, train_baseline_xg
    from .data import build_shot_epv_dataset
    from .trainer import ShotTrainer, test_shot_model


def _ensure_baseline_xg_artifacts():
    try:
        return load_baseline_xg()
    except FileNotFoundError:
        dataset = build_baseline_xg_dataset()
        return train_baseline_xg(dataset)


def Train() -> None:
    baseline_xg_artifacts = _ensure_baseline_xg_artifacts()
    dataset, dataset_summary = build_shot_epv_dataset(baseline_xg_artifacts=baseline_xg_artifacts)
    train_summary = ShotTrainer().run(dataset)
    print(json.dumps({"dataset_summary": dataset_summary, "train_summary": train_summary}, indent=2, default=str))


def Test() -> None:
    baseline_xg_artifacts = _ensure_baseline_xg_artifacts()
    dataset, dataset_summary = build_shot_epv_dataset(baseline_xg_artifacts=baseline_xg_artifacts)
    test_summary = test_shot_model(dataset)
    print(json.dumps({"dataset_summary": dataset_summary, "test_summary": test_summary}, indent=2, default=str))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        Train()
    elif mode == "test":
        Test()
    elif mode == "train_test":
        Train()
        Test()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [train|test|train_test]")
        sys.exit(1)