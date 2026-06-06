import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ActionSelection.trainer import train_action_selection_from_sources, test_action_selection_from_sources
else:
    from .trainer import train_action_selection_from_sources, test_action_selection_from_sources


def Train() -> None:
    result = train_action_selection_from_sources()
    print(json.dumps(result, indent=2, default=str))


def Test() -> None:
    result = test_action_selection_from_sources()
    print(json.dumps(result, indent=2, default=str))


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