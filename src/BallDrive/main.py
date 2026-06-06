import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from BallDrive.trainer import BallDriveTrainer, test_ball_drive_models
else:
    from .trainer import BallDriveTrainer, test_ball_drive_models


def Train() -> None:
    result = BallDriveTrainer().run()
    print(json.dumps(result, indent=2, default=str))


def Test() -> None:
    result = test_ball_drive_models()
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