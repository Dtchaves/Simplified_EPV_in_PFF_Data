import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Pass.Pass_sucess_probability.trainer import Train
    from Pass.Pass_sucess_probability.test import Test
else:
    from .trainer import Train
    from .test import Test


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


