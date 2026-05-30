from __future__ import annotations

import json
from pathlib import Path

from data import BallDriveDataConfig
from trainer import BallDriveTrainer, BallDriveTrainingConfig


def main() -> None:
    data_config = BallDriveDataConfig()
    train_config = BallDriveTrainingConfig(epochs=3)
    trainer = BallDriveTrainer(data_config=data_config, train_config=train_config)

    report = trainer.run()

    out_path = Path(__file__).resolve().parents[2] / "results" / "metrics" / "ball_drive_smoke_train.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps({"status": "ok", "report_path": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
