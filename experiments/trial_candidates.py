"""독립 실험 러너들이 공유하는 CIFAR-10 비교 후보."""

from pathlib import PurePosixPath


LEARNING_RATES = (0.01, 0.02, 0.05, 0.10)
DROPOUT_TRIALS = {"E14": 0.1, "E15": 0.2, "E16": 0.3, "E17": 0.4}
AUGMENTATION_TRIALS = {
    "E18": "flip", "E19": "crop", "E20": "flip_crop",
    "E21": "flip_crop_jitter", "E22": "flip_crop_cutout",
}
SCHEDULE_TRIALS = {"E23": (30, 1), "E24": (65, 1), "E25": (80, 1), "E26": (40, 2)}
ADAM_TRAINING = {"optimizer": "adam", "learning_rate": 0.001, "momentum": 0.0}
FINAL_SEEDS = (101, 202, 303)
FINAL_TEST_SIZE = 10_000


def add_official_test(manifest: dict):
    test_file = PurePosixPath(manifest["test_ranges"][0]["file"]).with_name("test_batch")
    manifest["final_test_ranges"] = [{"file": test_file.as_posix(), "start": 0, "end": FINAL_TEST_SIZE}]
