"""검증으로 확정한 두 CNN 구성의 성능을 기존 엑셀 양식에 기록한다."""

from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.trial_candidates import FINAL_SEEDS


TEMPLATE = Path(__file__).resolve().parents[1] / "CNN_Experiment_Plan.xlsx"
REPORT_NAME = "CNN_Results.xlsx"
MODEL_IDS = {"SmallCNN (E03)": "E03_smallcnn", "4층 CNN (E04)": "E04_cnn4"}


def load_cnn_workbook():
    from openpyxl import load_workbook

    return load_workbook(TEMPLATE)


def save_cnn_workbook(book, summaries: dict, stage_labels: dict, output_dir: Path):
    validation = book["검증 기록"]
    final = book["최종 비교"]
    final["A1"] = "각자 튜닝한 두 구성의 성능 비교  |  CIFAR-10 공식 테스트"
    final["A20"] = "각 모델의 검증 결과로 확정한 구성입니다. 성능 차이에는 구조와 선택된 학습 설정이 함께 반영됩니다."

    candidates = {}
    tests = {}
    for model_id, summary in summaries.items():
        for stage in summary["stages"].values():
            for candidate in stage["candidates"]:
                candidates[model_id, candidate["experiment_id"]] = candidate
        rows = summary["official_tests"]
        if sorted(row["seed"] for row in rows) != sorted(FINAL_SEEDS):
            raise ValueError(f"{model_id}의 공식 테스트에는 세 시드의 결과가 각각 한 번씩 필요합니다.")
        for row in rows:
            tests[model_id, row["seed"]] = row

    # E14 이후에는 두 모델의 실험 ID가 같으므로 항상 모델과 ID를 함께 사용한다.
    for name, _, trial, _, accuracy, loss, note in validation.iter_rows(min_row=7, max_row=44, max_col=7):
        candidate = candidates[MODEL_IDS[name.value], trial.value]
        accuracy.value = candidate["validation_accuracy"]
        loss.value = candidate["validation_loss"]
        if candidate.get("status") == "failed":
            note.value = f"실패 · {candidate['last_completed_round']}라운드 완료 · {candidate['reason']}"

    stage_ids = {label: stage for stage, label in stage_labels.items()}
    for name, label, chosen in validation.iter_rows(min_row=49, max_row=58, max_col=3):
        summary = summaries[MODEL_IDS[name.value]]
        chosen.value = summary["stages"][stage_ids[label.value]]["selected"]

    for name, seed, _, _, accuracy, loss in final.iter_rows(min_row=7, max_row=12, max_col=6):
        result = tests[MODEL_IDS[name.value], seed.value]
        accuracy.value, loss.value = result["accuracy"], result["loss"]
        if result.get("status") == "failed":
            final["G6"] = "실패 사유"
            final.column_dimensions["G"].width = 68
            final.cell(seed.row, 7, result["reason"])

    for name, rate, augmentation, schedule, dropout, optimizer in final.iter_rows(min_row=22, max_row=23, max_col=6):
        settings = summaries[MODEL_IDS[name.value]]["final_selection"]["settings"]
        training = settings["training"]
        rate.value = training["learning_rate"]
        augmentation.value = training["augmentation"]
        schedule.value = f"{settings['num_rounds']} × {training['local_epochs']}"
        dropout.value = settings["model_config"]["dropout"]
        optimizer.value = training["optimizer"].upper()

    book.calculation.calcMode = "auto"
    book.calculation.fullCalcOnLoad = True
    path = output_dir / REPORT_NAME
    with TemporaryDirectory(dir=output_dir) as directory:
        pending = Path(directory) / REPORT_NAME
        book.save(pending)
        pending.replace(path)
    return path
