import random
from collections import Counter, defaultdict


class DataSplitter:
    """Split CIFAR records into deterministic IID ranges or label-skew indices."""

    def split(
        self,
        segments: list[dict],
        client_count: int,
        seed: int,
        *,
        distribution: str = "iid",
        labels_by_file: dict[str, list[int]] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        if distribution == "non_iid":
            if labels_by_file is None:
                raise ValueError("Non-IID 분할에는 CIFAR-10 레이블이 필요합니다.")
            return self._split_by_label(segments, client_count, seed, labels_by_file)
        if distribution != "iid":
            raise ValueError("데이터 분포는 iid 또는 non_iid여야 합니다.")

        rng = random.Random(seed)
        ranges_by_client = [[] for _ in range(client_count)]

        for segment in segments:
            count = segment["end"] - segment["start"]
            boundaries = [0]
            for _ in range(client_count - 1):
                boundaries.append(rng.randint(0, count))
            boundaries.append(count)
            boundaries.sort()

            for client_id in range(client_count):
                start = segment["start"] + boundaries[client_id]
                end = segment["start"] + boundaries[client_id + 1]
                if start < end:
                    ranges_by_client[client_id].append(
                        {"file": segment["file"], "start": start, "end": end}
                    )
        return (
            [{"ranges": ranges} for ranges in ranges_by_client],
            [
                {
                    "client_id": client_id,
                    "num_samples": sum(item["end"] - item["start"] for item in ranges),
                    "class_counts": None,
                }
                for client_id, ranges in enumerate(ranges_by_client)
            ],
        )

    @staticmethod
    def _split_by_label(
        segments: list[dict],
        client_count: int,
        seed: int,
        labels_by_file: dict[str, list[int]],
    ) -> tuple[list[dict], list[dict]]:
        rng = random.Random(seed)
        assigned: list[dict[str, list[int]]] = [
            defaultdict(list) for _ in range(client_count)
        ]
        counts = [Counter() for _ in range(client_count)]
        owners_per_class = min(client_count, max(1, round(client_count * 2 / 10)))

        for segment in segments:
            file_name = segment["file"]
            labels = labels_by_file[file_name]
            start, end = segment["start"], segment["end"]
            if end > len(labels):
                raise ValueError(f"{file_name}의 레이블 수와 데이터 범위가 맞지 않습니다.")
            for label in range(10):
                indices = [
                    index
                    for index in range(start, end)
                    if labels[index] == label
                ]
                rng.shuffle(indices)
                owners = [
                    (label + offset) % client_count
                    for offset in range(owners_per_class)
                ]
                for position, sample_index in enumerate(indices):
                    client_id = owners[position % len(owners)]
                    assigned[client_id][file_name].append(sample_index)
                    counts[client_id][label] += 1

        assignments = []
        class_counts = []
        for client_id, files in enumerate(assigned):
            assignments.append(
                {
                    "indices_by_file": [
                        {"file": name, "indices": sorted(indices)}
                        for name, indices in sorted(files.items())
                        if indices
                    ]
                }
            )
            sample_counts = {str(label): counts[client_id][label] for label in range(10)}
            class_counts.append(
                {
                    "client_id": client_id,
                    "num_samples": sum(sample_counts.values()),
                    "class_counts": sample_counts,
                }
            )
        return assignments, class_counts
