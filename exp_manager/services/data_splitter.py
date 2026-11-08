import random


class DataSplitter:
    """훈련 데이터 구간을 시드 기반으로 클라이언트별 범위로 분할한다."""

    def split(self, segments: list[dict], client_count: int, seed: int) -> list[list[dict]]:
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
        return ranges_by_client
