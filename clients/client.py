class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset

    def train(self, global_model):
        # 전달받은 전역 모델로 로컬 학습
        local_model = global_model

        # 학습 코드
        return local_model

    def run(self, server_url):
        # 1. 서버에서 전역 모델 가져오기
        # 2. 로컬 학습
        # 3. 업데이트 제출

        pass
