async function control(path) {
  const response = await fetch(path, { method: 'POST' });
  if (!response.ok) {
    const result = await response.json();
    alert(result.error || '요청을 처리하지 못했습니다.');
  }
  await refresh();
}

async function saveExperimentConfig() {
  const response = await fetch('/api/experiment/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      experiment_id: document.getElementById('experiment-id').value,
      num_rounds: Number(document.getElementById('rounds').value),
      experiment_type: document.getElementById('experiment-type').value,
      dlg_training_rounds: Number(document.getElementById('dlg-training-rounds').value),
      use_pretrained_global_model: document.getElementById('use-pretrained-global-model').checked,
      pretrained_model_format: document.getElementById('pretrained-model-format').value,
      pretrained_model_data: document.getElementById('use-pretrained-global-model').checked
        ? document.getElementById('pretrained-model-data').value
        : '',
      dlg_iterations: Number(document.getElementById('dlg-iterations').value),
      dlg_client_id: document.getElementById('dlg-client').value === ''
        ? null
        : Number(document.getElementById('dlg-client').value),
      participation_rate: Number(document.getElementById('participation').value),
      seed: Number(document.getElementById('seed').value),
      local_epochs: Number(document.getElementById('local-epochs').value),
      batch_size: Number(document.getElementById('batch-size').value),
      learning_rate: Number(document.getElementById('learning-rate').value),
      device: document.getElementById('device').value,
      round_timeout: Number(document.getElementById('round-timeout').value),
      max_train_samples: document.getElementById('max-train-samples').value === ''
        ? null
        : Number(document.getElementById('max-train-samples').value),
      max_test_samples: document.getElementById('max-test-samples').value === ''
        ? null
        : Number(document.getElementById('max-test-samples').value),
      data_distribution: document.getElementById('distribution').value,
      dlg_noise_sigma: Number(document.getElementById('dlg-noise').value),
      dlg_defense_enabled: document.getElementById('dlg-defense-enabled').checked,
      defense: document.getElementById('defense').value,
      defense_enabled: document.getElementById('defense-enabled').checked,
      attack_enabled: document.getElementById('attack').checked,
      attack_scale: document.getElementById('attack-scale').value.trim() === ''
        ? -5
        : Number(document.getElementById('attack-scale').value),
      malicious_clients: document
        .getElementById('malicious')
        .value.split(',')
        .map((value) => Number(value.trim()))
        .filter((value) => Number.isInteger(value)),
      round_configs: readRoundConfigs(),
    }),
  });
  const savedConfig = await response.json();
  if (!response.ok) {
    alert(savedConfig.error || '설정을 저장하지 못했습니다.');
  } else if (savedConfig.pretrained_global_model_loaded) {
    document.getElementById('pretrained-model-status').textContent =
      `전역 모델 체크포인트를 저장했습니다 (버전 ${savedConfig.pretrained_global_model_version ?? 0}).`;
  }
  await refresh();
}

function stateText(value) {
  return value ? '<span class="ok">정상</span>' : '<span class="off">종료됨</span>';
}

const roundConfigsByExperimentType = {
  federated: [],
  dlg: [],
};

function applyExperimentTemplate(templateName) {
  const templates = {
    baseline: { participation: 1, delayClients: [], distribution: 'iid' },
    delay: { participation: 1, delayClients: [2, 5, 8], distribution: 'iid' },
    partial: { participation: 0.5, delayClients: [], distribution: 'iid' },
    'label-bias': { participation: 1, delayClients: [], distribution: 'non_iid' },
  };
  const template = templates[templateName];
  if (!template || document.getElementById('experiment-type').value !== 'federated') {
    return;
  }

  const savedConfigs = document.querySelector('#round-configs tr')
    ? readRoundConfigs()
    : roundConfigsByExperimentType.federated;
  buildRoundEditor(savedConfigs);

  document.getElementById('participation').value = template.participation;
  document.getElementById('distribution').value = template.distribution;
  document.querySelectorAll('#round-configs tr').forEach((row) => {
    row.querySelector('.round-participation').value = template.participation;
    row.querySelectorAll('.round-delay').forEach((checkbox) => {
      checkbox.checked = template.delayClients.includes(Number(checkbox.value));
    });
  });
  roundConfigsByExperimentType.federated = readRoundConfigs();
}

      async function refresh() {
        const response = await fetch('/api/status');
        const data = await response.json();
        const serverDetails = data.server.details || {};
        document.getElementById('server-status').innerHTML =
          `${data.server.url} · 프로세스: ${stateText(data.server.running)} · API: ${stateText(data.server.reachable)}`;
        const phaseLabels = {
          idle: '대기',
          checking_availability: '참여 가능 여부 확인',
          training: '클라이언트 학습',
          aggregating: '업데이트 집계',
          evaluating: '모델 평가',
          completed: '완료',
          failed: '실패',
        };
        const accuracy = Number.isFinite(serverDetails.last_accuracy)
          ? `${(serverDetails.last_accuracy * 100).toFixed(2)}%`
          : '—';
        const loss = Number.isFinite(serverDetails.last_loss)
          ? serverDetails.last_loss.toFixed(4)
          : '—';
        const participation = Number.isFinite(serverDetails.participation_rate)
          ? `${Math.round(serverDetails.participation_rate * 100)}%`
          : '—';
        const deadline = serverDetails.update_deadline
          ? new Date(serverDetails.update_deadline).toLocaleTimeString()
          : '—';
        document.getElementById('server-live-status').textContent = [
          `진행: ${phaseLabels[serverDetails.round_phase] || serverDetails.round_phase || '—'}`,
          `현재 라운드: ${serverDetails.current_round ?? '—'} · 모델 버전: ${serverDetails.global_model_version ?? '—'}`,
          `참여 가능: ${serverDetails.available_count ?? 0} · 선정: ${serverDetails.selected_count ?? 0} · 업데이트 수신: ${serverDetails.updates_received ?? 0}`,
          `참여율: ${participation} · 제출 마감: ${deadline}`,
          `공격: ${serverDetails.attack_enabled ? 'ON' : 'OFF'} · 방어: ${serverDetails.defense_enabled ? serverDetails.defense : 'OFF'}`,
          `지연 대상: ${(serverDetails.delayed_clients || []).join(', ') || '없음'}`,
          `최근 정확도: ${accuracy} · 손실: ${loss}`,
          serverDetails.round_message || '',
        ].join('\n');
        document.getElementById('clients').innerHTML = data.clients
          .map(
            (client) => {
              const details = client.details || {};
              const activity = client.round_activity || {};
              const serverRound = serverDetails.current_round;
              const training = details.training || {};
              const trainingMatchesRound = training.round === serverRound;
              const trainingStatus = trainingMatchesRound
                ? (activity.update_status === 'received' || activity.update_status === 'expired'
                  ? activity.update_status
                  : training.status)
                : activity.selected
                  ? 'waiting'
                  : 'not_selected';
              const epochProgress = trainingMatchesRound && training.epochs
                ? ` · epoch ${training.epoch}/${training.epochs}`
                : '';
              const trainedSamples = trainingMatchesRound
                ? training.processed_samples || 0
                : 0;
              const totalSamples = trainingMatchesRound
                ? training.total_samples || 0
                : activity.expected_samples ?? (
                  serverDetails.max_train_samples === null ||
                  serverDetails.max_train_samples === undefined
                    ? details.dataset_samples ?? 0
                    : Math.min(
                      details.dataset_samples ?? 0,
                      serverDetails.max_train_samples,
                    )
                );
              const remainingSamples = trainingMatchesRound
                ? training.remaining_samples || 0
                : totalSamples;
              const statusLabels = {
                idle: '대기',
                training: '학습 중',
                trained: '학습 완료',
                submitted: '제출 완료',
                received: '제출 완료',
                waiting: '선정 대기',
                expired: '마감 초과',
                not_selected: '미선정',
                failed: '오류',
              };
              const activityValue = (value) => value ? '예' : '아니오';
              return `
        <tr>
          <td>${client.client_id}</td>
          <td><a href="${client.url}/health" target="_blank">${client.url}</a></td>
          <td>${stateText(client.running)}</td>
          <td>${stateText(client.reachable)}</td>
          <td>${activity.available === undefined ? activityValue(details.available) : activityValue(activity.available)}</td>
          <td>${activityValue(Boolean(activity.selected))}</td>
          <td>${activityValue(Boolean(activity.malicious))}</td>
          <td>${activityValue(Boolean(activity.delayed))}</td>
          <td>${statusLabels[trainingStatus] || '—'}${epochProgress}</td>
          <td>${trainedSamples} / ${totalSamples} · 남음 ${remainingSamples}</td>
        </tr>
      `;
            },
          )
          .join('');

        const experimentResponse = await fetch('/api/experiment/status');
        const experiment = await experimentResponse.json();
        document.getElementById('experiment-status').textContent =
          `${experiment.status} · 라운드 ${experiment.current_round}/${experiment.total_rounds} · ${experiment.message}`;
        const maliciousSelections = experiment.malicious_clients_by_round || {};
        const selectedRounds = Object.entries(maliciousSelections)
          .filter(([, ids]) => ids.length)
          .map(([round, ids]) => `라운드 ${round}: 클라이언트 ${ids.join(', ')}`);
        document.getElementById('malicious-clients-status').textContent =
          selectedRounds.length ? selectedRounds.join(' · ') : '선정된 악성 클라이언트: 없음';
        const dlgSelections = experiment.dlg_participant_by_round || {};
        const dlgRounds = Object.entries(dlgSelections)
          .map(([round, clientId]) => `라운드 ${round}: 클라이언트 ${clientId}`);
        document.getElementById('dlg-participant-status').textContent =
          dlgRounds.length ? dlgRounds.join(' · ') : 'DLG 참가 클라이언트: 없음';
        document.getElementById('experiment-log').textContent = JSON.stringify(
          experiment.round_results.slice(-5),
          null,
          2,
        );
        updateRoundExecutionButtons(experiment.round_results || []);
      }

      async function refreshDataStatus() {
        const response = await fetch('/api/data/status');
        const data = await response.json();
        const dataDistributed = Boolean(data.configured);
        document.getElementById('start-experiment-button').disabled = !dataDistributed;
        document.getElementById('stop-experiment-button').disabled = !dataDistributed;
        document.getElementById('data-status').textContent =
          `${data.message} · 훈련 파일: ${data.config.train_files.join(', ') || '자동 탐색'} · 테스트 파일: ${data.config.test_files.join(', ') || '자동 탐색'}`;
      }

      async function distributeData() {
        const response = await fetch('/api/data/distribute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ seed: 42 }),
        });
        if (!response.ok) {
          alert((await response.json()).error);
        }
        await refreshDataStatus();
      }

      async function loadExperimentConfig() {
        const response = await fetch('/api/experiment/config');
        const config = await response.json();
        document.getElementById('experiment-id').value = config.experiment_id;
        document.getElementById('rounds').value = config.num_rounds;
        document.getElementById('experiment-type').value = config.experiment_type || 'federated';
        roundConfigsByExperimentType[document.getElementById('experiment-type').value] =
          config.round_configs || [];
        document.getElementById('dlg-training-rounds').value = config.dlg_training_rounds ?? 5;
        document.getElementById('use-pretrained-global-model').checked =
          Boolean(config.use_pretrained_global_model);
        document.getElementById('pretrained-model-format').value =
          config.pretrained_model_format || 'json';
        document.getElementById('pretrained-model-status').textContent =
          config.pretrained_global_model_loaded
            ? `저장된 전역 모델 체크포인트를 사용합니다 (버전 ${config.pretrained_global_model_version ?? 0}). 새 모델 데이터를 입력하면 교체됩니다.`
            : 'JSON은 weights 객체가 포함된 small_cnn_v1 체크포인트를 지원합니다. CSV/텍스트는 숫자 가중치를 모델 파라미터 순서로 입력합니다.';
        document.getElementById('dlg-iterations').value = config.dlg_iterations ?? 2;
        document.getElementById('dlg-client').value =
          config.dlg_client_id === null || config.dlg_client_id === undefined
            ? ''
            : config.dlg_client_id;
        document.getElementById('participation').value = config.participation_rate;
        document.getElementById('seed').value = config.seed;
          document.getElementById('local-epochs').value = config.local_epochs;
          document.getElementById('batch-size').value = config.batch_size;
          document.getElementById('learning-rate').value = config.learning_rate;
          document.getElementById('device').value = config.device;
          document.getElementById('round-timeout').value = config.round_timeout;
          document.getElementById('max-train-samples').value = config.max_train_samples ?? '';
          document.getElementById('max-test-samples').value = config.max_test_samples ?? '';
        document.getElementById('distribution').value = config.data_distribution;
        document.getElementById('dlg-noise').value = config.dlg_noise_sigma ?? 0;
        document.getElementById('dlg-defense-enabled').checked = Boolean(config.dlg_defense_enabled);
        document.getElementById('dlg-noise').disabled = !Boolean(config.dlg_defense_enabled);
          document.getElementById('defense').value = config.defense === 'none' ? 'norm_clipping' : config.defense;
        document.getElementById('defense-enabled').checked = Boolean(config.defense_enabled);
        document.getElementById('attack').checked = config.attack_enabled;
        document.getElementById('attack-scale').value = config.attack_scale ?? -5;
        document.getElementById('malicious').value = config.malicious_clients.join(',');
        buildRoundEditor(config.round_configs || []);
        updateConditionalSettings();
      }

      function selectExperimentTab(experimentType) {
        const currentType = document.getElementById('experiment-type').value;
        if (currentType === experimentType) {
          return;
        }

        roundConfigsByExperimentType[currentType] = readRoundConfigs();
        document.getElementById('experiment-type').value = experimentType;
        buildRoundEditor(roundConfigsByExperimentType[experimentType] || []);
        updateConditionalSettings();
      }

      function updateConditionalSettings() {
        const dlgEnabled = document.getElementById('experiment-type').value === 'dlg';
        const usePretrainedModel =
          dlgEnabled && document.getElementById('use-pretrained-global-model').checked;
        const roundsSetting = document.getElementById('rounds-setting');
        roundsSetting.classList.toggle('reserved-hidden', dlgEnabled);
        roundsSetting.setAttribute('aria-hidden', String(dlgEnabled));
        document.getElementById('rounds').disabled = dlgEnabled;
        document.getElementById('federated-tab-panel').hidden = dlgEnabled;
        document.getElementById('dlg-tab-panel').hidden = !dlgEnabled;
        document.getElementById('federated-tab').classList.toggle('active', !dlgEnabled);
        document.getElementById('federated-tab').setAttribute('aria-selected', String(!dlgEnabled));
        document.getElementById('dlg-tab').classList.toggle('active', dlgEnabled);
        document.getElementById('dlg-tab').setAttribute('aria-selected', String(dlgEnabled));
        document.getElementById('experiment-templates').hidden = dlgEnabled;

        const dlgDefenseEnabled = document.getElementById('dlg-defense-enabled').checked;
        document.getElementById('dlg-noise-setting').hidden =
          !dlgEnabled || !dlgDefenseEnabled;
        document.getElementById('dlg-noise').disabled =
          !dlgEnabled || !dlgDefenseEnabled;
        const dlgTrainingRoundsSetting =
          document.getElementById('dlg-training-rounds-setting');
        dlgTrainingRoundsSetting.classList.toggle(
          'reserved-hidden',
          usePretrainedModel,
        );
        dlgTrainingRoundsSetting.setAttribute(
          'aria-hidden',
          String(usePretrainedModel),
        );
        document.getElementById('dlg-training-rounds').disabled = usePretrainedModel;
        document.getElementById('pretrained-model-settings').hidden = !usePretrainedModel;
        updatePretrainedModelInput();

        const attackEnabled = document.getElementById('attack').checked;
        document.getElementById('attack-settings').hidden = !attackEnabled;

        const defenseEnabled = document.getElementById('defense-enabled').checked;
        document.getElementById('defense-method-setting').hidden =
          !defenseEnabled || dlgEnabled;
        document.getElementById('defense-enabled').disabled =
          dlgEnabled;
      }

      function updatePretrainedModelInput() {
        const textFormat =
          document.getElementById('pretrained-model-format').value === 'text';
        document.getElementById('pretrained-model-file-setting').hidden = textFormat;
        document.getElementById('pretrained-model-text-setting').hidden = !textFormat;
      }

      loadExperimentConfig();
      document.getElementById('dlg-defense-enabled').addEventListener('change', updateConditionalSettings);
      document.getElementById('use-pretrained-global-model').addEventListener('change', () => {
        if (document.getElementById('use-pretrained-global-model').checked) {
          roundConfigsByExperimentType.dlg = readRoundConfigs();
        }
        updateConditionalSettings();
        buildRoundEditor(roundConfigsByExperimentType.dlg);
      });
      document.getElementById('attack').addEventListener('change', updateConditionalSettings);
      document.getElementById('defense-enabled').addEventListener('change', updateConditionalSettings);
      document.getElementById('pretrained-model-format').addEventListener(
        'change',
        updatePretrainedModelInput,
      );
      document.getElementById('pretrained-model-file').addEventListener('change', async (event) => {
        const [file] = event.target.files;
        if (!file) {
          return;
        }
        if (file.size > 32 * 1024 * 1024) {
          alert('전역 모델 파일은 32 MiB 이하여야 합니다.');
          event.target.value = '';
          return;
        }
        document.getElementById('pretrained-model-data').value = await file.text();
        const extension = file.name.split('.').pop().toLowerCase();
        if (['json', 'csv', 'txt'].includes(extension)) {
          document.getElementById('pretrained-model-format').value =
            extension === 'txt' ? 'text' : extension;
        }
        updatePretrainedModelInput();
        document.getElementById('pretrained-model-status').textContent =
          `${file.name} 파일을 읽었습니다. 설정 저장을 눌러 모델 체크포인트를 등록하세요.`;
      });
      document.getElementById('round-configs').addEventListener('change', (event) => {
        const row = event.target.closest('tr');
        if (row) updateRoundVisibility(row);
      });
      refresh();
      refreshDataStatus();
      setInterval(() => {
        if (!document.hidden) refresh();
      }, 3000);
