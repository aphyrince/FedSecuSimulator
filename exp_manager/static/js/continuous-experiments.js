const continuousEnvironments = [
  { id: 'baseline', label: '기준', participation: 1, delays: [], distribution: 'iid' },
  { id: 'delay', label: '지연', participation: 1, delays: [2, 5, 8], distribution: 'iid' },
  { id: 'partial', label: '부분 참여', participation: 0.5, delays: [], distribution: 'iid' },
  { id: 'label-bias', label: '레이블 편향', participation: 1, delays: [], distribution: 'non_iid' },
];

const continuousSecurityScenarios = [
  { id: 'normal', label: '정상', attack: false, defense: false },
  { id: 'attack', label: '공격', attack: true, defense: false },
  {
    id: 'defense_norm_clipping',
    label: '방어 · 노름 클리핑',
    attack: false,
    defense: true,
    defenseMethod: 'norm_clipping',
  },
  {
    id: 'defense_coordinate_median',
    label: '방어 · 좌표별 중앙값',
    attack: false,
    defense: true,
    defenseMethod: 'coordinate_median',
  },
  {
    id: 'attack_defense_norm_clipping',
    label: '공격 + 방어 · 노름 클리핑',
    attack: true,
    defense: true,
    defenseMethod: 'norm_clipping',
  },
  {
    id: 'attack_defense_coordinate_median',
    label: '공격 + 방어 · 좌표별 중앙값',
    attack: true,
    defense: true,
    defenseMethod: 'coordinate_median',
  },
];
const continuousCaseCount =
  continuousEnvironments.length * continuousSecurityScenarios.length;
let continuousStartPending = false;
let continuousDlgCheckpointData = '';
let continuousDlgImagePixels = null;
let continuousDlgStartPending = false;
let continuousDlgRunning = false;
let continuousExperimentBusy = false;
let continuousLinkedImagePixels = null;
let continuousLinkedStartPending = false;
let continuousLinkedRunning = false;
let lastContinuousSummaryBatchId = '';
let continuousDataReady = false;
let importedContinuousResults = null;

function updateContinuousDlgStartButton() {
  const button = document.getElementById('start-continuous-dlg-button');
  if (button) {
    button.disabled = continuousDlgStartPending || continuousDlgRunning ||
      continuousExperimentBusy ||
      !continuousDlgCheckpointData || !continuousDlgImagePixels;
  }
  const linkedButton = document.getElementById('start-continuous-linked-button');
  if (linkedButton) {
    linkedButton.disabled = continuousLinkedStartPending || continuousLinkedRunning ||
      continuousExperimentBusy;
  }
}

function setContinuousDataReady(ready) {
  continuousDataReady = ready;
  updateContinuousDlgStartButton();
}

function setContinuousExperimentBusy(busy) {
  continuousExperimentBusy = busy;
  updateContinuousDlgStartButton();
}

function escapeAttribute(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[character]);
}

function continuousDefaults() {
  const value = (id) => document.getElementById(id).value;
  return {
    seed: value('seed'),
    localEpochs: value('local-epochs'),
    batchSize: value('batch-size'),
    learningRate: value('learning-rate'),
    device: value('device'),
    roundTimeout: value('round-timeout'),
    maxTrainSamples: value('max-train-samples'),
    maxTestSamples: value('max-test-samples'),
    maliciousClients: escapeAttribute(value('malicious')),
  };
}

function renderContinuousCaseSettings(defaults) {
  return `
    <details class="continuous-case-settings">
      <summary>학습·실행 상세 설정</summary>
      <div class="continuous-case-settings-grid">
        <label>시드<input class="continuous-seed" type="number" min="0" value="${defaults.seed}"></label>
        <label>로컬 epoch<input class="continuous-local-epochs" type="number" min="1" value="${defaults.localEpochs}"></label>
        <label>배치 크기<input class="continuous-batch-size" type="number" min="1" value="${defaults.batchSize}"></label>
        <label>학습률<input class="continuous-learning-rate" type="number" min="0.000001" step="any" value="${defaults.learningRate}"></label>
        <label>연산 장치
          <select class="continuous-device">
            <option value="cpu" ${defaults.device === 'cpu' ? 'selected' : ''}>CPU</option>
            <option value="cuda" ${defaults.device === 'cuda' ? 'selected' : ''}>CUDA</option>
            <option value="auto" ${defaults.device === 'auto' ? 'selected' : ''}>자동</option>
          </select>
        </label>
        <label>라운드 제출 제한 시간(초)<input class="continuous-round-timeout" type="number" min="1" value="${defaults.roundTimeout}"></label>
        <label>클라이언트별 학습 표본 제한<input class="continuous-max-train-samples" type="number" min="1" placeholder="비우면 전체" value="${defaults.maxTrainSamples}"></label>
        <label>평가 표본 제한<input class="continuous-max-test-samples" type="number" min="1" placeholder="비우면 전체" value="${defaults.maxTestSamples}"></label>
        <label>악성 클라이언트 후보<input class="continuous-malicious-clients" value="${defaults.maliciousClients}" placeholder="쉼표로 구분, 예: 0,1"></label>
      </div>
      <label class="continuous-custom-rounds-toggle">
        <input class="continuous-custom-rounds" type="checkbox">
        라운드별 설정 사용
      </label>
      <div class="continuous-round-editor" hidden></div>
    </details>
  `;
}

function resetContinuousCases(
  containerId = 'continuous-experiments',
  statusId = 'continuous-experiment-status',
) {
  const rounds = Number(document.getElementById('rounds').value) || 1;
  const attackScaleValue = document.getElementById('attack-scale').value;
  const attackScale = attackScaleValue === '' ? -5 : Number(attackScaleValue);
  const defense = document.getElementById('defense').value || 'norm_clipping';
  const defaults = continuousDefaults();
  const cards = continuousEnvironments.flatMap((environment) =>
    continuousSecurityScenarios.map((security) => `
      <article class="continuous-case-card" data-environment="${environment.id}" data-security="${security.id}">
        <div class="continuous-case-card-header">
          <span class="continuous-case-index"></span>
          <div class="continuous-case-title">
            <div class="continuous-case-badges">
              <span class="continuous-badge continuous-environment-badge">${environment.label}</span>
              <span class="continuous-badge continuous-security-badge">${security.label}</span>
            </div>
            <input class="continuous-name" value="${environment.label} · ${security.label}" maxlength="80" aria-label="실험 이름">
          </div>
        </div>
        <div class="continuous-case-fields">
          <label>라운드<input class="continuous-rounds" type="number" min="1" value="${rounds}"></label>
          <label>참여율<input class="continuous-participation" type="number" min="0.01" max="1" step="0.05" value="${environment.participation}"></label>
          <label>지연 클라이언트<input class="continuous-delays" value="${environment.delays.join(',')}" placeholder="예: 2,5,8"></label>
          <label>데이터 분포
          <select class="continuous-distribution">
            <option value="iid" ${environment.distribution === 'iid' ? 'selected' : ''}>IID</option>
            <option value="non_iid" ${environment.distribution === 'non_iid' ? 'selected' : ''}>Non-IID</option>
          </select>
          </label>
        </div>
        <div class="continuous-security-settings">
          <label class="continuous-toggle">
            <input class="continuous-attack" type="checkbox" ${security.attack ? 'checked' : ''}>
            공격 사용
          </label>
          <label>공격 배율<input class="continuous-attack-scale" type="number" step="any" value="${attackScale}"></label>
          <label class="continuous-toggle">
            <input class="continuous-defense-enabled" type="checkbox" ${security.defense ? 'checked' : ''}>
            방어 사용
          </label>
          <label>방어 방식
          <select class="continuous-defense">
            <option value="norm_clipping" ${(security.defenseMethod || defense) === 'norm_clipping' ? 'selected' : ''}>노름 클리핑</option>
            <option value="coordinate_median" ${(security.defenseMethod || defense) === 'coordinate_median' ? 'selected' : ''}>좌표별 중앙값</option>
          </select>
          </label>
        </div>
        ${renderContinuousCaseSettings(defaults)}
      </article>
    `),
  );
  const container = document.getElementById(containerId);
  container.innerHTML = cards.join('');
  container.querySelectorAll('.continuous-case-card').forEach((card, index) => {
    card.querySelector('.continuous-case-index').textContent =
      String(index + 1).padStart(2, '0');
    updateContinuousDefenseVisibility(card);
    updateContinuousRoundEditor(card);
  });
  document.getElementById(statusId).textContent =
    `연속 실험 ${continuousCaseCount}개가 준비되었습니다.`;
}

function updateContinuousDefenseVisibility(row) {
  const enabled = row.querySelector('.continuous-defense-enabled').checked;
  row.querySelector('.continuous-defense').disabled = !enabled;
}

function readContinuousRoundConfigs(row) {
  if (!row.querySelector('.continuous-custom-rounds').checked) {
    return [];
  }
  return Array.from(row.querySelectorAll('.continuous-round-config-row')).map((roundRow) => ({
    participation_rate: Number(roundRow.querySelector('.continuous-round-participation').value),
    attack_enabled: roundRow.querySelector('.continuous-round-attack').checked,
    delay_clients: Array.from(
      roundRow.querySelectorAll('.continuous-round-delay:checked'),
    ).map((checkbox) => Number(checkbox.value)),
    defense_enabled: roundRow.querySelector('.continuous-round-defense-enabled').checked,
    defense: roundRow.querySelector('.continuous-round-defense').value,
  }));
}

function updateContinuousRoundEditor(row) {
  const enabled = row.querySelector('.continuous-custom-rounds').checked;
  const editor = row.querySelector('.continuous-round-editor');
  editor.hidden = !enabled;
  if (!enabled) {
    return;
  }

  const previous = readContinuousRoundConfigs(row);
  const count = Math.max(1, Number(row.querySelector('.continuous-rounds').value) || 1);
  const defaultDelayClients = row.querySelector('.continuous-delays').value
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)
    .map(Number);
  editor.innerHTML = `
    <table>
      <thead>
        <tr><th>라운드</th><th>참여율</th><th>공격</th><th>지연 클라이언트</th><th>방어</th><th>방어 방식</th></tr>
      </thead>
      <tbody>
        ${Array.from({ length: count }, (_, index) => {
          const saved = previous[index] || {};
          const participation = saved.participation_rate ??
            Number(row.querySelector('.continuous-participation').value);
          const attack = saved.attack_enabled ??
            row.querySelector('.continuous-attack').checked;
          const delays = saved.delay_clients || defaultDelayClients;
          const defenseEnabled = saved.defense_enabled ??
            row.querySelector('.continuous-defense-enabled').checked;
          const defense = saved.defense ||
            row.querySelector('.continuous-defense').value;
          return `
            <tr class="continuous-round-config-row">
              <td>${index + 1}</td>
              <td><input class="continuous-round-participation" type="number" min="0.01" max="1" step="0.05" value="${participation}"></td>
              <td><input class="continuous-round-attack" type="checkbox" ${attack ? 'checked' : ''}></td>
              <td>
                <div class="client-checkboxes">
                  ${Array.from({ length: 10 }, (_, clientId) => `
                    <label title="클라이언트 ${clientId}">
                      <input class="continuous-round-delay" type="checkbox" value="${clientId}" ${delays.includes(clientId) ? 'checked' : ''}>
                      <span>${clientId}</span>
                    </label>
                  `).join('')}
                </div>
              </td>
              <td><input class="continuous-round-defense-enabled" type="checkbox" ${defenseEnabled ? 'checked' : ''}></td>
              <td>
                <select class="continuous-round-defense" ${defenseEnabled ? '' : 'disabled'}>
                  <option value="norm_clipping" ${defense === 'norm_clipping' ? 'selected' : ''}>노름 클리핑</option>
                  <option value="coordinate_median" ${defense === 'coordinate_median' ? 'selected' : ''}>좌표별 중앙값</option>
                </select>
              </td>
            </tr>
          `;
        }).join('')}
      </tbody>
    </table>
  `;
  editor.querySelectorAll('.continuous-round-config-row').forEach((roundRow) => {
    roundRow.querySelector('.continuous-round-defense-enabled').addEventListener('change', () => {
      roundRow.querySelector('.continuous-round-defense').disabled =
        !roundRow.querySelector('.continuous-round-defense-enabled').checked;
    });
  });
}

function readContinuousCases(containerId = 'continuous-experiments') {
  return Array.from(
    document.getElementById(containerId).querySelectorAll(':scope > .continuous-case-card'),
  ).map((row) => {
    const optionalNumber = (selector) => {
      const value = row.querySelector(selector).value;
      return value === '' ? null : Number(value);
    };
    const maliciousClients = row.querySelector('.continuous-malicious-clients').value
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean)
      .map(Number);
    return {
      name: row.querySelector('.continuous-name').value.trim(),
      environment_template: row.dataset.environment,
      security_scenario: row.dataset.security,
      num_rounds: Number(row.querySelector('.continuous-rounds').value),
      participation_rate: Number(row.querySelector('.continuous-participation').value),
      delay_clients: row.querySelector('.continuous-delays').value
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean)
        .map(Number),
      data_distribution: row.querySelector('.continuous-distribution').value,
      attack_enabled: row.querySelector('.continuous-attack').checked,
      attack_scale: Number(row.querySelector('.continuous-attack-scale').value),
      defense_enabled: row.querySelector('.continuous-defense-enabled').checked,
      defense: row.querySelector('.continuous-defense').value,
      seed: Number(row.querySelector('.continuous-seed').value),
      local_epochs: Number(row.querySelector('.continuous-local-epochs').value),
      batch_size: Number(row.querySelector('.continuous-batch-size').value),
      learning_rate: Number(row.querySelector('.continuous-learning-rate').value),
      device: row.querySelector('.continuous-device').value,
      round_timeout: Number(row.querySelector('.continuous-round-timeout').value),
      max_train_samples: optionalNumber('.continuous-max-train-samples'),
      max_test_samples: optionalNumber('.continuous-max-test-samples'),
      malicious_clients: maliciousClients,
      round_configs: readContinuousRoundConfigs(row),
    };
  });
}

async function startContinuousExperiments() {
  if (!continuousDataReady) {
    alert('연속 실험을 시작하기 전에 훈련/테스트 데이터 범위 분배를 실행하세요.');
    return;
  }
  if (continuousStartPending) {
    return;
  }
  continuousStartPending = true;
  const button = document.getElementById('start-continuous-button');
  button.disabled = true;
  try {
    if (document.getElementById('experiment-type').value !== 'federated') {
      selectExperimentTab('federated');
    }
    const saved = await saveExperimentConfig({
      experiment_type: 'federated',
      use_pretrained_global_model: false,
      pretrained_model_data: '',
      round_configs: [],
    });
    if (!saved) {
      return;
    }
    const response = await fetch('/api/experiment/continuous/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cases: readContinuousCases() }),
    });
    const result = await response.json();
    if (!response.ok) {
      alert(result.error || '연속 실험을 시작하지 못했습니다.');
      return;
    }
    renderContinuousExperimentStatus(result.continuous_experiment);
  } catch (error) {
    const message = `연속 실험 시작 요청에 실패했습니다: ${error.message}`;
    document.getElementById('continuous-experiment-status').textContent = message;
    alert(message);
  } finally {
    continuousStartPending = false;
    await refresh();
  }
}

async function startContinuousDlgExperiments() {
  if (continuousDlgStartPending || continuousDlgRunning) {
    return;
  }
  continuousDlgStartPending = true;
  updateContinuousDlgStartButton();
  document.getElementById('continuous-dlg-status').textContent =
    '체크포인트와 입력을 검증하고 있습니다...';
  try {
    const clientValue = document.getElementById('continuous-dlg-client').value;
    const response = await fetch('/api/experiment/continuous/dlg/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        checkpoint_data: continuousDlgCheckpointData,
        image: continuousDlgImagePixels,
        label: Number(document.getElementById('continuous-dlg-label').value),
        dlg_iterations: Number(
          document.getElementById('continuous-dlg-iterations').value,
        ),
        dlg_client_id: clientValue === '' ? null : Number(clientValue),
        noise_small: Number(
          document.getElementById('continuous-dlg-noise-small').value,
        ),
        noise_large: Number(
          document.getElementById('continuous-dlg-noise-large').value,
        ),
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      alert(result.error || '연속 DLG 실험을 시작하지 못했습니다.');
      document.getElementById('continuous-dlg-status').textContent =
        result.error || '연속 DLG 시작에 실패했습니다.';
      return;
    }
    renderContinuousExperimentStatus(result.continuous_experiment);
  } finally {
    continuousDlgStartPending = false;
    updateContinuousDlgStartButton();
    await refresh();
  }
}

function renderContinuousExperimentStatus(batch) {
  if (!batch) {
    return;
  }
  const active = ['running', 'stopping'].includes(batch.status);
  const current = batch.current_case || 0;
  const name = batch.current_case_name ? ` · ${batch.current_case_name}` : '';
  if (!active && lastContinuousSummaryBatchId !== batch.batch_id) {
    lastContinuousSummaryBatchId = batch.batch_id;
    refreshContinuousResults();
  }
  if (batch.experiment_type === 'dlg') {
    continuousDlgRunning = active;
    document.getElementById('continuous-dlg-status').textContent =
      `${batch.status} · ${current}/${batch.total_cases}${name} · 배치 ${batch.batch_id}`;
    document.getElementById('stop-continuous-dlg-button').disabled =
      batch.status !== 'running';
    document.getElementById('continuous-dlg-log').textContent = JSON.stringify(
      batch.cases || [],
      null,
      2,
    );
    updateContinuousDlgStartButton();
    return;
  }
  if (batch.experiment_type === 'federated_dlg') {
    continuousLinkedRunning = active;
    document.getElementById('reset-continuous-linked-button').disabled = active;
    document.querySelectorAll(
      '#continuous-linked-experiments input, #continuous-linked-experiments select',
    ).forEach((input) => {
      input.disabled = active;
    });
    if (!active) {
      document.querySelectorAll('#continuous-linked-experiments > .continuous-case-card')
        .forEach(updateContinuousDefenseVisibility);
      document.querySelectorAll('#continuous-linked-experiments .continuous-round-config-row')
        .forEach((row) => {
          row.querySelector('.continuous-round-defense').disabled =
            !row.querySelector('.continuous-round-defense-enabled').checked;
        });
    }
    document.getElementById('continuous-linked-status').textContent =
      `${batch.status} · ${current}/${batch.total_cases}${name} · 배치 ${batch.batch_id}`;
    document.getElementById('stop-continuous-linked-button').disabled =
      batch.status !== 'running';
    document.getElementById('continuous-linked-log').textContent = JSON.stringify(
      batch.cases || [],
      null,
      2,
    );
    updateContinuousDlgStartButton();
    return;
  }
  continuousDlgRunning = false;
  continuousLinkedRunning = false;
  document.getElementById('continuous-experiment-status').textContent =
    `${batch.status} · ${current}/${batch.total_cases}${name} · 배치 ${batch.batch_id}`;
  document.getElementById('reset-continuous-button').disabled = active;
  document.getElementById('stop-continuous-button').disabled = !active;
  document.querySelectorAll('#continuous-experiments input, #continuous-experiments select')
    .forEach((input) => {
      input.disabled = active;
    });
  if (!active) {
    document.querySelectorAll('#continuous-experiments > .continuous-case-card')
      .forEach(updateContinuousDefenseVisibility);
    document.querySelectorAll('.continuous-round-config-row').forEach((row) => {
      row.querySelector('.continuous-round-defense').disabled =
        !row.querySelector('.continuous-round-defense-enabled').checked;
    });
  }
  document.getElementById('continuous-experiment-log').textContent = JSON.stringify(
    batch.cases || [],
    null,
    2,
  );
  updateContinuousDlgStartButton();
}

async function startContinuousLinkedExperiments() {
  if (continuousLinkedStartPending || continuousLinkedRunning) {
    return;
  }
  if (!continuousDataReady) {
    alert('통합 실험을 시작하기 전에 훈련/테스트 데이터 범위 분배를 실행하세요.');
    return;
  }
  if (!continuousLinkedImagePixels) {
    alert('통합 실험의 공통 DLG 원본 이미지를 업로드하세요.');
    return;
  }
  const labelValue = document.getElementById('continuous-linked-label').value;
  if (labelValue === '') {
    alert('통합 실험의 공통 DLG 레이블을 선택하세요.');
    return;
  }
  continuousLinkedStartPending = true;
  updateContinuousDlgStartButton();
  document.getElementById('continuous-linked-status').textContent =
    '연합학습 24개와 정상 체크포인트 기반 DLG 12개를 준비하고 있습니다...';
  try {
    if (document.getElementById('experiment-type').value !== 'federated') {
      selectExperimentTab('federated');
    }
    const saved = await saveExperimentConfig({
      experiment_type: 'federated',
      use_pretrained_global_model: false,
      pretrained_model_data: '',
      round_configs: [],
    });
    if (!saved) {
      return;
    }
    const clientValue = document.getElementById('continuous-linked-client').value;
    const response = await fetch('/api/experiment/continuous/linked/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cases: readContinuousCases('continuous-linked-experiments'),
        dlg: {
          image: continuousLinkedImagePixels,
          label: Number(labelValue),
          dlg_iterations: Number(
            document.getElementById('continuous-linked-iterations').value,
          ),
          dlg_client_id: clientValue === '' ? null : Number(clientValue),
          noise_small: Number(
            document.getElementById('continuous-linked-noise-small').value,
          ),
          noise_large: Number(
            document.getElementById('continuous-linked-noise-large').value,
          ),
        },
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      document.getElementById('continuous-linked-status').textContent =
        result.error || '연결 실험 시작에 실패했습니다.';
      alert(result.error || '연결 실험을 시작하지 못했습니다.');
      return;
    }
    renderContinuousExperimentStatus(result.continuous_experiment);
  } catch (error) {
    const message = `통합 실험 시작 요청에 실패했습니다: ${error.message}`;
    document.getElementById('continuous-linked-status').textContent = message;
    alert(message);
  } finally {
    continuousLinkedStartPending = false;
    updateContinuousDlgStartButton();
    await refresh();
  }
}

function renderContinuousResults(cases, contextLabel) {
  const status = document.getElementById('continuous-results-status');
  const rows = document.getElementById('continuous-results-rows');
  const totalCount = document.getElementById('results-total-count');
  const completedCount = document.getElementById('results-completed-count');
  const flCount = document.getElementById('results-fl-count');
  const dlgCount = document.getElementById('results-dlg-count');
  const environmentFilter = document.getElementById('continuous-results-environment');
  const typeFilter = document.getElementById('continuous-results-type');
  const empty = document.getElementById('continuous-results-empty');
  if (!status || !rows || !totalCount || !completedCount || !flCount || !dlgCount) {
    return;
  }
  totalCount.textContent = String(cases.length);
  completedCount.textContent = String(cases.filter((item) => item.status === 'completed').length);
  flCount.textContent = String(cases.filter((item) => item.experiment_type !== 'dlg').length);
  dlgCount.textContent = String(cases.filter((item) => item.experiment_type === 'dlg').length);

  const environmentLabels = {
    baseline: ['기준 환경', 'env-baseline'],
    delay: ['지연 환경', 'env-delay'],
    partial: ['부분 참여', 'env-partial'],
    'label-bias': ['레이블 편향', 'env-label-bias'],
    dlg: ['DLG', 'env-dlg'],
  };
  const statusLabels = {
    completed: ['완료', 'is-completed'],
    running: ['실행 중', 'is-running'],
    failed: ['실패', 'is-failed'],
    stopped: ['중단', 'is-stopped'],
    pending: ['대기', 'is-pending'],
  };
  const renderAccuracy = (value) => {
    if (value == null || value === '') {
      return '—';
    }
    const accuracy = Number(value);
    if (!Number.isFinite(accuracy)) {
      return '—';
    }
    const percentage = Math.min(100, Math.max(0, accuracy * 100));
    return `<div class="result-accuracy"><strong>${percentage.toFixed(1)}%</strong><span class="result-accuracy-track" aria-hidden="true"><i style="width:${percentage}%"></i></span></div>`;
  };
  const renderRows = () => {
    const environment = environmentFilter?.value === 'all' ? '' : environmentFilter?.value || '';
    const type = typeFilter?.value === 'all' ? '' : typeFilter?.value || '';
    const filtered = cases.filter((item) =>
      (!environment || item.environment_template === environment) &&
      (!type || item.experiment_type === type)
    );
    rows.innerHTML = filtered.map((item) => {
      const [environmentLabel, environmentClass] = environmentLabels[item.environment_template] ||
        [item.environment_label || '기타', 'env-dlg'];
      const [statusLabel, statusClass] = statusLabels[item.status] || [item.status || '미확인', 'is-pending'];
      const isDlg = item.experiment_type === 'dlg';
      const sigma = item.noise_sigma == null ? NaN : Number(item.noise_sigma);
      const mse = item.mse == null ? NaN : Number(item.mse);
      return `
        <tr>
          <td><span class="result-environment-badge ${environmentClass}">${escapeAttribute(environmentLabel)}</span></td>
          <td>${escapeAttribute(item.security_label || item.name || '—')}</td>
          <td><span class="result-kind-pill ${isDlg ? 'dlg' : 'fl'}">${isDlg ? 'DLG' : '연합학습'}</span></td>
          <td>${renderAccuracy(item.accuracy)}</td>
          <td>${Number.isFinite(sigma) ? sigma.toFixed(4) : '—'}</td>
          <td>${Number.isFinite(mse) ? mse.toExponential(3) : '—'}</td>
          <td><span class="result-status-pill ${statusClass}">${escapeAttribute(statusLabel)}</span></td>
          <td><span class="result-experiment-id" title="${escapeAttribute(item.experiment_id || '')}">${escapeAttribute(item.experiment_id || '—')}</span></td>
        </tr>
      `;
    }).join('');
    if (empty) {
      empty.hidden = filtered.length > 0;
      empty.textContent = cases.length
        ? '선택한 필터에 해당하는 결과가 없습니다.'
        : '아직 표시할 통합 실험 결과가 없습니다.';
    }
    return filtered.length;
  };
  if (environmentFilter) {
    environmentFilter.onchange = renderRows;
  }
  if (typeFilter) {
    typeFilter.onchange = renderRows;
  }
  const displayedCount = renderRows();
  status.textContent = `${contextLabel} · ${displayedCount}개 표시`;
}

async function refreshContinuousResults() {
  if (importedContinuousResults) {
    renderContinuousResults(
      importedContinuousResults.cases,
      importedContinuousResults.context,
    );
    return;
  }
  const status = document.getElementById('continuous-results-status');
  if (!status) {
    return;
  }
  try {
    const response = await fetch('/api/results/continuous');
    const summary = await response.json();
    if (!response.ok) {
      throw new Error(summary.error || '결과 조회에 실패했습니다.');
    }
    const batch = summary.batch;
    if (!batch) {
      renderContinuousResults([], '저장된 통합 실험이 없습니다');
      return;
    }
    const batchStatus = {
      running: '실행 중',
      stopping: '중단 처리 중',
      completed: '완료',
      failed: '실패',
      stopped: '중단',
    }[batch.status] || batch.status || '상태 미확인';
    const startedAt = batch.started_at
      ? new Date(batch.started_at).toLocaleString()
      : '시작 시각 미기록';
    renderContinuousResults(
      summary.cases || [],
      `최신 통합 실험 ${batch.batch_id} · ${batchStatus} · ${startedAt}`,
    );
  } catch (error) {
    status.textContent = `결과를 불러오지 못했습니다: ${error.message}`;
  }
}

function showLatestContinuousResults() {
  importedContinuousResults = null;
  const input = document.getElementById('continuous-results-manifest');
  if (input) {
    input.value = '';
  }
  refreshContinuousResults();
}

function loadContinuousResultsManifest(manifest, fileName) {
  if (
    !manifest
    || manifest.experiment_type !== 'federated_dlg'
    || !Array.isArray(manifest.cases)
  ) {
    throw new Error('통합 실험의 manifest.json 파일을 선택하세요.');
  }
  const cases = manifest.cases.map((item) => {
    const configuration = item.configuration || {};
    return {
      ...item,
      batch_id: manifest.batch_id,
      batch_status: manifest.status,
      noise_sigma: item.noise_sigma ?? configuration.noise_sigma ?? configuration.dlg_noise_sigma,
    };
  });
  const startedAt = manifest.started_at
    ? new Date(manifest.started_at).toLocaleString()
    : '시작 시각 미기록';
  importedContinuousResults = {
    cases,
    context: `파일 ${fileName} · 통합 실험 ${manifest.batch_id || 'ID 미기록'} · ${startedAt}`,
  };
  renderContinuousResults(cases, importedContinuousResults.context);
}

document.getElementById('continuous-dlg-checkpoint').addEventListener('change', async (event) => {
  const [file] = event.target.files;
  continuousDlgCheckpointData = '';
  if (!file) {
    document.getElementById('continuous-dlg-checkpoint-status').textContent =
      '업로드한 체크포인트를 공격 및 두 노이즈 방어 실험에 그대로 사용합니다.';
    updateContinuousDlgStartButton();
    return;
  }
  if (file.size > 32 * 1024 * 1024) {
    alert('체크포인트 파일은 32 MiB 이하여야 합니다.');
    event.target.value = '';
    updateContinuousDlgStartButton();
    return;
  }
  try {
    const checkpointData = await file.text();
    JSON.parse(checkpointData);
    continuousDlgCheckpointData = checkpointData;
    document.getElementById('continuous-dlg-checkpoint-status').textContent =
      `${file.name} · 체크포인트를 읽었습니다. 세 실험이 이 모델을 공유합니다.`;
  } catch (error) {
    document.getElementById('continuous-dlg-checkpoint-status').textContent =
      '체크포인트 파일을 읽거나 JSON을 해석하지 못했습니다.';
    alert('유효한 JSON 체크포인트 파일을 선택하세요.');
  } finally {
    updateContinuousDlgStartButton();
  }
});

document.getElementById('continuous-dlg-original-image').addEventListener(
  'change',
  async (event) => {
    const [file] = event.target.files;
    continuousDlgImagePixels = null;
    if (!file) {
      document.getElementById('continuous-dlg-image-preview').hidden = true;
      document.getElementById('continuous-dlg-image-status').textContent =
        '업로드 이미지는 RGB 32×32 크기로 변환되어 모든 실험에 공통 사용됩니다.';
      updateContinuousDlgStartButton();
      return;
    }
    if (!file.type.startsWith('image/') || file.size > 10 * 1024 * 1024) {
      alert('이미지 파일(10 MiB 이하)을 선택하세요.');
      event.target.value = '';
      updateContinuousDlgStartButton();
      return;
    }
    try {
      const bitmap = await createImageBitmap(file);
      const canvas = document.createElement('canvas');
      canvas.width = 32;
      canvas.height = 32;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) {
        throw new Error('이미지 변환용 canvas를 사용할 수 없습니다.');
      }
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, 32, 32);
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = 'high';
      context.drawImage(bitmap, 0, 0, 32, 32);
      bitmap.close();
      const rgba = context.getImageData(0, 0, 32, 32).data;
      const pixels = [[], [], []];
      for (let index = 0; index < rgba.length; index += 4) {
        pixels[0].push(rgba[index] / 255);
        pixels[1].push(rgba[index + 1] / 255);
        pixels[2].push(rgba[index + 2] / 255);
      }
      continuousDlgImagePixels = pixels.flat();
      const preview = document.getElementById('continuous-dlg-image-preview');
      preview.src = canvas.toDataURL('image/png');
      preview.hidden = false;
      document.getElementById('continuous-dlg-image-status').textContent =
        `${file.name} · RGB 32×32 · 레이블 ${document.getElementById('continuous-dlg-label').value}`;
    } catch (error) {
      continuousDlgImagePixels = null;
      document.getElementById('continuous-dlg-image-status').textContent =
        `이미지를 변환하지 못했습니다: ${error.message}`;
      alert('이미지를 읽거나 32×32 크기로 변환하지 못했습니다.');
    } finally {
      updateContinuousDlgStartButton();
    }
  },
);

document.getElementById('continuous-dlg-label').addEventListener('change', () => {
  const file = document.getElementById('continuous-dlg-original-image').files[0];
  if (file && continuousDlgImagePixels) {
    document.getElementById('continuous-dlg-image-status').textContent =
      `${file.name} · RGB 32×32 · 레이블 ${document.getElementById('continuous-dlg-label').value}`;
  }
});

document.getElementById('continuous-linked-original-image').addEventListener(
  'change',
  async (event) => {
    const [file] = event.target.files;
    continuousLinkedImagePixels = null;
    if (!file) {
      document.getElementById('continuous-linked-image-preview').hidden = true;
      document.getElementById('continuous-linked-image-status').textContent =
        '업로드 이미지는 RGB 32×32로 변환되어 12개 DLG 실험에서 공통 사용됩니다.';
      updateContinuousDlgStartButton();
      return;
    }
    if (!file.type.startsWith('image/') || file.size > 10 * 1024 * 1024) {
      alert('이미지 파일(10 MiB 이하)을 선택하세요.');
      event.target.value = '';
      updateContinuousDlgStartButton();
      return;
    }
    try {
      const bitmap = await createImageBitmap(file);
      const canvas = document.createElement('canvas');
      canvas.width = 32;
      canvas.height = 32;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) {
        throw new Error('이미지 변환용 canvas를 사용할 수 없습니다.');
      }
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, 32, 32);
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = 'high';
      context.drawImage(bitmap, 0, 0, 32, 32);
      bitmap.close();
      const rgba = context.getImageData(0, 0, 32, 32).data;
      const pixels = [[], [], []];
      for (let index = 0; index < rgba.length; index += 4) {
        pixels[0].push(rgba[index] / 255);
        pixels[1].push(rgba[index + 1] / 255);
        pixels[2].push(rgba[index + 2] / 255);
      }
      continuousLinkedImagePixels = pixels.flat();
      const preview = document.getElementById('continuous-linked-image-preview');
      preview.src = canvas.toDataURL('image/png');
      preview.hidden = false;
      document.getElementById('continuous-linked-image-status').textContent =
        `${file.name} · RGB 32×32 · 레이블 ${document.getElementById('continuous-linked-label').value}`;
    } catch (error) {
      continuousLinkedImagePixels = null;
      document.getElementById('continuous-linked-image-status').textContent =
        `이미지를 변환하지 못했습니다: ${error.message}`;
      alert('이미지를 읽거나 32×32 크기로 변환하지 못했습니다.');
    } finally {
      updateContinuousDlgStartButton();
    }
  },
);

document.getElementById('continuous-linked-label').addEventListener('change', () => {
  const file = document.getElementById('continuous-linked-original-image').files[0];
  if (file && continuousLinkedImagePixels) {
    document.getElementById('continuous-linked-image-status').textContent =
      `${file.name} · RGB 32×32 · 레이블 ${document.getElementById('continuous-linked-label').value}`;
  }
});

function handleContinuousCaseChange(event) {
  const row = event.target.closest('.continuous-case-card');
  if (!row) {
    return;
  }
  if (event.target.matches('.continuous-defense-enabled')) {
    updateContinuousDefenseVisibility(row);
  } else if (event.target.matches('.continuous-custom-rounds')) {
    updateContinuousRoundEditor(row);
  } else if (event.target.matches('.continuous-rounds')) {
    updateContinuousRoundEditor(row);
  }
}

document.getElementById('continuous-experiments').addEventListener(
  'change',
  handleContinuousCaseChange,
);
document.getElementById('continuous-linked-experiments').addEventListener(
  'change',
  handleContinuousCaseChange,
);

document.getElementById('continuous-results-manifest').addEventListener(
  'change',
  async (event) => {
    const [file] = event.target.files;
    if (!file) {
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('통합 실험 manifest.json 파일은 10 MiB 이하여야 합니다.');
      event.target.value = '';
      return;
    }
    try {
      loadContinuousResultsManifest(JSON.parse(await file.text()), file.name);
    } catch (error) {
      alert(error.message || '통합 실험 결과 파일을 읽지 못했습니다.');
      event.target.value = '';
    }
  },
);

refreshContinuousResults();

resetContinuousCases();
resetContinuousCases('continuous-linked-experiments', 'continuous-linked-fl-status');
updateContinuousDlgStartButton();
