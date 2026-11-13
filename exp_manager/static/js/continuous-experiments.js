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

function updateContinuousDlgStartButton() {
  const button = document.getElementById('start-continuous-dlg-button');
  if (button) {
    button.disabled = continuousDlgStartPending || continuousDlgRunning ||
      continuousExperimentBusy ||
      !continuousDlgCheckpointData || !continuousDlgImagePixels;
  }
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

function resetContinuousCases() {
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
  const container = document.getElementById('continuous-experiments');
  container.innerHTML = cards.join('');
  container.querySelectorAll('.continuous-case-card').forEach((card, index) => {
    card.querySelector('.continuous-case-index').textContent =
      String(index + 1).padStart(2, '0');
    updateContinuousDefenseVisibility(card);
    updateContinuousRoundEditor(card);
  });
  document.getElementById('continuous-experiment-status').textContent =
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

function readContinuousCases() {
  return Array.from(
    document.querySelectorAll('#continuous-experiments > .continuous-case-card'),
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
    const message = `연속 DLG 시작 요청에 실패했습니다: ${error.message}`;
    document.getElementById('continuous-dlg-status').textContent = message;
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
  continuousDlgRunning = false;
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

document.getElementById('continuous-experiments').addEventListener('change', (event) => {
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
});

resetContinuousCases();
updateContinuousDlgStartButton();
