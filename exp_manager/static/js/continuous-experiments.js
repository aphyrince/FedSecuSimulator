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

function resetContinuousCases() {
  const rounds = Number(document.getElementById('rounds').value) || 1;
  const attackScaleValue = document.getElementById('attack-scale').value;
  const attackScale = attackScaleValue === '' ? -5 : Number(attackScaleValue);
  const defense = document.getElementById('defense').value || 'norm_clipping';
  const rows = continuousEnvironments.flatMap((environment) =>
    continuousSecurityScenarios.map((security) => `
      <tr data-environment="${environment.id}" data-security="${security.id}">
        <td class="continuous-index"></td>
        <td><input class="continuous-name" value="${environment.label} · ${security.label}" maxlength="80"></td>
        <td>${environment.label}</td>
        <td>${security.label}</td>
        <td><input class="continuous-rounds" type="number" min="1" value="${rounds}"></td>
        <td><input class="continuous-participation" type="number" min="0.01" max="1" step="0.05" value="${environment.participation}"></td>
        <td><input class="continuous-delays" value="${environment.delays.join(',')}" placeholder="예: 2,5,8"></td>
        <td>
          <select class="continuous-distribution">
            <option value="iid" ${environment.distribution === 'iid' ? 'selected' : ''}>IID</option>
            <option value="non_iid" ${environment.distribution === 'non_iid' ? 'selected' : ''}>Non-IID</option>
          </select>
        </td>
        <td><input class="continuous-attack" type="checkbox" ${security.attack ? 'checked' : ''}></td>
        <td><input class="continuous-attack-scale" type="number" step="any" value="${attackScale}"></td>
        <td><input class="continuous-defense-enabled" type="checkbox" ${security.defense ? 'checked' : ''}></td>
        <td>
          <select class="continuous-defense">
            <option value="norm_clipping" ${(security.defenseMethod || defense) === 'norm_clipping' ? 'selected' : ''}>노름 클리핑</option>
            <option value="coordinate_median" ${(security.defenseMethod || defense) === 'coordinate_median' ? 'selected' : ''}>좌표별 중앙값</option>
          </select>
        </td>
      </tr>
    `),
  );
  document.getElementById('continuous-experiments').innerHTML = rows.join('');
  document.querySelectorAll('#continuous-experiments tr').forEach((row, index) => {
    row.querySelector('.continuous-index').textContent = String(index + 1);
    updateContinuousDefenseVisibility(row);
  });
  document.getElementById('continuous-experiment-status').textContent =
    `연속 실험 ${continuousCaseCount}개가 준비되었습니다.`;
}

function updateContinuousDefenseVisibility(row) {
  const enabled = row.querySelector('.continuous-defense-enabled').checked;
  row.querySelector('.continuous-defense').disabled = !enabled;
}

function readContinuousCases() {
  return Array.from(document.querySelectorAll('#continuous-experiments tr')).map((row) => ({
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
  }));
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
  } finally {
    continuousStartPending = false;
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
  document.getElementById('continuous-experiment-status').textContent =
    `${batch.status} · ${current}/${batch.total_cases}${name} · 배치 ${batch.batch_id}`;
  document.getElementById('reset-continuous-button').disabled = active;
  document.querySelectorAll('#continuous-experiments input, #continuous-experiments select')
    .forEach((input) => {
      input.disabled = active;
    });
  if (!active) {
    document.querySelectorAll('#continuous-experiments tr')
      .forEach(updateContinuousDefenseVisibility);
  }
  document.getElementById('continuous-experiment-log').textContent = JSON.stringify(
    batch.cases || [],
    null,
    2,
  );
}

document.getElementById('continuous-experiments').addEventListener('change', (event) => {
  const row = event.target.closest('tr');
  if (row && event.target.matches('.continuous-defense-enabled')) {
    updateContinuousDefenseVisibility(row);
  }
});

resetContinuousCases();
