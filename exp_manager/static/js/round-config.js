      function updateRoundVisibility(row) {
        const defenseEnabled = row.querySelector('.round-defense-enabled').checked;
        const defenseToggle = row.querySelector('.round-defense-enabled');
        const defenseMethod = row.querySelector('.round-defense');

        defenseToggle.disabled = false;
        defenseMethod.disabled = !defenseEnabled;

        row.classList.toggle(
          'non-attack-round',
          !row.querySelector('.round-attack').checked,
        );
      }

      function updateRoundExecutionButtons(roundResults = []) {
        const executedRounds = new Set(
          roundResults.map((result) => Number(result.round)).filter(Number.isInteger),
        );
        const rows = Array.from(document.querySelectorAll('#round-configs tr'));
        const nextRound = rows.find((row) =>
          !executedRounds.has(Number(row.dataset.round)),
        );

        rows.forEach((row) => {
          const round = Number(row.dataset.round);
          const button = row.querySelector('.round-execute-button');
          const executed = executedRounds.has(round);
          const current = nextRound === row;
          button.disabled = executed || !current;
          button.textContent = executed ? '실행됨' : current ? '실행' : '대기';
          button.title = executed
            ? `라운드 ${round}은(는) 이미 실행되었습니다.`
            : current
              ? `라운드 ${round} 실행`
              : `이전 라운드 실행 후 활성화됩니다.`;
        });
      }

      function buildRoundEditor(savedConfigs = []) {
        const isDlgExperiment = document.getElementById('experiment-type').value === 'dlg';
        const usePretrainedGlobalModel =
          isDlgExperiment && document.getElementById('use-pretrained-global-model').checked;
        const normalCount = isDlgExperiment
          ? Number(document.getElementById('dlg-training-rounds').value) || 5
          : Number(document.getElementById('rounds').value) || 1;
        const count = usePretrainedGlobalModel
          ? 1
          : Math.max(1, normalCount + (isDlgExperiment ? 1 : 0));
        const defaultParticipation = Number(document.getElementById('participation').value);
        const defaultAttack = document.getElementById('attack').checked;
        const defaultDefense = document.getElementById('defense').value;
          const defaultDefenseEnabled = document.getElementById('defense-enabled').checked;
        document.getElementById('round-configs').innerHTML = Array.from(
          { length: count },
          (_, index) => {
            const round = index + 1;
            const saved = savedConfigs[index] || {};
            const delayClients = saved.delay_clients || [];
            return `
              <tr data-round="${round}">
                <td>${round}</td>
                <td><input class="round-participation" type="number" min="0.01" max="1" step="0.05" value="${saved.participation_rate ?? defaultParticipation}"></td>
                <td><input class="round-attack" type="checkbox" ${saved.attack_enabled ?? defaultAttack ? 'checked' : ''}></td>
                <td>
                  <div class="client-checkboxes">
                    ${Array.from({ length: 10 }, (_, clientId) =>
                      `<label title="클라이언트 ${clientId}">
                        <input class="round-delay" type="checkbox" value="${clientId}" ${delayClients.includes(clientId) ? 'checked' : ''}>
                        <span>${clientId}</span>
                      </label>`
                    ).join('')}
                  </div>
                </td>
                <td>
                    <input class="round-defense-enabled" type="checkbox" ${(saved.defense_enabled ?? defaultDefenseEnabled) ? 'checked' : ''}>
                </td>
                <td>
                  <select class="round-defense">
                    <option value="norm_clipping" ${saved.defense === 'norm_clipping' || (!saved.defense && defaultDefense === 'norm_clipping') ? 'selected' : ''}>노름 클리핑</option>
                    <option value="coordinate_median" ${saved.defense === 'coordinate_median' || (!saved.defense && defaultDefense === 'coordinate_median') ? 'selected' : ''}>좌표별 중앙값</option>
                  </select>
                </td>
                <td><button class="round-execute-button" onclick="executeSingleRound(${round})">실행</button></td>
              </tr>`;
          },
        ).join('');
        document.querySelectorAll('#round-configs tr').forEach(updateRoundVisibility);
      }

      function readRoundConfigs() {
        return Array.from(document.querySelectorAll('#round-configs tr')).map((row) => ({
          participation_rate: Number(row.querySelector('.round-participation').value),
          attack_enabled: row.querySelector('.round-attack').checked,
          delay_clients: Array.from(row.querySelectorAll('.round-delay:checked'))
            .map((checkbox) => Number(checkbox.value)),
          defense: row.querySelector('.round-defense').value,
          defense_enabled: row.querySelector('.round-defense-enabled').checked,
          dlg_noise_sigma: Number(document.getElementById('dlg-noise').value),
          dlg_defense_enabled: document.getElementById('dlg-defense-enabled').checked,
        }));
      }

      async function executeSingleRound(round) {
        await saveExperimentConfig();
        const row = document.querySelector(`#round-configs tr[data-round="${round}"]`);
        const response = await fetch(`/api/rounds/${round}/execute`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            experiment_id: document.getElementById('experiment-id').value,
            round: round,
            participation_rate: Number(row.querySelector('.round-participation').value),
            attack_enabled: row.querySelector('.round-attack').checked,
            delay_clients: Array.from(row.querySelectorAll('.round-delay:checked'))
              .map((checkbox) => Number(checkbox.value)),
            defense: row.querySelector('.round-defense').value,
            defense_enabled: row.querySelector('.round-defense-enabled').checked,
            dlg_noise_sigma: Number(document.getElementById('dlg-noise').value),
            dlg_defense_enabled: document.getElementById('dlg-defense-enabled').checked,
          }),
        });
        if (!response.ok) {
          alert((await response.json()).error);
        }
        await refresh();
      }
