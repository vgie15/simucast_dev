// ── Card 3 shell wrapper ──────────────────────────────────────────────────────
function card3Shell(paramsHtml) {
  return `
<div class="app-card">
  <div class="card-step-header">
    <div style="display:flex; align-items:center; gap:.8rem;">
      <div class="card-step-num">3</div>
      <span class="card-step-title">Model Configuration</span>
      <span style="font-size:.78rem; color:#6b7280; border:1px solid #e5e7eb; border-radius:20px; padding:.2rem .7rem;">Optional / Advanced</span>
    </div>
  </div>
  <p style="font-size:.85rem; color:#6b7280; margin-bottom:1.2rem;">Fine-tune hyperparameters for the selected model</p>
  <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:1rem 1.2rem; margin-bottom:1.5rem; display:flex; align-items:flex-start; gap:.6rem;">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" style="flex-shrink:0;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
    <div>
      <div style="font-weight:700; font-size:.88rem; color:#1e40af; margin-bottom:.2rem;">Advanced Feature</div>
      <div style="font-size:.82rem; color:#3b82f6;">Default parameters work well for most cases. Adjust these only if you understand their impact on model performance.</div>
    </div>
  </div>
  <form method="POST" action="/build-model/retrain">
    ${paramsHtml}
    <div style="margin-top:1.5rem; padding:.8rem 1rem; background:#f9fafb; border-radius:8px; font-size:.78rem; color:#6b7280;">
      Note: Changing parameters will not re-evaluate models. Click "Retrain Selected Model" to apply these settings.
    </div>
    <div style="display:flex; justify-content:flex-end; margin-top:1rem;">
      <button type="submit" style="display:flex; align-items:center; gap:.5rem; background:var(--orange); color:white; padding:.7rem 1.3rem; border-radius:10px; font-size:.88rem; font-weight:600; border:none; cursor:pointer;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
        Retrain Selected Model
      </button>
    </div>
  </form>
</div>`;
}

// ── Card 3 params per model ───────────────────────────────────────────────────
const card3Params = {
  ridge_regression: `
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem;">
      <div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:.6rem;">
          <span style="font-weight:700; font-size:.9rem; color:#1a1a2e;">Training Data Split</span>
          <span id="splitVal" style="background:#f3f4f6; border:1px solid #e5e7eb; border-radius:8px; padding:.2rem .6rem; font-size:.85rem; font-weight:600;">80%</span>
        </div>
        <input type="range" name="test_size" id="splitSlider" min="60" max="90" value="80" step="5"
          oninput="document.getElementById('splitVal').textContent = this.value + '%'"
          style="width:100%; accent-color:var(--orange);">
        <div style="display:flex; justify-content:space-between; font-size:.75rem; color:#9ca3af; margin-top:.3rem;">
          <span>60% Train</span><span>90% Train</span>
        </div>
        <p style="font-size:.75rem; color:#6b7280; margin-top:.5rem; line-height:1.5;">Higher training split may improve learning but makes testing less stable on small datasets.</p>
      </div>
      <div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:.6rem;">
          <span style="font-weight:700; font-size:.9rem; color:#1a1a2e;">Regularization Strength (Alpha)</span>
          <span id="alphaVal" style="background:#f3f4f6; border:1px solid #e5e7eb; border-radius:8px; padding:.2rem .6rem; font-size:.85rem; font-weight:600;">1.0</span>
        </div>
        <input type="range" name="ridge_alpha" id="alphaSlider" min="-2" max="2" value="0" step="0.1"
          oninput="const a=Math.pow(10,parseFloat(this.value)); document.getElementById('alphaVal').textContent=a<1?a.toFixed(3):a<10?a.toFixed(2):a.toFixed(1);"
          style="width:100%; accent-color:var(--orange);">
        <div style="display:flex; justify-content:space-between; font-size:.75rem; color:#9ca3af; margin-top:.3rem;">
          <span>0.01 (Less regularized)</span><span>100 (More regularized)</span>
        </div>
        <p style="font-size:.75rem; color:#6b7280; margin-top:.5rem; line-height:1.5;">Higher alpha creates a simpler model and reduces overfitting. Lower alpha behaves more like standard linear regression.</p>
      </div>
    </div>`,

  random_forest_regressor: `
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem;">
      <div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:.6rem;">
          <span style="font-weight:700; font-size:.9rem; color:#1a1a2e;">Training Data Split</span>
          <span id="splitVal" style="background:#f3f4f6; border:1px solid #e5e7eb; border-radius:8px; padding:.2rem .6rem; font-size:.85rem; font-weight:600;">80%</span>
        </div>
        <input type="range" name="test_size" id="splitSlider" min="60" max="90" value="80" step="5"
          oninput="document.getElementById('splitVal').textContent = this.value + '%'"
          style="width:100%; accent-color:var(--orange);">
        <div style="display:flex; justify-content:space-between; font-size:.75rem; color:#9ca3af; margin-top:.3rem;">
          <span>60% Train</span><span>90% Train</span>
        </div>
        <p style="font-size:.75rem; color:#6b7280; margin-top:.5rem; line-height:1.5;">Higher training split may improve learning but makes testing less stable on small datasets.</p>
      </div>
      <div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:.6rem;">
          <span style="font-weight:700; font-size:.9rem; color:#1a1a2e;">Number of Trees</span>
          <span id="treesVal" style="background:#f3f4f6; border:1px solid #e5e7eb; border-radius:8px; padding:.2rem .6rem; font-size:.85rem; font-weight:600;">300</span>
        </div>
        <input type="range" name="n_estimators" id="treesSlider" min="100" max="500" value="300" step="50"
          oninput="document.getElementById('treesVal').textContent = this.value"
          style="width:100%; accent-color:var(--orange);">
        <div style="display:flex; justify-content:space-between; font-size:.75rem; color:#9ca3af; margin-top:.3rem;">
          <span>100 (Fast)</span><span>500 (Accurate)</span>
        </div>
        <p style="font-size:.75rem; color:#6b7280; margin-top:.5rem; line-height:1.5;">More trees improve stability but increase training time.</p>
      </div>
      <div>
        <div style="font-weight:700; font-size:.9rem; color:#1a1a2e; margin-bottom:.6rem;">Max Tree Depth</div>
        <div style="position:relative;">
          <select name="max_depth" style="width:100%; padding:.85rem 1.2rem; border:1px solid #e5e7eb; border-radius:10px; font-size:.9rem; color:#1a1a2e; appearance:none; background:white; cursor:pointer;">
            <option value="None">Auto (None) - let trees grow fully</option>
            <option value="5">5 - very simple</option>
            <option value="8">8 - simple</option>
            <option value="12">12 - moderate</option>
            <option value="16">16 - complex</option>
          </select>
          <svg style="position:absolute; right:1rem; top:50%; transform:translateY(-50%); pointer-events:none;" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
        <p style="font-size:.75rem; color:#6b7280; margin-top:.5rem; line-height:1.5;">Limiting depth prevents overfitting. Use Auto only if your dataset is large and clean.</p>
      </div>
      <div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:.6rem;">
          <span style="font-weight:700; font-size:.9rem; color:#1a1a2e;">Min Samples Leaf</span>
          <span id="leafVal" style="background:#f3f4f6; border:1px solid #e5e7eb; border-radius:8px; padding:.2rem .6rem; font-size:.85rem; font-weight:600;">1</span>
        </div>
        <input type="range" name="min_samples_leaf" id="leafSlider" min="1" max="10" value="1" step="1"
          oninput="document.getElementById('leafVal').textContent = this.value"
          style="width:100%; accent-color:var(--orange);">
        <div style="display:flex; justify-content:space-between; font-size:.75rem; color:#9ca3af; margin-top:.3rem;">
          <span>1 (Complex)</span><span>10 (Simple)</span>
        </div>
        <p style="font-size:.75rem; color:#6b7280; margin-top:.5rem; line-height:1.5;">Higher values reduce overfitting on noisy data.</p>
      </div>
    </div>`
};

// ── Update card 3 ─────────────────────────────────────────────────────────────
function updateCard3(modelKey) {
  const wrapper = document.getElementById('card3-wrapper');
  if (!wrapper) return;
  const params = card3Params[modelKey] || card3Params['ridge_regression'];
  wrapper.innerHTML = card3Shell(params);
}

// ── Select model row ──────────────────────────────────────────────────────────
function selectModel(index) {
  const result = modelData[index];
  if (!result || !result.success) return;

  modelData.forEach((_, i) => {
    const radio = document.getElementById('radio-' + i);
    const row = document.getElementById('row-' + i);
    if (radio) { radio.style.borderColor = '#d1d5db'; radio.style.background = ''; }
    if (row) row.style.background = '';
  });

  const radio = document.getElementById('radio-' + index);
  const row = document.getElementById('row-' + index);
  if (radio) {
    radio.style.borderColor = 'var(--orange)';
    radio.style.background = 'radial-gradient(circle, var(--orange) 45%, white 46%)';
  }
  if (row) row.style.background = '#fffbf5';

  fetch('/build-model/select-model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `model_key=${result.model_key}&artifact_path=${encodeURIComponent(result.artifact_path)}`
  });

  updateCard3(result.model_key);
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
  if (typeof modelData === 'undefined' || !modelData.length) return;

  // Find currently selected or recommended model
  const selectedKey = (modelData.find(r => r.recommended) || modelData[0])?.model_key || 'ridge_regression';
  updateCard3(selectedKey);
});

// ── Toggle quality dropdowns ──────────────────────────────────────────────────
function toggleQuality(id) {
  const panel = document.getElementById('panel-' + id);
  const arrow = document.getElementById('arrow-' + id);
  const isOpen = panel.style.display === 'block';
  panel.style.display = isOpen ? 'none' : 'block';
  arrow.classList.toggle('open', !isOpen);
}