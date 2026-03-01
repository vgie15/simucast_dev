function selectModel(index) {
  const result = modelData[index];
  if (!result || !result.success) return;

  modelData.forEach((_, i) => {
    const radio = document.getElementById('radio-' + i);
    const row = document.getElementById('row-' + i);
    if (radio) {
      radio.style.borderColor = '#d1d5db';
      radio.style.background = '';
    }
    if (row) row.style.background = '';
  });

  const radio = document.getElementById('radio-' + index);
  const row = document.getElementById('row-' + index);
  if (radio) {
    radio.style.borderColor = 'var(--orange)';
    radio.style.background = 'radial-gradient(circle, var(--orange) 45%, white 46%)';
  }
  if (row) row.style.background = '#fffbf5';

  document.getElementById('selectedModelName').textContent = result.model_name;
  document.getElementById('selectedModelKey').value = result.model_key;
  document.getElementById('selectedArtifactPath').value = result.artifact_path;
}