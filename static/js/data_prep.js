function toggleSynthetic() {
  const box = document.getElementById('syntheticCheck');
  const options = document.getElementById('syntheticOptions');
  setTimeout(() => {
    options.style.display = box.checked ? 'block' : 'none';
  }, 10);
}

const fileInput = document.getElementById('fileInput');
if (fileInput) {
  fileInput.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    document.getElementById('dropDefault').style.display = 'none';
    document.getElementById('dropPreview').style.display = 'block';
    document.getElementById('dropFileName').textContent = file.name;
    document.getElementById('dropFileSize').textContent = sizeMB + ' MB';
  });
}

function toggleQuality(id) {
  const panel = document.getElementById('panel-' + id);
  const arrow = document.getElementById('arrow-' + id);
  const isOpen = panel.style.display === 'block';
  panel.style.display = isOpen ? 'none' : 'block';
  arrow.classList.toggle('open', !isOpen);
}

function dismissToast() {
  const toast = document.getElementById('uploadToast');
  if (!toast) return;
  toast.style.opacity = '0';
  setTimeout(() => toast.style.display = 'none', 500);
  sessionStorage.setItem('toastDismissed', '1');
}

const toast = document.getElementById('uploadToast');
if (toast) {
  if (sessionStorage.getItem('toastDismissed') === '1') {
    toast.style.display = 'none';
  } else {
    setTimeout(dismissToast, 4000);
  }
}

document.addEventListener('DOMContentLoaded', function () {
  const removeRowsCheck = document.getElementById('removeRowsCheck');
  const imputationChecks = document.querySelectorAll('.imputation-check');

  if (!removeRowsCheck) return;

  removeRowsCheck.addEventListener('change', function () {
    imputationChecks.forEach(cb => {
      cb.disabled = this.checked;
      cb.checked = false;
      cb.closest('label').style.opacity = this.checked ? '0.4' : '1';
      cb.closest('label').style.pointerEvents = this.checked ? 'none' : 'auto';
    });
  });
});