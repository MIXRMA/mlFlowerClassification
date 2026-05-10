'use strict';
let activeTab    = 'upload';
let selectedFile = null;
let urlDebounce  = null;

function switchTab(tab) {
  activeTab = tab;

  document.getElementById('panelUpload').style.display = tab === 'upload' ? '' : 'none';
  document.getElementById('panelUrl').style.display    = tab === 'url'    ? '' : 'none';

  document.getElementById('tabUpload').classList.toggle('tab--active', tab === 'upload');
  document.getElementById('tabUrl').classList.toggle('tab--active',    tab === 'url');

  selectedFile = null;
  updateClassifyButton();
}

function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.add('dropzone--active');
}

function handleDragLeave(e) {
  document.getElementById('dropzone').classList.remove('dropzone--active');
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.remove('dropzone--active');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    setFile(file);
  }
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) setFile(file);
}

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    const previewContainer = document.getElementById('previewContainer');
    const previewImg       = document.getElementById('previewImg');
    previewImg.src         = e.target.result;
    previewContainer.style.display = '';

    // Swap icon for a checkmark
    document.querySelector('.dropzone__icon svg').innerHTML =
      '<polyline points="20 6 9 17 4 12" stroke-width="2"/>';
    document.querySelector('.dropzone__primary').textContent = file.name;
    document.querySelector('.dropzone__secondary').textContent =
      `${(file.size / 1024).toFixed(1)} KB · click to change`;
  };
  reader.readAsDataURL(file);
  updateClassifyButton();
}

function handleUrlInput(e) {
  clearTimeout(urlDebounce);
  const url = e.target.value.trim();

  updateClassifyButton();

  if (!url) {
    hideUrlPreview();
    return;
  }

  urlDebounce = setTimeout(() => {
    attemptUrlPreview(url);
  }, 600);
}

function attemptUrlPreview(url) {
  const img     = document.getElementById('urlPreviewImg');
  const preview = document.getElementById('urlPreview');

  img.onload  = () => { preview.style.display = ''; };
  img.onerror = () => { preview.style.display = 'none'; };
  img.src     = url;
}

function hideUrlPreview() {
  document.getElementById('urlPreview').style.display = 'none';
}

function updateClassifyButton() {
  const btn = document.getElementById('btnClassify');

  if (activeTab === 'upload') {
    btn.disabled = !selectedFile;
  } else {
    const url = document.getElementById('urlInput').value.trim();
    btn.disabled = !url;
  }
}

async function classify() {
  setLoading(true);
  hideResults();

  try {
    const formData = new FormData();

    if (activeTab === 'upload' && selectedFile) {
      formData.append('image', selectedFile);
    } else if (activeTab === 'url') {
      const url = document.getElementById('urlInput').value.trim();
      formData.append('imageUrl', url);
    } else {
      throw new Error('No input provided.');
    }

    const response = await fetch('/predict', {
      method: 'POST',
      body:   formData,
    });

    const data = await response.json();

    if (data.success) {
      renderResults(data.predictions);
    } else {
      renderError(data.error || 'Unknown error from server.');
    }

  } catch (err) {
    renderError(err.message || 'Network error — is the Flask server running?');
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  const btn     = document.getElementById('btnClassify');
  const text    = document.querySelector('.btn-classify__text');
  const spinner = document.getElementById('spinner');

  btn.disabled      = isLoading;
  text.textContent  = isLoading ? 'Identifying…' : 'Identify Species';
  spinner.style.display = isLoading ? '' : 'none';
}

function renderResults(predictions) {
  if (!predictions || predictions.length === 0) {
    renderError('No predictions returned.');
    return;
  }

  const section = document.getElementById('resultsSection');
  section.style.display = '';

  // Hide error card
  document.getElementById('errorCard').style.display = 'none';
  document.getElementById('resultHero').style.display = '';

  const top = predictions[0];

  document.getElementById('heroName').textContent =
    formatClassName(top.name);
  document.getElementById('heroConf').textContent =
    `${top.confidence.toFixed(2)}% confidence`;

  requestAnimationFrame(() => {
    setTimeout(() => {
      document.getElementById('heroBar').style.width = `${top.confidence}%`;
    }, 50);
  });

  const list = document.getElementById('resultList');
  list.innerHTML = '';

  predictions.slice(1).forEach((pred, i) => {
    const item = document.createElement('div');
    item.className = 'result-item';
    item.style.animationDelay = `${i * 0.07}s`;

    item.innerHTML = `
      <div class="result-item__fill"></div>
      <div class="result-item__rank">#${pred.rank}</div>
      <div class="result-item__name">${formatClassName(pred.name)}</div>
      <div class="result-item__conf">${pred.confidence.toFixed(2)}%</div>
    `;
    list.appendChild(item);

    // Animate fill bar
    requestAnimationFrame(() => {
      setTimeout(() => {
        item.querySelector('.result-item__fill').style.width =
          `${pred.confidence}%`;
      }, 100 + i * 70);
    });
  });

  section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderError(message) {
  const section = document.getElementById('resultsSection');
  section.style.display = '';

  document.getElementById('resultHero').style.display = 'none';
  document.getElementById('resultList').innerHTML = '';

  const errorCard = document.getElementById('errorCard');
  errorCard.style.display = 'flex';
  document.getElementById('errorMsg').textContent = message;
}

function hideResults() {
  document.getElementById('resultsSection').style.display = 'none';
}

function formatClassName(name) {
  return name
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, c => c.toUpperCase())
    .trim();
}

function resetUI() {
  hideResults();
  selectedFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('previewContainer').style.display = 'none';
  document.getElementById('previewImg').src = '';
  document.querySelector('.dropzone__primary').textContent = 'Drop any flower image here';
  document.querySelector('.dropzone__secondary').textContent =
    'or click to browse — any size, format, or colour profile';
  document.querySelector('.dropzone__icon svg').innerHTML =
    '<path d="M12 5v14M5 12l7-7 7 7"/>';
  document.getElementById('urlInput').value = '';
  hideUrlPreview();

  updateClassifyButton();

  document.querySelector('.shell').scrollIntoView({ behavior: 'smooth' });
}

(function initPetalCanvas() {
  const canvas = document.getElementById('petalCanvas');
  const ctx    = canvas.getContext('2d');

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const PETAL_COUNT = 28;
  const petals = [];

  function makePetal() {
    return {
      x:       Math.random() * canvas.width,
      y:       Math.random() * canvas.height * 1.2 - canvas.height * 0.1,
      vx:      (Math.random() - 0.5) * 0.4,
      vy:      Math.random() * 0.5 + 0.2,
      size:    Math.random() * 14 + 6,
      rot:     Math.random() * Math.PI * 2,
      rotV:    (Math.random() - 0.5) * 0.012,
      opacity: Math.random() * 0.4 + 0.1,
      hue:     Math.random() > 0.5 ? 32 : 130,  // amber or green
    };
  }

  for (let i = 0; i < PETAL_COUNT; i++) petals.push(makePetal());

  function drawPetal(p) {
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot);
    ctx.globalAlpha = p.opacity;

    ctx.beginPath();
    const s = p.size;
    ctx.moveTo(0, -s);
    ctx.bezierCurveTo(s * 0.6, -s * 0.5, s * 0.6, s * 0.5, 0, s);
    ctx.bezierCurveTo(-s * 0.6, s * 0.5, -s * 0.6, -s * 0.5, 0, -s);
    ctx.closePath();

    const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, s);
    grad.addColorStop(0, `hsla(${p.hue}, 50%, 55%, 0.9)`);
    grad.addColorStop(1, `hsla(${p.hue}, 35%, 30%, 0.1)`);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.restore();
  }

  function tick() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    petals.forEach(p => {
      p.x   += p.vx + Math.sin(Date.now() * 0.0007 + p.y * 0.01) * 0.3;
      p.y   += p.vy;
      p.rot += p.rotV;

      if (p.y > canvas.height + 30) {
        p.y  = -30;
        p.x  = Math.random() * canvas.width;
      }
      if (p.x < -30 || p.x > canvas.width + 30) {
        p.x = Math.random() * canvas.width;
      }

      drawPetal(p);
    });

    requestAnimationFrame(tick);
  }

  tick();
})();