/* LTX 2.5 Video Enhancement Studio — ReDetail 2.0 Frontend Controller */

let currentSource = null;
let currentSettings = {};
let activeJobId = null;
let schemaData = null;
let presetsData = {};
let sseInstaller = null;
let sseJobs = null;
let jobsRefreshRequest = null;
let jobMonitorEpoch = 0;
let submissionInFlight = false;
let installerRefreshRequest = null;

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initAccordions();
  initDropzone();
  initControls();
  initSplitSlider();
  initSSEInstaller();
  initSSEJobs();
  loadPresetsAndSchema();
  startHardwarePolling();
  refreshJobsList();
  refreshModelsList();
});

// ==================== NAVIGATION TABS ====================
function initNavigation() {
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
      tab.classList.add('active');
      const target = document.getElementById(`tab-${tab.dataset.tab}`);
      if (target) {
        target.style.display = 'flex';
        if (tab.dataset.tab === 'queue') refreshJobsList();
        if (tab.dataset.tab === 'system') { refreshModelsList(); updateHardwareTelemetry(); }
      }
    });
  });

  document.getElementById('refresh-btn').addEventListener('click', () => {
    updateHardwareTelemetry();
    refreshJobsList();
    refreshModelsList();
  });

  // Modal close
  document.getElementById('close-modal-btn').addEventListener('click', () => {
    const modal = document.getElementById('video-modal');
    modal.style.display = 'none';
    document.getElementById('modal-video-player').pause();
  });
}

// ==================== ACCORDIONS ====================
function initAccordions() {
  document.querySelectorAll('.accordion-header').forEach(hdr => {
    hdr.addEventListener('click', () => {
      hdr.parentElement.classList.toggle('open');
    });
  });
}

// ==================== DROPZONE & FILE UPLOAD ====================
function initDropzone() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
  });
}

function uploadFile(file) {
  const dropContent = document.querySelector('.dropzone-content');
  const progressWrap = document.getElementById('upload-progress-wrap');
  const progressFill = document.getElementById('upload-progress-fill');
  const statusText = document.getElementById('upload-status-text');

  dropContent.style.display = 'none';
  progressWrap.style.display = 'block';
  progressFill.style.width = '10%';
  statusText.innerText = `Uploading ${file.name}...`;

  const formData = new FormData();
  formData.append('video', file);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload', true);
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 90);
      progressFill.style.width = `${pct}%`;
      statusText.innerText = `Uploading... ${pct}%`;
    }
  };

  xhr.onload = () => {
    if (xhr.status === 200) {
      const data = JSON.parse(xhr.responseText);
      progressFill.style.width = '100%';
      statusText.innerText = 'Upload complete! Probing video streams...';
      setTimeout(() => {
        progressWrap.style.display = 'none';
        dropContent.style.display = 'flex';
        renderSourceMedia(data);
      }, 400);
    } else {
      alert('Upload failed: ' + xhr.responseText);
      progressWrap.style.display = 'none';
      dropContent.style.display = 'flex';
    }
  };

  xhr.onerror = () => {
    alert('Upload failed due to network error.');
    progressWrap.style.display = 'none';
    dropContent.style.display = 'flex';
  };

  xhr.send(formData);
}

function renderSourceMedia(data) {
  clearJobMonitor();
  currentSource = data;
  const card = document.getElementById('source-media-card');
  const player = document.getElementById('source-player');
  const m = data.metadata || {};

  player.src = data.url;
  document.getElementById('meta-res').innerText = `${m.width || '?'}x${m.height || '?'}`;
  document.getElementById('meta-fps').innerText = `${m.fps || 24} fps`;
  document.getElementById('meta-dur').innerText = `${m.duration ? m.duration.toFixed(1) : '?'}s (${m.frames || '?'} frames)`;
  document.getElementById('meta-audio').innerText = m.has_audio ? 'Audio: Original' : 'Audio: None';

  // Render Filmstrip & Scene cuts
  const filmstrip = document.getElementById('timeline-filmstrip');
  filmstrip.innerHTML = '';
  (data.thumbnails || []).forEach(th => {
    const img = document.createElement('img');
    img.src = th.url;
    filmstrip.appendChild(img);
  });

  const markers = document.getElementById('scene-markers');
  markers.innerHTML = '';
  const dur = m.duration || 1.0;
  (data.scene_cuts || []).forEach(c => {
    const marker = document.createElement('div');
    marker.className = 'cut-marker';
    marker.style.left = `${(c / dur) * 100}%`;
    marker.title = `Scene cut at ${c.toFixed(2)}s`;
    markers.appendChild(marker);
  });

  // Timeline track seek
  const track = document.getElementById('timeline-track');
  track.onclick = (e) => {
    const rect = track.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    player.currentTime = frac * dur;
  };

  player.ontimeupdate = () => {
    const frac = (player.currentTime / dur) * 100;
    document.getElementById('timeline-playhead').style.left = `${frac}%`;
    document.getElementById('timeline-time').innerText = `${player.currentTime.toFixed(1)}s / ${dur.toFixed(1)}s`;
  };

  card.style.display = 'block';
  document.getElementById('enhance-btn').disabled = false;
  document.getElementById('test-render-btn').disabled = false;

  // Reset comparison cards and idle status when new video is uploaded
  const compCard = document.getElementById('comparison-card');
  if (compCard) compCard.style.display = 'none';
  const colorCard = document.getElementById('color-qa-card');
  if (colorCard) colorCard.style.display = 'none';
  const activeCard = document.getElementById('active-job-card');
  if (activeCard) activeCard.style.display = 'none';
  const idleCard = document.getElementById('idle-monitor-card');
  if (idleCard) {
    idleCard.style.display = 'block';
    idleCard.innerHTML = `
      <div class="idle-icon">🎬</div>
      <h3>Video Loaded: ${data.filename}</h3>
      <p style="margin-top:8px; color:var(--fg-muted);">Source: ${m.width || '?'}x${m.height || '?'} @ ${m.fps || 24} fps (${m.duration ? m.duration.toFixed(1) : '?'}s)</p>
      <p style="margin-top:6px; font-weight:600; color:var(--accent, #6366f1);">Click "✨ Enhance Video (Full Pass)" below to start processing, or "🧪 Quick Test" for a 17-frame test.</p>
    `;
  }

  updateGeometryCalculation();
}

// ==================== CONTROLS & GEOMETRY ====================
function initControls() {
  // Mode selection
  document.querySelectorAll('.mode-card').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-card').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset.mode;
      document.getElementById('restore-first-row').style.display = (mode === 'refine') ? 'block' : 'none';
      updateGeometryCalculation();
    });
  });

  // Resolution Mode
  document.querySelectorAll('.res-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.res-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const isScale = tab.dataset.resmode === 'scale';
      document.getElementById('scale-control-wrap').style.display = isScale ? 'block' : 'none';
      document.getElementById('preset-control-wrap').style.display = isScale ? 'none' : 'block';
      updateGeometryCalculation();
    });
  });

  // Scale slider
  const scaleSlider = document.getElementById('scale-slider');
  scaleSlider.addEventListener('input', () => {
    document.getElementById('scale-val').innerText = `${parseFloat(scaleSlider.value).toFixed(2)}x`;
    updateGeometryCalculation();
  });

  document.getElementById('long-edge-select').addEventListener('change', updateGeometryCalculation);

  // Guide Strength
  const guideSlider = document.getElementById('guide-slider');
  guideSlider.addEventListener('input', () => {
    const val = parseFloat(guideSlider.value);
    document.getElementById('guide-val').innerText = val.toFixed(2);
    document.getElementById('guide-warning').style.display = (val < 0.6) ? 'block' : 'none';
  });

  // LoRA slider
  const loraSlider = document.getElementById('lora-slider');
  loraSlider.addEventListener('input', () => {
    document.getElementById('lora-val').innerText = parseFloat(loraSlider.value).toFixed(2);
  });

  // Color strength slider
  const colorStrSlider = document.getElementById('color-str-slider');
  colorStrSlider.addEventListener('input', () => {
    document.getElementById('color-str-val').innerText = parseFloat(colorStrSlider.value).toFixed(2);
  });

  // CRF slider
  const crfSlider = document.getElementById('crf-slider');
  crfSlider.addEventListener('input', () => {
    document.getElementById('crf-val').innerText = crfSlider.value;
  });

  // Seed dice
  document.getElementById('seed-dice-btn').addEventListener('click', () => {
    document.getElementById('seed-input').value = Math.floor(Math.random() * 2147483647);
  });

  // Range checkbox
  document.getElementById('range-enable-chk').addEventListener('change', (e) => {
    document.getElementById('range-inputs').style.display = e.target.checked ? 'flex' : 'none';
    document.getElementById('range-highlight').style.display = e.target.checked ? 'block' : 'none';
  });

  // Precision select (INT8 vs GGUF)
  document.getElementById('precision-select').addEventListener('change', (e) => {
    document.getElementById('gguf-file-row').style.display = (e.target.value === 'gguf') ? 'block' : 'none';
  });

  // Cached conditioning toggle
  document.getElementById('cached-cond-chk').addEventListener('change', (e) => {
    document.getElementById('custom-encoder-sub').style.display = e.target.checked ? 'none' : 'block';
  });

  // Sigmas mode
  document.getElementById('sigmas-mode-select').addEventListener('change', (e) => {
    document.getElementById('custom-sigmas-row').style.display = (e.target.value === 'custom') ? 'block' : 'none';
  });

  // Tile mode
  document.getElementById('tile-mode-select').addEventListener('change', (e) => {
    document.getElementById('manual-tiles-row').style.display = (e.target.value === 'manual') ? 'flex' : 'none';
  });

  // Action Buttons
  document.getElementById('test-render-btn').addEventListener('click', () => submitJob(true));
  document.getElementById('enhance-btn').addEventListener('click', () => submitJob(false));
  document.getElementById('cancel-active-btn').addEventListener('click', cancelActiveJob);

  // Preset Selection
  document.getElementById('preset-select').addEventListener('change', (e) => {
    applyPreset(e.target.value);
  });
}

async function updateGeometryCalculation() {
  if (!currentSource || !currentSource.metadata) return;
  const w = currentSource.metadata.width || 1920;
  const h = currentSource.metadata.height || 1080;
  const mode = document.querySelector('.mode-card.active').dataset.mode;
  const resMode = document.querySelector('.res-tab.active').dataset.resmode;
  const scale = parseFloat(document.getElementById('scale-slider').value);
  const longEdge = parseInt(document.getElementById('long-edge-select').value);

  try {
    const res = await fetch('/api/geometry', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ width: w, height: h, mode, resolution_mode: resMode, scale, preset_long_edge: longEdge })
    });
    const g = await res.json();
    document.getElementById('target-res-badge').innerText = `${g.tw} x ${g.th}`;
    document.getElementById('geom-grid').innerText = `/${g.grid}`;
    document.getElementById('geom-aerr').innerText = `${g.aspect_error}%`;
    document.getElementById('geom-actual').innerText = `${g.scale_actual}x`;
  } catch (err) {
    console.error('Geometry calculation error:', err);
  }
}

// ==================== PRESETS ====================
async function loadPresetsAndSchema() {
  try {
    const res = await fetch('/api/settings/schema');
    const data = await res.json();
    schemaData = data;
    presetsData = data.presets || {};

    // Populate dropdowns with discovered models from server ComfyUI
    const disc = data.discovered || {};
    populateSelect('gguf-file-select', disc.ggufs);
    populateSelect('encoder-select', disc.encoders);
    populateSelect('sampler-select', disc.samplers);

    // Load custom presets
    const pRes = await fetch('/api/presets');
    const customPresets = await pRes.json();
    const select = document.getElementById('preset-select');
    select.innerHTML = '';
    for (const [key, p] of Object.entries(customPresets)) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.innerText = p.label;
      select.appendChild(opt);
    }
  } catch (err) {
    console.error('Failed to load schema & presets:', err);
  }
}

function populateSelect(id, list) {
  const el = document.getElementById(id);
  if (!el || !Array.isArray(list) || list.length === 0) return;
  const currentVal = el.value;
  el.innerHTML = '';
  list.forEach(item => {
    const opt = document.createElement('option');
    opt.value = item;
    opt.innerText = item;
    el.appendChild(opt);
  });
  if (currentVal && list.includes(currentVal)) {
    el.value = currentVal;
  }
}

function applyPreset(presetKey) {
  const p = presetsData[presetKey];
  if (!p || !p.settings) return;
  const s = p.settings;

  if (s.mode) {
    document.querySelectorAll('.mode-card').forEach(c => {
      c.classList.toggle('active', c.dataset.mode === s.mode);
    });
  }
  if (s.scale !== undefined) {
    document.getElementById('scale-slider').value = s.scale;
    document.getElementById('scale-val').innerText = `${s.scale.toFixed(2)}x`;
  }
  if (s.guide_strength !== undefined) {
    document.getElementById('guide-slider').value = s.guide_strength;
    document.getElementById('guide-val').innerText = s.guide_strength.toFixed(2);
  }
  if (s.color_enabled !== undefined) document.getElementById('color-enable-chk').checked = s.color_enabled;
  if (s.color_mode) document.getElementById('color-mode-select').value = s.color_mode;
  if (s.color_strength !== undefined) {
    document.getElementById('color-str-slider').value = s.color_strength;
    document.getElementById('color-str-val').innerText = s.color_strength.toFixed(2);
  }
  if (s.precision) document.getElementById('precision-select').value = s.precision;
  if (s.cached_cond !== undefined) document.getElementById('cached-cond-chk').checked = s.cached_cond;

  updateGeometryCalculation();
}

// ==================== SUBMIT & RUN ====================
function collectSettings(isTest = false) {
  const activeMode = document.querySelector('.mode-card.active');
  const mode = activeMode?.dataset?.mode || 'refine';

  const activeRes = document.querySelector('.res-tab.active');
  const resMode = activeRes?.dataset?.resmode || 'scale';

  const getVal = (id, def = '') => {
    const el = document.getElementById(id);
    return (el && el.value !== undefined && el.value !== null) ? el.value : def;
  };
  const getChecked = (id, def = false) => {
    const el = document.getElementById(id);
    return el ? Boolean(el.checked) : def;
  };
  const getFloat = (id, def = 0.0) => {
    const v = parseFloat(getVal(id, def));
    return isNaN(v) ? def : v;
  };
  const getInt = (id, def = 0) => {
    const v = parseInt(getVal(id, def), 10);
    return isNaN(v) ? def : v;
  };

  const s = {
    mode,
    restore_first: getChecked('restore-first-chk', false),
    resolution_mode: resMode,
    scale: getFloat('scale-slider', 1.5),
    preset_long_edge: getInt('long-edge-select', 1920),
    guide_strength: getFloat('guide-slider', 1.0),
    lora_strength: getFloat('lora-slider', 1.0),
    seed: getInt('seed-input', 42),
    randomize_seed: getChecked('seed-randomize-chk', false),

    // Color consistency
    color_enabled: getChecked('color-enable-chk', true),
    color_auto: getChecked('color-auto-chk', true),
    color_mode: getVal('color-mode-select', 'source') || 'source',
    color_strength: getFloat('color-str-slider', 0.7),
    protect_faces: getChecked('protect-faces-chk', true),
    protect_skin: getChecked('protect-skin-chk', false) ? 0.6 : 0.0,

    // Precision & Encoder
    precision: getVal('precision-select', 'int8') || 'int8',
    gguf_file: getVal('gguf-file-select', 'LTX-2.5-Distilled-Q4_K_M.gguf') || 'LTX-2.5-Distilled-Q4_K_M.gguf',
    cached_cond: getChecked('cached-cond-chk', true),
    encoder: (getVal('encoder-select', '') || 'gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors'),
    clip_device: getVal('clip-dev-select', 'default') || 'default',

    // Prompts
    prompt_positive: getVal('pos-prompt', ''),
    prompt_negative: getVal('neg-prompt', ''),

    // Sampler
    sampler_name: (getVal('sampler-select', 'euler') || 'euler'),
    sigmas_mode: (getVal('sigmas-mode-select', 'distilled') || 'distilled'),
    sigmas_custom: getVal('custom-sigmas-input', '1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0'),

    // Tiles & Decoders
    tile_mode: getVal('tile-mode-select', 'auto') || 'auto',
    tile_w: getInt('tile-w', 1024),
    tile_h: getInt('tile-h', 576),
    decode_tile: getInt('decode-tile-input', 0),

    // Memory
    budget_fmp: getFloat('budget-input', 0.0),
    snap_to_cuts: getChecked('snap-cuts-chk', true),

    // Audio & Export
    audio_mode: getVal('audio-mode-select', 'original') || 'original',
    container: getVal('container-select', 'mp4') || 'mp4',
    vcodec: getVal('codec-select', 'libx264') || 'libx264',
    crf: getInt('crf-slider', 14),

    // Test mode
    test_enabled: isTest,
    test_frames: 17
  };

  // Range in/out
  if (getChecked('range-enable-chk', false)) {
    s.range_start_s = getFloat('range-in', 0.0);
    s.range_end_s = getFloat('range-out', 0.0);
  }

  return s;
}

async function readJobSubmission(res) {
  const body = await res.text();
  const status = `HTTP ${res.status}`;
  let data;
  try {
    data = JSON.parse(body);
  } catch (_) {
    const reason = /^\s*</.test(body)
      ? 'The server or connection gateway returned an HTML page instead of the LTX API response.'
      : 'LTX returned an unreadable API response.';
    throw new Error(`${status}: ${reason}`);
  }
  if (!res.ok || !data?.success) {
    throw new Error(`${status}: ${typeof data?.error === 'string' ? data.error : 'LTX did not accept the request.'}`);
  }
  if (!data.job || !/^ltx_[a-zA-Z0-9_-]+$/.test(data.job.id || '')) {
    throw new Error(`${status}: The response did not include a valid job receipt.`);
  }
  return data;
}

async function postJobSubmission(url, payload) {
  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort(), 45000);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      ...(payload === undefined ? {} : {body: JSON.stringify(payload)}),
      signal: controller.signal
    });
    return await readJobSubmission(res);
  } catch (err) {
    if (controller.signal.aborted) throw new Error('The submission connection timed out after 45 seconds.');
    throw err;
  } finally {
    clearTimeout(deadline);
  }
}

function submissionFailureMessage(err) {
  return 'Submission could not be confirmed: ' + (err.message || err) +
    '\nYour source video and settings are unchanged. Check Queue & History before submitting again to avoid a duplicate render.';
}

async function submitJob(isTest = false) {
  if (submissionInFlight) return;
  if (!currentSource) {
    alert('Please upload a source video first.');
    return;
  }

  const btn = isTest ? document.getElementById('test-render-btn') : document.getElementById('enhance-btn');
  const originalHtml = btn ? btn.innerHTML : '';
  const buttons = ['test-render-btn', 'enhance-btn'].map(id => document.getElementById(id)).filter(Boolean);
  const disabledBefore = buttons.map(button => button.disabled);
  submissionInFlight = true;
  buttons.forEach(button => { button.disabled = true; });
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⏳ Starting Render...</span>';
  }

  try {
    const settings = collectSettings(isTest);
    console.log('Submitting job with settings:', settings);

    // Hide comparison and idle cards, show active card immediately
    const compCard = document.getElementById('comparison-card');
    if (compCard) compCard.style.display = 'none';
    const idleCard = document.getElementById('idle-monitor-card');
    if (idleCard) idleCard.style.display = 'none';
    const activeCard = document.getElementById('active-job-card');
    if (activeCard) {
      activeCard.style.display = 'block';
      document.getElementById('job-title').innerText = `Job: ${currentSource.filename}`;
      document.getElementById('active-stage-text').innerText = 'Starting render pipeline...';
      document.getElementById('active-progress-fill').style.width = '2%';
      document.getElementById('active-progress-pct').innerText = '0%';
      document.getElementById('active-eta-text').innerText = 'Initializing...';
    }

    const data = await postJobSubmission('/api/jobs', {
      input_video_path: currentSource.filepath,
      settings,
      name: currentSource.filename
    });
    rememberActiveJob(data.job.id);
    displayActiveJob(data.job);
    refreshJobsList();
  } catch (err) {
    console.error('Submission error:', err);
    alert(submissionFailureMessage(err));
    const activeCard = document.getElementById('active-job-card');
    if (activeCard) activeCard.style.display = 'none';
    const idleCard = document.getElementById('idle-monitor-card');
    if (idleCard) idleCard.style.display = 'block';
  } finally {
    submissionInFlight = false;
    buttons.forEach((button, index) => { button.disabled = disabledBefore[index]; });
    if (btn) {
      btn.innerHTML = originalHtml;
    }
  }
}

async function cancelActiveJob() {
  if (!activeJobId) return;
  try {
    await fetch(`/api/jobs/${activeJobId}/cancel`, { method: 'POST' });
  } catch (err) {
    console.error('Cancel error:', err);
  }
}

// ==================== SSE INSTALLER EVENTS ====================
function usesQuickTunnel() {
  return new URL(location.href).hostname.endsWith('.trycloudflare.com');
}

function refreshInstallerStatus() {
  if (!installerRefreshRequest) {
    installerRefreshRequest = (async () => {
      const controller = new AbortController();
      const deadline = setTimeout(() => controller.abort(), 10000);
      try {
        const res = await fetch('/api/installer/status', {cache: 'no-store', signal: controller.signal});
        if (!res.ok) throw new Error('Installer status HTTP ' + res.status);
        renderInstallerProgress(await res.json());
      } catch (err) {
        console.error('Could not refresh installer status:', err);
      } finally {
        clearTimeout(deadline);
      }
    })().finally(() => { installerRefreshRequest = null; });
  }
  return installerRefreshRequest;
}

function initSSEInstaller() {
  // Quick Tunnels buffer SSE indefinitely; use bounded, coalesced snapshots.
  if (usesQuickTunnel()) {
    refreshInstallerStatus();
    setInterval(() => refreshInstallerStatus(), 5000);
    return;
  }
  sseInstaller = new EventSource('/api/installer/events');
  sseInstaller.onmessage = (e) => {
    try {
      const doc = JSON.parse(e.data);
      renderInstallerProgress(doc);
    } catch (err) {
      console.error('SSE parse error:', err);
    }
  };
}

function renderInstallerProgress(doc) {
  const banner = document.getElementById('install-banner');
  const isRunning = doc.status === 'running' || doc.current !== null;

  if (!isRunning && doc.status === 'done') {
    banner.style.display = 'none';
    return;
  }

  banner.style.display = 'block';
  const cur = doc.current;
  const overall = doc.overall || {};

  if (cur) {
    document.getElementById('banner-file-name').innerText = `Downloading: ${cur.label || cur.file}`;
    const mbDone = (cur.done / 1e9).toFixed(2);
    const mbTotal = (cur.total / 1e9).toFixed(2);
    const speedMb = ((cur.speed_bps || 0) / 1e6).toFixed(1);
    const etaText = cur.eta_s ? `${Math.floor(cur.eta_s / 60)}m ${cur.eta_s % 60}s` : '--';

    document.getElementById('banner-file-meta').innerText = `${mbDone} GB / ${mbTotal} GB • ${speedMb} MB/s (Source: ${cur.source || 'CDN'})`;
    document.getElementById('banner-file-pct').innerText = `${cur.pct.toFixed(1)}%`;
    document.getElementById('banner-speed-eta').innerText = `${speedMb} MB/s • ETA: ${etaText}`;
    document.getElementById('banner-file-fill').style.width = `${cur.pct}%`;
  } else {
    document.getElementById('banner-file-name').innerText = `Setup in progress (${doc.status})...`;
    document.getElementById('banner-file-meta').innerText = 'Configuring paths & neural modules...';
    document.getElementById('banner-file-fill').style.width = '100%';
    document.getElementById('banner-file-pct').innerText = '';
    document.getElementById('banner-speed-eta').innerText = '';
  }

  const oPct = (overall.pct || 0).toFixed(1);
  document.getElementById('banner-overall-fill').style.width = `${oPct}%`;
  document.getElementById('banner-overall-pct').innerText = `${oPct}%`;

  // Action area for token requirement
  const actArea = document.getElementById('banner-action-area');
  const actions = doc.actions || [];
  const tokenAct = actions.find(a => a.id === 'hf_token');
  if (tokenAct) {
    actArea.innerHTML = `
      <input type="password" id="banner-token-input" placeholder="Hugging Face Token" class="mini-input" style="width:140px;">
      <button class="button primary smallbtn" onclick="saveBannerToken()">Save</button>
    `;
  } else {
    actArea.innerHTML = '';
  }
}

async function saveBannerToken() {
  const input = document.getElementById('banner-token-input');
  if (!input || !input.value.trim()) return;
  await fetch('/api/installer/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token: input.value.trim() })
  });
  alert('Token saved! Downloader will resume shortly.');
}

// ==================== SSE JOB PROGRESS ====================
function initSSEJobs() {
  try {
    const saved = new URL(location.href).searchParams.get('job') || sessionStorage.getItem('ltx.activeJobId');
    if (saved && /^ltx_[a-zA-Z0-9_-]+$/.test(saved)) activeJobId = saved;
  } catch (_) {}
  if (!usesQuickTunnel()) {
    sseJobs = new EventSource('/api/jobs/events');
    // A reconnect does not replay the server's previous state transitions.
    sseJobs.onopen = () => refreshJobsList();
    sseJobs.onerror = () => refreshJobsList();
    sseJobs.onmessage = (e) => {
      try {
        const job = JSON.parse(e.data);
        if (activeJobId && job.id === activeJobId) {
          jobMonitorEpoch++;
          displayActiveJob(job);
        }
        refreshJobsList();
      } catch (err) {
        console.error('Job SSE error:', err);
      }
    };
  }
  // Keep monitoring even when a proxy silently drops an idle SSE connection.
  setInterval(() => refreshJobsList(), 5000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshJobsList(); });
  window.addEventListener('online', () => refreshJobsList());
}

function rememberActiveJob(id) {
  activeJobId = id;
  jobMonitorEpoch++;
  try {
    sessionStorage.setItem('ltx.activeJobId', id);
    const url = new URL(location.href);
    url.searchParams.set('job', id);
    history.replaceState(null, '', url.href);
  } catch (_) {}
}

function clearJobMonitor() {
  activeJobId = null;
  jobMonitorEpoch++;
  try {
    sessionStorage.removeItem('ltx.activeJobId');
    const url = new URL(location.href);
    url.searchParams.delete('job');
    history.replaceState(null, '', url.href);
  } catch (_) {}
}

function reconcileJobMonitor(jobs) {
  let selected = jobs.find(job => job.id === activeJobId);
  if (!selected && !currentSource) {
    selected = jobs.find(job => job.status === 'running') || jobs.find(job => job.status === 'queued');
    if (selected) rememberActiveJob(selected.id);
  }
  if (selected) displayActiveJob(selected);
}

function displayActiveJob(job) {
  const card = document.getElementById('active-job-card');
  const idle = document.getElementById('idle-monitor-card');

  if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
    if (job.status === 'completed') {
      showComparisonResult(job);
      return;
    }
    if (job.status === 'failed') {
      card.style.display = 'block';
      idle.style.display = 'none';
      document.getElementById('job-title').innerText = `Job Failed: ${job.name || job.id}`;
      document.getElementById('active-stage-text').innerHTML = `<span style="color:#ef4444; font-weight:600;">Error: ${job.error || 'Execution failed. Check Queue tab for logs.'}</span>`;
      document.getElementById('active-progress-fill').style.backgroundColor = '#ef4444';
      return;
    }
    setTimeout(() => {
      if (activeJobId !== job.id) return;
      card.style.display = 'none';
      idle.style.display = 'block';
    }, 4000);
    return;
  }

  // Handle queued state
  if (job.status === 'queued') {
    idle.style.display = 'none';
    card.style.display = 'block';
    const badge = document.querySelector('.active-job-badge');
    if (badge) {
      const pos = job.queue_position || 1;
      badge.innerText = `⏳ IN QUEUE (#${pos})`;
      badge.style.background = 'rgba(245, 158, 11, 0.2)';
      badge.style.color = '#f59e0b';
    }
    document.getElementById('job-title').innerText = `Job: ${job.name || job.id}`;
    document.getElementById('active-stage-text').innerText = job.stage || 'Waiting in queue...';
    document.getElementById('active-stage-text').style.color = '#f59e0b';
    document.getElementById('active-progress-fill').style.width = '100%';
    document.getElementById('active-progress-fill').style.backgroundColor = '#f59e0b';
    document.getElementById('active-progress-pct').innerText = `Queued (Position #${job.queue_position || 1})`;
    document.getElementById('active-step-info').innerText = 'Waiting for GPU';
    document.getElementById('active-eta-text').innerText = job.eta_text || 'In Queue';
    document.getElementById('pill-chunk').innerText = `Queue: #${job.queue_position || 1}`;
    document.getElementById('pill-mode').innerText = `Mode: ${job.settings ? job.settings.mode : 'Refine'}`;
    document.getElementById('pill-target').innerText = `Target: ${job.geometry ? `${job.geometry.tw}x${job.geometry.th}` : '--'}`;
    document.getElementById('pill-elapsed').innerText = 'Queued';
    return;
  }

  // Running state
  idle.style.display = 'none';
  card.style.display = 'block';
  const badge = document.querySelector('.active-job-badge');
  if (badge) {
    badge.innerText = 'ACTIVE RENDER';
    badge.style.background = 'rgba(214, 255, 32, 0.15)';
    badge.style.color = 'var(--accent, #d6ff20)';
  }
  document.getElementById('active-stage-text').style.color = 'var(--muted)';
  document.getElementById('active-progress-fill').style.backgroundColor = 'var(--accent, #d6ff20)';

  document.getElementById('job-title').innerText = `Job: ${job.name || job.id}`;
  document.getElementById('active-stage-text').innerText = job.stage || 'Processing...';

  const pct = (job.progress || 0).toFixed(1);
  document.getElementById('active-progress-fill').style.width = `${pct}%`;
  document.getElementById('active-progress-pct').innerText = `${pct}%`;

  const step = job.step || 0;
  const maxStep = job.max_steps || 8;
  document.getElementById('active-step-info').innerText = `Step ${step}/${maxStep}`;
  document.getElementById('active-eta-text').innerText = `ETA: ${job.eta_text || 'Calculating...'}`;

  document.getElementById('pill-chunk').innerText = `Chunk: ${job.current_chunk || 1}/${job.total_chunks || 1}`;
  document.getElementById('pill-mode').innerText = `Mode: ${job.settings ? job.settings.mode : 'Refine'}`;
  document.getElementById('pill-target').innerText = `Target: ${job.geometry ? `${job.geometry.tw}x${job.geometry.th}` : '--'}`;
  document.getElementById('pill-elapsed').innerText = `Elapsed: ${job.elapsed_seconds ? `${job.elapsed_seconds}s` : '--'}`;
}

function showComparisonResult(job) {
  const card = document.getElementById('active-job-card');
  if (card) card.style.display = 'none';
  const idle = document.getElementById('idle-monitor-card');
  if (idle) idle.style.display = 'none';

  const compCard = document.getElementById('comparison-card');
  const qaCard = document.getElementById('color-qa-card');

  if (job.preview_url) {
    document.getElementById('comp-img-before').src = job.preview_url;
    document.getElementById('comp-img-after').src = job.preview_url;
    compCard.style.display = 'block';
  }

  if (job.output_video_url) {
    const dlBtn = document.getElementById('download-result-btn');
    dlBtn.href = job.output_video_url;
    dlBtn.download = `enhanced_${job.id}.mp4`;

    document.getElementById('view-result-btn').onclick = () => {
      openVideoModal(job.output_video_url, `Enhanced Output (${job.id})`);
    };
  }

  // Color QA metrics
  if (job.color_analysis) {
    const qa = job.color_analysis;
    qaCard.style.display = 'block';
    const vBadge = document.getElementById('qa-verdict-badge');
    vBadge.innerText = qa.verdict.toUpperCase();
    vBadge.className = `badge ${qa.verdict === 'ok' ? 'highlight-badge' : ''}`;
    document.getElementById('qa-advice-text').innerText = qa.advice || '';

    const fl = qa.flicker || {};
    document.getElementById('q-flicker-l').innerText = `${fl.ratio_L ? fl.ratio_L.toFixed(2) : '1.00'}x`;
    const dr = qa.drift || {};
    document.getElementById('q-chroma-shift').innerText = `${dr.mean_chroma ? dr.mean_chroma.toFixed(2) : '0.00'}`;
    const seams = (qa.seams || []).filter(s => s.flag).length;
    document.getElementById('q-seams').innerText = `${seams} steps detected`;
  }
}

// ==================== INTERACTIVE SPLIT SLIDER ====================
function initSplitSlider() {
  const wrapper = document.getElementById('comp-wrapper');
  const overlay = document.getElementById('comp-overlay');
  const divider = document.getElementById('comp-divider');
  let isDragging = false;

  const setSplit = (x) => {
    const rect = wrapper.getBoundingClientRect();
    const pos = Math.max(0, Math.min(rect.width, x - rect.left));
    const pct = (pos / rect.width) * 100;
    overlay.style.width = `${pct}%`;
    divider.style.left = `${pct}%`;
  };

  divider.addEventListener('mousedown', () => isDragging = true);
  window.addEventListener('mouseup', () => isDragging = false);
  window.addEventListener('mousemove', (e) => {
    if (isDragging) setSplit(e.clientX);
  });

  // Touch support
  divider.addEventListener('touchstart', () => isDragging = true);
  window.addEventListener('touchend', () => isDragging = false);
  window.addEventListener('touchmove', (e) => {
    if (isDragging && e.touches.length > 0) setSplit(e.touches[0].clientX);
  });
}

// ==================== HARDWARE & JOBS LIST ====================
async function startHardwarePolling() {
  updateHardwareTelemetry();
  setInterval(updateHardwareTelemetry, 5000);
}

async function updateHardwareTelemetry() {
  try {
    const res = await fetch('/api/status');
    const s = await res.json();
    const connDot = document.getElementById('conn-dot');
    const connText = document.getElementById('conn-text');
    const vramPill = document.getElementById('vram-pill');

    if (s.active_job) {
      connDot.className = 'dot busy';
      connText.innerText = 'LTX Rendering';
    } else if (s.h3_busy) {
      connDot.className = 'dot busy';
      connText.innerText = 'H3 Studio Busy (Paused)';
    } else {
      connDot.className = 'dot';
      connText.innerText = 'RTX 5090 Ready';
    }

    const gpu = s.gpu || {};
    if (gpu.available) {
      const usedGb = ((gpu.vram_used_mb || 0) / 1024).toFixed(1);
      const totalGb = ((gpu.vram_total_mb || 32768) / 1024).toFixed(0);
      vramPill.innerText = `VRAM: ${usedGb} / ${totalGb} GB`;

      document.getElementById('tele-gpu-name').innerText = gpu.name || 'NVIDIA RTX 5090';
      document.getElementById('tele-vram-text').innerText = `${usedGb} / ${totalGb} GB`;
      document.getElementById('tele-util-text').innerText = `${gpu.util || 0}%`;
      const pct = (gpu.vram_used_mb / gpu.vram_total_mb) * 100;
      document.getElementById('tele-vram-fill').style.width = `${pct}%`;
    }

    const ram = s.ram || {};
    document.getElementById('tele-ram-text').innerText = `${ram.available_gb || 0} GB Free`;
    const disk = s.disk || {};
    document.getElementById('tele-disk-text').innerText = `${disk.free_gb || 0} GB Free`;
    document.getElementById('tele-h3-text').innerText = s.h3_busy ? 'Running (Active)' : 'Idle';

    const comfy = s.comfy || {};
    document.getElementById('worker-status-badge').innerText = comfy.running ? 'Running' : 'Stopped';
    document.getElementById('worker-pid').innerText = comfy.pid || '--';

    document.getElementById('queue-count-badge').innerText = (s.queue_length || 0) + (s.active_job ? 1 : 0);
  } catch (err) {
    document.getElementById('conn-dot').className = 'dot err';
    document.getElementById('conn-text').innerText = 'Server Disconnected';
  }
}

function refreshJobsList() {
  if (!jobsRefreshRequest) jobsRefreshRequest = loadJobsList().finally(() => { jobsRefreshRequest = null; });
  return jobsRefreshRequest;
}

async function loadJobsList() {
  const epoch = jobMonitorEpoch;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const res = await fetch('/api/jobs', {cache: 'no-store', signal: controller.signal});
    if (!res.ok) throw new Error('Could not refresh job status: HTTP ' + res.status);
    const jobs = await res.json();
    if (!Array.isArray(jobs)) throw new Error('Invalid jobs response');
    // A delayed poll must not replace a newer SSE event or newly submitted job.
    if (epoch === jobMonitorEpoch) reconcileJobMonitor(jobs);
    const tbody = document.getElementById('jobs-table-body');
    const pendingCount = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
    const badge = document.getElementById('queue-count-badge');
    if (badge) badge.innerText = pendingCount;
    if (!jobs || jobs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty-cell">No enhancement jobs yet.</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    jobs.forEach(j => {
      const tr = document.createElement('tr');
      const dur = j.elapsed_seconds ? `${j.elapsed_seconds}s` : (j.eta_text || '--');
      const timeStr = new Date(j.created_at * 1000).toLocaleTimeString();
      const geomStr = j.geometry ? `${j.geometry.tw}x${j.geometry.th}` : '--';
      const isFailedOrCancelled = j.status === 'failed' || j.status === 'cancelled';
      const isQueued = j.status === 'queued';

      tr.innerHTML = `
        <td><strong>${j.id}</strong></td>
        <td>${j.name || Path(j.input_video_path).name}</td>
        <td><span class="badge">${j.settings ? j.settings.mode : 'refine'}</span></td>
        <td>${geomStr}</td>
        <td>${timeStr}</td>
        <td>${dur}</td>
        <td><span class="badge ${j.status === 'completed' ? 'highlight-badge' : (isQueued ? 'queue-badge' : '')}">${j.status}${isQueued && j.queue_position ? ` (#${j.queue_position})` : ''}</span></td>
        <td>
          ${isQueued ? `<button class="button danger smallbtn" onclick="cancelJob('${j.id}')">Cancel</button>` : ''}
          ${isFailedOrCancelled ? `<button class="button alt smallbtn" onclick="retryJob('${j.id}')">🔄 Re-queue</button>` : ''}
          ${j.output_video_url ? `<button class="button alt smallbtn" onclick="openVideoModal('${j.output_video_url}', '${j.id}')">▶ Play</button>` : ''}
          ${j.output_video_url ? `<a class="button alt smallbtn" href="${j.output_video_url}" download>⬇ Save</a>` : ''}
          <button class="button danger smallbtn" onclick="deleteJob('${j.id}')">Delete</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to refresh jobs:', err);
  } finally {
    clearTimeout(timeout);
  }
}

async function retryJob(id) {
  if (submissionInFlight) return;
  submissionInFlight = true;
  try {
    const data = await postJobSubmission(`/api/jobs/${id}/retry`);
    if (data.success && data.job) {
      rememberActiveJob(data.job.id);
      displayActiveJob(data.job);
      refreshJobsList();
      alert(`Job re-queued successfully as ${data.job.id}!`);
    } else {
      alert('Failed to re-queue job: ' + (data.error || 'Unknown error'));
    }
  } catch (err) {
    alert(submissionFailureMessage(err));
  } finally {
    submissionInFlight = false;
  }
}

async function cancelJob(id) {
  try {
    await fetch(`/api/jobs/${id}/cancel`, { method: 'POST' });
    refreshJobsList();
  } catch (err) {
    console.error('Cancel error:', err);
  }
}

async function refreshModelsList() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const tbody = document.getElementById('models-table-body');
    if (!data.models) return;

    tbody.innerHTML = '';
    data.models.forEach(m => {
      const tr = document.createElement('tr');
      const szGb = (m.size / 1e9).toFixed(2);
      const dlGb = (m.downloaded / 1e9).toFixed(2);
      const statusBadge = m.ready
        ? '<span class="badge highlight-badge">Ready & Verified</span>'
        : (m.downloaded > 0 ? `<span class="badge">In Progress (${dlGb} GB)</span>` : '<span class="badge">Missing</span>');

      tr.innerHTML = `
        <td><strong>${m.label}</strong></td>
        <td><code>${m.name}</code></td>
        <td>${szGb} GB</td>
        <td>${dlGb} GB</td>
        <td>${statusBadge}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to load models list:', err);
  }
}

async function deleteJob(id) {
  if (!confirm(`Delete job record ${id}?`)) return;
  await fetch(`/api/jobs/${id}`, { method: 'DELETE' });
  refreshJobsList();
}

function openVideoModal(url, title) {
  const modal = document.getElementById('video-modal');
  const player = document.getElementById('modal-video-player');
  document.getElementById('modal-video-title').innerText = title;
  player.src = url;
  modal.style.display = 'flex';
  player.play();
}
