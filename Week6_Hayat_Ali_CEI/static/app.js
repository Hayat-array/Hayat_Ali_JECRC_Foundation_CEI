/* =====================================================================
   app.js — Frontend logic for CNN-ViT Image Denoiser
   Handles: Draw mode | Upload mode | Preset mode | API calls | Rendering
   ===================================================================== */

document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ──────────────────────────────────────────────────────
    const loadingOverlay = document.getElementById('loading-overlay');
    const loaderText     = document.getElementById('loader-text');

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const modeDraw   = document.getElementById('mode-draw');
    const modeUpload = document.getElementById('mode-upload');
    const modePreset = document.getElementById('mode-preset');

    // Draw mode
    const drawCanvas  = document.getElementById('draw-canvas');
    const dCtx        = drawCanvas.getContext('2d');
    const toolPen     = document.getElementById('tool-pen');
    const toolEraser  = document.getElementById('tool-eraser');
    const toolClear   = document.getElementById('tool-clear');
    const brushSlider = document.getElementById('brush-size');
    const brushLabel  = document.getElementById('brush-label');

    // Upload mode
    const uploadZone    = document.getElementById('upload-zone');
    const fileInput     = document.getElementById('file-input');
    const browseBtn     = document.getElementById('browse-btn');
    const uploadPreview = document.getElementById('upload-preview');
    const previewImg    = document.getElementById('preview-img');
    const removeUpload  = document.getElementById('remove-upload');

    // Preset mode
    const digitBtns     = document.querySelectorAll('.digit-btn');
    const btnRandom     = document.getElementById('btn-random');
    const presetPreviewWrap = document.getElementById('preset-preview-wrap');
    const presetCanvas  = document.getElementById('preset-canvas');
    const presetLabel   = document.getElementById('preset-label');

    // Shared
    const noiseSlider = document.getElementById('noise-factor');
    const noiseLabel  = document.getElementById('noise-label');
    const btnDenoise  = document.getElementById('btn-denoise');

    // Results
    const placeholderState = document.getElementById('placeholder-state');
    const activeResults    = document.getElementById('active-results');
    const resultBadge      = document.getElementById('result-badge');
    const outClean    = document.getElementById('out-clean');
    const outNoisy    = document.getElementById('out-noisy');
    const outDenoised = document.getElementById('out-denoised');
    const noisyEta    = document.getElementById('noisy-eta');

    const mPsnr = document.getElementById('m-psnr');
    const mSsim = document.getElementById('m-ssim');
    const mMse  = document.getElementById('m-mse');
    const barPsnr = document.getElementById('bar-psnr');
    const barSsim = document.getElementById('bar-ssim');
    const barMse  = document.getElementById('bar-mse');

    // ── State ─────────────────────────────────────────────────────────
    let currentMode    = 'draw';        // 'draw' | 'upload' | 'preset'
    let drawTool       = 'pen';         // 'pen' | 'eraser'
    let brushSize      = 18;
    let isDrawing      = false;

    let uploadedFile   = null;          // File object (upload mode)
    let presetGrid     = null;          // 28×28 float array (preset mode)

    // ── Hidden 28×28 canvas for pixel sampling ────────────────────────
    const offCanvas = document.createElement('canvas');
    offCanvas.width = offCanvas.height = 28;
    const offCtx = offCanvas.getContext('2d');

    // ─────────────────────────────────────────────────────────────────
    //  TAB SWITCHING
    // ─────────────────────────────────────────────────────────────────
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            if (mode === currentMode) return;
            currentMode = mode;

            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            modeDraw.classList.toggle('hidden',   mode !== 'draw');
            modeUpload.classList.toggle('hidden', mode !== 'upload');
            modePreset.classList.toggle('hidden', mode !== 'preset');
        });
    });

    // ─────────────────────────────────────────────────────────────────
    //  DRAW MODE
    // ─────────────────────────────────────────────────────────────────
    function initDraw() {
        dCtx.fillStyle = '#000';
        dCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    }
    initDraw();

    function getPos(e) {
        const r = drawCanvas.getBoundingClientRect();
        const cx = e.touches ? e.touches[0].clientX : e.clientX;
        const cy = e.touches ? e.touches[0].clientY : e.clientY;
        return { x: cx - r.left, y: cy - r.top };
    }

    drawCanvas.addEventListener('mousedown',  e => { isDrawing = true; drawAt(e); });
    drawCanvas.addEventListener('mousemove',  e => drawAt(e));
    drawCanvas.addEventListener('mouseup',    () => { isDrawing = false; dCtx.beginPath(); });
    drawCanvas.addEventListener('mouseleave', () => { isDrawing = false; dCtx.beginPath(); });
    drawCanvas.addEventListener('touchstart', e => { isDrawing = true; drawAt(e); }, { passive: false });
    drawCanvas.addEventListener('touchmove',  e => { e.preventDefault(); drawAt(e); }, { passive: false });
    drawCanvas.addEventListener('touchend',   () => { isDrawing = false; dCtx.beginPath(); });

    function drawAt(e) {
        if (!isDrawing) return;
        const { x, y } = getPos(e);
        dCtx.lineWidth  = brushSize;
        dCtx.lineCap    = 'round';
        dCtx.lineJoin   = 'round';
        dCtx.strokeStyle = drawTool === 'pen' ? '#fff' : '#000';
        dCtx.lineTo(x, y);
        dCtx.stroke();
        dCtx.beginPath();
        dCtx.moveTo(x, y);
    }

    toolPen.addEventListener('click', () => {
        drawTool = 'pen';
        toolPen.classList.add('active');
        toolEraser.classList.remove('active');
    });
    toolEraser.addEventListener('click', () => {
        drawTool = 'eraser';
        toolEraser.classList.add('active');
        toolPen.classList.remove('active');
    });
    toolClear.addEventListener('click', initDraw);

    brushSlider.addEventListener('input', () => {
        brushSize = parseInt(brushSlider.value);
        brushLabel.textContent = brushSize + 'px';
    });

    // Extract a 28×28 grayscale float grid from draw canvas
    function getDrawGrid() {
        offCtx.fillStyle = '#000';
        offCtx.fillRect(0, 0, 28, 28);
        offCtx.drawImage(drawCanvas, 0, 0, 28, 28);
        const data = offCtx.getImageData(0, 0, 28, 28).data;
        const grid = [];
        for (let y = 0; y < 28; y++) {
            const row = [];
            for (let x = 0; x < 28; x++) {
                row.push(data[(y * 28 + x) * 4] / 255.0);
            }
            grid.push(row);
        }
        return grid;
    }

    // Draw a 28×28 preset grid back onto the main draw canvas
    function renderGridOnDraw(grid) {
        dCtx.fillStyle = '#000';
        dCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
        const cw = drawCanvas.width / 28;
        const ch = drawCanvas.height / 28;
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const v = Math.max(0, Math.min(1, grid[y][x]));
                if (v > 0.02) {
                    const c = Math.floor(v * 255);
                    dCtx.fillStyle = `rgb(${c},${c},${c})`;
                    dCtx.fillRect(x * cw, y * ch, cw + 1, ch + 1);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  UPLOAD MODE
    // ─────────────────────────────────────────────────────────────────
    uploadZone.addEventListener('click', () => fileInput.click());
    browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });

    uploadZone.addEventListener('dragover', e => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('image/')) setUploadedFile(f);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) setUploadedFile(fileInput.files[0]);
    });

    function setUploadedFile(file) {
        uploadedFile = file;
        const url = URL.createObjectURL(file);
        previewImg.src = url;
        uploadZone.classList.add('hidden');
        uploadPreview.classList.remove('hidden');
    }

    removeUpload.addEventListener('click', () => {
        uploadedFile = null;
        fileInput.value = '';
        previewImg.src = '';
        uploadPreview.classList.add('hidden');
        uploadZone.classList.remove('hidden');
    });

    // ─────────────────────────────────────────────────────────────────
    //  PRESET MODE
    // ─────────────────────────────────────────────────────────────────
    async function loadPreset(digit) {
        showLoading(digit !== null
            ? `Fetching MNIST test sample for digit "${digit}"…`
            : 'Loading random MNIST test sample…');
        try {
            const url = digit !== null ? `/api/sample?digit=${digit}` : '/api/sample';
            const res = await fetch(url);
            if (!res.ok) { const e = await res.json(); throw new Error(e.error); }
            const data = await res.json();
            presetGrid = data.image;
            renderGridOnCanvas(presetCanvas, data.image);
            presetPreviewWrap.classList.remove('hidden');
            presetLabel.textContent = `digit: ${data.label}`;
        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            hideLoading();
        }
    }

    digitBtns.forEach(btn => btn.addEventListener('click', () => loadPreset(parseInt(btn.dataset.digit))));
    btnRandom.addEventListener('click', () => loadPreset(null));

    function renderGridOnCanvas(targetCanvas, grid) {
        const ctx2 = targetCanvas.getContext('2d');
        const W = targetCanvas.width, H = targetCanvas.height;
        const cw = W / 28, ch = H / 28;
        ctx2.fillStyle = '#000';
        ctx2.fillRect(0, 0, W, H);
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const v = Math.max(0, Math.min(1, grid[y][x]));
                if (v > 0.02) {
                    const c = Math.floor(v * 255);
                    ctx2.fillStyle = `rgb(${c},${c},${c})`;
                    ctx2.fillRect(x * cw, y * ch, cw + 1, ch + 1);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  SHARED: NOISE SLIDER
    // ─────────────────────────────────────────────────────────────────
    noiseSlider.addEventListener('input', () => {
        noiseLabel.textContent = parseFloat(noiseSlider.value).toFixed(2);
    });

    // ─────────────────────────────────────────────────────────────────
    //  LOADING HELPERS
    // ─────────────────────────────────────────────────────────────────
    function showLoading(msg) {
        loaderText.textContent = msg || 'Processing…';
        loadingOverlay.classList.add('active');
    }
    function hideLoading() {
        loadingOverlay.classList.remove('active');
    }

    // ─────────────────────────────────────────────────────────────────
    //  SHOW RESULTS
    // ─────────────────────────────────────────────────────────────────
    function showResults(data, eta, label) {
        const nf = parseFloat(eta).toFixed(2);
        noisyEta.textContent = `η = ${nf}`;
        resultBadge.textContent = label || 'Custom input';

        // Set PNG images (returned as base64)
        outClean.src    = `data:image/png;base64,${data.clean_png}`;
        outNoisy.src    = `data:image/png;base64,${data.noisy_png}`;
        outDenoised.src = `data:image/png;base64,${data.reconstructed_png}`;

        // Metrics
        const { mse, psnr, ssim } = data.metrics;
        mPsnr.textContent = psnr.toFixed(2);
        mSsim.textContent = ssim.toFixed(4);
        mMse.textContent  = mse.toFixed(5);

        barPsnr.style.width = `${Math.min(100, (psnr / 30) * 100)}%`;
        barSsim.style.width = `${Math.min(100, ssim * 100)}%`;
        barMse.style.width  = `${Math.min(100, (mse / 0.15) * 100)}%`;

        placeholderState.classList.add('hidden');
        activeResults.classList.remove('hidden');

        // Scroll results into view on mobile
        activeResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // ─────────────────────────────────────────────────────────────────
    //  MAIN DENOISING FLOW
    // ─────────────────────────────────────────────────────────────────
    btnDenoise.addEventListener('click', async () => {
        const eta = noiseSlider.value;

        if (currentMode === 'upload') {
            // ── Full image upload path ──
            if (!uploadedFile) {
                alert('Please select an image file to upload first.');
                return;
            }
            showLoading('Encoding and processing uploaded image…');
            try {
                const b64 = await fileToBase64(uploadedFile);
                // Strip data:image/...;base64, prefix
                const rawB64 = b64.split(',')[1];

                const res = await fetch('/api/denoise-image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_b64: rawB64, noise_factor: parseFloat(eta) })
                });
                const data = await handleApiResponse(res);
                showResults(data, eta, `Uploaded image (${uploadedFile.name})`);
            } catch (err) {
                alert(`Error: ${err.message}`);
            } finally {
                hideLoading();
            }

        } else if (currentMode === 'preset') {
            // ── Preset / MNIST path ──
            if (!presetGrid) {
                alert('Please select a digit or load a random MNIST preset first.');
                return;
            }
            showLoading('Running CNN-ViT denoising on MNIST sample…');
            try {
                const res = await fetch('/api/denoise', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: presetGrid, noise_factor: parseFloat(eta) })
                });
                const data = await handleApiResponse(res);
                showResults(data, eta, `MNIST preset — digit "${presetLabel.textContent.split(': ')[1]}"`);
            } catch (err) {
                alert(`Error: ${err.message}`);
            } finally {
                hideLoading();
            }

        } else {
            // ── Draw mode ──
            const grid = getDrawGrid();
            const hasPixels = grid.some(row => row.some(v => v > 0.04));
            if (!hasPixels) {
                alert('Please draw something on the canvas first!');
                return;
            }
            showLoading('Running CNN-ViT denoising on drawn input…');
            try {
                const res = await fetch('/api/denoise', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: grid, noise_factor: parseFloat(eta) })
                });
                const data = await handleApiResponse(res);
                showResults(data, eta, 'Hand-drawn input');
            } catch (err) {
                alert(`Error: ${err.message}`);
            } finally {
                hideLoading();
            }
        }
    });

    // ─────────────────────────────────────────────────────────────────
    //  HELPERS
    // ─────────────────────────────────────────────────────────────────
    async function handleApiResponse(res) {
        if (res.status === 503) {
            const e = await res.json();
            throw new Error(e.error || 'Model not ready yet. Please wait for training to finish.');
        }
        if (!res.ok) {
            const e = await res.json();
            throw new Error(e.error || `Server error ${res.status}`);
        }
        return res.json();
    }

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload  = e => resolve(e.target.result);
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }
});
