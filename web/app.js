// ===================================================================
// COLREG SOUND SIGNAL CLASSIFIER - FRONTEND APPLICATION
// ===================================================================

const API_BASE_URL = 'http://localhost:5000/api';

// Global state
let currentAudioFile = null;
let recordedAudioBlob = null;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimer = null;
let recordingStartTime = 0;
let audioContext = null;
let analyser = null;
let animationId = null;

// ===================================================================
// UTILITY FUNCTIONS
// ===================================================================

function addLog(message, type = 'info') {
    const logTerminal = document.getElementById('logTerminal');
    const timestamp = new Date().toLocaleTimeString();

    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;

    logTerminal.appendChild(entry);
    logTerminal.scrollTop = logTerminal.scrollHeight;
}

function addTrainingLog(message, type = 'info') {
    const trainingLog = document.getElementById('trainingLog');
    const timestamp = new Date().toLocaleTimeString();

    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;

    trainingLog.appendChild(entry);
    trainingLog.scrollTop = trainingLog.scrollHeight;
}

function clearLog() {
    document.getElementById('logTerminal').innerHTML = '';
    addLog('System log cleared', 'system');
}

function showResults(data) {
    const resultsPanel = document.getElementById('resultsPanel');
    resultsPanel.style.display = 'block';

    document.getElementById('resultSignal').textContent = data.predicted_class.toUpperCase();
    document.getElementById('resultConfidence').textContent = `${data.confidence}%`;
    document.getElementById('confidenceFill').style.width = `${data.confidence}%`;
    document.getElementById('processingTime').textContent = `${data.processing_time}s`;
    document.getElementById('audioDuration').textContent = `${data.audio_duration}s`;

    // Scroll to results
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResults() {
    document.getElementById('resultsPanel').style.display = 'none';
}

function updateSystemStatus(status, text) {
    const badge = document.getElementById('systemStatusBadge');
    const statusText = document.getElementById('statusText');

    statusText.textContent = text;

    if (status === 'ready') {
        badge.style.borderColor = 'var(--accent-green)';
        badge.style.background = 'rgba(85, 255, 85, 0.1)';
        statusText.style.color = 'var(--accent-green)';
    } else if (status === 'warning') {
        badge.style.borderColor = 'var(--accent-gold)';
        badge.style.background = 'rgba(245, 192, 65, 0.1)';
        statusText.style.color = 'var(--accent-gold)';
    } else {
        badge.style.borderColor = 'var(--danger-red)';
        badge.style.background = 'rgba(255, 85, 85, 0.1)';
        statusText.style.color = 'var(--danger-red)';
    }
}

// ===================================================================
// FILE UPLOAD HANDLERS
// ===================================================================

function setupFileUpload() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const classifyBtn = document.getElementById('classifyBtn');
    const clearFileBtn = document.getElementById('clearFileBtn');
    const fileInfo = document.getElementById('fileInfo');

    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('active');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('active');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('active');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Classify button
    classifyBtn.addEventListener('click', classifyUploadedFile);

    // Clear button
    clearFileBtn.addEventListener('click', () => {
        currentAudioFile = null;
        fileInput.value = '';
        fileInfo.textContent = 'No file selected';
        classifyBtn.disabled = true;
        clearFileBtn.disabled = true;
        uploadZone.classList.remove('active');
        hideResults();
        addLog('File selection cleared', 'system');
    });
}

function handleFileSelect(file) {
    if (!file.name.toLowerCase().endsWith('.wav')) {
        addLog('ERROR: Only .wav files are supported', 'error');
        return;
    }

    currentAudioFile = file;
    document.getElementById('fileInfo').textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
    document.getElementById('classifyBtn').disabled = false;
    document.getElementById('clearFileBtn').disabled = false;
    document.getElementById('uploadZone').classList.add('active');

    addLog(`File loaded: ${file.name}`, 'success');
}

async function classifyUploadedFile() {
    if (!currentAudioFile) {
        addLog('ERROR: No file selected', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', currentAudioFile);

    addLog('Sending file to classifier...', 'warning');
    document.getElementById('classifyBtn').disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            addLog(`✓ Classification complete: ${data.predicted_class}`, 'success');
            addLog(`  Confidence: ${data.confidence}% | Processing: ${data.processing_time}s`, 'info');
            showResults(data);
        } else {
            addLog(`ERROR: ${data.error}`, 'error');
        }
    } catch (error) {
        addLog(`ERROR: ${error.message}`, 'error');
        addLog('⚠ Make sure the API server is running on http://localhost:5000', 'warning');
    } finally {
        document.getElementById('classifyBtn').disabled = false;
    }
}

// ===================================================================
// AUDIO RECORDING HANDLERS
// ===================================================================

function setupAudioRecorder() {
    const recordBtn = document.getElementById('recordBtn');
    const classifyRecordingBtn = document.getElementById('classifyRecordingBtn');
    const clearRecordingBtn = document.getElementById('clearRecordingBtn');

    recordBtn.addEventListener('click', toggleRecording);
    classifyRecordingBtn.addEventListener('click', classifyRecording);
    clearRecordingBtn.addEventListener('click', clearRecording);
}

async function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Setup audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.fftSize = 2048;

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            recordedAudioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            document.getElementById('classifyRecordingBtn').disabled = false;
            document.getElementById('clearRecordingBtn').disabled = false;
            addLog(`✓ Recording complete: ${(recordedAudioBlob.size / 1024).toFixed(2)} KB`, 'success');

            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        };

        mediaRecorder.start();
        recordingStartTime = Date.now();

        // UI updates
        document.getElementById('recordBtn').classList.add('recording');
        document.getElementById('recordIcon').textContent = '■';

        addLog('🎤 Recording started... (20 second limit)', 'warning');

        // Start timer
        recordingTimer = setInterval(updateRecordingTimer, 100);

        // Start waveform visualization
        visualizeAudio();

        // Auto-stop after 20 seconds
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                stopRecording();
                addLog('⚠ Recording stopped: 20 second limit reached', 'warning');
            }
        }, 20000);

    } catch (error) {
        addLog(`ERROR: Microphone access denied - ${error.message}`, 'error');
        addLog('⚠ Please allow microphone access in your browser settings', 'warning');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());

        clearInterval(recordingTimer);

        document.getElementById('recordBtn').classList.remove('recording');
        document.getElementById('recordIcon').textContent = '●';
        document.getElementById('recordTimer').textContent = '00:00';

        if (audioContext) {
            audioContext.close();
        }
    }
}

function updateRecordingTimer() {
    const elapsed = (Date.now() - recordingStartTime) / 1000;
    const minutes = Math.floor(elapsed / 60);
    const seconds = Math.floor(elapsed % 60);
    document.getElementById('recordTimer').textContent =
        `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

function visualizeAudio() {
    const canvas = document.getElementById('waveformCanvas');
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        animationId = requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        ctx.fillStyle = '#010711';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2;
        ctx.strokeStyle = '#55ff55';
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }

    draw();
}

async function classifyRecording() {
    if (!recordedAudioBlob) {
        addLog('ERROR: No recording available', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', recordedAudioBlob, 'recording.wav');

    addLog('Classifying recorded audio...', 'warning');
    document.getElementById('classifyRecordingBtn').disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            addLog(`✓ Classification complete: ${data.predicted_class}`, 'success');
            addLog(`  Confidence: ${data.confidence}% | Processing: ${data.processing_time}s`, 'info');
            showResults(data);
        } else {
            addLog(`ERROR: ${data.error}`, 'error');
        }
    } catch (error) {
        addLog(`ERROR: ${error.message}`, 'error');
    } finally {
        document.getElementById('classifyRecordingBtn').disabled = false;
    }
}

function clearRecording() {
    recordedAudioBlob = null;
    document.getElementById('classifyRecordingBtn').disabled = true;
    document.getElementById('clearRecordingBtn').disabled = true;

    // Clear waveform
    const canvas = document.getElementById('waveformCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#010711';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    addLog('Recording deleted', 'system');
    hideResults();
}

// ===================================================================
// PAGE NAVIGATION
// ===================================================================

function setupNavigation() {
    const navTabs = document.querySelectorAll('.nav-tab');

    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetPage = tab.dataset.page;

            // Update active tab
            navTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Show target page
            document.querySelectorAll('.page').forEach(page => {
                page.style.display = 'none';
            });
            document.getElementById(`page-${targetPage}`).style.display = 'block';

            addLog(`→ Switched to ${targetPage.toUpperCase()} module`, 'system');
        });
    });
}

// ===================================================================
// TRAINING CONTROLS
// ===================================================================

function setupTrainingControls() {
    document.getElementById('runFullPipelineBtn').addEventListener('click', runFullPipeline);
    document.getElementById('generateCleanBtn').addEventListener('click', generateCleanData);
    document.getElementById('generateNoisyBtn').addEventListener('click', generateNoisyData);
    document.getElementById('trainModelBtn').addEventListener('click', trainModel);
}

async function runFullPipeline() {
    addTrainingLog('=== STARTING FULL TRAINING PIPELINE ===', 'warning');
    addTrainingLog('This will take 15-45 minutes depending on your hardware...', 'info');

    const btn = document.getElementById('runFullPipelineBtn');
    btn.disabled = true;
    btn.textContent = '⏳ RUNNING PIPELINE...';

    try {
        const response = await fetch(`${API_BASE_URL}/train/full_pipeline`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            addTrainingLog('=== PIPELINE COMPLETE ===', 'success');
            addTrainingLog('✓ Model trained successfully!', 'success');

            // Display output
            if (data.output) {
                data.output.split('\n').forEach(line => {
                    if (line.trim()) addTrainingLog(line, 'info');
                });
            }

            // Update status
            document.getElementById('modelStatus').textContent = 'Trained';
            document.getElementById('modelStatus').style.color = 'var(--accent-green)';
            document.getElementById('lastTraining').textContent = new Date().toLocaleString();

            updateSystemStatus('ready', 'SYSTEM READY');
            addLog('✓ Model training complete! You can now classify audio.', 'success');

        } else {
            addTrainingLog('=== PIPELINE FAILED ===', 'error');
            addTrainingLog(`ERROR: ${data.error}`, 'error');
            if (data.output) {
                addTrainingLog('Output:', 'warning');
                data.output.split('\n').forEach(line => {
                    if (line.trim()) addTrainingLog(line, 'error');
                });
            }
        }
    } catch (error) {
        addTrainingLog(`ERROR: ${error.message}`, 'error');
        addTrainingLog('⚠ Make sure the API server is running', 'warning');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>🚀</span> RUN FULL PIPELINE';
    }
}

async function generateCleanData() {
    addTrainingLog('Generating clean training data...', 'warning');
    const btn = document.getElementById('generateCleanBtn');
    btn.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/train/generate_clean`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            addTrainingLog('✓ Clean data generation complete!', 'success');
            if (data.output) {
                data.output.split('\n').forEach(line => {
                    if (line.trim()) addTrainingLog(line, 'info');
                });
            }
        } else {
            addTrainingLog(`ERROR: ${data.error}`, 'error');
        }
    } catch (error) {
        addTrainingLog(`ERROR: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
    }
}

async function generateNoisyData() {
    addTrainingLog('Generating noisy training data...', 'warning');
    const btn = document.getElementById('generateNoisyBtn');
    btn.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/train/generate_noisy`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            addTrainingLog('✓ Noisy data generation complete!', 'success');
            if (data.output) {
                data.output.split('\n').forEach(line => {
                    if (line.trim()) addTrainingLog(line, 'info');
                });
            }
        } else {
            addTrainingLog(`ERROR: ${data.error}`, 'error');
        }
    } catch (error) {
        addTrainingLog(`ERROR: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
    }
}

async function trainModel() {
    addTrainingLog('Starting model training...', 'warning');
    addTrainingLog('This may take 20-40 minutes...', 'info');

    const btn = document.getElementById('trainModelBtn');
    btn.disabled = true;
    btn.textContent = '⏳ TRAINING...';

    try {
        const response = await fetch(`${API_BASE_URL}/train/train`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            addTrainingLog('✓ Model training complete!', 'success');

            if (data.output) {
                data.output.split('\n').forEach(line => {
                    if (line.trim()) addTrainingLog(line, 'info');
                });
            }

            document.getElementById('modelStatus').textContent = 'Trained';
            document.getElementById('modelStatus').style.color = 'var(--accent-green)';
            document.getElementById('lastTraining').textContent = new Date().toLocaleString();

            updateSystemStatus('ready', 'SYSTEM READY');

        } else {
            addTrainingLog(`ERROR: ${data.error}`, 'error');
        }
    } catch (error) {
        addTrainingLog(`ERROR: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>3️⃣</span> TRAIN MODEL';
    }
}

// ===================================================================
// SYSTEM STATUS CHECK
// ===================================================================

async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        const data = await response.json();

        // Update system info page
        document.getElementById('deviceType').textContent = data.device || 'Unknown';
        document.getElementById('modelLoaded').textContent = data.model_loaded ? 'Yes' : 'No';
        document.getElementById('apiStatus').textContent = 'Connected';

        document.getElementById('deviceType').style.color = 'var(--accent-green)';
        document.getElementById('apiStatus').style.color = 'var(--accent-green)';

        if (data.model_loaded) {
            document.getElementById('modelLoaded').style.color = 'var(--accent-green)';
            document.getElementById('modelStatus').textContent = 'Loaded';
            document.getElementById('modelStatus').style.color = 'var(--accent-green)';

            updateSystemStatus('ready', 'SYSTEM READY');
            addLog('✓ System ready | Model loaded successfully', 'success');
            addLog(`  Device: ${data.device} | Classes: ${data.classes}`, 'info');
        } else {
            document.getElementById('modelLoaded').style.color = 'var(--accent-gold)';
            document.getElementById('modelStatus').textContent = 'Not Trained';
            document.getElementById('modelStatus').style.color = 'var(--accent-gold)';

            updateSystemStatus('warning', 'MODEL NOT LOADED');
            addLog('⚠ Model not loaded. Please train the model first.', 'warning');
            addLog('  Go to TRAINING tab and click "RUN FULL PIPELINE"', 'info');
        }

    } catch (error) {
        updateSystemStatus('error', 'API OFFLINE');
        document.getElementById('apiStatus').textContent = 'Disconnected';
        document.getElementById('apiStatus').style.color = 'var(--danger-red)';

        addLog('✗ Cannot connect to API server', 'error');
        addLog('  Please start the server: python web/api_server.py', 'warning');
    }
}

// ===================================================================
// INITIALIZATION
// ===================================================================

function initialize() {
    addLog('=== COLREG SOUND SIGNAL CLASSIFIER v2.4.1 ===', 'system');
    addLog('Kongsberg Maritime Systems | CNN+GRU Architecture', 'system');
    addLog('System initialized', 'success');

    setupNavigation();
    setupFileUpload();
    setupAudioRecorder();
    setupTrainingControls();

    document.getElementById('clearLogBtn').addEventListener('click', clearLog);

    // Check system status
    setTimeout(checkSystemStatus, 500);
}

// Start application when DOM is ready
document.addEventListener('DOMContentLoaded', initialize);