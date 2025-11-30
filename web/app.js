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
    if (!file.name.endsWith('.wav')) {
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
            addLog(`Classification complete: ${data.predicted_class}`, 'success');
            addLog(`Confidence: ${data.confidence}% | Processing time: ${data.processing_time}s`, 'info');
            showResults(data);
        } else {
            addLog(`ERROR: ${data.error}`, 'error');
        }
    } catch (error) {
        addLog(`ERROR: ${error.message}`, 'error');
        addLog('Make sure the API server is running on http://localhost:5000', 'warning');
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
            addLog(`Recording complete: ${(recordedAudioBlob.size / 1024).toFixed(2)} KB`, 'success');

            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        };

        mediaRecorder.start();
        recordingStartTime = Date.now();

        // UI updates
        document.getElementById('recordBtn').classList.add('recording');
        document.getElementById('recordIcon').textContent = '■';

        addLog('Recording started...', 'warning');

        // Start timer
        recordingTimer = setInterval(updateRecordingTimer, 100);

        // Start waveform visualization
        visualizeAudio();

        // Auto-stop after 20 seconds
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                stopRecording();
                addLog('Recording stopped: 20 second limit reached', 'warning');
            }
        }, 20000);

    } catch (error) {
        addLog(`ERROR: Microphone access denied - ${error.message}`, 'error');
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
}

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
}
for (let i = 0; i < bufferLength; i++) {
    const v = data
}