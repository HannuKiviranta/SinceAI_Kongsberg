const API_BASE_URL = '/api'; // Relative path works in Docker

let currentFilename = null; // Stores the filename on the server
let mediaRecorder = null;
let audioChunks = [];
let recordingStartTime = 0;
let recordingTimerInterval = null;

// --- LOGGING UTILS ---
function addLog(msg, type = 'info') {
    const term = document.getElementById('logTerminal');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="timestamp">[${new Date().toLocaleTimeString()}]</span> ${msg}`;
    term.appendChild(entry);
    term.scrollTop = term.scrollHeight;
}

// --- INIT ---
document.addEventListener('DOMContentLoaded', () => {
    setupFileUpload();
    setupRecorder();
    
    document.getElementById('classifyBtn').addEventListener('click', classifySignal);
    document.getElementById('clearFileBtn').addEventListener('click', resetUI);
});

// --- FILE UPLOAD ---
function setupFileUpload() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');

    zone.addEventListener('click', () => input.click());
    
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('active'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('active'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('active');
        if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });

    input.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileSelect(e.target.files[0]);
    });
}

async function handleFileSelect(file) {
    addLog(`Loading audio: ${file.name}...`, 'info');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        // 1. Send to Preview Endpoint
        addLog('Generating spectrogram visualization...', 'warning');
        const response = await fetch(`${API_BASE_URL}/preview`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.success) {
            // 2. Display Image
            const img = document.getElementById('spectrogramImage');
            img.src = data.spectrogram_image;
            document.getElementById('spectrogramContainer').style.display = 'block';
            
            // 3. Update UI State
            document.getElementById('fileInfo').textContent = `Loaded: ${file.name} (${data.duration.toFixed(1)}s)`;
            document.getElementById('classifyBtn').disabled = false;
            document.getElementById('clearFileBtn').disabled = false;
            
            // 4. Store filename for classification
            currentFilename = data.temp_filename;
            
            addLog('Spectrogram generated. Ready to classify.', 'success');
        } else {
            addLog(`Error: ${data.error}`, 'error');
        }
    } catch (e) {
        addLog(`Upload failed: ${e.message}`, 'error');
    }
}

// --- CLASSIFICATION ---
async function classifySignal() {
    if (!currentFilename) return;

    const btn = document.getElementById('classifyBtn');
    btn.disabled = true;
    btn.innerHTML = '<span>⌛</span> ANALYZING...';
    addLog('Running inference model...', 'warning');

    try {
        const formData = new FormData();
        formData.append('filename', currentFilename); // Send filename, not file

        const response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.success) {
            showResults(data);
            addLog(`Prediction: ${data.prediction} (${data.confidence}%)`, 'success');
        } else {
            addLog(`Classification error: ${data.error}`, 'error');
        }
    } catch (e) {
        addLog(`Network error: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>▶</span> ANALYZE SIGNAL';
    }
}

function showResults(data) {
    const panel = document.getElementById('resultsPanel');
    panel.style.display = 'block';
    document.getElementById('resultSignal').textContent = data.prediction;
    document.getElementById('resultConfidence').textContent = `${data.confidence}%`;
    document.getElementById('confidenceFill').style.width = `${data.confidence}%`;
}

function resetUI() {
    currentFilename = null;
    document.getElementById('spectrogramContainer').style.display = 'none';
    document.getElementById('spectrogramImage').src = '';
    document.getElementById('fileInfo').textContent = 'No signal loaded.';
    document.getElementById('resultsPanel').style.display = 'none';
    document.getElementById('classifyBtn').disabled = true;
    document.getElementById('clearFileBtn').disabled = true;
    document.getElementById('fileInput').value = '';
    addLog('Interface reset.', 'system');
}

// --- RECORDER ---
function setupRecorder() {
    const btn = document.getElementById('recordBtn');
    
    btn.addEventListener('click', async () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            startRecording();
        }
    });
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: 'audio/wav' });
            // Create a File object from the Blob
            const file = new File([blob], `rec_${Date.now()}.wav`, { type: "audio/wav" });
            handleFileSelect(file); // Pass to the same handler as upload
            
            document.getElementById('recordBtn').classList.remove('recording');
            document.getElementById('recordTimer').textContent = "00:00";
            clearInterval(recordingTimerInterval);
        };

        mediaRecorder.start();
        recordingStartTime = Date.now();
        recordingTimerInterval = setInterval(() => {
            const secs = Math.floor((Date.now() - recordingStartTime) / 1000);
            document.getElementById('recordTimer').textContent = `00:${secs.toString().padStart(2, '0')}`;
        }, 1000);

        document.getElementById('recordBtn').classList.add('recording');
        addLog('Microphone recording started...', 'warning');

    } catch (e) {
        addLog(`Mic Access Error: ${e.message}`, 'error');
    }
}

function stopRecording() {
    if (mediaRecorder) mediaRecorder.stop();
}