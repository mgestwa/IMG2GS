import { initViewer, loadPly, loadPlyFromBuffer, resizeViewer } from './viewer.js';

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const viewerContainer = document.getElementById('viewer-container');
const downloadBtn = document.getElementById('download-btn');
const resetBtn = document.getElementById('reset-btn');

let currentPlyUrl = null;

// Drag & Drop
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

resetBtn.addEventListener('click', () => {
    viewerContainer.style.display = 'none';
    dropZone.style.display = 'flex';
    fileInput.value = '';
    currentPlyUrl = null;
    statusText.textContent = "Processing... (Usually takes 5-10s)";
});

downloadBtn.addEventListener('click', () => {
    if (currentPlyUrl) {
        const link = document.createElement('a');
        link.href = currentPlyUrl;
        link.download = 'splat.ply';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});

async function handleFile(file) {
    // 1. Check if it is a PLY file (Local Viewing)
    if (file.name.toLowerCase().endsWith('.ply')) {
        console.log("Local PLY file detected:", file.name);

        dropZone.style.display = 'none';
        statusBar.style.display = 'none';
        viewerContainer.style.display = 'block';
        statusText.textContent = "Loading Local PLY...";

        resizeViewer();

        try {
            const buffer = await file.arrayBuffer();
            await loadPlyFromBuffer(buffer);
            statusText.textContent = "Loaded: " + file.name;
        } catch (err) {
            console.error(err);
            alert("Failed to load local PLY.");
            dropZone.style.display = 'flex';
            viewerContainer.style.display = 'none';
        }
        return;
    }

    // 2. Existing Image Upload Logic
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (JPG, PNG) or a 3D model (.ply).');
        return;
    }

    // UI Updates
    dropZone.style.display = 'none';
    statusBar.style.display = 'flex';
    statusText.textContent = "Uploading & Processing...";

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Processing failed');

        const data = await response.json();
        currentPlyUrl = data.url;

        // Success
        statusText.textContent = "Loading Viewer...";
        viewerContainer.style.display = 'block';
        statusBar.style.display = 'none';

        resizeViewer();
        await loadPly(currentPlyUrl);

    } catch (err) {
        console.error(err);
        statusText.textContent = "Error: " + err.message;
        setTimeout(() => {
            statusBar.style.display = 'none';
            dropZone.style.display = 'flex';
        }, 3000);
    }
}

// Init viewer once on load (it will wait for data)
initViewer(document.getElementById('viewer-canvas'));
