import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let scene, camera, renderer, controls;
let pointCloud;

export function initViewer(canvas) {
    // PREVENT CRASH: If renderer already exists, do not create another context.
    // This fixes "Too many active WebGL contexts" error.
    if (renderer) return;

    try {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050505);

        // Camera
        camera = new THREE.PerspectiveCamera(55, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
        camera.position.set(0, 0, 0);

        // Renderer with CRASH PROTECTION
        // 1. alpha: false might help performance
        // 2. powerPreference: "high-performance" is good but "default" is safer if crashing
        renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true,
            alpha: false,
            powerPreference: "default"
        });

        renderer.setSize(canvas.clientWidth, canvas.clientHeight);

        // CRITICAL FIX: Cap pixel ratio to 2.0. 
        // 4K screens have ratio=3 or 4, creating massive buffers that crash the GPU.
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));

        // CINEMATIC LOOK
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.0;

        // Controls
        controls = new OrbitControls(camera, canvas);
        controls.enableDamping = true;

        // Fix for "Cannot read properties of undefined (reading 'target')"
        // Initialize target to a safe default
        controls.target.set(0, 0, 5);

        // Auto-resize
        window.addEventListener('resize', resizeViewer, false);

        animate();

    } catch (e) {
        console.error("CRITICAL: Failed to initialize WebGL Viewer", e);
        alert("Your browser could not start the 3D engine. Try refreshing.");
    }
}

export function resizeViewer() {
    if (!camera || !renderer) return;
    const canvas = renderer.domElement;
    // We need to ensure the parent has size, so we look at the parent or just rely on the canvas filling it
    // If canvas is 100% w/h, clientWidth depends on parent.
    const width = canvas.parentElement.clientWidth;
    const height = canvas.parentElement.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

export async function loadPly(url) {
    console.log("Loading PLY from:", url);
    try {
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        await loadPlyFromBuffer(buffer);
    } catch (e) {
        console.error("Error loading PLY URL:", e);
        alert("Failed to load 3D model from URL. Check console.");
    }
}

export async function loadPlyFromBuffer(buffer) {
    if (pointCloud) {
        scene.remove(pointCloud);
        if (pointCloud.geometry) pointCloud.geometry.dispose();
        if (pointCloud.material) pointCloud.material.dispose();
    }

    try {
        const geometry = parsePly(buffer);
        console.log("PLY Parsed. Vertices:", geometry.attributes.position.count);

        if (geometry.attributes.position.count === 0) {
            console.warn("Point cloud is empty!");
            return;
        }

        // Create a soft gaussian texture
        const sprite = textureFromCanvas();

        const material = new THREE.PointsMaterial({
            size: 0.1,             // Smaller size for dense meshes
            vertexColors: true,
            map: sprite,
            alphaTest: 0.1,
            transparent: false,
            opacity: 1.0,
            sizeAttenuation: true
        });

        pointCloud = new THREE.Points(geometry, material);
        pointCloud.rotation.x = Math.PI; // Flip Y/Z coordinate system if needed

        scene.add(pointCloud);

        if (controls) {
            controls.target.set(0, 0, 0); // Center on origin
            controls.update();
        }

        if (camera) {
            // Reset camera to a reasonable distance
            camera.position.set(0, 0, 5);
            camera.lookAt(0, 0, 0);
        }

    } catch (e) {
        console.error("Error parsing PLY buffer:", e);
        alert("Failed to parse PLY file. Ensure it is a valid 3DGS binary PLY.");
    }
}

function parsePly(buffer) {
    const textDecoder = new TextDecoder();
    const headerLimit = 4000;
    const headerText = textDecoder.decode(buffer.slice(0, headerLimit));

    // Look for various newline combinations
    let headerEndIndex = headerText.indexOf('end_header\n');
    let headerOffset = 'end_header\n'.length;

    if (headerEndIndex === -1) {
        headerEndIndex = headerText.indexOf('end_header\r\n');
        headerOffset = 'end_header\r\n'.length;
    }

    if (headerEndIndex === -1) throw new Error("PLY header not found");

    const bodyStart = headerEndIndex + headerOffset;

    const lines = headerText.substring(0, headerEndIndex).split(/\r?\n/);
    let vertexCount = 0;

    lines.forEach(line => {
        if (line.trim().startsWith('element vertex')) {
            const parts = line.trim().split(/\s+/);
            vertexCount = parseInt(parts[2]);
        }
    });

    console.log(`Parsing ${vertexCount} vertices starting at byte ${bodyStart}`);

    const dataView = new DataView(buffer, bodyStart);
    const floatSize = 4;
    // x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, s0,s1,s2, r0,r1,r2,r3
    const stride = 17 * floatSize;

    const positions = [];
    const colors = [];

    // Safety check for buffer size
    const expectedBytes = vertexCount * stride;
    if (buffer.byteLength - bodyStart < expectedBytes) {
        console.warn(`Buffer too small! Expected ${expectedBytes}, got ${buffer.byteLength - bodyStart}. Truncating count.`);
        vertexCount = Math.floor((buffer.byteLength - bodyStart) / stride);
    }

    const C0 = 0.28209479177387814;

    for (let i = 0; i < vertexCount; i++) {
        const offset = i * stride;

        // Positions (x, y, z)
        const x = dataView.getFloat32(offset, true);
        const y = dataView.getFloat32(offset + 4, true);
        const z = dataView.getFloat32(offset + 8, true);

        // Colors (f_dc 0,1,2 offsets: 24, 28, 32)
        const f_dc_0 = dataView.getFloat32(offset + 24, true);
        const f_dc_1 = dataView.getFloat32(offset + 28, true);
        const f_dc_2 = dataView.getFloat32(offset + 32, true);

        // Convert SH DC to RGB
        let r = 0.5 + C0 * f_dc_0;
        let g = 0.5 + C0 * f_dc_1;
        let b = 0.5 + C0 * f_dc_2;

        positions.push(x, y, z);
        colors.push(r, g, b); // Three.js handles clamping float colors [0..1]
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    return geometry;
}

function textureFromCanvas() {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const context = canvas.getContext('2d');

    // Draw a radial gradient (white to transparent)
    const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.4, 'rgba(255,255,255,0.8)');
    gradient.addColorStop(1, 'rgba(0,0,0,0)');

    context.fillStyle = gradient;
    context.fillRect(0, 0, 32, 32);

    const texture = new THREE.CanvasTexture(canvas);
    return texture;
}
