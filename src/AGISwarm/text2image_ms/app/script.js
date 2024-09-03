let ws = new WebSocket(WEBSOCKET_URL);
let currentRequestID = '';

ws.onopen = function () {
    console.log("WebSocket connection established");
};

ws.onmessage = function (event) {
    const data = JSON.parse(event.data);
    console.log('Message received:', data);
    if (data["type"] == "generation_complete") {
        enableGenerateButton();
        return;
    };
    if (data["type"] == "waiting") {
        msg = data["msg"];
        console.log(msg);
        return;
    }

    imgElement = document.getElementById('image-output');
    let img = imgElement.querySelector('img');
    if (!img) {
        img = document.createElement('img');
        imgElement.appendChild(img);
    }
    img.src = data['latents'];
    console.log('Image updated:', data);
};

ws.onclose = function (event) {
    console.log("WebSocket connection closed");
};

function decodeBase64Image(base64String) {
    // Remove data URL prefix if present
    const base64Data = base64String.replace(/^data:image\/\w+;base64,/, '');

    // Decode base64
    const binaryString = atob(base64Data);

    // Create Uint8Array
    const uint8Array = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        uint8Array[i] = binaryString.charCodeAt(i);
    }

    // Create Blob
    return new Blob([uint8Array], { type: 'image/png' });
};


function sendMessage() {
    const prompt = document.getElementById('prompt').value;
    const negative_prompt = document.getElementById('negative_prompt').value;
    const num_inference_steps = document.getElementById('num_inference_steps').value;
    const guidance_scale = document.getElementById('guidance_scale').value;
    const width = document.getElementById('width').value;
    const height = document.getElementById('height').value;
    const seed = document.getElementById('seed').value;

    const message = {
        prompt: prompt,
        negative_prompt: negative_prompt,
        num_inference_steps: parseInt(num_inference_steps),
        guidance_scale: parseFloat(guidance_scale),
        width: parseInt(width),
        height: parseInt(height),
        seed: parseInt(seed)
    };

    ws.send(JSON.stringify(message));
    console.log('Message sent:', message);
    disableGenerateButton();
};

function updateStatus(step, total_steps) {
    const statusElement = document.getElementById('status');
    if (!statusElement) {
        const statusElement = document.createElement('div');
        statusElement.id = 'status';
        document.getElementById('image-output').appendChild(statusElement);
    }
    statusElement.textContent = `Generating: ${step}/${total_steps}`;
}


function updateImage(base64Image) {
    const img = document.getElementById('generated-image');
    if (!img) {
        const img = document.createElement('img');
        img.id = 'generated-image';
        document.body.appendChild(img);
    }
    img.src = `data:image/png;base64,${base64Image}`;
}

function disableGenerateButton() {
    const button = document.getElementById('send-btn');
    button.disabled = true;
    button.textContent = 'Generating...';
}

function enableGenerateButton() {
    const button = document.getElementById('send-btn');
    button.disabled = false;
    button.textContent = 'Generate';
}

function resetForm() {
    document.getElementById('num_inference_steps').value = '50';
    document.getElementById('guidance_scale').value = '7.5';
    document.getElementById('width').value = '512';
    document.getElementById('height').value = '512';
    document.getElementById('seed').value = '-1';
}

const menuToggle = document.getElementById('menu-toggle');
const configContainer = document.querySelector('.config-container');

menuToggle.addEventListener('click', () => {
    configContainer.classList.toggle('show');
});

document.addEventListener('click', (event) => {
    if (!configContainer.contains(event.target) && !menuToggle.contains(event.target)) {
        configContainer.classList.remove('show');
    }
});