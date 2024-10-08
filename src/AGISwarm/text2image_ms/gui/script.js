let ws = new WebSocket(WEBSOCKET_URL);
let abort_url = ABORT_URL;
let currentRequestID = '';
let idle = true;

ws.onopen = function () {
    console.log("WebSocket connection established");
};

ws.onmessage = function (event) {
    const data = JSON.parse(event.data);
    currentRequestID = data["task_id"];
    console.log('Message received:', data);
    switch (data["status"]) {
        case "starting":
            updateStatus("Starting generation");
            disableGenerateButton();
            break;
        case "waiting":
            queue_pos = data["content"]["queue_pos"];
            updateStatus("<br>" + "<span style='color:blue;'>You are in position " + queue_pos + " in the queue</span>" + "<br>");
            enableAbortButton();
            break;
        case "running":
            updateStatus(data["content"]["step"] + " / " + data["content"]["total_steps"]);
            if ("image" in data["content"]){
                updateImage(data["content"]["image"]);
            }
            enableAbortButton();
            break;
        case "finished":
            updateStatus("");
            enableGenerateButton();
            break;
        case "aborted":
            updateStatus("<br>" + "<span style='color:red;'>Generation aborted</span>" + "<br>");
            enableGenerateButton();
            break;
        case "error":
            updateStatus("<br>" + "<span style='color:red;'>Error in generation</span>" + "<br>");
            enableGenerateButton();
            break;
    }
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


function get_user_message_container(prompt, negative_prompt) {
    let user_message_container = document.createElement('div');
    user_message_container.classList.add('message-container');
    user_message = document.createElement('pre');
    user_message.classList.add('message');
    user_message.classList.add('prompt-message');
    user_message.textContent = prompt;
    user_message_container.appendChild(user_message);
    if (negative_prompt != '') {
        let negative_prompt_message_container = document.createElement('div');
        negative_prompt_message = document.createElement('pre');
        negative_prompt_message.classList.add('message');
        negative_prompt_message.classList.add('negative-prompt-message');
        negative_prompt_message.textContent = negative_prompt;
        negative_prompt_message_container.appendChild(negative_prompt_message);
        user_message_container.appendChild(negative_prompt_message_container);
    }
    return user_message_container;
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

    user_message_container = get_user_message_container(prompt, negative_prompt);
    document.getElementById('chat-output').insertBefore(user_message_container, document.getElementById('chat-output').firstChild);

    ws.send(JSON.stringify(message));
    console.log('Message sent:', message);
    disableGenerateButton();
};


function abortGeneration() {
    console.log(currentRequestID)
    fetch(abort_url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'task_id': currentRequestID })
    })
        .then(response => response.text())
        .catch(error => console.error('Error aborting generation:', error));
}

function sendButtonClick() {
    document.getElementById('send-btn').disabled = true;
    if (idle) {
        sendMessage();
    }
    else {
        abortGeneration();
    }
}

function enterSend(event) {
    if (event.key === 'Enter' && !event.ctrlKey) {
        event.preventDefault();
        if (idle){
            document.getElementById('send-btn').disabled = true;
            sendMessage();
        }
    } else if (event.key === 'Enter' && event.ctrlKey) {
        // Allow new line with Ctrl+Enter
        this.value += '\n';
    }
}

document.getElementById('prompt').addEventListener('keydown', enterSend);
document.getElementById('negative_prompt').addEventListener('keydown', enterSend);

function updateStatus(message) {
    botMessage = get_bot_message_container();
    statusElement = botMessage.querySelector('#status');
    if (!statusElement) {
        statusElement = document.createElement('pre');
        statusElement.id = 'status';
        botMessage.appendChild(statusElement);
    }
    statusElement.innerHTML = message;
};


function get_bot_message_container() {
    const chatOutput = document.getElementById('chat-output');
    // Check if the bot message div already exists
    let botMessageContainer = chatOutput.firstElementChild;
    let botMessage = botMessageContainer ? botMessageContainer.firstElementChild : null;
    if (!botMessage || !botMessage.classList.contains('bot-message')) {
        botMessageContainer = document.createElement('div');
        botMessageContainer.classList.add('message-container');
        botMessage = document.createElement('pre');
        botMessage.classList.add('message');
        botMessage.classList.add('bot-message');
        botMessage.appendChild(document.createElement('img'));
        botMessageContainer.appendChild(botMessage);

        chatOutput.insertBefore(botMessageContainer, chatOutput.firstChild);
    }
    return botMessage;
};

function scroll() {
    const chatOutput = document.getElementById('chat-output');
    const isAtBottom = chatOutput.scrollHeight - chatOutput.clientHeight <= chatOutput.scrollTop + 1;
    if (isAtBottom) {
        // If the user is at the bottom, scroll to the bottom
        chatOutput.scrollTop = chatOutput.scrollHeight;
    }
};

function updateImage(base64Image) {
    botMessage = get_bot_message_container();
    img = botMessage.querySelector('img');
    img.src = base64Image;
    console.log('Image updated');
    scroll();
};

function disableGenerateButton() {
    document.getElementById('send-btn').disabled = true;
};

function enableGenerateButton() {
    document.getElementById('send-btn').style.backgroundColor = "#363d46";
    document.getElementById('send-btn').innerHTML = '<i class="fa fa-paper-plane"></i>';
    document.getElementById('send-btn').disabled = false;
    idle = true;
};

function enableAbortButton() {
    if (!idle) {
        return;
    }
    document.getElementById('send-btn').style.backgroundColor = "#363d46";
    document.getElementById('send-btn').innerHTML = '<i class="fa fa-stop"></i>';
    document.getElementById('send-btn').disabled = false;
    idle = false;
}

function resetForm() {
    document.getElementById('num_inference_steps').value = '50';
    document.getElementById('guidance_scale').value = '7.5';
    document.getElementById('width').value = '512';
    document.getElementById('height').value = '512';
    document.getElementById('seed').value = '-1';
};

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