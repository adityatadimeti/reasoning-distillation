// Connect to WebSocket server
const socket = io();
const modelOutputElem = document.getElementById('model-output');
const statusElem = document.getElementById('connection-status');
const experimentsInfoElem = document.getElementById('experiment-details');
const problemsListElem = document.getElementById('problems-list');
const currentProblemElem = document.getElementById('current-problem');

// Current active problem
let activeProblemId = null;
// Store problem output by problem ID
let problemOutputs = {};

// Handle connection status
socket.on('connect', () => {
    statusElem.textContent = 'Connected';
    statusElem.style.color = '#28a745';
    
    // Signal to the server that we're ready to receive problem outputs
    setTimeout(() => {
        console.log('Sending client_ready signal to server');
        socket.emit('client_ready');
    }, 500);  // Small delay to ensure connection is fully established
});

socket.on('disconnect', () => {
    statusElem.textContent = 'Disconnected';
    statusElem.style.color = '#dc3545';
});

socket.on('status', (data) => {
    statusElem.textContent = data.message;
});

// Handle experiment status updates
socket.on('experiment_status', (data) => {
    // Update experiment information
    let html = `
        <p><strong>Experiment:</strong> ${data.experiment_name || 'N/A'}</p>
        <p><strong>Progress:</strong> ${data.completed || 0}/${data.total || 0} problems</p>
        <p><strong>Status:</strong> ${data.status || 'Starting...'}</p>
    `;
    
    if (data.config) {
        html += '<details><summary><strong>Configuration</strong></summary><pre>';
        html += JSON.stringify(data.config, null, 2);
        html += '</pre></details>';
    }
    
    experimentsInfoElem.innerHTML = html;
});

// Handle problem status updates
socket.on('problem_status', (data) => {
    const { problem_id, status } = data;
    
    // Check if we already have this problem in the list
    const existingProblem = document.getElementById(`problem-${problem_id}`);
    
    if (existingProblem) {
        // Update existing problem card
        existingProblem.className = `problem-card ${status}`;
        existingProblem.querySelector('.problem-status').textContent = status;
    } else {
        // Create new problem card
        const problemCard = document.createElement('div');
        problemCard.id = `problem-${problem_id}`;
        problemCard.className = `problem-card ${status}`;
        
        problemCard.innerHTML = `
            <strong>${problem_id}</strong>
            <div class="problem-status">${status}</div>
        `;
        
        // Initialize output storage
        problemOutputs[problem_id] = '';
        
        // Add click handler to view problem output
        problemCard.addEventListener('click', () => {
            // Remove active class from current active problem
            if (activeProblemId) {
                const activeElem = document.getElementById(`problem-${activeProblemId}`);
                if (activeElem) activeElem.classList.remove('active');
            }
            
            // Set this problem as active
            activeProblemId = problem_id;
            problemCard.classList.add('active');
            
            // Update output display
            currentProblemElem.textContent = problem_id;
            updateModelOutput(problem_id);
        });
        
        problemsListElem.appendChild(problemCard);
        
        // Auto-select first problem
        if (!activeProblemId) {
            problemCard.click();
        }
    }
});

// Handle model output chunks
socket.on('model_output', (data) => {
    const { problem_id, chunk } = data;
    
    console.log(`Received chunk for problem_id: ${problem_id}`);
    
    // Store the chunk
    if (!problemOutputs[problem_id]) {
        problemOutputs[problem_id] = '';
        console.log(`Initializing output storage for problem_id: ${problem_id}`);
    }
    problemOutputs[problem_id] += chunk;
    
    // If this is the active problem, update the display
    if (problem_id === activeProblemId) {
        console.log(`Updating display for active problem: ${problem_id}`);
        updateModelOutput(problem_id);
    } else {
        console.log(`Not updating display. Active: ${activeProblemId}, Received: ${problem_id}`);
    }
    
    // Auto-select this problem if no problem is currently selected
    if (!activeProblemId) {
        console.log(`No active problem, attempting to select: ${problem_id}`);
        const problemCard = document.getElementById(`problem-${problem_id}`);
        if (problemCard) {
            console.log(`Found problem card, clicking: ${problem_id}`);
            problemCard.click();
        } else {
            console.log(`Problem card not found for: ${problem_id}`);
        }
    }
});

// Format and display model output with answer highlighting
function updateModelOutput(problemId) {
    if (!problemOutputs[problemId]) {
        modelOutputElem.textContent = 'No output yet.';
        return;
    }
    
    let formattedOutput = problemOutputs[problemId];
    
    // Highlight <think> sections
    formattedOutput = formattedOutput.replace(
        /<think>([\s\S]*?)<\/think>/g, 
        '<div class="think-section"><strong>&lt;think&gt;</strong>$1<strong>&lt;/think&gt;</strong></div>'
    );
    
    // Highlight boxed answers
    formattedOutput = formattedOutput.replace(
        /\\boxed\{([^{}]+)\}/g,
        '<span class="answer-highlight">\\boxed{$1}</span>'
    );
    
    modelOutputElem.innerHTML = formattedOutput;
    
    // Scroll to the bottom
    modelOutputElem.scrollTop = modelOutputElem.scrollHeight;
} 