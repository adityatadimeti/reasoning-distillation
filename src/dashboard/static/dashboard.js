// Connect to WebSocket server
const socket = io();
const modelOutputElem = document.getElementById('model-output');
const modelOutputHeaderElem = document.getElementById('model-output-header');
const modelOutputContainerElem = document.getElementById('model-output-container');
const statusElem = document.getElementById('connection-status');
const experimentsInfoElem = document.getElementById('experiment-details');
const problemsListElem = document.getElementById('problems-list');
const currentProblemElem = document.getElementById('current-problem');
const answerInfoElem = document.getElementById('answer-info');
const answerContentElem = document.querySelector('#answer-info .answer-content');
const summaryHeaderElem = document.getElementById('summary-header');
const summaryContainerElem = document.getElementById('summary-container');
const summaryInfoElem = document.getElementById('summary-info');
const summaryContentElem = document.querySelector('#summary-info .summary-content');

// Current active problem
let activeProblemId = null;
// Store problem output by problem ID
let problemOutputs = {};
// Store answer information by problem ID
let answerInfo = {};
// Store summary information by problem ID
let summaryInfo = {};

// Store the complete experiment state
let experimentState = {
    experiment_name: 'N/A',
    completed: 0,
    total: 0,
    status: 'Starting...',
    config: null
};

// Setup collapsible sections
document.addEventListener('DOMContentLoaded', function() {
    // Setup model output toggle
    modelOutputHeaderElem.addEventListener('click', function() {
        toggleSection(modelOutputContainerElem, modelOutputHeaderElem.querySelector('.toggle-btn'));
    });
    
    // Setup summary toggle
    summaryHeaderElem.addEventListener('click', function() {
        toggleSection(summaryContainerElem, summaryHeaderElem.querySelector('.toggle-btn'));
    });
    
    // Initialize toggle buttons to match their content state
    document.querySelectorAll('.toggle-btn').forEach(button => {
        const contentElement = button.closest('.section-header').nextElementSibling;
        updateToggleButton(button, contentElement);
    });
});

// Function to toggle collapsible sections
function toggleSection(contentElement, buttonElement) {
    if (contentElement.classList.contains('collapsed')) {
        // Expand section
        contentElement.classList.remove('collapsed');
    } else {
        // Collapse section
        contentElement.classList.add('collapsed');
    }
    
    // Update button appearance to match content state
    updateToggleButton(buttonElement, contentElement);
}

// Helper function to ensure button state matches content state
function updateToggleButton(buttonElement, contentElement) {
    if (contentElement.classList.contains('collapsed')) {
        buttonElement.textContent = '+';
        buttonElement.classList.add('collapsed');
        buttonElement.setAttribute('aria-expanded', 'false');
        buttonElement.setAttribute('title', 'Expand section');
    } else {
        buttonElement.textContent = '−'; // Using minus sign (U+2212)
        buttonElement.classList.remove('collapsed');
        buttonElement.setAttribute('aria-expanded', 'true');
        buttonElement.setAttribute('title', 'Collapse section');
    }
}

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
    console.log(`Received experiment_status event. Status: ${data.status}`);
    console.log(`Has config: ${Boolean(data.config)}`);
    
    // Update only the fields that are present in the data
    if (data.experiment_name !== undefined) experimentState.experiment_name = data.experiment_name;
    if (data.completed !== undefined) experimentState.completed = data.completed;
    if (data.total !== undefined) experimentState.total = data.total;
    if (data.status !== undefined) experimentState.status = data.status;
    if (data.config !== undefined) experimentState.config = data.config;
    
    // Update experiment information
    let html = `
        <p><strong>Experiment:</strong> ${experimentState.experiment_name}</p>
        <p><strong>Progress:</strong> ${experimentState.completed}/${experimentState.total} problems</p>
        <p><strong>Status:</strong> ${experimentState.status}</p>
    `;
    
    if (experimentState.config) {
        console.log('Adding config details to UI');
        html += '<details class="config-details">';
        html += '<summary class="config-summary"><strong>Config</strong></summary>';
        html += '<div class="config-container">';
        
        // Main experiment config
        html += '<div class="config-section">';
        html += '<h4>Experiment Settings</h4>';
        html += '<table class="config-table">';
        
        // Display basic experiment settings
        const basicConfigKeys = [
            'experiment_name', 'results_dir', 'data_path', 'save_intermediate',
            'dashboard_port', 'reasoning_model', 'summarizer_type'
        ];
        
        basicConfigKeys.forEach(key => {
            if (experimentState.config[key] !== undefined) {
                html += `<tr><td>${key}</td><td>${experimentState.config[key]}</td></tr>`;
            }
        });
        html += '</table></div>';
        
        // Generation parameters
        html += '<div class="config-section">';
        html += '<h4>Generation Parameters</h4>';
        html += '<table class="config-table">';
        
        const genParamKeys = [
            'max_tokens', 'temperature', 'top_p', 'top_k', 
            'presence_penalty', 'frequency_penalty'
        ];
        
        genParamKeys.forEach(key => {
            if (experimentState.config[key] !== undefined) {
                html += `<tr><td>${key}</td><td>${experimentState.config[key]}</td></tr>`;
            }
        });
        html += '</table></div>';
        
        // Summarization parameters
        html += '<div class="config-section">';
        html += '<h4>Summarization Parameters</h4>';
        html += '<table class="config-table">';
        
        const summaryParamKeys = [
            'enable_summarization', 'summary_max_tokens', 'summary_temperature', 
            'summary_top_p', 'summary_top_k', 'summary_presence_penalty', 
            'summary_frequency_penalty'
        ];
        
        summaryParamKeys.forEach(key => {
            if (experimentState.config[key] !== undefined) {
                html += `<tr><td>${key}</td><td>${experimentState.config[key]}</td></tr>`;
            }
        });
        html += '</table></div>';
        
        // Prompt Templates
        html += '<div class="config-section">';
        html += '<h4>Prompt Templates</h4>';
        
        // Show reasoning prompt template if available
        if (experimentState.config.reasoning_prompt_template) {
            html += '<details class="prompt-details">';
            html += '<summary>Reasoning Prompt</summary>';
            html += `<pre class="prompt-template">${experimentState.config.reasoning_prompt_template}</pre>`;
            html += '</details>';
        }
        
        // Show summarization prompt template if available
        if (experimentState.config.summarize_prompt_template) {
            html += '<details class="prompt-details">';
            html += '<summary>Summarization Prompt</summary>';
            html += `<pre class="prompt-template">${experimentState.config.summarize_prompt_template}</pre>`;
            html += '</details>';
        }
        
        html += '</div>';
        
        // Raw config for advanced users
        html += '<details class="raw-config-details">';
        html += '<summary>Raw Configuration JSON</summary>';
        html += `<pre class="raw-config">${JSON.stringify(experimentState.config, null, 2)}</pre>`;
        html += '</details>';
        
        html += '</div>'; // end config-container
        html += '</details>'; // end main details
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
            updateAnswerInfo(problem_id);
            updateSummaryInfo(problem_id);
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
    
    // console.log(`Received chunk for problem_id: ${problem_id}`);
    
    // Store the chunk
    if (!problemOutputs[problem_id]) {
        problemOutputs[problem_id] = '';
        console.log(`Initializing output storage for problem_id: ${problem_id}`);
    }
    problemOutputs[problem_id] += chunk;
    
    // If this is the active problem, update the display
    if (problem_id === activeProblemId) {
        // console.log(`Updating display for active problem: ${problem_id}`);
        updateModelOutput(problem_id);
    } else {
        // console.log(`Not updating display. Active: ${activeProblemId}, Received: ${problem_id}`);
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

// Handle answer information
socket.on('answer_info', (data) => {
    const { problem_id, extracted_answer, correct_answer, is_correct } = data;
    
    console.log(`Received answer info for problem_id: ${problem_id}`);
    
    // Store the answer information
    answerInfo[problem_id] = {
        extractedAnswer: extracted_answer,
        correctAnswer: correct_answer,
        isCorrect: is_correct
    };
    
    // If this is the active problem, update the display
    if (problem_id === activeProblemId) {
        updateAnswerInfo(problem_id);
    }
});

// Handle summary information
socket.on('reasoning_summary', (data) => {
    const { problem_id, summary } = data;
    
    console.log(`Received summary for problem_id: ${problem_id}`);
    
    // Store the summary information
    summaryInfo[problem_id] = summary;
    
    // If this is the active problem, update the display
    if (problem_id === activeProblemId) {
        updateSummaryInfo(problem_id);
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

// Display answer information
function updateAnswerInfo(problemId) {
    if (!answerInfo[problemId]) {
        answerInfoElem.style.display = 'none';
        return;
    }
    
    const { extractedAnswer, correctAnswer, isCorrect } = answerInfo[problemId];
    
    // Create HTML for answer information
    let html = `
        <div class="answer-pair">
            <div class="answer-label">Extracted Answer:</div>
            <div class="answer-value">${extractedAnswer}</div>
        </div>
        <div class="answer-pair">
            <div class="answer-label">Correct Answer:</div>
            <div class="answer-value">${correctAnswer}</div>
        </div>
        <div class="answer-pair">
            <div class="answer-label">Status:</div>
            <div class="answer-value ${isCorrect ? 'correct' : 'incorrect'}">
                ${isCorrect ? 'Correct ✓' : 'Incorrect ✗'}
            </div>
        </div>
    `;
    
    answerContentElem.innerHTML = html;
    answerInfoElem.style.display = 'block';
}

// Display summary information
function updateSummaryInfo(problemId) {
    if (!summaryInfo[problemId]) {
        summaryInfoElem.style.display = 'none';
        summaryContainerElem.classList.add('collapsed');
        const toggleBtn = summaryHeaderElem.querySelector('.toggle-btn');
        updateToggleButton(toggleBtn, summaryContainerElem);
        return;
    }
    
    summaryContentElem.textContent = summaryInfo[problemId];
    summaryInfoElem.style.display = 'block';
    
    // Make sure the section is expanded when new content is available
    summaryContainerElem.classList.remove('collapsed');
    const toggleBtn = summaryHeaderElem.querySelector('.toggle-btn');
    updateToggleButton(toggleBtn, summaryContainerElem);
} 