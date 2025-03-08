// Legacy Dashboard.js 
// This file is maintained for backward compatibility
// It forwards to the new modular dashboard files

console.warn('dashboard.js is deprecated. Using modular dashboard files instead.');

// Nothing to do here as the modular files are loaded directly in the HTML
// This file is just kept to avoid breaking any existing code that might reference it

// Check if we're in static mode (viewing saved results)
console.log('DEBUG: DASHBOARD_STATIC_MODE typeof =', typeof window.DASHBOARD_STATIC_MODE);
console.log('DEBUG: DASHBOARD_STATIC_MODE value =', window.DASHBOARD_STATIC_MODE);
console.log('DEBUG: staticResults typeof =', typeof window.staticResults);
console.log('DEBUG: staticResults defined =', window.staticResults !== undefined);

const isStaticMode = window.DASHBOARD_STATIC_MODE === true || typeof window.staticResults !== 'undefined';

console.log('Dashboard initializing, static mode:', isStaticMode);
console.log('Static results available:', Boolean(window.staticResults));

// Connect to WebSocket server only if not in static mode and io is defined
const socket = !isStaticMode && (typeof io !== 'undefined') ? io() : null;
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
// Store problem outputs by problem ID and iteration
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
    console.log('DOM loaded, static mode:', isStaticMode);
    console.log('Static results available:', Boolean(window.staticResults));
    
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
    
    // If we have static results, initialize the dashboard with them
    if (isStaticMode && window.staticResults) {
        console.log('Initializing static dashboard...');
        initializeStaticDashboard(window.staticResults);
    } else if (isStaticMode) {
        console.error('Static mode detected but no static results found!');
    }
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

// Only set up socket event handlers if not in static mode
if (!isStaticMode && socket) {
    // Handle connection status
    socket.on('connect', () => {
        console.log('WebSocket connected');
        statusElem.textContent = 'Connected';
        statusElem.style.color = '#28a745';
        
        // Signal to the server that we're ready to receive problem outputs
        setTimeout(() => {
            console.log('Sending client_ready signal to server');
            socket.emit('client_ready');
            
            // Also request current experiment status
            console.log('Requesting current experiment status');
            socket.emit('request_current_status');
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
        // Update only the fields that are present in the data
        if (data.experiment_name !== undefined) experimentState.experiment_name = data.experiment_name;
        if (data.completed !== undefined) experimentState.completed = data.completed;
        if (data.total !== undefined) experimentState.total = data.total;
        if (data.status !== undefined) experimentState.status = data.status;
        if (data.config !== undefined) experimentState.config = data.config;
        
        updateExperimentInfoDisplay();
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
            problemOutputs[problem_id] = {};
            
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
        const { problem_id, chunk, iteration = 0 } = data;
        
        // Initialize problem structure if needed
        if (!problemOutputs[problem_id]) {
            problemOutputs[problem_id] = {};
        }
        
        // Initialize iteration if needed
        if (!problemOutputs[problem_id][iteration]) {
            problemOutputs[problem_id][iteration] = {
                reasoning: '',
                summary: '',
                answer: '',
                correct_answer: '',
                is_correct: false
            };
        }
        
        // Store the chunk
        problemOutputs[problem_id][iteration].reasoning += chunk;
        
        // If this is the active problem, update the display
        if (problem_id === activeProblemId) {
            updateModelOutput(problem_id);
        }
    });
    
    // Handle summary updates
    socket.on('summary', (data) => {
        const { problem_id, summary, iteration = 0 } = data;
        
        // Initialize problem structure if needed
        if (!problemOutputs[problem_id]) {
            problemOutputs[problem_id] = {};
        }
        
        // Initialize iteration if needed
        if (!problemOutputs[problem_id][iteration]) {
            problemOutputs[problem_id][iteration] = {
                reasoning: '',
                summary: '',
                answer: '',
                correct_answer: '',
                is_correct: false
            };
        }
        
        // Store the summary
        problemOutputs[problem_id][iteration].summary = summary;
        
        // If this is the active problem, update the display
        if (problem_id === activeProblemId) {
            updateModelOutput(problem_id);
        }
    });
    
    // Handle answer information
    socket.on('answer_info', (data) => {
        const { problem_id, answer, correct_answer, is_correct, iteration = 0 } = data;
        
        // Initialize problem structure if needed
        if (!problemOutputs[problem_id]) {
            problemOutputs[problem_id] = {};
        }
        
        // Initialize iteration if needed
        if (!problemOutputs[problem_id][iteration]) {
            problemOutputs[problem_id][iteration] = {
                reasoning: '',
                summary: '',
                answer: '',
                correct_answer: '',
                is_correct: false
            };
        }
        
        // Store the answer information
        problemOutputs[problem_id][iteration].answer = answer;
        problemOutputs[problem_id][iteration].correct_answer = correct_answer;
        problemOutputs[problem_id][iteration].is_correct = is_correct;
        
        // If this is the active problem, update the display
        if (problem_id === activeProblemId) {
            updateModelOutput(problem_id);
        }
    });
    
    // Handle reasoning summary
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
}

// Update the model output display
function updateModelOutput(problemId) {
    if (!problemOutputs[problemId]) {
        modelOutputElem.textContent = 'No output yet.';
        return;
    }
    
    let formattedContent = '';
    
    // Get the number of iterations
    const iterations = Object.keys(problemOutputs[problemId])
        .map(Number)
        .sort((a, b) => a - b);
    
    // Display each iteration
    for (const iteration of iterations) {
        const iterData = problemOutputs[problemId][iteration];
        
        // Display the iteration heading
        formattedContent += `<h3>Iteration ${iteration}</h3>`;
        
        // Display the reasoning
        if (iterData.reasoning) {
            formattedContent += '<div class="reasoning-section">';
            formattedContent += '<h4>Reasoning</h4>';
            formattedContent += formatReasoning(iterData.reasoning);
            
            // Display the answer if available
            if (iterData.answer) {
                const correctClass = iterData.is_correct ? 'answer-correct' : 'answer-incorrect';
                formattedContent += `<div class="answer-section ${correctClass}">`;
                formattedContent += `<h4>Answer: ${iterData.answer}</h4>`;
                formattedContent += `<div>Correct answer: ${iterData.correct_answer}</div>`;
                formattedContent += `<div>Status: ${iterData.is_correct ? 'Correct' : 'Incorrect'}</div>`;
                formattedContent += '</div>';
            }
            
            formattedContent += '</div>';
        }
        
        // Display the summary if available (and not the last iteration)
        if (iterData.summary) {
            formattedContent += '<div class="summary-section">';
            formattedContent += '<h4>Summary</h4>';
            formattedContent += iterData.summary;
            formattedContent += '</div>';
        }
    }
    
    modelOutputElem.innerHTML = formattedContent;
    
    // Scroll to the bottom
    modelOutputElem.scrollTop = modelOutputElem.scrollHeight;
}

// Helper function to format reasoning text
function formatReasoning(text) {
    let formatted = text;
    
    // Highlight <think> sections
    formatted = formatted.replace(
        /<think>([\s\S]*?)<\/think>/g, 
        '<div class="think-section"><strong>&lt;think&gt;</strong>$1<strong>&lt;/think&gt;</strong></div>'
    );
    
    // Highlight boxed answers
    formatted = formatted.replace(
        /\\boxed\{([^{}]+)\}/g,
        '<span class="answer-highlight">\\boxed{$1}</span>'
    );
    
    return formatted;
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

// Helper function to render config section HTML
function renderConfigSection(config) {
    let html = '<details class="config-details">';
    html += '<summary class="config-summary"><strong>Configuration</strong></summary>';
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
        if (config[key] !== undefined) {
            html += `<tr><td>${key}</td><td>${config[key]}</td></tr>`;
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
        if (config[key] !== undefined) {
            html += `<tr><td>${key}</td><td>${config[key]}</td></tr>`;
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
        if (config[key] !== undefined) {
            html += `<tr><td>${key}</td><td>${config[key]}</td></tr>`;
        }
    });
    html += '</table></div>';
    
    // Prompt Templates
    html += '<div class="config-section">';
    html += '<h4>Prompt Templates</h4>';
    
    // Show reasoning prompt template if available
    if (config.reasoning_prompt_template) {
        html += '<details class="prompt-details">';
        html += '<summary>Reasoning Prompt</summary>';
        html += `<pre class="prompt-template">${config.reasoning_prompt_template}</pre>`;
        html += '</details>';
    }
    
    // Show summarization prompt template if available
    if (config.summarize_prompt_template) {
        html += '<details class="prompt-details">';
        html += '<summary>Summarization Prompt</summary>';
        html += `<pre class="prompt-template">${config.summarize_prompt_template}</pre>`;
        html += '</details>';
    }
    
    html += '</div>';
    
    // Raw config for advanced users
    html += '<details class="raw-config-details">';
    html += '<summary>Raw Configuration JSON</summary>';
    html += `<pre class="raw-config">${JSON.stringify(config, null, 2)}</pre>`;
    html += '</details>';
    
    html += '</div>'; // end config-container
    html += '</details>'; // end main details
    
    return html;
}

// Helper function to initialize the dashboard with static results
function initializeStaticDashboard(results) {
    console.log('Initializing dashboard with static results');
    
    // Set experiment state from results
    if (results.experiment_name) {
        experimentState.experiment_name = results.experiment_name;
    }
    if (results.config) {
        experimentState.config = results.config;
    }
    
    // Update experiment information display
    let html = `
        <p><strong>Experiment:</strong> ${experimentState.experiment_name}</p>
        <p><strong>Status:</strong> Completed</p>
    `;
    
    if (experimentState.config) {
        html += renderConfigSection(experimentState.config);
    }
    
    experimentsInfoElem.innerHTML = html;
    
    // Process results
    if (results.results && Array.isArray(results.results)) {
        experimentState.total = results.results.length;
        experimentState.completed = results.results.length;
        
        // Create problem cards
        results.results.forEach(result => {
            const problemId = result.problem_id;
            const isCorrect = result.initial_correct;
            
            // Add to problem list
            const problemCard = document.createElement('div');
            problemCard.className = `problem-card ${isCorrect ? 'correct' : 'incorrect'}`;
            problemCard.innerHTML = `
                <div class="problem-id">${problemId}</div>
                <div class="problem-status">${isCorrect ? 'correct' : 'incorrect'}</div>
            `;
            problemCard.setAttribute('data-problem-id', problemId);
            problemCard.addEventListener('click', function() {
                document.querySelectorAll('.problem-card').forEach(card => {
                    card.classList.remove('active');
                });
                this.classList.add('active');
                
                const clickedProblemId = this.getAttribute('data-problem-id');
                activeProblemId = clickedProblemId;
                
                // Display problem information
                currentProblemElem.textContent = clickedProblemId;
                
                // Show reasoning
                if (result.initial_reasoning) {
                    problemOutputs[problemId] = result.initial_reasoning;
                    modelOutputElem.textContent = result.initial_reasoning;
                }
                
                // Show answer info
                answerInfo[problemId] = {
                    extracted: result.initial_answer,
                    correct: result.correct_answer,
                    isCorrect: result.initial_correct
                };
                updateAnswerInfo(problemId);
                
                // Show summary if available
                if (result.summary) {
                    summaryInfo[problemId] = result.summary;
                    updateSummaryInfo(problemId);
                }
            });
            problemsListElem.appendChild(problemCard);
            
            // Store data
            if (result.initial_reasoning) {
                problemOutputs[problemId] = result.initial_reasoning;
            }
            
            answerInfo[problemId] = {
                extracted: result.initial_answer,
                correct: result.correct_answer,
                isCorrect: result.initial_correct
            };
            
            if (result.summary) {
                summaryInfo[problemId] = result.summary;
            }
        });
        
        // Auto-select the first problem if any exist
        if (results.results.length > 0) {
            const firstProblemCard = document.querySelector('.problem-card');
            if (firstProblemCard) {
                firstProblemCard.click();
            }
        }
    }
}

// Helper function to update experiment information display
function updateExperimentInfoDisplay() {
    // Update experiment information
    let html = `
        <p><strong>Experiment:</strong> ${experimentState.experiment_name}</p>
        <p><strong>Progress:</strong> ${experimentState.completed}/${experimentState.total} problems</p>
        <p><strong>Status:</strong> ${experimentState.status}</p>
    `;
    
    if (experimentState.config) {
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
} 