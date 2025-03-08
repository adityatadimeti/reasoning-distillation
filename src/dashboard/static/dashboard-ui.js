// Dashboard UI Components
// Handles UI manipulation, rendering, and display functions

// Toggle section visibility
function toggleSection(contentElement, buttonElement) {
    if (contentElement.classList.contains('collapsed')) {
        contentElement.classList.remove('collapsed');
    } else {
        contentElement.classList.add('collapsed');
    }
    
    updateToggleButton(buttonElement, contentElement);
}

// Update toggle button appearance
function updateToggleButton(buttonElement, contentElement) {
    if (contentElement.classList.contains('collapsed')) {
        buttonElement.textContent = '+';
        buttonElement.setAttribute('aria-expanded', 'false');
        buttonElement.setAttribute('title', 'Expand section');
    } else {
        buttonElement.textContent = '−';
        buttonElement.setAttribute('aria-expanded', 'true');
        buttonElement.setAttribute('title', 'Collapse section');
    }
}

// Update model output display
function updateModelOutput(problemId) {
    const outputs = DashboardCore.getProblemOutputs(problemId);
    const modelOutputElem = document.getElementById('model-output');
    
    if (!outputs || Object.keys(outputs).length === 0) {
        modelOutputElem.textContent = 'No output available';
        return;
    }
    
    // Currently shows only the first iteration's reasoning
    // This will be updated in the new UI with collapsible iterations
    const iteration = 0;
    if (outputs[iteration] && outputs[iteration].reasoning) {
        modelOutputElem.innerHTML = formatReasoning(outputs[iteration].reasoning);
    } else {
        modelOutputElem.textContent = 'No reasoning available';
    }
}

// Format reasoning text for display
function formatReasoning(text) {
    if (!text) return '';
    
    // Replace special patterns
    let formatted = text
        .replace(/\\boxed\{(.*?)\}/g, '<span class="boxed">$1</span>')
        .replace(/\\begin\{align\}([\s\S]*?)\\end\{align\}/g, '<div class="math-align">$1</div>')
        .replace(/\$(.*?)\$/g, '<span class="inline-math">$1</span>')
        .replace(/\n/g, '<br>');
    
    return formatted;
}

// Update answer information display (legacy)
function updateAnswerInfo(problemId) {
    const answerInfoElem = document.getElementById('answer-info');
    const answerContentElem = answerInfoElem.querySelector('.answer-content');
    
    const answerInfo = window.answerInfo[problemId];
    if (!answerInfo) {
        answerInfoElem.style.display = 'none';
        return;
    }
    
    // Create the answer info HTML
    let html = `<div class="answer-row">`;
    html += `<div class="answer-label">Model answer:</div>`;
    html += `<div class="answer-value">${answerInfo.answer || 'No answer extracted'}</div>`;
    html += `</div>`;
    
    html += `<div class="answer-row">`;
    html += `<div class="answer-label">Correct answer:</div>`;
    html += `<div class="answer-value">${answerInfo.correct_answer}</div>`;
    html += `</div>`;
    
    html += `<div class="answer-row">`;
    html += `<div class="answer-label">Status:</div>`;
    html += `<div class="answer-value ${answerInfo.is_correct ? 'correct' : 'incorrect'}">`;
    html += answerInfo.is_correct ? 'Correct' : 'Incorrect';
    html += `</div>`;
    html += `</div>`;
    
    answerContentElem.innerHTML = html;
    answerInfoElem.style.display = 'block';
}

// Update summary information display
function updateSummaryInfo(problemId) {
    const summaryInfo = DashboardCore.getSummaryInfo(problemId);
    const summaryInfoElem = document.getElementById('summary-info');
    const summaryContentElem = document.querySelector('#summary-info .summary-content');
    const summaryContainerElem = document.getElementById('summary-container');
    const summaryHeaderElem = document.getElementById('summary-header');
    
    if (!summaryInfo) {
        summaryInfoElem.style.display = 'none';
        summaryContainerElem.classList.add('collapsed');
        const toggleBtn = summaryHeaderElem.querySelector('.toggle-btn');
        updateToggleButton(toggleBtn, summaryContainerElem);
        return;
    }
    
    summaryContentElem.textContent = summaryInfo;
    summaryInfoElem.style.display = 'block';
    
    // Expand the summary section
    summaryContainerElem.classList.remove('collapsed');
    const toggleBtn = summaryHeaderElem.querySelector('.toggle-btn');
    updateToggleButton(toggleBtn, summaryContainerElem);
}

// Render configuration section
function renderConfigSection(config) {
    if (!config) return '';
    
    let html = '<details class="config-details">';
    html += '<summary class="config-summary"><strong>Configuration</strong></summary>';
    html += '<div class="config-content">';
    
    // Model configuration
    html += '<div class="config-section">';
    html += '<h4>Model Configuration</h4>';
    
    const modelParamKeys = [
        'reasoning_model', 'summarizer_type', 'summarizer_model',
        'max_tokens', 'temperature', 'top_p', 'top_k',
        'presence_penalty', 'frequency_penalty'
    ];
    
    modelParamKeys.forEach(key => {
        if (key in config) {
            html += `<div class="config-item">`;
            html += `<span class="config-key">${key}:</span>`;
            html += `<span class="config-value">${config[key]}</span>`;
            html += `</div>`;
        }
    });
    
    html += '</div>';
    
    // Summarization configuration
    html += '<div class="config-section">';
    html += '<h4>Summarization Configuration</h4>';
    
    const summaryParamKeys = [
        'enable_summarization', 'summary_max_tokens', 'summary_temperature',
        'summary_top_p', 'summary_top_k', 'summary_presence_penalty',
        'summary_frequency_penalty'
    ];
    
    summaryParamKeys.forEach(key => {
        if (key in config) {
            html += `<div class="config-item">`;
            html += `<span class="config-key">${key}:</span>`;
            html += `<span class="config-value">${config[key]}</span>`;
            html += `</div>`;
        }
    });
    
    html += '</div>';
    
    // Prompt templates
    if (config.reasoning_prompt_template) {
        html += '<div class="config-section">';
        html += '<h4>Prompt Templates</h4>';
        html += '<details>';
        html += '<summary>Reasoning Prompt</summary>';
        html += `<pre class="prompt-template">${config.reasoning_prompt_template}</pre>`;
        html += '</details>';
        html += '</div>';
    }
    
    if (config.summarize_prompt_template) {
        html += '<div class="config-section">';
        html += '<details>';
        html += '<summary>Summarization Prompt</summary>';
        html += `<pre class="prompt-template">${config.summarize_prompt_template}</pre>`;
        html += '</details>';
        html += '</div>';
    }
    
    // Raw config
    html += '<div class="config-section">';
    html += '<details>';
    html += '<summary>Raw Configuration JSON</summary>';
    html += `<pre class="raw-config">${JSON.stringify(config, null, 2)}</pre>`;
    html += '</details>';
    html += '</div>';
    
    html += '</div>'; // End config-content
    html += '</details>'; // End config-details
    
    return html;
}

// Create a problem card in the problems list
function createProblemCard(problemId, title, status) {
    const problemsList = document.getElementById('problems-list');
    
    // Check if card already exists
    const existingCard = document.getElementById(`problem-${problemId}`);
    if (existingCard) {
        // Just update the status
        existingCard.className = `problem-card ${status}`;
        return;
    }
    
    // Create new card
    const card = document.createElement('div');
    card.id = `problem-${problemId}`;
    card.className = `problem-card ${status}`;
    card.textContent = title;
    
    // Add click handler
    card.addEventListener('click', () => {
        // Remove active class from all cards
        document.querySelectorAll('.problem-card').forEach(card => {
            card.classList.remove('active');
        });
        
        // Add active class to this card
        card.classList.add('active');
        
        // Set as active problem and update display
        DashboardCore.setActiveProblem(problemId);
        updateProblemDisplay(problemId);
    });
    
    problemsList.appendChild(card);
}

// Update the display for a problem
function updateProblemDisplay(problemId) {
    // Update current problem display
    document.getElementById('current-problem').textContent = `Problem: ${problemId}`;
    
    // Use the new iterations UI
    DashboardIterations.updateIterationsUI(problemId);
    
    // Update legacy answer info for backward compatibility
    if (window.answerInfo && window.answerInfo[problemId]) {
        updateAnswerInfo(problemId);
    }
}

// Update experiment information display
function updateExperimentInfoDisplay(experimentState) {
    const experimentDetails = document.getElementById('experiment-details');
    
    let html = '<div class="experiment-status">';
    html += `<div><strong>Status:</strong> <span class="status-value">${experimentState.status}</span></div>`;
    
    if (experimentState.totalProblems > 0) {
        const percentComplete = Math.round((experimentState.completedProblems / experimentState.totalProblems) * 100);
        html += `<div><strong>Progress:</strong> ${experimentState.completedProblems}/${experimentState.totalProblems} (${percentComplete}%)</div>`;
        
        // Add progress bar
        html += '<div class="progress-bar-container">';
        html += `<div class="progress-bar" style="width: ${percentComplete}%"></div>`;
        html += '</div>';
    }
    
    html += '</div>'; // End experiment-status
    
    // Add configuration section if available
    if (experimentState.config && Object.keys(experimentState.config).length > 0) {
        html += renderConfigSection(experimentState.config);
    }
    
    experimentDetails.innerHTML = html;
}

// Hide an element by CSS selector
function hideElement(selector) {
    const element = document.querySelector(selector);
    if (element) {
        element.style.display = 'none';
    }
}

// Show an element by CSS selector
function showElement(selector) {
    const element = document.querySelector(selector);
    if (element) {
        element.style.display = '';
    }
}

// Set up UI event listeners
function setupUIEventListeners() {
    // Set up collapsible sections
    document.querySelectorAll('.section-header').forEach(header => {
        const toggleBtn = header.querySelector('.toggle-btn');
        const contentElem = header.nextElementSibling;
        
        header.addEventListener('click', () => {
            toggleSection(contentElem, toggleBtn);
        });
    });
}

// Dashboard UI exports
window.DashboardUI = {
    toggleSection,
    updateToggleButton,
    updateModelOutput,
    formatReasoning,
    updateAnswerInfo,
    updateSummaryInfo,
    renderConfigSection,
    createProblemCard,
    updateProblemDisplay,
    updateExperimentInfoDisplay,
    hideElement,
    showElement,
    setupUIEventListeners
};

// Initialize UI event listeners
document.addEventListener('DOMContentLoaded', setupUIEventListeners); 