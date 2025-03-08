// Dashboard Iterations UI
// Handles the new UI with collapsible iterations

// Create or update the iterations UI
function updateIterationsUI(problemId) {
    console.log(`Updating iterations UI for problem ${problemId}`, {
        globalOutputs: window.problemOutputs,
        problemOutputs: window.problemOutputs[problemId]
    });
    
    // Use the global problemOutputs variable
    const outputs = window.problemOutputs && window.problemOutputs[problemId];
    const outputContainer = document.getElementById('iterations-container');
    
    if (!outputs || Object.keys(outputs).length === 0) {
        console.warn(`No outputs found for problem ${problemId}`);
        outputContainer.innerHTML = '<div class="no-data">No iterations available</div>';
        return;
    }
    
    // Sort iterations by their numerical iteration number
    const sortedIterations = Object.keys(outputs)
        .map(Number)
        .sort((a, b) => a - b);
    
    console.log(`Found ${sortedIterations.length} iterations for problem ${problemId}:`, sortedIterations);
    
    let html = '';
    
    // Create a collapsible section for each iteration
    sortedIterations.forEach(iteration => {
        const iterData = outputs[iteration];
        console.log(`Processing iteration ${iteration}:`, iterData);
        
        html += `<div class="iteration-section" id="iteration-${iteration}-section">`;
        html += `<div class="section-header" id="iteration-${iteration}-header">`;
        html += `<h3>Iteration ${iteration}${iteration === 0 ? ' (Initial)' : ''}</h3>`;
        html += `<button class="toggle-btn" aria-expanded="true" title="Collapse section">−</button>`;
        html += `</div>`;
        html += `<div class="section-content" id="iteration-${iteration}-content">`;
        
        // Show answer info if available
        if (iterData.answer) {
            html += `<div class="iteration-answer">`;
            html += `<h4>Answer</h4>`;
            html += `<div class="answer-row">`;
            html += `<div class="answer-label">Model answer:</div>`;
            html += `<div class="answer-value">${iterData.answer || 'No answer extracted'}</div>`;
            html += `</div>`;
            
            if (iterData.correct_answer) {
                html += `<div class="answer-row">`;
                html += `<div class="answer-label">Correct answer:</div>`;
                html += `<div class="answer-value">${iterData.correct_answer}</div>`;
                html += `</div>`;
                
                html += `<div class="answer-row">`;
                html += `<div class="answer-label">Status:</div>`;
                html += `<div class="answer-value ${iterData.is_correct ? 'correct' : 'incorrect'}">`;
                html += iterData.is_correct ? 'Correct' : 'Incorrect';
                html += `</div>`;
                html += `</div>`;
            }
            
            html += `</div>`; // End iteration-answer
        }
        
        // Show reasoning first (before summary)
        if (iterData.reasoning) {
            html += `<div class="iteration-reasoning">`;
            html += `<h4>Reasoning</h4>`;
            html += `<div class="reasoning-content">${DashboardUI.formatReasoning(iterData.reasoning)}</div>`;
            html += `</div>`; // End iteration-reasoning
        }
        
        // Show summary if available
        if (iterData.summary) {
            html += `<div class="iteration-summary">`;
            html += `<h4>Summary</h4>`;
            html += `<div class="summary-content">${iterData.summary}</div>`;
            html += `</div>`; // End iteration-summary
        }
        
        html += `</div>`; // End section-content
        html += `</div>`; // End iteration-section
    });
    
    // Update the DOM
    outputContainer.innerHTML = html;
    console.log(`Updated iterations UI with ${sortedIterations.length} iterations`);
    
    // Set up toggle listeners for each iteration section
    sortedIterations.forEach(iteration => {
        const header = document.getElementById(`iteration-${iteration}-header`);
        const content = document.getElementById(`iteration-${iteration}-content`);
        const toggleBtn = header.querySelector('.toggle-btn');
        
        if (header && content && toggleBtn) {
            header.addEventListener('click', () => {
                DashboardUI.toggleSection(content, toggleBtn);
            });
        } else {
            console.error(`Failed to set up toggle listener for iteration ${iteration}`);
        }
    });
}

// Dashboard Iterations exports
window.DashboardIterations = {
    updateIterationsUI
}; 