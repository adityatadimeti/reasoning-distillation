// Dashboard Experiment Controls
// Handles experiment start/stop and model selection

// Track experiment state
let experimentRunning = false;

// Make the function explicitly global at the top of the file
window.startExperiment = function() {
    console.log("DEBUG: startExperiment global function called");
    
    const startButton = document.getElementById('start-experiment');
    const reasoningModelSelect = document.getElementById('reasoning-model');
    const summarizationModelSelect = document.getElementById('summarization-model');
    const statusMessage = document.getElementById('experiment-status-message');
    
    if (!startButton || !reasoningModelSelect || !summarizationModelSelect) {
        console.error('Experiment control elements not found');
        return;
    }
    
    // Disable button and update UI
    startButton.disabled = true;
    startButton.textContent = 'Running...';
    if (statusMessage) statusMessage.textContent = 'Starting experiment...';
    
    // Get selected models
    const reasoningModel = reasoningModelSelect.value;
    const summarizationModel = summarizationModelSelect.value;
    
    console.log(`Starting experiment with reasoning model: ${reasoningModel}, summarization model: ${summarizationModel}`);
    
    // Call API to start experiment
    console.log("DEBUG: About to call fetch API");
    fetch('/api/start_experiment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            reasoning_model: reasoningModel,
            summarizer_model: summarizationModel,
            summarizer_type: 'external'
        })
    })
    .then(response => {
        addModelDebugInfo(`API Response status: ${response.status}`);
        return response.json();
    })
    .then(data => {
        addModelDebugInfo(`API Response data: ${JSON.stringify(data)}`);
        
        // Check for error status
        if (data.status === 'error') {
            // Re-enable the button
            this.disabled = false;
            this.textContent = 'Start Run';
            
            // If the error is "already running", we need to reset the experimentRunning flag
            if (data.message && data.message.includes('already running')) {
                addModelDebugInfo('Resetting experiment running state...');
                // Reset experiment state
                experimentRunning = false;
                
                // Try to reset the server's state by calling a reset endpoint
                fetch('/api/reset_experiment_state', { method: 'POST' })
                .then(response => response.json())
                .then(resetData => {
                    addModelDebugInfo(`Reset response: ${JSON.stringify(resetData)}`);
                })
                .catch(error => {
                    addModelDebugInfo(`Reset error: ${error}`);
                });
            }
        } else {
            experimentRunning = true;
        }
    })
    .catch(error => {
        addModelDebugInfo(`Error: ${error.message}`);
        // Re-enable button
        this.disabled = false;
        this.textContent = 'Start Run';
        experimentRunning = false;
    });
};

// Add a console log to verify script is loading
console.log("DEBUG: dashboard-experiment.js loaded and startExperiment function defined globally");

// Initialize experiment controls
function initExperimentControls() {
    const startButton = document.getElementById('start-experiment');
    const reasoningModelSelect = document.getElementById('reasoning-model');
    const summarizationModelSelect = document.getElementById('summarization-model');
    const statusMessage = document.getElementById('experiment-status-message');
    
    console.log("DEBUG: Initializing experiment controls");
    console.log("DEBUG: Start button found:", !!startButton);
    console.log("DEBUG: Reasoning model select found:", !!reasoningModelSelect);
    console.log("DEBUG: Summarization model select found:", !!summarizationModelSelect);
    
    if (!startButton || !reasoningModelSelect || !summarizationModelSelect) {
        console.error('Experiment control elements not found');
        return;
    }
    
    // Set up start button click handler
    startButton.addEventListener('click', function() {
        console.log("DEBUG: Start button clicked");
        
        if (experimentRunning) {
            console.log('Experiment already running');
            return;
        }
        
        // Disable button and update UI
        startButton.disabled = true;
        startButton.textContent = 'Running...';
        statusMessage.textContent = 'Starting experiment...';
        
        // Get selected models
        const reasoningModel = reasoningModelSelect.value;
        const summarizationModel = summarizationModelSelect.value;
        
        console.log(`Starting experiment with reasoning model: ${reasoningModel}, summarization model: ${summarizationModel}`);
        
        // Create payload with exact keys matching backend expectations
        const payload = {
            reasoning_model: reasoningModel,
            summarizer_model: summarizationModel,
            summarizer_type: 'external'
        };
        
        console.log("DEBUG: Sending API request with payload:", payload);
        
        // Call API to start experiment
        fetch('/api/start_experiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(response => {
            addModelDebugInfo(`API Response status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            addModelDebugInfo(`API Response data: ${JSON.stringify(data)}`);
            
            // Check for error status
            if (data.status === 'error') {
                // Re-enable the button
                this.disabled = false;
                this.textContent = 'Start Run';
                
                // If the error is "already running", we need to reset the experimentRunning flag
                if (data.message && data.message.includes('already running')) {
                    addModelDebugInfo('Resetting experiment running state...');
                    // Reset experiment state
                    experimentRunning = false;
                    
                    // Try to reset the server's state by calling a reset endpoint
                    fetch('/api/reset_experiment_state', { method: 'POST' })
                    .then(response => response.json())
                    .then(resetData => {
                        addModelDebugInfo(`Reset response: ${JSON.stringify(resetData)}`);
                    })
                    .catch(error => {
                        addModelDebugInfo(`Reset error: ${error}`);
                    });
                }
            } else {
                experimentRunning = true;
            }
        })
        .catch(error => {
            addModelDebugInfo(`Error: ${error.message}`);
            // Re-enable button
            this.disabled = false;
            this.textContent = 'Start Run';
            experimentRunning = false;
        });
    });
    
    // Listen for experiment status updates
    if (typeof io !== 'undefined') {
        console.log("DEBUG: Socket.io found, initializing socket connection");
        socket = io();
        
        socket.on('connect', function() {
            console.log("DEBUG: Socket connected");
        });
        
        socket.on('experiment_status', function(data) {
            console.log('Experiment status update:', data);
            
            if (data.status === 'Completed') {
                // Reset UI when experiment completes
                startButton.disabled = false;
                startButton.textContent = 'Start Run';
                statusMessage.textContent = 'Experiment completed.';
                experimentRunning = false;
            } else if (data.status === 'Running') {
                // Update UI for running state
                startButton.disabled = true;
                startButton.textContent = 'Running...';
                statusMessage.textContent = 'Experiment running...';
                experimentRunning = true;
            }
        });
        
        socket.on('status', function(data) {
            console.log('Status update:', data);
            if (data.message && data.message.includes('Experiment completed')) {
                // Reset UI when experiment completes
                startButton.disabled = false;
                startButton.textContent = 'Start Run';
                statusMessage.textContent = 'Experiment completed.';
                experimentRunning = false;
            }
        });
    } else {
        console.error("DEBUG: Socket.io not found!");
    }
}

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('DEBUG: DOMContentLoaded fired');
    alert('Dashboard JavaScript loaded - click OK to continue');
    
    // Test direct button access
    const startButton = document.getElementById('start-experiment');
    if (startButton) {
        console.log('DEBUG: Found start button by ID');
        
        // Add a direct click handler regardless of other initialization
        startButton.addEventListener('click', function() {
            console.log('DEBUG: Direct button click handler fired');
            alert('Button clicked!');
        });
    } else {
        console.error('DEBUG: Could not find start button by ID!');
    }
    
    // Continue with regular initialization
    initExperimentControls();
});

// Make sure socket is defined
let socket;
if (!window.DASHBOARD_STATIC_MODE && typeof io !== 'undefined') {
    console.log("DEBUG: Creating socket in global scope");
    socket = io();
} else {
    console.log("DEBUG: Not creating socket - Static mode:", window.DASHBOARD_STATIC_MODE, "io defined:", typeof io !== 'undefined');
}
