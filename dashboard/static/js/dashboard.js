// Dashboard JavaScript

// Global variables
let eventSource = null;
let lastUpdateTime = 0;
let currentData = null;
let selectedProblemId = null;
let selectedTab = 'details';
let pollingInterval = null;
let connectionStatus = 'disconnected';

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    initDashboard();
});

// Initialize the dashboard
function initDashboard() {
    // Set up event listeners
    setupEventListeners();
    
    // Try to connect to SSE
    connectToEventSource();
    
    // Initial data fetch
    fetchDashboardData();
    
    // Set up polling as fallback
    setupPolling();
}

// Set up event listeners
function setupEventListeners() {
    // Tab navigation
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            switchTab(tabId);
        });
    });
    
    // Refresh button
    const refreshButton = document.getElementById('refresh-button');
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            fetchDashboardData();
        });
    }
}

// Connect to Server-Sent Events
function connectToEventSource() {
    if (eventSource) {
        eventSource.close();
    }
    
    try {
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        eventSource = new EventSource(`/api/events?_=${timestamp}`);
        
        // Track reconnection attempts
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        eventSource.onopen = function() {
            console.log('Connected to event source');
            updateConnectionStatus('connected');
            reconnectAttempts = 0; // Reset reconnect attempts on successful connection
            
            // Periodically check status to keep connection alive
            if (window.statusCheckInterval) {
                clearInterval(window.statusCheckInterval);
            }
            
            // Check status every 30 seconds instead of continuously
            window.statusCheckInterval = setInterval(() => {
                fetch('/api/status')
                    .then(response => response.json())
                    .catch(error => console.error('Status check error:', error));
            }, 30000);
        };
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.last_update > lastUpdateTime) {
                lastUpdateTime = data.last_update;
                updateDashboard(data);
            }
        };
        
        eventSource.onerror = function() {
            console.error('Error connecting to event source');
            updateConnectionStatus('disconnected');
            
            // Clear status check interval on error
            if (window.statusCheckInterval) {
                clearInterval(window.statusCheckInterval);
            }
            
            // Close the connection
            eventSource.close();
            
            // Implement exponential backoff for reconnection
            reconnectAttempts++;
            const delay = Math.min(30000, Math.pow(2, reconnectAttempts) * 1000);
            
            console.log(`Connection error. Reconnecting in ${delay/1000} seconds (attempt ${reconnectAttempts}/${maxReconnectAttempts})...`);
            
            // Try to reconnect with increasing delay, up to max attempts
            if (reconnectAttempts <= maxReconnectAttempts) {
                setTimeout(connectToEventSource, delay);
            } else {
                console.error('Max reconnection attempts reached. Please refresh the page.');
            }
        };
    } catch (error) {
        console.error('Failed to connect to event source:', error);
        updateConnectionStatus('disconnected');
    }
}

// Set up polling as fallback
function setupPolling() {
    // Clear existing interval if any
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    // Poll every 30 seconds instead of 5 seconds
    pollingInterval = setInterval(function() {
        if (connectionStatus === 'disconnected') {
            fetchDashboardData();
        }
    }, 30000);
}

// Fetch dashboard data from API
function fetchDashboardData() {
    fetch('/api/data')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.last_update > lastUpdateTime) {
                lastUpdateTime = data.last_update;
                updateDashboard(data);
            }
        })
        .catch(error => {
            console.error('Error fetching dashboard data:', error);
        });
}

// Update the dashboard with new data
function updateDashboard(data) {
    // Store current data
    currentData = data;
    
    // Update experiment status
    updateExperimentStatus(data);
    
    // Update problem list
    updateProblemList(data);
    
    // Update API calls
    updateApiCalls(data);
    
    // Update selected problem if any
    if (selectedProblemId && data.problems[selectedProblemId]) {
        updateProblemDetails(selectedProblemId);
    } else if (data.current_problem && data.problems[data.current_problem]) {
        // Select current problem if none selected
        selectedProblemId = data.current_problem;
        updateProblemDetails(selectedProblemId);
    }
}

// Update experiment status
function updateExperimentStatus(data) {
    const statusElement = document.getElementById('experiment-status');
    if (statusElement) {
        statusElement.textContent = data.status;
        statusElement.className = 'status-badge badge-' + data.status;
    }
    
    // Update start time
    const startTimeElement = document.getElementById('start-time');
    if (startTimeElement && data.start_time) {
        const startDate = new Date(data.start_time * 1000);
        startTimeElement.textContent = formatDateTime(startDate);
    }
    
    // Update last update time
    const lastUpdateElement = document.getElementById('last-update');
    if (lastUpdateElement && data.last_update) {
        const updateDate = new Date(data.last_update * 1000);
        lastUpdateElement.textContent = formatDateTime(updateDate);
    }
    
    // Update problem count
    const problemCountElement = document.getElementById('problem-count');
    if (problemCountElement) {
        const problemCount = Object.keys(data.problems).length;
        problemCountElement.textContent = problemCount;
    }
    
    // Update completed problem count
    const completedCountElement = document.getElementById('completed-count');
    if (completedCountElement) {
        const completedCount = Object.values(data.problems).filter(p => p.status === 'completed').length;
        completedCountElement.textContent = completedCount;
    }
}

// Update problem list
function updateProblemList(data) {
    const problemListElement = document.getElementById('problem-list');
    if (!problemListElement) return;
    
    // Clear existing list
    problemListElement.innerHTML = '';
    
    // Add problems to list
    Object.entries(data.problems).forEach(([problemId, problem]) => {
        const problemItem = document.createElement('div');
        problemItem.className = 'problem-item';
        if (problemId === selectedProblemId) {
            problemItem.classList.add('active');
        }
        
        problemItem.innerHTML = `
            <div>
                <span class="problem-id">${problemId}</span>
                <span class="status-badge badge-${problem.status} problem-status">${problem.status}</span>
            </div>
            <div>
                <span class="timestamp">${problem.start_time ? formatTime(new Date(problem.start_time * 1000)) : ''}</span>
            </div>
        `;
        
        problemItem.addEventListener('click', function() {
            document.querySelectorAll('.problem-item').forEach(item => {
                item.classList.remove('active');
            });
            this.classList.add('active');
            selectedProblemId = problemId;
            updateProblemDetails(problemId);
        });
        
        problemListElement.appendChild(problemItem);
    });
}

// Update problem details
function updateProblemDetails(problemId) {
    if (!currentData || !currentData.problems[problemId]) return;
    
    const problem = currentData.problems[problemId];
    
    // Update problem header
    const problemHeaderElement = document.getElementById('problem-header');
    if (problemHeaderElement) {
        problemHeaderElement.textContent = `Problem ${problemId}`;
    }
    
    // Update problem status
    const problemStatusElement = document.getElementById('problem-status');
    if (problemStatusElement) {
        problemStatusElement.textContent = problem.status;
        problemStatusElement.className = 'status-badge badge-' + problem.status;
    }
    
    // Update problem question
    const questionElement = document.getElementById('problem-question');
    if (questionElement && problem.question) {
        questionElement.textContent = problem.question;
    }
    
    // Update iterations
    updateIterations(problem);
}

// Update iterations
function updateIterations(problem) {
    const iterationsContainer = document.getElementById('iterations-container');
    if (!iterationsContainer) return;
    
    // Clear existing iterations
    iterationsContainer.innerHTML = '';
    
    // Add iterations
    if (problem.iterations) {
        const iterations = Object.entries(problem.iterations).sort(([a], [b]) => parseInt(a) - parseInt(b));
        
        iterations.forEach(([iterationNum, iteration]) => {
            const iterationContainer = document.createElement('div');
            iterationContainer.className = 'iteration-container';
            
            // Create iteration header
            const iterationHeader = document.createElement('div');
            iterationHeader.className = 'iteration-header';
            iterationHeader.innerHTML = `
                <div>
                    <strong>Iteration ${iterationNum}</strong>
                    ${iteration.confidence ? ` - Confidence: ${(iteration.confidence * 100).toFixed(1)}%` : ''}
                </div>
                <div>
                    ${iteration.timestamp ? `<span class="timestamp">${formatTime(new Date(iteration.timestamp * 1000))}</span>` : ''}
                </div>
            `;
            
            // Create iteration body
            const iterationBody = document.createElement('div');
            iterationBody.className = 'iteration-body';
            
            // Add reasoning
            if (iteration.reasoning) {
                const reasoningContainer = document.createElement('div');
                reasoningContainer.className = 'reasoning-container';
                reasoningContainer.textContent = iteration.reasoning;
                
                const reasoningHeader = document.createElement('h4');
                reasoningHeader.textContent = 'Reasoning';
                
                iterationBody.appendChild(reasoningHeader);
                iterationBody.appendChild(reasoningContainer);
            }
            
            // Add summary if available
            if (iteration.summary) {
                const summaryContainer = document.createElement('div');
                summaryContainer.className = 'summary-container';
                summaryContainer.textContent = iteration.summary;
                
                const summaryHeader = document.createElement('h4');
                summaryHeader.textContent = 'Summary';
                
                iterationBody.appendChild(summaryHeader);
                iterationBody.appendChild(summaryContainer);
            }
            
            // Add answer
            if (iteration.answer) {
                const answerContainer = document.createElement('div');
                answerContainer.className = 'answer-container';
                answerContainer.textContent = `Answer: ${iteration.answer}`;
                
                const answerHeader = document.createElement('h4');
                answerHeader.textContent = 'Extracted Answer';
                
                iterationBody.appendChild(answerHeader);
                iterationBody.appendChild(answerContainer);
            }
            
            // Add error information
            if (iteration.errors_detected !== undefined) {
                const errorsContainer = document.createElement('div');
                errorsContainer.innerHTML = `
                    <p><strong>Errors detected:</strong> ${iteration.errors_detected ? 'Yes' : 'No'}</p>
                    ${iteration.should_continue !== undefined ? `<p><strong>Should continue:</strong> ${iteration.should_continue ? 'Yes' : 'No'}</p>` : ''}
                `;
                
                iterationBody.appendChild(errorsContainer);
            }
            
            // Add toggle functionality
            iterationHeader.addEventListener('click', function() {
                const isActive = iterationBody.classList.contains('active');
                
                // Close all iteration bodies
                document.querySelectorAll('.iteration-body').forEach(body => {
                    body.classList.remove('active');
                });
                
                // Toggle this one
                if (!isActive) {
                    iterationBody.classList.add('active');
                }
            });
            
            // Append to container
            iterationContainer.appendChild(iterationHeader);
            iterationContainer.appendChild(iterationBody);
            iterationsContainer.appendChild(iterationContainer);
        });
    } else {
        iterationsContainer.innerHTML = '<p>No iterations available yet.</p>';
    }
}

// Update API calls
function updateApiCalls(data) {
    const apiCallsContainer = document.getElementById('api-calls-container');
    if (!apiCallsContainer) return;
    
    // Clear existing calls
    apiCallsContainer.innerHTML = '';
    
    // Add API calls
    if (data.api_calls && data.api_calls.length > 0) {
        // Sort by timestamp (newest first)
        const sortedCalls = [...data.api_calls].sort((a, b) => b.timestamp - a.timestamp);
        
        sortedCalls.forEach((call, index) => {
            const callItem = document.createElement('div');
            callItem.className = 'api-call-item';
            
            // Create call header
            const callHeader = document.createElement('div');
            callHeader.className = 'api-call-header';
            callHeader.innerHTML = `
                <div>
                    <strong>${call.endpoint}</strong>
                </div>
                <div>
                    ${call.timestamp ? `<span class="timestamp">${formatTime(new Date(call.timestamp * 1000))}</span>` : ''}
                </div>
            `;
            
            // Create call body
            const callBody = document.createElement('div');
            callBody.className = 'api-call-body';
            
            // Add payload
            if (call.payload) {
                const payloadHeader = document.createElement('h4');
                payloadHeader.textContent = 'Payload';
                
                const payloadContainer = document.createElement('div');
                payloadContainer.className = 'code-block';
                payloadContainer.textContent = JSON.stringify(call.payload, null, 2);
                
                callBody.appendChild(payloadHeader);
                callBody.appendChild(payloadContainer);
            }
            
            // Add response
            if (call.response) {
                const responseHeader = document.createElement('h4');
                responseHeader.textContent = 'Response';
                
                const responseContainer = document.createElement('div');
                responseContainer.className = 'code-block';
                
                // Limit response size for performance
                const responseStr = JSON.stringify(call.response, null, 2);
                responseContainer.textContent = responseStr.length > 10000 
                    ? responseStr.substring(0, 10000) + '... (truncated)'
                    : responseStr;
                
                callBody.appendChild(responseHeader);
                callBody.appendChild(responseContainer);
            }
            
            // Add toggle functionality
            callHeader.addEventListener('click', function() {
                const isActive = callBody.classList.contains('active');
                
                // Close all call bodies
                document.querySelectorAll('.api-call-body').forEach(body => {
                    body.classList.remove('active');
                });
                
                // Toggle this one
                if (!isActive) {
                    callBody.classList.add('active');
                }
            });
            
            // Append to container
            callItem.appendChild(callHeader);
            callItem.appendChild(callBody);
            apiCallsContainer.appendChild(callItem);
        });
    } else {
        apiCallsContainer.innerHTML = '<p>No API calls recorded yet.</p>';
    }
}

// Switch between tabs
function switchTab(tabId) {
    // Update selected tab
    selectedTab = tabId;
    
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        if (button.getAttribute('data-tab') === tabId) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        if (content.getAttribute('data-tab') === tabId) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
}

// Update connection status
function updateConnectionStatus(status) {
    connectionStatus = status;
    
    const statusIndicator = document.getElementById('connection-indicator');
    const statusText = document.getElementById('connection-status');
    
    if (statusIndicator && statusText) {
        if (status === 'connected') {
            statusIndicator.className = 'status-indicator status-connected';
            statusText.textContent = 'Connected';
        } else {
            statusIndicator.className = 'status-indicator status-disconnected';
            statusText.textContent = 'Disconnected';
        }
    }
}

// Format date and time
function formatDateTime(date) {
    return date.toLocaleString();
}

// Format time only
function formatTime(date) {
    return date.toLocaleTimeString();
}

// Format duration in seconds to readable format
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
} 