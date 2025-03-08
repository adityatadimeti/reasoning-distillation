// Core Dashboard Functionality
// Handles initialization, global variables, and core functionality

// Ensure global state variables exist
if (!window.problemOutputs) window.problemOutputs = {};
if (!window.problemStatuses) window.problemStatuses = {};
if (!window.summaryInfo) window.summaryInfo = {}; 
if (!window.answerInfo) window.answerInfo = {};

// Log globals for debugging
console.log('Global state initialized:', {
    problemOutputs: window.problemOutputs,
    problemStatuses: window.problemStatuses,
    summaryInfo: window.summaryInfo,
    answerInfo: window.answerInfo
});

// Store private socket reference
let socket = null;
let activeProblemId = null;

// Experiment state
let experimentState = {
    status: 'Starting',
    totalProblems: 0,
    completedProblems: 0,
    config: {}
};

// Initialize the dashboard
function initializeDashboard() {
    console.log('Initializing dashboard');
    
    // Check if we're in static mode (viewing saved results)
    if (window.DASHBOARD_STATIC_MODE) {
        console.log('Dashboard initialized in static mode');
        if (window.staticResults) {
            console.log('Static results found, initializing static dashboard');
            DashboardUI.hideElement('#connection-status');
            initializeStaticDashboard(window.staticResults);
        } else {
            console.error('Static mode active but no results found');
        }
        return;
    }
    
    // Dynamic mode - connect to Socket.IO
    setupSocketConnection();
}

// Set up socket connection and event handlers
function setupSocketConnection() {
    console.log('Setting up socket connection');
    
    // Connect to the server
    socket = io();
    
    // Handle connection events
    socket.on('connect', () => {
        console.log('Connected to server');
        document.getElementById('connection-status').textContent = 'Connected';
        document.getElementById('connection-status').className = 'connected';
        
        // Let the server know we're ready to receive data
        socket.emit('client_ready');
        
        // Request current experiment status
        socket.emit('request_current_status');
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.getElementById('connection-status').className = 'disconnected';
    });
    
    // Register all event handlers
    DashboardEvents.registerEventHandlers(socket);
}

// Initialize the dashboard with static results
function initializeStaticDashboard(results) {
    console.log('Initializing static dashboard with results:', results);
    
    const { experiment_name, config, results: problemResults } = results;
    
    // Set experiment state
    experimentState = {
        status: 'Completed',
        totalProblems: problemResults.length,
        completedProblems: problemResults.length,
        config: config || {}
    };
    
    // Update experiment info display
    DashboardUI.updateExperimentInfoDisplay(experimentState);
    
    // Process and display each problem
    problemResults.forEach(result => {
        const problemId = result.problem_id;
        
        // Store problem status
        window.problemStatuses[problemId] = result.initial_correct ? 'correct' : 'incorrect';
        
        // Create problem card
        DashboardUI.createProblemCard(problemId, `${problemId}: ${result.question.substring(0, 50)}...`, 
            window.problemStatuses[problemId]);
        
        // Store problem outputs
        processStaticProblemResult(problemId, result);
    });
    
    // Select the first problem
    const firstProblemId = problemResults[0]?.problem_id;
    if (firstProblemId) {
        activeProblemId = firstProblemId;
        DashboardUI.updateProblemDisplay(firstProblemId);
    }
}

// Process static problem result
function processStaticProblemResult(problemId, result) {
    // Create iterations structure
    window.problemOutputs[problemId] = {};
    
    // Process all iterations
    if (result.iterations && result.iterations.length > 0) {
        result.iterations.forEach(iteration => {
            window.problemOutputs[problemId][iteration.iteration] = {
                reasoning: iteration.reasoning,
                summary: iteration.summary || '',
                answer: iteration.answer || '',
                correct_answer: result.correct_answer,
                is_correct: iteration.correct
            };
        });
    } else {
        // Legacy format support
        window.problemOutputs[problemId][0] = {
            reasoning: result.initial_reasoning,
            summary: '',
            answer: result.initial_answer,
            correct_answer: result.correct_answer,
            is_correct: result.initial_correct
        };
        
        if (result.summary) {
            window.summaryInfo[problemId] = result.summary;
        }
    }
    
    // Store answer info
    window.answerInfo[problemId] = {
        answer: result.initial_answer,
        correct_answer: result.correct_answer,
        is_correct: result.initial_correct
    };
}

// Set active problem
function setActiveProblem(problemId) {
    activeProblemId = problemId;
}

// Get active problem ID
function getActiveProblem() {
    return activeProblemId;
}

// Get experiment state
function getExperimentState() {
    return experimentState;
}

// Update experiment state
function updateExperimentState(data) {
    // Update only the provided fields
    for (const key in data) {
        experimentState[key] = data[key];
    }
}

// Dashboard Core exports
window.DashboardCore = {
    initializeDashboard,
    initializeStaticDashboard,
    setActiveProblem,
    getActiveProblem,
    getExperimentState,
    updateExperimentState
};

// Initialize when the document is ready
document.addEventListener('DOMContentLoaded', initializeDashboard); 