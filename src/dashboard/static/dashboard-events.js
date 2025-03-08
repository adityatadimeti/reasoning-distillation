// Dashboard Event Handlers
// Handles socket.io events and other event-driven functionality

// Register all event handlers
function registerEventHandlers(socket) {
    // Handle experiment status updates
    socket.on('experiment_status', (data) => {
        console.log('Received experiment status:', data);
        
        // Update experiment state
        const experimentState = DashboardCore.getExperimentState();
        DashboardCore.updateExperimentState({
            status: data.status || experimentState.status,
            totalProblems: data.total || experimentState.totalProblems,
            completedProblems: data.completed || experimentState.completedProblems
        });
        
        // Add config if available
        if (data.config) {
            DashboardCore.updateExperimentState({ config: data.config });
        }
        
        // Update the UI
        DashboardUI.updateExperimentInfoDisplay(DashboardCore.getExperimentState());
    });
    
    // Handle problem status updates
    socket.on('problem_status', (data) => {
        const { problem_id, status, question } = data;
        console.log(`Problem ${problem_id} status: ${status}`);
        
        // Create a new problem card if needed
        const title = question 
            ? `${problem_id}: ${question.substring(0, 50)}...` 
            : problem_id;
            
        DashboardUI.createProblemCard(problem_id, title, status);
        
        // If this is the first problem and none are active yet, select it
        if (!DashboardCore.getActiveProblem()) {
            // Simulate a click on the problem card
            const problemCard = document.getElementById(`problem-${problem_id}`);
            if (problemCard) {
                problemCard.click();
            }
        }
    });
    
    // Handle model output chunks
    socket.on('model_output', (data) => {
        const { problem_id, chunk, iteration = 0 } = data;
        
        console.log(`Received model output chunk for problem ${problem_id}, iteration ${iteration}`);
        
        // Ensure we have the global problemOutputs
        if (!window.problemOutputs) {
            window.problemOutputs = {};
        }
        
        // Initialize problem if needed
        if (!window.problemOutputs[problem_id]) {
            window.problemOutputs[problem_id] = {};
        }
        
        // Initialize iteration if needed
        if (!window.problemOutputs[problem_id][iteration]) {
            window.problemOutputs[problem_id][iteration] = {
                reasoning: '',
                summary: '',
                answer: '',
                correct_answer: '',
                is_correct: false
            };
        }
        
        // Add the chunk to the iteration's reasoning
        window.problemOutputs[problem_id][iteration].reasoning += chunk;
        
        // Log output state for debugging
        console.log(`Problem ${problem_id}, iteration ${iteration} output updated`, {
            outputLength: window.problemOutputs[problem_id][iteration].reasoning.length
        });
        
        // If this is the active problem, update the display
        if (problem_id === DashboardCore.getActiveProblem()) {
            DashboardIterations.updateIterationsUI(problem_id);
        }
    });
    
    // Handle summary updates
    socket.on('summary', (data) => {
        const { problem_id, summary, iteration = 0 } = data;
        
        console.log(`Received summary for problem ${problem_id}, iteration ${iteration}`);
        
        // Ensure we have the global problemOutputs
        if (!window.problemOutputs) {
            window.problemOutputs = {};
        }
        
        // Initialize problem if needed
        if (!window.problemOutputs[problem_id]) {
            window.problemOutputs[problem_id] = {};
        }
        
        // Initialize iteration if needed
        if (!window.problemOutputs[problem_id][iteration]) {
            window.problemOutputs[problem_id][iteration] = {
                reasoning: '',
                summary: summary,
                answer: '',
                correct_answer: '',
                is_correct: false
            };
        } else {
            window.problemOutputs[problem_id][iteration].summary = summary;
        }
        
        // Store summary info (for backward compatibility)
        window.summaryInfo = window.summaryInfo || {};
        window.summaryInfo[problem_id] = summary;
        
        // If this is the active problem, update the display
        if (problem_id === DashboardCore.getActiveProblem()) {
            DashboardIterations.updateIterationsUI(problem_id);
        }
    });
    
    // Handle answer updates
    socket.on('answer_info', (data) => {
        const { problem_id, answer, correct_answer, is_correct, iteration = 0 } = data;
        
        console.log(`Received answer for problem ${problem_id}, iteration ${iteration}:`, answer);
        
        // Ensure we have the global problemOutputs
        if (!window.problemOutputs) {
            window.problemOutputs = {};
        }
        
        // Initialize problem if needed
        if (!window.problemOutputs[problem_id]) {
            window.problemOutputs[problem_id] = {};
        }
        
        // Initialize iteration if needed
        if (!window.problemOutputs[problem_id][iteration]) {
            window.problemOutputs[problem_id][iteration] = {
                reasoning: '',
                summary: '',
                answer: answer,
                correct_answer: correct_answer,
                is_correct: is_correct
            };
        } else {
            window.problemOutputs[problem_id][iteration].answer = answer;
            window.problemOutputs[problem_id][iteration].correct_answer = correct_answer;
            window.problemOutputs[problem_id][iteration].is_correct = is_correct;
        }
        
        // Update answer info (for backward compatibility)
        window.answerInfo = window.answerInfo || {};
        window.answerInfo[problem_id] = {
            answer: answer,
            correct_answer: correct_answer,
            is_correct: is_correct
        };
        
        // Update problem status
        const status = is_correct ? 'correct' : 'incorrect';
        DashboardUI.createProblemCard(problem_id, problem_id, status);
        
        // If this is the active problem, update the display
        if (problem_id === DashboardCore.getActiveProblem()) {
            DashboardIterations.updateIterationsUI(problem_id);
        }
    });
}

// Dashboard Events exports
window.DashboardEvents = {
    registerEventHandlers
}; 