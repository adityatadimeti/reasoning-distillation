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
        
        // Check if this is a summarizing status
        if (status && status.includes('summarizing')) {
            // Extract the iteration number
            const iterMatch = status.match(/iter(\d+)-summarizing/);
            if (iterMatch && iterMatch[1]) {
                const iteration = parseInt(iterMatch[1], 10);
                
                // Ensure we have the problemOutputs structure
                if (!window.problemOutputs) window.problemOutputs = {};
                if (!window.problemOutputs[problem_id]) window.problemOutputs[problem_id] = {};
                if (!window.problemOutputs[problem_id][iteration]) {
                    window.problemOutputs[problem_id][iteration] = {
                        reasoning: '',
                        summary: '',
                        answer: '',
                        correct_answer: '',
                        is_correct: false,
                        summaryStreaming: true  // Mark as streaming
                    };
                } else {
                    window.problemOutputs[problem_id][iteration].summaryStreaming = true;
                }
                
                // Update UI if this is the active problem
                if (problem_id === DashboardCore.getActiveProblem()) {
                    DashboardIterations.updateIterationsUI(problem_id);
                }
            }
        }
        
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
    
    // Helper function to maintain scroll position when updating content
    function updateUIWithScrollPreservation(problemId) {
        // Only update if this is the active problem
        if (problemId === DashboardCore.getActiveProblem()) {
            // Store the current scroll position
            const container = document.getElementById('iterations-container');
            const scrollTop = container ? container.scrollTop : 0;
            const scrollHeight = container ? container.scrollHeight : 0;
            const clientHeight = container ? container.clientHeight : 0;
            const wasAtBottom = scrollHeight - scrollTop <= clientHeight + 50; // 50px threshold
            
            // Update the UI
            DashboardIterations.updateIterationsUI(problemId, false); // Pass false to not handle scroll in updateIterationsUI
            
            // Now restore scroll position with a slightly longer delay to ensure rendering is complete
            setTimeout(() => {
                if (container) {
                    if (wasAtBottom) {
                        container.scrollTop = container.scrollHeight; // Scroll to bottom
                    } else {
                        container.scrollTop = scrollTop; // Maintain scroll position
                    }
                }
            }, 50); // Slightly longer delay to ensure content is fully rendered
        }
    }
    
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
                is_correct: false,
                summaryStreaming: false
            };
        }
        
        // Add the chunk to the iteration's reasoning
        window.problemOutputs[problem_id][iteration].reasoning += chunk;
        
        // Log output state for debugging
        console.log(`Problem ${problem_id}, iteration ${iteration} output updated`, {
            outputLength: window.problemOutputs[problem_id][iteration].reasoning.length
        });
        
        // Check if this is the active problem and the reasoning content element exists
        if (problem_id === DashboardCore.getActiveProblem()) {
            const reasoningContentElement = document.getElementById(`reasoning-${iteration}-content`);
            if (reasoningContentElement) {
                // Direct DOM update for content - much faster than rebuilding
                const reasoningContent = reasoningContentElement.querySelector('.reasoning-content');
                if (reasoningContent) {
                    reasoningContent.innerHTML = DashboardUI.formatReasoning(window.problemOutputs[problem_id][iteration].reasoning);
                    return; // Skip full UI rebuild if we updated directly
                }
            }
            
            // Fall back to full UI update if direct update wasn't possible
            updateUIWithScrollPreservation(problem_id);
        }
    });
    
    // Handle summary updates
    socket.on('summary', (data) => {
        const { problem_id, summary, iteration = 0 } = data;
        
        console.log(`Received complete summary for problem ${problem_id}, iteration ${iteration}`);
        
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
                is_correct: false,
                summaryStreaming: false  // Mark as not streaming
            };
        } else {
            window.problemOutputs[problem_id][iteration].summary = summary;
            window.problemOutputs[problem_id][iteration].summaryStreaming = false;  // Mark as not streaming
        }
        
        // Store summary info (for backward compatibility)
        window.summaryInfo = window.summaryInfo || {};
        window.summaryInfo[problem_id] = summary;
        
        // If this is the active problem, update the display
        if (problem_id === DashboardCore.getActiveProblem()) {
            DashboardIterations.updateIterationsUI(problem_id);
        }
    });
    
    // Handle streaming summary chunks
    socket.on('summary_chunk', (data) => {
        const { problem_id, chunk, iteration = 0 } = data;
        
        console.log(`Received summary chunk for problem ${problem_id}, iteration ${iteration}`);
        
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
                is_correct: false,
                summaryStreaming: true  // Mark as streaming
            };
        }
        
        // Add the chunk to the iteration's summary
        window.problemOutputs[problem_id][iteration].summary += chunk;
        window.problemOutputs[problem_id][iteration].summaryStreaming = true;  // Mark as streaming
        
        // Log output state for debugging
        console.log(`Problem ${problem_id}, iteration ${iteration} summary updated`, {
            summaryLength: window.problemOutputs[problem_id][iteration].summary.length
        });
        
        // Update summary info (for backward compatibility)
        window.summaryInfo = window.summaryInfo || {};
        window.summaryInfo[problem_id] = window.problemOutputs[problem_id][iteration].summary;
        
        // Check if this is the active problem and the summary content element exists
        if (problem_id === DashboardCore.getActiveProblem()) {
            const summaryContentElement = document.getElementById(`summary-${iteration}-content`);
            if (summaryContentElement) {
                // Direct DOM update for content - much faster than rebuilding
                const summaryContent = summaryContentElement.querySelector('.summary-content');
                if (summaryContent) {
                    summaryContent.textContent = window.problemOutputs[problem_id][iteration].summary;
                    return; // Skip full UI rebuild if we updated directly
                }
            }
            
            // Fall back to full UI update if direct update wasn't possible
            updateUIWithScrollPreservation(problem_id);
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
                is_correct: is_correct,
                summaryStreaming: false
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
        
        // Update UI with scroll preservation
        updateUIWithScrollPreservation(problem_id);
    });
}

// Dashboard Events exports
window.DashboardEvents = {
    registerEventHandlers
};