// Dashboard Iterations UI
// Handles the new UI with collapsible iterations

// Update the answer progression section to show how answers change across iterations
function updateAnswerProgressionUI(problemId) {
    const outputs = window.problemOutputs && window.problemOutputs[problemId];
    const progressionContainer = document.querySelector('.progression-content');
    
    if (!outputs || Object.keys(outputs).length === 0) {
        progressionContainer.innerHTML = '<div class="no-data">No answer progression available</div>';
        return;
    }
    
    // Sort iterations by their numerical iteration number
    const sortedIterations = Object.keys(outputs)
        .map(Number)
        .sort((a, b) => a - b);
        
    // Check if we have a final answer in the problem data
    const problemData = window.problemData && window.problemData[problemId];
    const hasFinalAnswer = problemData && 'final_answer' in problemData && problemData.final_answer !== null;
    
    // If we have a final answer, prepare to show it in the progression
    if (hasFinalAnswer) {
        console.log(`Problem ${problemId} has final answer: ${problemData.final_answer}`); 
    }
    
    let html = '';
    
    // Track correctness across iterations for problem card coloring
    let firstIterationCorrect = false;
    let anyLaterIterationCorrect = false;
    let anyLaterIterationIncorrect = false;
    let allCorrect = true;
    let allIncorrect = true;
    
    // Create a progression item for each iteration
    sortedIterations.forEach((iteration, index) => {
        const iterData = outputs[iteration];
        const answer = iterData.answer || 'No answer';
        const isCorrect = iterData.is_correct === true;
        
        // Mark if this is using the answer from final summary
        const isFromFinalSummary = hasFinalAnswer && 
                                   index === sortedIterations.length - 1 && 
                                   problemData.final_answer === answer;
        
        // Update tracking variables for problem card coloring
        if (index === 0) {
            firstIterationCorrect = isCorrect;
        } else {
            if (isCorrect) anyLaterIterationCorrect = true;
            if (!isCorrect) anyLaterIterationIncorrect = true;
        }
        
        // If this is the last iteration, track its correctness
        if (index === sortedIterations.length - 1) {
            lastIterationCorrect = isCorrect;
        }
        
        if (isCorrect) allIncorrect = false;
        if (!isCorrect) allCorrect = false;
        
        // Create a progression item with appropriate styling based on correctness
        html += `<span class="progression-item ${isCorrect ? 'correct' : 'incorrect'} ${isFromFinalSummary ? 'final-summary' : ''}">`;
        html += `Iter ${iteration}: ${answer}`;
        html += isFromFinalSummary ? ' (final summary)' : '';
        html += `</span>`;
        
        // If we're at the last iteration and there's a final summary answer different from the iteration answer
        if (hasFinalAnswer && 
            index === sortedIterations.length - 1 && 
            problemData.final_answer !== answer) {
            const finalIsCorrect = problemData.final_correct === true;
            html += `<span class="progression-item ${finalIsCorrect ? 'correct' : 'incorrect'} final-summary">`;
            html += `Final: ${problemData.final_answer}`;
            html += ` (final summary)`;
            html += `</span>`;
        }
    });
    
    progressionContainer.innerHTML = html || '<div class="no-data">No answer progression available</div>';
    
    // Update problem card color based on answer progression
    const problemCard = document.getElementById(`problem-${problemId}`);
    if (problemCard) {
        // Log debugging information
        console.log(`Problem ${problemId} status:`, {
            firstIterationCorrect,
            anyLaterIterationCorrect,
            anyLaterIterationIncorrect,
            allCorrect,
            allIncorrect,
            currentClasses: [...problemCard.classList],
            iterations: sortedIterations.map(iter => ({
                iteration: iter,
                isCorrect: outputs[iter].is_correct === true,
                answer: outputs[iter].answer,
                correctAnswer: outputs[iter].correct_answer,
                rawData: outputs[iter]
            }))
        });
        
        // Log each iteration's correctness explicitly
        sortedIterations.forEach((iter, idx) => {
            const iterData = outputs[iter];
            console.log(`Problem ${problemId}, Iteration ${iter}: `, {
                isCorrect: iterData.is_correct === true,
                answer: iterData.answer,
                correctAnswer: iterData.correct_answer,
                rawIsCorrect: iterData.is_correct
            });
        });
        
        // Remove existing status classes
        problemCard.classList.remove('correct', 'incorrect', 'improved', 'improved-final', 'regressed', 'regressed-final');
        
        // Apply new status class based on progression
        if (allCorrect) {
            problemCard.classList.add('correct');
            console.log(`Problem ${problemId}: Setting to 'correct' (all iterations correct)`);
        } else if (allIncorrect) {
            problemCard.classList.add('incorrect');
            console.log(`Problem ${problemId}: Setting to 'incorrect' (all iterations incorrect)`);
        } else if (!firstIterationCorrect && anyLaterIterationCorrect) {
            if (lastIterationCorrect) {
                // Started incorrect, had correct answers, and ended with correct answer
                problemCard.classList.add('improved-final');
                console.log(`Problem ${problemId}: Setting to 'improved-final' (started incorrect, ended correct)`);
            } else {
                // Started incorrect, had correct answers, but ended with incorrect answer
                problemCard.classList.add('improved');
                console.log(`Problem ${problemId}: Setting to 'improved' (started incorrect, had correct answers, but ended incorrect)`);
            }
        } else if (firstIterationCorrect && anyLaterIterationIncorrect) {
            if (lastIterationCorrect) {
                // Started correct, had incorrect answers, but ended with correct answer
                problemCard.classList.add('regressed-final');
                console.log(`Problem ${problemId}: Setting to 'regressed-final' (started correct, had incorrect answers, ended correct)`);
            } else {
                // Started correct, had incorrect answers, and ended with incorrect answer
                problemCard.classList.add('regressed');
                console.log(`Problem ${problemId}: Setting to 'regressed' (started correct, had incorrect answers, ended incorrect)`);
            }
        } else {
            console.log(`Problem ${problemId}: No condition matched, keeping current styling`);
        }
        
        // Log final classes
        console.log(`Problem ${problemId} final classes:`, [...problemCard.classList]);
    }
}

// Create or update the iterations UI
function updateIterationsUI(problemId, handleScroll = true) {
    console.log(`Updating iterations UI for problem ${problemId}`, {
        globalOutputs: window.problemOutputs,
        problemOutputs: window.problemOutputs[problemId]
    });
    
    // Remember scroll position (only if handleScroll is true)
    let scrollTop = 0;
    let wasAtBottom = false;
    const container = document.getElementById('iterations-container');
    
    if (handleScroll && container) {
        scrollTop = container.scrollTop;
        const scrollHeight = container.scrollHeight;
        const clientHeight = container.clientHeight;
        wasAtBottom = scrollHeight - scrollTop <= clientHeight + 50; // 50px threshold
    }
    
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
            html += `<div class="subsection-header" id="reasoning-${iteration}-header">`;
            html += `<h4>Reasoning</h4>`;
            html += `<button class="toggle-btn" aria-expanded="true" title="Collapse section">−</button>`;
            html += `</div>`;
            html += `<div class="subsection-content" id="reasoning-${iteration}-content">`;
            html += `<div class="reasoning-content">${DashboardUI.formatReasoning(iterData.reasoning)}</div>`;
            
            // Add finish reason display if available
            if (iterData.finish_reason) {
                if (iterData.finish_reason === 'length') {
                    html += `<div class="finish-reason warning">
                        <span class="warning-icon">⚠️</span> 
                        Output was cut off because it reached the maximum token limit.
                        The reasoning and/or answer may be incomplete.
                    </div>`;
                } else if (iterData.finish_reason !== 'stop') {
                    html += `<div class="finish-reason">
                        Completion reason: ${iterData.finish_reason}
                    </div>`;
                }
            }
            
            html += `</div>`; // End subsection-content
            html += `</div>`; // End iteration-reasoning
        }
        
        // Show summary if available
        if (iterData.summary) {
            html += `<div class="iteration-summary">`;
            html += `<div class="subsection-header" id="summary-${iteration}-header">`;
            html += `<h4>Summary${iterData.summaryStreaming ? ' <span class="streaming-text">(streaming...)</span>' : ''}</h4>`;
            html += `<button class="toggle-btn" aria-expanded="true" title="Collapse section">−</button>`;
            html += `</div>`;
            html += `<div class="subsection-content" id="summary-${iteration}-content">`;
            html += `<div class="summary-content">${DashboardUI.formatReasoning(iterData.summary)}</div>`;
            
            // Add summary finish reason display if available
            if (iterData.summary_finish_reason) {
                if (iterData.summary_finish_reason === 'length') {
                    html += `<div class="finish-reason warning">
                        <span class="warning-icon">⚠️</span> 
                        Summary was cut off because it reached the maximum token limit.
                        The summary may be incomplete.
                    </div>`;
                } else if (iterData.summary_finish_reason !== 'stop') {
                    html += `<div class="finish-reason">
                        Completion reason: ${iterData.summary_finish_reason}
                    </div>`;
                }
            }
            
            html += `</div>`; // End subsection-content
            html += `</div>`; // End iteration-summary
        }
        
        // Check if this is the last iteration and we have a final summary
        if (iteration === sortedIterations[sortedIterations.length - 1] && iterData.final_summary) {
            html += `<div class="iteration-final-summary">`;
            html += `<div class="subsection-header" id="final-summary-${iteration}-header">`;
            html += `<h4>Final Summary</h4>`;
            html += `<button class="toggle-btn" aria-expanded="true" title="Collapse section">−</button>`;
            html += `</div>`;
            html += `<div class="subsection-content" id="final-summary-${iteration}-content">`;
            html += `<div class="summary-content">${DashboardUI.formatReasoning(iterData.final_summary)}</div>`;
            
            // Add final summary answer if available
            if (iterData.final_summary_answer) {
                const isCorrect = iterData.final_summary_correct === true;
                html += `<div class="summary-answer ${isCorrect ? 'correct' : 'incorrect'}">`;
                html += `<strong>Final Summary Answer:</strong> ${iterData.final_summary_answer}`;
                html += `</div>`;
            }
            
            // Add finish reason display if available
            if (iterData.summary_finish_reason) {
                if (iterData.summary_finish_reason === 'length') {
                    html += `<div class="finish-reason warning">
                        <span class="warning-icon">⚠️</span> 
                        Final summary was cut off because it reached the maximum token limit.
                        The summary may be incomplete.
                    </div>`;
                } else if (iterData.summary_finish_reason !== 'stop') {
                    html += `<div class="finish-reason">
                        Completion reason: ${iterData.summary_finish_reason}
                    </div>`;
                }
            }
            
            html += `</div>`; // End subsection-content
            html += `</div>`; // End iteration-final-summary
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
        
        // Set up toggle listeners for reasoning section
        const reasoningHeader = document.getElementById(`reasoning-${iteration}-header`);
        const reasoningContent = document.getElementById(`reasoning-${iteration}-content`);
        if (reasoningHeader && reasoningContent) {
            const reasoningToggleBtn = reasoningHeader.querySelector('.toggle-btn');
            if (reasoningToggleBtn) {
                reasoningHeader.addEventListener('click', () => {
                    DashboardUI.toggleSection(reasoningContent, reasoningToggleBtn);
                });
            }
        }
        
        // Set up toggle listeners for summary section
        const summaryHeader = document.getElementById(`summary-${iteration}-header`);
        const summaryContent = document.getElementById(`summary-${iteration}-content`);
        if (summaryHeader && summaryContent) {
            const summaryToggleBtn = summaryHeader.querySelector('.toggle-btn');
            if (summaryToggleBtn) {
                summaryHeader.addEventListener('click', () => {
                    DashboardUI.toggleSection(summaryContent, summaryToggleBtn);
                });
            }
        }
        
        // Set up toggle listeners for final summary section if it exists
        const finalSummaryHeader = document.getElementById(`final-summary-${iteration}-header`);
        const finalSummaryContent = document.getElementById(`final-summary-${iteration}-content`);
        if (finalSummaryHeader && finalSummaryContent) {
            const finalSummaryToggleBtn = finalSummaryHeader.querySelector('.toggle-btn');
            if (finalSummaryToggleBtn) {
                finalSummaryHeader.addEventListener('click', () => {
                    DashboardUI.toggleSection(finalSummaryContent, finalSummaryToggleBtn);
                });
            }
        }
    });
    
    // Restore scroll position only if handleScroll is true
    if (handleScroll && container) {
        setTimeout(() => {
            if (wasAtBottom) {
                container.scrollTop = container.scrollHeight; // Scroll to bottom
            } else {
                container.scrollTop = scrollTop; // Maintain scroll position
            }
        }, 50);
    }
}

// Dashboard Iterations exports
window.DashboardIterations = {
    updateIterationsUI,
    updateAnswerProgressionUI
}; 