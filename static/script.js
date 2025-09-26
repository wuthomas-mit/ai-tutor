const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('message-input');
let pastedImages = []; // Array to store pasted images

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Handle message submission
messageInput.addEventListener('keydown', async function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const message = this.value.trim();
        if (message || pastedImages.length > 0) {
            await sendMessage(message);
            this.value = '';
            this.style.height = 'auto';
            // Clear images after sending
            pastedImages = [];
            updateImagePreviewDisplay();
        }
    }
});

// Handle paste events
document.addEventListener('paste', function(e) {
    if (document.activeElement === messageInput) {
        const items = e.clipboardData.items;
        let imageFound = false;
        
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                imageFound = true;
                const blob = items[i].getAsFile();
                const reader = new FileReader();
                
                reader.onload = function(event) {
                    const base64data = event.target.result;
                    pastedImages.push({
                        data: base64data,
                        type: blob.type
                    });
                    updateImagePreviewDisplay();
                };
                
                reader.readAsDataURL(blob);
            }
        }
        
        // If we found and handled an image, prevent the default paste behavior
        if (imageFound) {
            e.preventDefault();
        }
    }
});

// Remove the second DOMContentLoaded event listener and merge it with the first one
document.addEventListener('DOMContentLoaded', async function() {
    messageInput.focus();
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
});

// Global variable to store the currently selected source type
let currentSourceType = 'default';

// Function to add the source type selector for Version B
// Create image preview container
function createImagePreviewContainer() {
    let container = document.getElementById('image-preview-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'image-preview-container';
        container.className = 'image-preview-container';
        document.querySelector('.input-container').insertBefore(container, messageInput);
    }
    return container;
}

// Update image preview display
function updateImagePreviewDisplay() {
    const container = createImagePreviewContainer();
    
    // Clear existing previews
    container.innerHTML = '';
    
    // Add new previews
    pastedImages.forEach((img, index) => {
        const preview = document.createElement('div');
        preview.className = 'image-preview';
        
        const image = document.createElement('img');
        image.src = img.data;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-image-btn';
        removeBtn.innerHTML = '×';
        removeBtn.addEventListener('click', () => {
            pastedImages.splice(index, 1);
            updateImagePreviewDisplay();
        });
        
        preview.appendChild(image);
        preview.appendChild(removeBtn);
        container.appendChild(preview);
    });
    
    // Show/hide container based on whether there are images
    container.style.display = pastedImages.length > 0 ? 'flex' : 'none';
}

function createFeedbackButtons() {
    const container = document.createElement('div');
    container.className = 'feedback-buttons';
    
    const thumbsUp = document.createElement('button');
    thumbsUp.className = 'feedback-button';
    thumbsUp.innerHTML = '👍';
    
    const thumbsDown = document.createElement('button');
    thumbsDown.className = 'feedback-button';
    thumbsDown.innerHTML = '👎';
    
    thumbsUp.addEventListener('click', function() {
        if (!this.disabled && !thumbsDown.disabled) {
            this.classList.add('active-positive');
            thumbsDown.classList.remove('active-negative');
            
            // Disable both buttons after feedback is given
            this.disabled = true;
            thumbsDown.disabled = true;
            
            saveFeedback(true);
        }
    });
    
    thumbsDown.addEventListener('click', function() {
        if (!this.disabled && !thumbsUp.disabled) {
            this.classList.add('active-negative');
            thumbsUp.classList.remove('active-positive');
            
            // Disable both buttons after feedback is given
            this.disabled = true;
            thumbsUp.disabled = true;
            
            saveFeedback(false);
        }
    });
    
    container.appendChild(thumbsUp);
    container.appendChild(thumbsDown);
    return container;
}

// Function to convert HTML back to plain text for feedback storage
function htmlToPlainText(html) {
    // Create a temporary div element
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = html;
    
    // Get the text content, which will strip all HTML tags
    let plainText = tempDiv.textContent || tempDiv.innerText || '';
    
    // Clean up extra whitespace and line breaks
    plainText = plainText.replace(/\s+/g, ' ').trim();
    
    return plainText;
}

async function saveFeedback(type) {
    try {
        const lastBotMessage = document.querySelector('.bot-message:last-of-type');
        const originalResponse = lastBotMessage.dataset.originalResponse;
        const threadId = lastBotMessage.dataset.threadId;
        const userEmail = localStorage.getItem('userEmail');
        
        // Convert HTML response to plain text for cleaner database storage
        const plainTextResponse = htmlToPlainText(originalResponse);
        
        await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                thread_id: threadId,
                message: plainTextResponse,
                feedback: type,
                user_email: userEmail
            }),
        });
        
        console.log('Feedback saved successfully');
    } catch (error) {
        console.error('Error saving feedback:', error);
    }
}

// Modified sendMessage function to include source type
async function sendMessage(message) {
    // Disable input while processing
    messageInput.disabled = true;
    
    // Get session info from localStorage
    const userEmail = localStorage.getItem('userEmail');
    const sessionId = localStorage.getItem('sessionId');
    const selectedProblemSet = localStorage.getItem('selectedProblemSet');
    
    if (!userEmail || !sessionId) {
        alert('Please log in again');
        window.location.href = '/';
        return;
    }
    
    // Create payload with text, images, and source type
    const payload = {
        message: message,
        source_type: currentSourceType || "default",  // Include the selected source type
        user_id: userEmail,  // Use email as user_id
        session_id: sessionId,
        problem_set: selectedProblemSet || null  // Include selected problem set
    };
    
    if (pastedImages.length > 0) {
        payload.image_data = pastedImages[0]?.data; // For now, just use first image
    }
    
    // Add user message to chat with image previews if any
    const userMessageElement = document.createElement('div');
    userMessageElement.className = 'message user-message';
    
    let messageHTML = `<div class="message-content">${escapeHtml(message)}`;
    
    // Add image previews to the message
    if (pastedImages.length > 0) {
        messageHTML += `<div class="sent-images">`;
        pastedImages.forEach(img => {
            messageHTML += `<img src="${img.data}" class="sent-image">`;
        });
        messageHTML += `</div>`;
    }
    
    messageHTML += `</div>`;
    userMessageElement.innerHTML = messageHTML;
    messagesContainer.appendChild(userMessageElement);

    try {
        // Add loading indicator
        const loadingElement = document.createElement('div');
        loadingElement.className = 'message bot-message';
        loadingElement.innerHTML = `
            <div class="message-content">
                <div class="loading-indicator">Thinking...</div>
            </div>
        `;
        messagesContainer.appendChild(loadingElement);

        // Scroll to show loading indicator
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Send message to server
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Remove loading indicator
        messagesContainer.removeChild(loadingElement);

        // Check if this is an A/B test response
        if (data.response && data.response.includes('<!-- ') && data.response.includes(' -->')) {
            const abTestMatch = data.response.match(/<!-- (.*?) -->/);
            if (abTestMatch) {
                try {
                    const abTestData = JSON.parse(abTestMatch[1]);
                    if (abTestData.type === 'ab_test_response') {
                        // Handle A/B test response
                        handleABTestResponse(abTestData);
                        return;
                    }
                } catch (e) {
                    console.log('Not an A/B test response, proceeding normally');
                }
            }
        }

        // Handle normal response
        handleNormalResponse(data);

    } catch (error) {
        console.error('Error:', error);
        // Add error message to chat
        const errorMessageElement = document.createElement('div');
        errorMessageElement.className = 'message bot-message';
        errorMessageElement.innerHTML = `
            <div class="message-content error">
                <p>Sorry, an error occurred. Please try again.</p>
            </div>
        `;
        messagesContainer.appendChild(errorMessageElement);
    } finally {
        // Re-enable input
        messageInput.disabled = false;
        messageInput.focus();
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// Function to handle normal responses
function handleNormalResponse(data) {
    // Add bot message to chat
    const botMessageElement = document.createElement('div');
    botMessageElement.className = 'message bot-message';
    
    // Store thread_id and original response text for feedback purposes
    botMessageElement.dataset.threadId = data.thread_id;
    botMessageElement.dataset.originalResponse = data.response;
    
    // Convert markdown and render LaTeX
    const renderedMessage = marked.parse(data.response);
    
    botMessageElement.innerHTML = `
        <div class="message-content">${renderedMessage}</div>
    `;
    
    // Add feedback buttons after the message content
    const feedbackButtons = createFeedbackButtons();
    botMessageElement.appendChild(feedbackButtons);
    
    messagesContainer.appendChild(botMessageElement);

    // Render LaTeX in the new message
    renderMathInElement(botMessageElement, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
        ],
        throwOnError: false
    });
}

// Function to handle A/B test responses
function handleABTestResponse(abTestData) {
    // Create A/B test comparison container
    const abTestElement = document.createElement('div');
    abTestElement.className = 'message ab-test-message';
    
    const controlContent = marked.parse(abTestData.control.content);
    const treatmentContent = marked.parse(abTestData.treatment.content);
    
    abTestElement.innerHTML = `
        <div class="ab-test-container">
            <div class="ab-test-header">
                <h3>A/B Test: Choose the Better Response</h3>
                <p>Please compare the two responses below and select which one you think is better.</p>
            </div>
            
            <div class="ab-test-responses">
                <div class="ab-test-response" data-variant="control">
                    <div class="ab-test-response-header">
                        <span class="response-label">Response A</span>
                    </div>
                    <div class="ab-test-response-content">${controlContent}</div>
                    <button class="ab-test-select-btn" onclick="selectABTestResponse('${abTestData.metadata.thread_id}', 'control', this)">
                        Select Response A
                    </button>
                </div>
                
                <div class="ab-test-response" data-variant="treatment">
                    <div class="ab-test-response-header">
                        <span class="response-label">Response B</span>
                    </div>
                    <div class="ab-test-response-content">${treatmentContent}</div>
                    <button class="ab-test-select-btn" onclick="selectABTestResponse('${abTestData.metadata.thread_id}', 'treatment', this)">
                        Select Response B
                    </button>
                </div>
            </div>
            
            <div class="ab-test-feedback">
                <label for="ab-test-reason">Before selecting, explain your choice.</label>
                <textarea id="ab-test-reason" placeholder="Your feedback helps improve the AI tutor..."></textarea>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(abTestElement);
    
    // Render LaTeX in both responses
    const responseContents = abTestElement.querySelectorAll('.ab-test-response-content');
    responseContents.forEach(content => {
        renderMathInElement(content, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    });
    
    // Store the A/B test data for later submission
    abTestElement.abTestData = abTestData;
}

// Function to handle A/B test response selection
async function selectABTestResponse(threadId, chosenVariant, buttonElement) {
    const abTestContainer = buttonElement.closest('.ab-test-container');
    const reasonTextarea = abTestContainer.querySelector('#ab-test-reason');
    const reason = reasonTextarea.value.trim();
    
    // Get the stored A/B test data
    const abTestElement = buttonElement.closest('.ab-test-message');
    const abTestData = abTestElement.abTestData;
    
    try {
        // Get user email from localStorage
        const userEmail = localStorage.getItem('userEmail');
        
        // Send the choice to the backend
        const response = await fetch('/ab-test-choice', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                thread_id: threadId,
                chosen_variant: chosenVariant,
                reason: reason,
                ab_test_data: abTestData,
                user_email: userEmail
            }),
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const result = await response.json();
        
        // Update UI to show selection was made
        const allButtons = abTestContainer.querySelectorAll('.ab-test-select-btn');
        allButtons.forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = '0.5';
        });
        
        // Highlight the chosen response
        const chosenResponse = abTestContainer.querySelector(`[data-variant="${chosenVariant}"]`);
        chosenResponse.classList.add('selected');
        
        // Add confirmation message
        const confirmationDiv = document.createElement('div');
        confirmationDiv.className = 'ab-test-confirmation';
        confirmationDiv.innerHTML = `
            <p>✅ Thank you! Your choice has been recorded and will help improve the AI tutor.</p>
            <p>Selected: Response ${chosenVariant === 'control' ? 'A' : 'B'}</p>
        `;
        abTestContainer.appendChild(confirmationDiv);
        
        // Disable the reason textarea
        reasonTextarea.disabled = true;
        
    } catch (error) {
        console.error('Error submitting A/B test choice:', error);
        alert('Failed to submit your choice. Please try again.');
    }
}

// Utility function to escape HTML and prevent XSS
function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Function to handle automatic scrolling
function handleScroll() {
    const isAtBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop <= messagesContainer.clientHeight + 100;
    return isAtBottom;
}

// Add scroll event listener
messagesContainer.addEventListener('scroll', handleScroll);

// Add logout functionality
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + L for logout
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        window.location.href = '/logout';
    }
});