const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('message-input');

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
        if (message) {
            await sendMessage(message);
            this.value = '';
            this.style.height = 'auto';
        }
    }
});

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
        if (!this.classList.contains('active-positive')) {
            this.classList.add('active-positive');
            thumbsDown.classList.remove('active-negative');
            saveFeedback('positive');
        }
    });
    
    thumbsDown.addEventListener('click', function() {
        if (!this.classList.contains('active-negative')) {
            this.classList.add('active-negative');
            thumbsUp.classList.remove('active-positive');
            saveFeedback('negative');
        }
    });
    
    container.appendChild(thumbsUp);
    container.appendChild(thumbsDown);
    return container;
}

async function saveFeedback(type) {
    try {
        await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ feedback: type }),
        });
    } catch (error) {
        console.error('Error saving feedback:', error);
    }
}

async function sendMessage(message) {
    // Disable input while processing
    messageInput.disabled = true;
    
    // Add user message to chat
    const userMessageElement = document.createElement('div');
    userMessageElement.className = 'message user-message';
    userMessageElement.innerHTML = `
        <div class="message-content">${escapeHtml(message)}</div>
    `;
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
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Remove loading indicator
        messagesContainer.removeChild(loadingElement);

        // Add bot message to chat
        const botMessageElement = document.createElement('div');
        botMessageElement.className = 'message bot-message';
        
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

// Utility function to escape HTML and prevent XSS
function escapeHtml(unsafe) {
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

// Initialize by focusing input and scrolling to bottom
document.addEventListener('DOMContentLoaded', function() {
    messageInput.focus();
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
});

// Add logout functionality
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + L for logout
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        window.location.href = '/logout';
    }
});