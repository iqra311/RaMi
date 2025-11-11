document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const sendButton = document.getElementById('send-button');
    const clientSelector = document.getElementById('client-selector');
    const sessionIdSpan = document.getElementById('session-id');
    const refreshButton = document.getElementById('refresh-button');
    const languageToggle = document.getElementById('language-toggle'); 


    let currentSessionId;
    let currentLanguage = 'en';
    function updateUIText() {
        if (currentLanguage === 'en') {
            document.documentElement.lang = 'en';
            document.documentElement.dir = 'ltr';
            document.querySelector('label[for="client-selector"]').textContent = 'Select Client:';
            userInput.placeholder = 'Ask a question...';
            sendButton.textContent = 'Send';
        } else {
            document.documentElement.lang = 'ar';
            document.documentElement.dir = 'rtl';
            document.querySelector('label[for="client-selector"]').textContent = 'اختر العميل:';
            userInput.placeholder = 'اطرح سؤالاً...';
            sendButton.textContent = 'إرسال';
        }
    }
    function startNewSession() {
        currentSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        sessionIdSpan.textContent = currentSessionId;
        chatBox.innerHTML = '';
        
        const welcomeMessage = currentLanguage === 'en'
            ? "Hello! I'm your Relationship Manager assistant. Please select a client and ask me a question about them."
            : "مرحباً! أنا مساعد مدير العلاقات الخاص بك. يرجى اختيار عميل وطرح سؤالك عنه.";
        appendMessage(welcomeMessage, 'bot-message', false);
        
        console.log("New session started:", currentSessionId);
    }

    refreshButton.addEventListener('click', startNewSession);

    languageToggle.addEventListener('change', () => {
        currentLanguage = languageToggle.checked ? 'ar' : 'en';
        console.log(`Language changed to: ${currentLanguage}`);
        updateUIText();
        startNewSession(); 
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        const clientId = clientSelector.value;
        if (!clientId) {
            const alertMessage = currentLanguage === 'en' ? "Please select a client first." : "يرجى اختيار عميل أولاً.";
            alert(alertMessage);
            return;
        }

        appendMessage(query, 'user-message', false);
        userInput.value = '';
        
        toggleFormState(true);
        const thinkingIndicator = appendMessage("Thinking...", "bot-message thinking", false);

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: query,
                    session_id: currentSessionId,
                    client_id: clientId,
                    language: currentLanguage 
                }),
            });


            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.json();
            thinkingIndicator.remove();
            appendMessage(data.answer, 'bot-message', true); 

        } catch (error) {
            thinkingIndicator.remove();
            appendMessage(`Sorry, an error occurred: ${error.message}`, 'bot-message', false);
        } finally {
            toggleFormState(false);
        }
    });

    function appendMessage(text, className, addFeedback) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        
        const p = document.createElement('p');
        p.innerHTML = text; 
        messageDiv.appendChild(p);

        if (addFeedback) {
            const feedbackArea = document.createElement('div');
            feedbackArea.className = 'feedback-area';

            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'feedback-buttons';
            
            const likeBtn = createFeedbackButton('👍', 'Like');
            const dislikeBtn = createFeedbackButton('👎', 'Dislike');
            const commentBtn = createFeedbackButton('💬', 'Comment');

            buttonContainer.append(likeBtn, dislikeBtn, commentBtn);
            feedbackArea.appendChild(buttonContainer);
            messageDiv.appendChild(feedbackArea);

            likeBtn.addEventListener('click', () => {
                likeBtn.classList.toggle('selected');
                dislikeBtn.classList.remove('selected');
                console.log('Feedback: Liked');
            });
            
            dislikeBtn.addEventListener('click', () => {
                dislikeBtn.classList.toggle('selected');
                likeBtn.classList.remove('selected');
                console.log('Feedback: Disliked');
            });
            
            commentBtn.addEventListener('click', () => {
                const existingCommentBox = feedbackArea.querySelector('.comment-box');
                if (existingCommentBox) {
                    existingCommentBox.remove();
                    commentBtn.classList.remove('selected');
                } else {
                    commentBtn.classList.add('selected');
                    const commentBox = createCommentBox(feedbackArea, commentBtn);
                    feedbackArea.appendChild(commentBox);
                }
            });
        }

        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return messageDiv;
    }

    function createFeedbackButton(emoji, title) {
        const btn = document.createElement('button');
        btn.className = 'feedback-btn';
        btn.innerHTML = emoji;
        btn.title = title;
        return btn;
    }

    function createCommentBox(feedbackArea, commentBtn) {
        const commentBox = document.createElement('div');
        commentBox.className = 'comment-box';

        const textarea = document.createElement('textarea');
        textarea.placeholder = 'Provide additional feedback...';
        textarea.rows = 2;

        const submitBtn = document.createElement('button');
        submitBtn.textContent = 'Submit';

        submitBtn.addEventListener('click', () => {
            const commentText = textarea.value.trim();
            if (commentText) {
                console.log(`Feedback Comment Submitted: "${commentText}"`);
                commentBox.innerHTML = '<p style="font-size: 0.9em; color: green;"><em>Thanks for your feedback!</em></p>';
            }
        });
        
        commentBox.append(textarea, submitBtn);
        return commentBox;
    }

    function toggleFormState(isLoading) {
        userInput.disabled = isLoading;
        sendButton.disabled = isLoading;
        clientSelector.disabled = isLoading;
    }

    startNewSession();
    updateUIText();

});
