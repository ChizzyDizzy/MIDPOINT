<template>
  <div class="chat-window">
    <!-- Messages Area -->
    <div class="messages-container" ref="messagesContainer">
      <div
        v-for="message in messages"
        :key="message.id"
        class="message"
        :class="[
          message.sender,
          {
            'crisis-message': message.safety?.risk_level === 'immediate' || message.safety?.risk_level === 'high',
            'error-message': message.isError,
            'welcome-message': message.isWelcome
          }
        ]"
      >
        <div class="message-avatar">
          {{ message.sender === 'user' ? '👤' : '🧠' }}
        </div>
        <div class="message-content">
          <div class="message-text">{{ message.text }}</div>
          <div class="message-meta">
            <span class="message-time">{{ formatTime(message.timestamp) }}</span>
            <span v-if="message.aiPowered" class="ai-badge">AI</span>
            <span v-if="message.safety?.risk_level && message.safety.risk_level !== 'none'"
                  class="risk-badge"
                  :class="message.safety.risk_level">
              {{ message.safety.risk_level }}
            </span>
          </div>
        </div>
      </div>

      <!-- Loading Indicator -->
      <div v-if="isLoading" class="message bot">
        <div class="message-avatar">🧠</div>
        <div class="message-content">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    </div>

    <!-- Input Area -->
    <div class="input-area">
      <textarea
        v-model="inputMessage"
        @keydown.enter.exact.prevent="sendMessage"
        @keydown.enter.shift.exact="inputMessage += '\n'"
        placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
        class="message-input"
        rows="1"
        :disabled="isLoading"
      ></textarea>
      <button
        @click="sendMessage"
        class="send-button"
        :disabled="!inputMessage.trim() || isLoading"
      >
        <span v-if="!isLoading">Send</span>
        <span v-else>...</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue'

// Props
const props = defineProps({
  messages: {
    type: Array,
    required: true
  },
  isLoading: {
    type: Boolean,
    default: false
  }
})

// Emits
const emit = defineEmits(['send-message'])

// State
const inputMessage = ref('')
const messagesContainer = ref(null)

// Methods
const sendMessage = () => {
  if (!inputMessage.value.trim() || props.isLoading) return

  emit('send-message', inputMessage.value)
  inputMessage.value = ''

  // Auto-scroll to bottom after sending
  nextTick(() => {
    scrollToBottom()
  })
}

const scrollToBottom = () => {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

// Watch for new messages and scroll
watch(() => props.messages.length, () => {
  nextTick(() => {
    scrollToBottom()
  })
})
</script>

<style scoped>
.chat-window {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: white;
  border-radius: 12px;
  box-shadow: var(--shadow-lg);
  overflow: hidden;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  display: flex;
  gap: 0.75rem;
  animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message.user {
  flex-direction: row-reverse;
}

.message-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  flex-shrink: 0;
  background: var(--background);
}

.message.user .message-avatar {
  background: var(--primary-color);
}

.message.bot .message-avatar {
  background: var(--secondary-color);
}

.message-content {
  max-width: 70%;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.message.user .message-content {
  align-items: flex-end;
}

.message-text {
  background: var(--background);
  padding: 0.75rem 1rem;
  border-radius: 12px;
  white-space: pre-wrap;
  word-wrap: break-word;
  line-height: 1.5;
}

.message.user .message-text {
  background: var(--primary-color);
  color: white;
}

.message.bot .message-text {
  background: #F0F4F8;
}

.message.crisis-message .message-text {
  background: #FFE5E5;
  border-left: 4px solid var(--danger-color);
}

.message.error-message .message-text {
  background: #FFF3CD;
  border-left: 4px solid var(--warning-color);
}

.message.welcome-message .message-text {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.message-meta {
  display: flex;
  gap: 0.5rem;
  font-size: 0.75rem;
  color: var(--text-secondary);
  padding: 0 0.5rem;
}

.message-time {
  font-size: 0.7rem;
}

.ai-badge,
.risk-badge {
  padding: 0.125rem 0.375rem;
  border-radius: 4px;
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
}

.ai-badge {
  background: var(--secondary-color);
  color: white;
}

.risk-badge {
  color: white;
}

.risk-badge.low {
  background: #3498db;
}

.risk-badge.medium {
  background: var(--warning-color);
}

.risk-badge.high,
.risk-badge.immediate {
  background: var(--danger-color);
}

/* Typing Indicator */
.typing-indicator {
  display: flex;
  gap: 4px;
  padding: 0.75rem 1rem;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--text-secondary);
  animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    opacity: 0.3;
    transform: translateY(0);
  }
  30% {
    opacity: 1;
    transform: translateY(-10px);
  }
}

/* Input Area */
.input-area {
  display: flex;
  gap: 0.75rem;
  padding: 1rem;
  border-top: 1px solid var(--border-color);
  background: white;
}

.message-input {
  flex: 1;
  padding: 0.75rem 1rem;
  border: 2px solid var(--border-color);
  border-radius: 8px;
  font-family: inherit;
  font-size: 1rem;
  resize: none;
  max-height: 120px;
  transition: border-color 0.3s;
}

.message-input:focus {
  outline: none;
  border-color: var(--primary-color);
}

.message-input:disabled {
  background: var(--background);
  cursor: not-allowed;
}

.send-button {
  padding: 0.75rem 1.5rem;
  background: var(--primary-color);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  min-width: 80px;
}

.send-button:hover:not(:disabled) {
  background: var(--primary-dark);
  transform: translateY(-2px);
  box-shadow: var(--shadow);
}

.send-button:active:not(:disabled) {
  transform: translateY(0);
}

.send-button:disabled {
  background: var(--text-secondary);
  cursor: not-allowed;
  opacity: 0.5;
}

/* Mobile Responsive */
@media (max-width: 768px) {
  .messages-container {
    padding: 1rem;
  }

  .message-content {
    max-width: 85%;
  }

  .input-area {
    padding: 0.75rem;
  }

  .send-button {
    padding: 0.75rem 1rem;
    min-width: 60px;
  }
}
</style>
