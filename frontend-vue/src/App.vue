<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="header-content">
        <h1 class="app-title">
          <span class="emoji">🧠</span>
          SafeMind AI
        </h1>
        <p class="app-subtitle">Mental Health Awareness Chatbot</p>
      </div>
      <div class="header-status">
        <span class="status-indicator" :class="{ active: isConnected }"></span>
        <span class="status-text">{{ isConnected ? 'Connected' : 'Offline' }}</span>
      </div>
    </header>

    <!-- Disclaimer Banner -->
    <div class="disclaimer-banner">
      <strong>⚠️ Important:</strong> This chatbot provides information and support,
      not medical diagnosis or treatment. If you're in crisis, call
      <strong>1333</strong> (Sri Lanka Crisis Hotline) immediately.
    </div>

    <!-- Main Chat Area -->
    <main class="chat-container">
      <ChatWindow
        :messages="messages"
        :isLoading="isLoading"
        @send-message="handleSendMessage"
      />
    </main>

    <!-- Footer -->
    <footer class="app-footer">
      <p>
        Made with ❤️ for mental health awareness |
        <a href="#" @click.prevent="showResources = true">Emergency Resources</a>
      </p>
    </footer>

    <!-- Resources Modal -->
    <ResourcesModal
      v-if="showResources"
      @close="showResources = false"
    />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import ChatWindow from './components/ChatWindow.vue'
import ResourcesModal from './components/ResourcesModal.vue'
import { chatService } from './services/api'
import { generateSessionId } from './utils/helpers'

// State
const messages = ref([])
const isLoading = ref(false)
const isConnected = ref(false)
const showResources = ref(false)
const sessionId = ref(generateSessionId())

// Check API connection
const checkConnection = async () => {
  try {
    await chatService.healthCheck()
    isConnected.value = true
  } catch (error) {
    isConnected.value = false
    console.error('API connection failed:', error)
  }
}

// Handle sending message
const handleSendMessage = async (messageText) => {
  if (!messageText.trim()) return

  // Add user message to chat
  const userMessage = {
    id: Date.now(),
    text: messageText,
    sender: 'user',
    timestamp: new Date().toISOString()
  }
  messages.value.push(userMessage)

  // Show loading
  isLoading.value = true

  try {
    // Send to API
    const response = await chatService.sendMessage(messageText, sessionId.value)

    // Add bot response to chat
    const botMessage = {
      id: Date.now() + 1,
      text: response.response,
      sender: 'bot',
      timestamp: response.timestamp,
      safety: response.safety,
      aiPowered: response.ai_powered
    }
    messages.value.push(botMessage)

    // Show resources modal if crisis detected
    if (response.safety.risk_level === 'high' || response.safety.risk_level === 'immediate') {
      showResources.value = true
    }

  } catch (error) {
    console.error('Error sending message:', error)

    // Add error message
    const errorMessage = {
      id: Date.now() + 1,
      text: 'Sorry, I encountered an error. Please try again or contact support if the issue persists.',
      sender: 'bot',
      timestamp: new Date().toISOString(),
      isError: true
    }
    messages.value.push(errorMessage)
  } finally {
    isLoading.value = false
  }
}

// Add welcome message
const addWelcomeMessage = () => {
  messages.value.push({
    id: Date.now(),
    text: `Hello! I'm SafeMind AI, your mental health awareness companion. I'm here to listen and provide support in a safe, non-judgmental space.

How are you feeling today?`,
    sender: 'bot',
    timestamp: new Date().toISOString(),
    isWelcome: true
  })
}

// Lifecycle
onMounted(() => {
  checkConnection()
  addWelcomeMessage()

  // Periodic connection check
  setInterval(checkConnection, 30000) // Check every 30s
})
</script>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.app-header {
  background: white;
  padding: 1rem 2rem;
  box-shadow: var(--shadow);
  display: flex;
  justify-content: space-between;
  align-items: center;
  z-index: 10;
}

.header-content {
  flex: 1;
}

.app-title {
  font-size: 1.8rem;
  color: var(--primary-color);
  margin: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.emoji {
  font-size: 2rem;
}

.app-subtitle {
  color: var(--text-secondary);
  font-size: 0.9rem;
  margin: 0;
}

.header-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: var(--text-secondary);
  transition: background-color 0.3s;
}

.status-indicator.active {
  background-color: var(--secondary-color);
  box-shadow: 0 0 8px var(--secondary-color);
}

.status-text {
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.disclaimer-banner {
  background: #FFF3CD;
  color: #856404;
  padding: 0.75rem 2rem;
  border-bottom: 2px solid #FFEAA7;
  font-size: 0.9rem;
  text-align: center;
}

.chat-container {
  flex: 1;
  overflow: hidden;
  padding: 1rem;
}

.app-footer {
  background: white;
  padding: 1rem;
  text-align: center;
  font-size: 0.85rem;
  color: var(--text-secondary);
  border-top: 1px solid var(--border-color);
}

.app-footer a {
  color: var(--primary-color);
  text-decoration: none;
  font-weight: 500;
}

.app-footer a:hover {
  text-decoration: underline;
}

@media (max-width: 768px) {
  .app-header {
    padding: 0.75rem 1rem;
    flex-direction: column;
    gap: 0.5rem;
  }

  .app-title {
    font-size: 1.4rem;
  }

  .disclaimer-banner {
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
  }

  .chat-container {
    padding: 0.5rem;
  }
}
</style>
