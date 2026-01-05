/**
 * SafeMind AI - API Service
 * Handles all communication with the FastAPI backend
 */

import axios from 'axios'

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('Response Error:', error.response?.status, error.message)

    // Handle specific error cases
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response

      switch (status) {
        case 400:
          throw new Error(data.detail || 'Invalid request')
        case 404:
          throw new Error('Resource not found')
        case 500:
          throw new Error('Server error. Please try again later.')
        default:
          throw new Error(data.detail || 'An error occurred')
      }
    } else if (error.request) {
      // Request made but no response
      throw new Error('Cannot connect to server. Please check your connection.')
    } else {
      // Something else happened
      throw new Error('An unexpected error occurred')
    }
  }
)

/**
 * Chat Service
 */
export const chatService = {
  /**
   * Send a message to the chatbot
   * @param {string} message - User's message
   * @param {string} sessionId - Session identifier
   * @param {string} culture - Cultural context (default: south_asian)
   * @returns {Promise} Response data
   */
  async sendMessage(message, sessionId, culture = 'south_asian') {
    try {
      const response = await apiClient.post('/chat', {
        message,
        session_id: sessionId,
        culture
      })
      return response.data
    } catch (error) {
      console.error('Send message error:', error)
      throw error
    }
  },

  /**
   * Get session history
   * @param {string} sessionId - Session identifier
   * @returns {Promise} Session data
   */
  async getSession(sessionId) {
    try {
      const response = await apiClient.get(`/session/${sessionId}`)
      return response.data
    } catch (error) {
      console.error('Get session error:', error)
      throw error
    }
  },

  /**
   * Export session conversation
   * @param {string} sessionId - Session identifier
   * @returns {Promise} Export data
   */
  async exportSession(sessionId) {
    try {
      const response = await apiClient.get(`/session/${sessionId}/export`)
      return response.data
    } catch (error) {
      console.error('Export session error:', error)
      throw error
    }
  },

  /**
   * Get emergency resources
   * @param {string} culture - Cultural context
   * @returns {Promise} Resources data
   */
  async getResources(culture = 'south_asian') {
    try {
      const response = await apiClient.get('/resources', {
        params: { culture }
      })
      return response.data
    } catch (error) {
      console.error('Get resources error:', error)
      throw error
    }
  },

  /**
   * Health check
   * @returns {Promise} Health status
   */
  async healthCheck() {
    try {
      const response = await apiClient.get('/health')
      return response.data
    } catch (error) {
      console.error('Health check error:', error)
      throw error
    }
  },

  /**
   * Get system status
   * @returns {Promise} System status
   */
  async getSystemStatus() {
    try {
      const response = await apiClient.get('/system/status')
      return response.data
    } catch (error) {
      console.error('Get system status error:', error)
      throw error
    }
  },

  /**
   * Test endpoint
   * @param {string} message - Test message
   * @returns {Promise} Test results
   */
  async test(message = 'I feel anxious') {
    try {
      const response = await apiClient.post('/test', { message })
      return response.data
    } catch (error) {
      console.error('Test error:', error)
      throw error
    }
  }
}

export default apiClient
