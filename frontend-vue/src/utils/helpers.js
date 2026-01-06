/**
 * SafeMind AI - Utility Helper Functions
 */

/**
 * Generate a unique session ID
 * @returns {string} Session ID
 */
export function generateSessionId() {
  const timestamp = Date.now()
  const random = Math.random().toString(36).substring(2, 9)
  return `session-${timestamp}-${random}`
}

/**
 * Format timestamp to readable time
 * @param {string|Date} timestamp
 * @returns {string} Formatted time
 */
export function formatTime(timestamp) {
  if (!timestamp) return ''

  const date = new Date(timestamp)
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

/**
 * Format timestamp to readable date and time
 * @param {string|Date} timestamp
 * @returns {string} Formatted date and time
 */
export function formatDateTime(timestamp) {
  if (!timestamp) return ''

  const date = new Date(timestamp)
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

/**
 * Truncate text to specified length
 * @param {string} text
 * @param {number} maxLength
 * @returns {string} Truncated text
 */
export function truncate(text, maxLength = 100) {
  if (!text || text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

/**
 * Sanitize user input
 * @param {string} input
 * @returns {string} Sanitized input
 */
export function sanitizeInput(input) {
  if (!input) return ''

  return input
    .trim()
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<[^>]*>/g, '')
}

/**
 * Detect if risk level is critical
 * @param {string} riskLevel
 * @returns {boolean}
 */
export function isCriticalRisk(riskLevel) {
  return ['high', 'immediate'].includes(riskLevel?.toLowerCase())
}

/**
 * Get risk level color
 * @param {string} riskLevel
 * @returns {string} Color code
 */
export function getRiskColor(riskLevel) {
  const colors = {
    none: '#95a5a6',
    low: '#3498db',
    medium: '#f39c12',
    high: '#e74c3c',
    immediate: '#c0392b'
  }
  return colors[riskLevel?.toLowerCase()] || colors.none
}

/**
 * Save to local storage
 * @param {string} key
 * @param {any} value
 */
export function saveToLocalStorage(key, value) {
  try {
    const serialized = JSON.stringify(value)
    localStorage.setItem(key, serialized)
  } catch (error) {
    console.error('Error saving to localStorage:', error)
  }
}

/**
 * Load from local storage
 * @param {string} key
 * @param {any} defaultValue
 * @returns {any}
 */
export function loadFromLocalStorage(key, defaultValue = null) {
  try {
    const serialized = localStorage.getItem(key)
    if (serialized === null) return defaultValue
    return JSON.parse(serialized)
  } catch (error) {
    console.error('Error loading from localStorage:', error)
    return defaultValue
  }
}

/**
 * Clear local storage
 * @param {string} key - Optional specific key to clear
 */
export function clearLocalStorage(key = null) {
  try {
    if (key) {
      localStorage.removeItem(key)
    } else {
      localStorage.clear()
    }
  } catch (error) {
    console.error('Error clearing localStorage:', error)
  }
}

/**
 * Debounce function
 * @param {Function} func
 * @param {number} wait
 * @returns {Function}
 */
export function debounce(func, wait = 300) {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}

/**
 * Check if browser supports required features
 * @returns {boolean}
 */
export function checkBrowserSupport() {
  const checks = {
    localStorage: typeof Storage !== 'undefined',
    fetch: typeof fetch !== 'undefined',
    es6: typeof Symbol !== 'undefined'
  }

  return Object.values(checks).every(check => check)
}

/**
 * Copy text to clipboard
 * @param {string} text
 * @returns {Promise<boolean>}
 */
export async function copyToClipboard(text) {
  try {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(text)
      return true
    } else {
      // Fallback for older browsers
      const textArea = document.createElement('textarea')
      textArea.value = text
      textArea.style.position = 'fixed'
      textArea.style.left = '-999999px'
      document.body.appendChild(textArea)
      textArea.select()
      const success = document.execCommand('copy')
      document.body.removeChild(textArea)
      return success
    }
  } catch (error) {
    console.error('Error copying to clipboard:', error)
    return false
  }
}
