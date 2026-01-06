<template>
  <div class="modal-overlay" @click="close">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h2>🆘 Emergency Resources</h2>
        <button class="close-button" @click="close">&times;</button>
      </div>

      <div class="modal-body">
        <div class="crisis-warning">
          <strong>⚠️ If you are in immediate danger, call 119 now.</strong>
        </div>

        <div class="resources-section">
          <h3>🇱🇰 Sri Lankan Crisis Hotlines</h3>

          <div class="resource-card">
            <div class="resource-icon">📞</div>
            <div class="resource-details">
              <h4>National Mental Health Crisis Hotline</h4>
              <p class="resource-number">1333</p>
              <p class="resource-info">Free confidential crisis support • Available 24/7</p>
            </div>
          </div>

          <div class="resource-card">
            <div class="resource-icon">🆘</div>
            <div class="resource-details">
              <h4>Emergency Services</h4>
              <p class="resource-number">119</p>
              <p class="resource-info">Police, ambulance, fire • Available 24/7</p>
            </div>
          </div>

          <div class="resource-card">
            <div class="resource-icon">💚</div>
            <div class="resource-details">
              <h4>Sumithrayo (Emotional Support)</h4>
              <p class="resource-number">011-2696666</p>
              <p class="resource-info">Volunteer emotional support • Available 24/7</p>
            </div>
          </div>

          <div class="resource-card">
            <div class="resource-icon">🏥</div>
            <div class="resource-details">
              <h4>Mental Health Helpline</h4>
              <p class="resource-number">1926</p>
              <p class="resource-info">Mental health information and referrals • Daily</p>
            </div>
          </div>
        </div>

        <div class="resources-section">
          <h3>🌐 Online Resources</h3>
          <ul class="resource-list">
            <li><strong>NIMH Sri Lanka:</strong> National Institute of Mental Health</li>
            <li><strong>Sumithrayo.org:</strong> Online emotional support resources</li>
            <li><strong>MindCare SL:</strong> Mental health awareness platform</li>
          </ul>
        </div>

        <div class="disclaimer-box">
          <strong>Important:</strong> This chatbot is NOT a replacement for professional
          mental health care. The resources above connect you with trained professionals
          who can provide immediate support.
        </div>
      </div>

      <div class="modal-footer">
        <button class="primary-button" @click="close">I Understand</button>
      </div>
    </div>
  </div>
</template>

<script setup>
// Emits
const emit = defineEmits(['close'])

// Methods
const close = () => {
  emit('close')
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 1rem;
  animation: fadeIn 0.3s;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.modal-content {
  background: white;
  border-radius: 12px;
  max-width: 600px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  animation: slideUp 0.3s;
}

@keyframes slideUp {
  from {
    transform: translateY(50px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.modal-header {
  padding: 1.5rem;
  border-bottom: 2px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-header h2 {
  margin: 0;
  color: var(--danger-color);
  font-size: 1.5rem;
}

.close-button {
  background: none;
  border: none;
  font-size: 2rem;
  cursor: pointer;
  color: var(--text-secondary);
  line-height: 1;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: all 0.3s;
}

.close-button:hover {
  background: var(--background);
  color: var(--text-primary);
}

.modal-body {
  padding: 1.5rem;
}

.crisis-warning {
  background: var(--danger-color);
  color: white;
  padding: 1rem;
  border-radius: 8px;
  text-align: center;
  margin-bottom: 1.5rem;
  font-size: 1.1rem;
}

.resources-section {
  margin-bottom: 2rem;
}

.resources-section h3 {
  color: var(--primary-color);
  margin-bottom: 1rem;
  font-size: 1.2rem;
}

.resource-card {
  display: flex;
  gap: 1rem;
  padding: 1rem;
  background: var(--background);
  border-radius: 8px;
  margin-bottom: 0.75rem;
  transition: all 0.3s;
}

.resource-card:hover {
  background: #E8EEF4;
  transform: translateX(4px);
}

.resource-icon {
  font-size: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 50px;
  height: 50px;
  background: white;
  border-radius: 50%;
  flex-shrink: 0;
}

.resource-details {
  flex: 1;
}

.resource-details h4 {
  margin: 0 0 0.25rem 0;
  color: var(--text-primary);
  font-size: 1rem;
}

.resource-number {
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--primary-color);
  margin: 0.25rem 0;
  letter-spacing: 0.05em;
}

.resource-info {
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin: 0.25rem 0 0 0;
}

.resource-list {
  list-style: none;
  padding: 0;
}

.resource-list li {
  padding: 0.75rem;
  background: var(--background);
  border-radius: 6px;
  margin-bottom: 0.5rem;
  font-size: 0.95rem;
}

.disclaimer-box {
  background: #FFF3CD;
  color: #856404;
  padding: 1rem;
  border-radius: 8px;
  border-left: 4px solid var(--warning-color);
  font-size: 0.9rem;
  line-height: 1.5;
}

.modal-footer {
  padding: 1rem 1.5rem;
  border-top: 1px solid var(--border-color);
  display: flex;
  justify-content: flex-end;
}

.primary-button {
  padding: 0.75rem 2rem;
  background: var(--primary-color);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s;
}

.primary-button:hover {
  background: var(--primary-dark);
  transform: translateY(-2px);
  box-shadow: var(--shadow);
}

.primary-button:active {
  transform: translateY(0);
}

@media (max-width: 768px) {
  .modal-content {
    max-height: 95vh;
    margin: 0.5rem;
  }

  .modal-header h2 {
    font-size: 1.2rem;
  }

  .resource-card {
    flex-direction: column;
    text-align: center;
  }

  .resource-icon {
    margin: 0 auto;
  }
}
</style>
