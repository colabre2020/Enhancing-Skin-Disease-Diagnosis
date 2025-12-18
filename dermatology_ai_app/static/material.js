/**
 * Material Design JavaScript Components
 * Handles interactive behaviors, animations, and micro-interactions
 */

class MaterialDesign {
  constructor() {
    this.init();
  }

  init() {
    this.setupRippleEffects();
    this.setupFormFields();
    this.setupFileUpload();
    this.setupSnackbar();
    this.setupProgressIndicators();
    this.setupCards();
    this.setupTooltips();
  }

  // Ripple effect implementation
  setupRippleEffects() {
    document.addEventListener('click', (e) => {
      const rippleElement = e.target.closest('.md-ripple, .md-button, .md-card, .md-chip');
      if (!rippleElement) return;

      this.createRipple(e, rippleElement);
    });
  }

  createRipple(event, element) {
    const circle = document.createElement('span');
    const diameter = Math.max(element.clientWidth, element.clientHeight);
    const radius = diameter / 2;

    const rect = element.getBoundingClientRect();
    const left = event.clientX - rect.left - radius;
    const top = event.clientY - rect.top - radius;

    circle.style.width = circle.style.height = `${diameter}px`;
    circle.style.left = `${left}px`;
    circle.style.top = `${top}px`;
    circle.classList.add('md-ripple-effect');

    // Remove existing ripple
    const existingRipple = element.querySelector('.md-ripple-effect');
    if (existingRipple) {
      existingRipple.remove();
    }

    element.appendChild(circle);

    // Add CSS for ripple animation
    if (!document.getElementById('ripple-styles')) {
      const style = document.createElement('style');
      style.id = 'ripple-styles';
      style.textContent = `
        .md-ripple-effect {
          position: absolute;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.6);
          transform: scale(0);
          animation: md-ripple-animation 0.6s linear;
          pointer-events: none;
        }

        @keyframes md-ripple-animation {
          to {
            transform: scale(4);
            opacity: 0;
          }
        }
      `;
      document.head.appendChild(style);
    }

    setTimeout(() => {
      circle.remove();
    }, 600);
  }

  // Enhanced form field interactions
  setupFormFields() {
    const textFields = document.querySelectorAll('.md-text-field__input');
    
    textFields.forEach(input => {
      // Handle focus states
      input.addEventListener('focus', (e) => {
        e.target.closest('.md-text-field').classList.add('md-text-field--focused');
        this.animateLabel(e.target, 'focus');
      });

      input.addEventListener('blur', (e) => {
        e.target.closest('.md-text-field').classList.remove('md-text-field--focused');
        this.animateLabel(e.target, 'blur');
      });

      // Handle input validation
      input.addEventListener('input', (e) => {
        this.validateField(e.target);
      });

      // Initial state check
      if (input.value) {
        this.animateLabel(input, 'focus');
      }
    });
  }

  animateLabel(input, action) {
    const label = input.nextElementSibling;
    if (!label || !label.classList.contains('md-text-field__label')) return;

    if (action === 'focus' || input.value) {
      label.style.transform = 'translateY(-24px) scale(0.75)';
      label.style.color = 'var(--md-primary-600)';
    } else {
      label.style.transform = 'translateY(0) scale(1)';
      label.style.color = 'var(--md-on-surface-variant)';
    }
  }

  validateField(input) {
    const textField = input.closest('.md-text-field');
    
    // Remove existing validation classes
    textField.classList.remove('md-text-field--error', 'md-text-field--success');

    // Basic validation
    if (input.hasAttribute('required') && !input.value.trim()) {
      textField.classList.add('md-text-field--error');
      this.showFieldError(textField, 'This field is required');
    } else if (input.type === 'email' && input.value && !this.isValidEmail(input.value)) {
      textField.classList.add('md-text-field--error');
      this.showFieldError(textField, 'Please enter a valid email address');
    } else {
      textField.classList.add('md-text-field--success');
      this.hideFieldError(textField);
    }
  }

  isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }

  showFieldError(textField, message) {
    let errorElement = textField.querySelector('.md-text-field__error');
    if (!errorElement) {
      errorElement = document.createElement('div');
      errorElement.className = 'md-text-field__error';
      textField.appendChild(errorElement);
    }
    errorElement.textContent = message;
    errorElement.style.display = 'block';
  }

  hideFieldError(textField) {
    const errorElement = textField.querySelector('.md-text-field__error');
    if (errorElement) {
      errorElement.style.display = 'none';
    }
  }

  // Enhanced file upload with Material Design
  setupFileUpload() {
    const uploadAreas = document.querySelectorAll('.md-file-upload');
    
    uploadAreas.forEach(uploadArea => {
      const input = uploadArea.querySelector('input[type="file"]');
      const dropZone = uploadArea.querySelector('.md-file-upload__drop-zone');
      
      if (!input || !dropZone) return;

      // Drag and drop handlers
      ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, this.preventDefaults);
      });

      ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
          dropZone.classList.add('md-file-upload--dragover');
        });
      });

      ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
          dropZone.classList.remove('md-file-upload--dragover');
        });
      });

      dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        this.handleFiles(files, uploadArea);
      });

      input.addEventListener('change', (e) => {
        this.handleFiles(e.target.files, uploadArea);
      });
    });
  }

  preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  handleFiles(files, uploadArea) {
    ([...files]).forEach(file => {
      this.processFile(file, uploadArea);
    });
  }

  processFile(file, uploadArea) {
    const previewArea = uploadArea.querySelector('.md-file-upload__preview');
    if (!previewArea) return;

    const fileItem = document.createElement('div');
    fileItem.className = 'md-file-item md-fade-in';
    
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        fileItem.innerHTML = `
          <div class="md-file-item__image">
            <img src="${e.target.result}" alt="${file.name}" />
          </div>
          <div class="md-file-item__info">
            <div class="md-file-item__name">${file.name}</div>
            <div class="md-file-item__size">${this.formatFileSize(file.size)}</div>
          </div>
          <button class="md-button md-button--text md-file-item__remove" type="button">
            <span class="material-icons">close</span>
          </button>
        `;
        
        // Add remove functionality
        const removeBtn = fileItem.querySelector('.md-file-item__remove');
        removeBtn.addEventListener('click', () => {
          fileItem.classList.add('md-fade-out');
          setTimeout(() => fileItem.remove(), 300);
        });
      };
      reader.readAsDataURL(file);
    }

    previewArea.appendChild(fileItem);
  }

  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Snackbar notifications
  setupSnackbar() {
    window.showSnackbar = (message, action = null, duration = 4000) => {
      this.showSnackbar(message, action, duration);
    };
  }

  showSnackbar(message, action = null, duration = 4000) {
    // Remove existing snackbar
    const existingSnackbar = document.querySelector('.md-snackbar');
    if (existingSnackbar) {
      existingSnackbar.remove();
    }

    const snackbar = document.createElement('div');
    snackbar.className = 'md-snackbar';
    
    let actionHTML = '';
    if (action) {
      actionHTML = `<button class="md-snackbar__action" onclick="${action.callback}">${action.text}</button>`;
    }

    snackbar.innerHTML = `
      <div class="md-snackbar__message">${message}</div>
      ${actionHTML}
    `;

    document.body.appendChild(snackbar);

    // Show with animation
    requestAnimationFrame(() => {
      snackbar.classList.add('md-snackbar--show');
    });

    // Auto hide
    setTimeout(() => {
      snackbar.classList.remove('md-snackbar--show');
      setTimeout(() => snackbar.remove(), 300);
    }, duration);
  }

  // Progress indicators
  setupProgressIndicators() {
    // Linear progress
    window.setLinearProgress = (percentage) => {
      const progressBars = document.querySelectorAll('.md-progress-linear__bar');
      progressBars.forEach(bar => {
        bar.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
      });
    };

    // Circular progress
    window.setCircularProgress = (percentage) => {
      const circles = document.querySelectorAll('.md-progress-circular__circle');
      circles.forEach(circle => {
        const radius = circle.r.baseVal.value;
        const circumference = 2 * Math.PI * radius;
        const strokeDashoffset = circumference - (percentage / 100) * circumference;
        
        circle.style.strokeDasharray = circumference;
        circle.style.strokeDashoffset = strokeDashoffset;
      });
    };
  }

  // Enhanced card interactions
  setupCards() {
    const cards = document.querySelectorAll('.md-card');
    
    cards.forEach(card => {
      // Add hover animations
      card.addEventListener('mouseenter', () => {
        if (!card.classList.contains('md-card--no-hover')) {
          card.style.transform = 'translateY(-4px) scale(1.02)';
        }
      });

      card.addEventListener('mouseleave', () => {
        card.style.transform = 'translateY(0) scale(1)';
      });

      // Add focus handling for accessibility
      if (card.tabIndex >= 0) {
        card.addEventListener('focus', () => {
          card.style.outline = '2px solid var(--md-primary-600)';
          card.style.outlineOffset = '2px';
        });

        card.addEventListener('blur', () => {
          card.style.outline = 'none';
        });
      }
    });
  }

  // Tooltip system
  setupTooltips() {
    const tooltipTriggers = document.querySelectorAll('[data-tooltip]');
    
    tooltipTriggers.forEach(trigger => {
      trigger.addEventListener('mouseenter', (e) => {
        this.showTooltip(e.target);
      });

      trigger.addEventListener('mouseleave', () => {
        this.hideTooltip();
      });

      trigger.addEventListener('focus', (e) => {
        this.showTooltip(e.target);
      });

      trigger.addEventListener('blur', () => {
        this.hideTooltip();
      });
    });
  }

  showTooltip(element) {
    const tooltipText = element.getAttribute('data-tooltip');
    if (!tooltipText) return;

    // Remove existing tooltip
    this.hideTooltip();

    const tooltip = document.createElement('div');
    tooltip.className = 'md-tooltip md-fade-in';
    tooltip.textContent = tooltipText;
    tooltip.id = 'md-tooltip';

    document.body.appendChild(tooltip);

    // Position tooltip
    const rect = element.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    
    let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
    let top = rect.top - tooltipRect.height - 8;

    // Adjust if tooltip goes off screen
    if (left < 8) left = 8;
    if (left + tooltipRect.width > window.innerWidth - 8) {
      left = window.innerWidth - tooltipRect.width - 8;
    }
    if (top < 8) {
      top = rect.bottom + 8;
    }

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  hideTooltip() {
    const tooltip = document.getElementById('md-tooltip');
    if (tooltip) {
      tooltip.remove();
    }
  }

  // Utility functions for external use
  static showLoading(message = 'Processing...') {
    let loadingOverlay = document.getElementById('md-loading-overlay');
    
    if (!loadingOverlay) {
      loadingOverlay = document.createElement('div');
      loadingOverlay.id = 'md-loading-overlay';
      loadingOverlay.className = 'md-loading-overlay';
      loadingOverlay.innerHTML = `
        <div class="md-loading-content">
          <div class="md-progress-circular">
            <svg class="md-progress-circular__svg" viewBox="0 0 48 48">
              <circle class="md-progress-circular__circle" cx="24" cy="24" r="20" stroke-dasharray="125.664" stroke-dashoffset="125.664" style="animation: md-circular-rotate 2s linear infinite;"></circle>
            </svg>
          </div>
          <div class="md-loading-message">${message}</div>
        </div>
      `;
      document.body.appendChild(loadingOverlay);
    } else {
      loadingOverlay.querySelector('.md-loading-message').textContent = message;
    }

    loadingOverlay.style.display = 'flex';
    requestAnimationFrame(() => {
      loadingOverlay.classList.add('md-loading-overlay--show');
    });
  }

  static hideLoading() {
    const loadingOverlay = document.getElementById('md-loading-overlay');
    if (loadingOverlay) {
      loadingOverlay.classList.remove('md-loading-overlay--show');
      setTimeout(() => {
        loadingOverlay.style.display = 'none';
      }, 300);
    }
  }

  // Theme switching
  static setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('md-theme', theme);
  }

  static getTheme() {
    return localStorage.getItem('md-theme') || 'light';
  }

  // Accessibility helpers
  static announceToScreenReader(message) {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  }
}

// Initialize Material Design when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new MaterialDesign();
  
  // Load saved theme
  const savedTheme = MaterialDesign.getTheme();
  MaterialDesign.setTheme(savedTheme);
});

// Add additional CSS for new components
const additionalStyles = `
  .md-fade-out {
    animation: md-fadeOut 0.3s ease-out forwards;
  }

  @keyframes md-fadeOut {
    to { opacity: 0; transform: scale(0.9); }
  }

  .md-file-item {
    display: flex;
    align-items: center;
    padding: var(--md-spacing-md);
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: var(--md-radius-md);
    margin-bottom: var(--md-spacing-sm);
    background-color: var(--md-surface);
    transition: all var(--md-transition-normal);
  }

  .md-file-item__image {
    width: 48px;
    height: 48px;
    border-radius: var(--md-radius-sm);
    overflow: hidden;
    margin-right: var(--md-spacing-md);
  }

  .md-file-item__image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .md-file-item__info {
    flex: 1;
  }

  .md-file-item__name {
    font-weight: 500;
    margin-bottom: 2px;
  }

  .md-file-item__size {
    font-size: 12px;
    color: var(--md-on-surface-variant);
  }

  .md-file-upload--dragover {
    border-color: var(--md-primary-600);
    background-color: var(--md-primary-50);
  }

  .md-tooltip {
    position: absolute;
    background-color: #616161;
    color: white;
    padding: 8px 16px;
    border-radius: var(--md-radius-sm);
    font-size: 12px;
    white-space: nowrap;
    z-index: 9999;
    pointer-events: none;
    box-shadow: var(--md-shadow-2);
  }

  .md-loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    opacity: 0;
    transition: opacity var(--md-transition-normal);
  }

  .md-loading-overlay--show {
    opacity: 1;
  }

  .md-loading-content {
    background-color: var(--md-surface);
    padding: var(--md-spacing-xl);
    border-radius: var(--md-radius-lg);
    text-align: center;
    box-shadow: var(--md-shadow-3);
    max-width: 300px;
  }

  .md-loading-message {
    margin-top: var(--md-spacing-md);
    color: var(--md-on-surface);
  }

  @keyframes md-circular-rotate {
    0% { stroke-dashoffset: 125.664; }
    50% { stroke-dashoffset: 31.416; }
    100% { stroke-dashoffset: 125.664; }
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .md-text-field--error .md-text-field__input {
    border-color: var(--md-error-500);
  }

  .md-text-field--error .md-text-field__label {
    color: var(--md-error-500);
  }

  .md-text-field--success .md-text-field__input {
    border-color: var(--md-success-500);
  }

  .md-text-field__error {
    color: var(--md-error-500);
    font-size: 12px;
    margin-top: 4px;
    display: none;
  }
`;

// Inject additional styles
const styleElement = document.createElement('style');
styleElement.textContent = additionalStyles;
document.head.appendChild(styleElement);