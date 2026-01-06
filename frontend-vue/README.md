# SafeMind AI - Vue.js Frontend

Mental Health Awareness Chatbot frontend built with Vue 3 and Composition API.

## Features

- ✅ Real-time chat interface
- ✅ Crisis detection and intervention
- ✅ Emergency resources modal
- ✅ Responsive design (mobile-friendly)
- ✅ Session management
- ✅ Message history
- ✅ Loading states and animations
- ✅ Accessibility support

## Tech Stack

- **Vue 3** - Progressive JavaScript framework
- **Composition API** - Modern Vue development
- **Vite** - Fast build tool
- **Axios** - HTTP client
- **Pinia** - State management (optional)

## Prerequisites

- Node.js 16+ and npm
- Backend API running on http://localhost:8000

## Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env

# Start development server
npm run dev
```

The application will be available at http://localhost:3000

## Build for Production

```bash
# Create production build
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend-vue/
├── src/
│   ├── components/
│   │   ├── ChatWindow.vue       # Main chat interface
│   │   └── ResourcesModal.vue   # Emergency resources
│   ├── services/
│   │   └── api.js               # API client
│   ├── utils/
│   │   └── helpers.js           # Utility functions
│   ├── App.vue                  # Main app component
│   ├── main.js                  # App entry point
│   └── style.css                # Global styles
├── public/                      # Static assets
├── index.html                   # HTML template
├── vite.config.js              # Vite configuration
└── package.json                 # Dependencies
```

## API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000/api`.

Available endpoints:
- POST `/api/chat` - Send message
- GET `/api/health` - Health check
- GET `/api/resources` - Get resources
- GET `/api/session/:id` - Get session

See `src/services/api.js` for full API documentation.

## Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
VITE_API_URL=http://localhost:8000/api
VITE_APP_NAME=SafeMind AI
VITE_APP_VERSION=2.0.0
```

## Development

```bash
# Run dev server with hot reload
npm run dev

# Lint and fix files
npm run lint
```

## Components

### ChatWindow.vue
Main chat interface component with:
- Message display (user/bot)
- Risk level indicators
- Typing indicator
- Input area with keyboard shortcuts

### ResourcesModal.vue
Emergency resources modal showing:
- Sri Lankan crisis hotlines
- Emergency services
- Mental health resources
- Disclaimers

## Customization

### Colors
Edit CSS variables in `src/style.css`:
```css
:root {
  --primary-color: #4A90E2;
  --secondary-color: #50E3C2;
  --danger-color: #E74C3C;
  /* ... */
}
```

### API URL
Edit in `.env`:
```env
VITE_API_URL=https://your-api-url.com/api
```

## Keyboard Shortcuts

- **Enter** - Send message
- **Shift + Enter** - New line

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Troubleshooting

### Cannot connect to backend
1. Verify backend is running on port 8000
2. Check CORS settings in backend
3. Verify VITE_API_URL in .env

### Build errors
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

## License

MIT License - See LICENSE file

## Author

Chirath Sanduwara Wijesinghe (CB011568)
Staffordshire University
