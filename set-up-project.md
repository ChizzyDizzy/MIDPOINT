# SafeMind Project Setup Guide

This guide provides comprehensive instructions for setting up, running, and evaluating the SafeMind mental health support platform.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: Version 3.11 or higher
- **Node.js**: Version 16.0 or higher
- **npm**: Version 8.0 or higher
- **RAM**: Minimum 4GB (8GB recommended for model training)
- **Disk Space**: At least 2GB free space

### Development Tools
- Git (for version control)
- Text editor or IDE (VS Code, PyCharm, etc.)
- Terminal/Command Prompt access

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd MIDPOINT
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
cd backend
python3 -m venv venv
```

Activate the virtual environment:
- **macOS/Linux**: `source venv/bin/activate`
- **Windows**: `venv\Scripts\activate`

Install required packages:
```bash
pip install -r requirements.txt
```

#### Configure Environment Variables

Create a configuration file:
```bash
cp .env.example .env
```

Edit `.env` and configure the following settings:

```env
# Response Engine Configuration
BACKEND_MODE=openai          # Options: openai, huggingface, local, fallback
FLASK_ENV=development        # Use 'production' for deployment

# Crisis Detection Settings
CRISIS_DETECTION_THRESHOLD=0.7
ENABLE_RESPONSES=True
DEFAULT_CULTURE=south_asian

# External Service Integration (Optional)
# If using external NLP services, add API credentials here
API_KEY=your-api-key-here
MODEL_NAME=gpt-3.5-turbo    # or other model identifier
```

**Backend Mode Options:**

| Mode | Description | Requirements |
|------|-------------|--------------|
| `openai` | External NLP service (best quality) | API key required |
| `huggingface` | Hugging Face model API | API key required |
| `local` | Locally trained model | Model training required |
| `fallback` | Template-based responses | No external dependencies |

### 3. Frontend Setup

Open a new terminal window:

```bash
cd frontend
npm install
```

Configure the API endpoint in `frontend/src/services/api.js` if needed (defaults to `http://localhost:5001`).

### 4. Data Preparation

The project includes pre-configured datasets in the `data/` directory:

- `mental_health_dataset.json` - Training data with 1,500+ examples
- `enhanced_crisis_patterns.json` - Crisis detection patterns
- `response_templates.json` - Template responses
- `cultural_templates.json` - Cultural context data

No additional data preparation is needed for basic operation.

## Running the Application

### Start the Backend Server

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python3 app_improved.py
```

The server will start on `http://localhost:5001`

You should see:
```
 * Running on http://127.0.0.1:5001
 * Debug mode: on
```

### Start the Frontend Application

In a new terminal:

```bash
cd frontend
npm start
```

The application will open automatically at `http://localhost:3000`

## Model Training (Optional)

For running a locally trained model instead of external services:

### Generate Expanded Dataset

```bash
cd scripts
python3 expand_dataset.py
```

This creates an expanded training dataset with paraphrased examples.

### Train the Model

```bash
cd backend
source venv/bin/activate
python3 train_model.py
```

**Training Parameters:**
- **Model**: DialoGPT (Microsoft conversational model)
- **Epochs**: 3-5 recommended
- **Batch Size**: 4-8 depending on available RAM
- **Learning Rate**: 5e-5

**Training Time:**
- CPU: 4-8 hours for full dataset
- GPU: 30-60 minutes (recommended for faster training)

### Cloud Training Option

For faster training with free GPU access:

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the training script and dataset
3. Enable GPU runtime: Runtime → Change runtime type → GPU (T4)
4. Run the training cells
5. Download the trained model files
6. Place model files in `backend/models/`

### Configure Local Model

After training, update `.env`:

```env
BACKEND_MODE=local
MODEL_PATH=./models/trained_model
```

## Testing

### Run Automated Tests

```bash
cd backend
source venv/bin/activate
python3 test_mvp.py
```

**Test Coverage:**
- Low-risk conversation handling
- Medium-risk detection (anxiety, stress)
- High-risk detection (self-harm indicators)
- Immediate risk detection (suicidal ideation)
- Cultural context understanding (A/L stress, family pressure)
- Positive message responses
- Academic stress scenarios
- Work-related stress

### Evaluation Metrics

The test suite reports:
- **Crisis Detection Accuracy**: Percentage of correctly identified risk levels
- **Response Appropriateness**: Manual review recommended
- **Latency**: Average response time
- **Pass/Fail Status**: Overall system health

**Target Metrics:**
- Crisis detection accuracy: ≥ 90%
- Response time: < 2 seconds
- All tests passing

### Manual Testing

Test the following scenarios through the interface:

1. **Low Risk**: "I'm feeling a bit stressed about my studies"
2. **Medium Risk**: "I can't sleep, I feel hopeless about everything"
3. **High Risk**: "I want to hurt myself, nobody cares"
4. **Immediate Risk**: "I'm going to end it tonight"
5. **Cultural Context**: "My parents are disappointed in my A/L results"
6. **Positive**: "I feel better after talking to my friends"

Verify:
- Appropriate crisis detection level
- Emergency resources displayed for high/immediate risk
- Culturally sensitive responses
- Mood tracking updates correctly

## Troubleshooting

### Backend Issues

**Port 5001 already in use:**
```bash
# Find and kill the process
lsof -ti:5001 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :5001   # Windows (note the PID, then: taskkill /PID <pid> /F)
```

**Module not found errors:**
```bash
pip install -r requirements.txt --upgrade
```

**Database/session errors:**
```bash
rm -rf __pycache__
rm -rf instance/
```

### Frontend Issues

**Port 3000 already in use:**
- The browser will prompt to use a different port (3001)
- Or kill the existing process

**npm install fails:**
```bash
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

**CORS errors:**
- Verify backend is running on port 5001
- Check `frontend/src/services/api.js` has correct API URL

### Model Training Issues

**Out of memory:**
- Reduce batch size in `train_model.py`
- Use cloud training with GPU
- Close other applications

**Training loss not decreasing:**
- Increase number of epochs
- Verify dataset quality
- Check learning rate (try 3e-5 or 7e-5)

**Model not loading:**
- Verify `MODEL_PATH` in `.env`
- Check model files exist and are complete
- Re-train if necessary

## Configuration Reference

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `BACKEND_MODE` | openai, huggingface, local, fallback | openai | Response generation backend |
| `FLASK_ENV` | development, production | development | Flask environment |
| `API_KEY` | string | - | External service API key |
| `MODEL_NAME` | string | gpt-3.5-turbo | Model identifier |
| `CRISIS_DETECTION_THRESHOLD` | 0.0-1.0 | 0.7 | Sensitivity threshold |
| `ENABLE_RESPONSES` | True, False | True | Enable response generation |
| `DEFAULT_CULTURE` | south_asian, general | south_asian | Cultural context |
| `MODEL_PATH` | path | ./models | Local model directory |

### Crisis Detection Levels

| Level | Risk Indicators | System Response |
|-------|----------------|-----------------|
| **Low** | Stress, mild anxiety, daily worries | Supportive conversation, coping strategies |
| **Medium** | Persistent sadness, hopelessness, sleep issues | Empathy, professional help suggestion |
| **High** | Self-harm thoughts, isolation, burden feelings | Crisis resources, immediate support |
| **Immediate** | Suicidal ideation, active planning, means access | Emergency alert, hotline display, urgent intervention |

### Response Templates

Located in `data/response_templates.json`:
- Organized by topic (anxiety, depression, stress, etc.)
- Culturally adapted responses
- Fallback responses when external services unavailable

## Deployment Considerations

### Production Setup

1. Set `FLASK_ENV=production` in `.env`
2. Use a production WSGI server (Gunicorn, uWSGI)
3. Enable HTTPS for secure communication
4. Set up proper logging and monitoring
5. Configure firewall rules
6. Use environment variables for sensitive data

### Security

- Never commit `.env` file to version control
- Rotate API keys regularly
- Validate and sanitize all user input
- Implement rate limiting
- Use secure session management
- Regular security audits

### Monitoring

Track the following metrics:
- Response latency
- Crisis detection accuracy
- User session duration
- Error rates
- Resource utilization

## Dataset Information

### Training Data Format

```json
{
  "conversations": [
    {
      "user": "User message",
      "bot": "System response",
      "context": "conversation context",
      "risk_level": "low|medium|high|immediate"
    }
  ]
}
```

### Expanding the Dataset

To add custom training examples:

1. Edit `data/mental_health_dataset.json`
2. Follow the existing format
3. Include diverse scenarios
4. Balance risk levels
5. Re-train the model

```bash
python3 scripts/expand_dataset.py
python3 backend/train_model.py
```

## Performance Optimization

### Backend Optimization
- Use caching for common responses
- Implement request queuing for high load
- Optimize crisis detection algorithms
- Use async processing where appropriate

### Frontend Optimization
- Enable production build: `npm run build`
- Implement lazy loading for components
- Optimize image and asset sizes
- Use service workers for offline capability

## Support and Maintenance

### Regular Maintenance Tasks
- Update dependencies monthly
- Review and update crisis patterns
- Analyze conversation logs for improvements
- Monitor system performance
- Backup conversation data

### Evaluation Schedule
- Weekly: Review crisis detection accuracy
- Monthly: Analyze user feedback
- Quarterly: Update training dataset
- Annually: Full system audit

## Project Structure Reference

```
MIDPOINT/
├── backend/                    # Python Flask backend
│   ├── app_improved.py         # Main application server
│   ├── ai_model_free.py        # Response generation logic
│   ├── enhanced_safety_detector.py  # Crisis detection system
│   ├── context_manager.py      # Session management
│   ├── cultural_adapter.py     # Cultural adaptation
│   ├── config.py               # Configuration loader
│   ├── train_model.py          # Model training script
│   ├── test_mvp.py             # Test suite
│   ├── .env.example            # Environment template
│   └── requirements.txt        # Python dependencies
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── App.js              # Main component
│   │   ├── components/         # UI components
│   │   └── services/           # API integration
│   └── package.json            # Node dependencies
├── data/                       # Datasets and configs
│   ├── mental_health_dataset.json
│   ├── enhanced_crisis_patterns.json
│   ├── response_templates.json
│   └── cultural_templates.json
├── scripts/                    # Utility scripts
│   └── expand_dataset.py
├── README.md                   # Project overview
└── set-up-project.md           # This file
```

## Additional Resources

### Crisis Hotlines (Sri Lanka)
- **National Crisis Line**: 1333
- **Sumithrayo**: 011-2696666
- **Emergency Services**: 119

### Technical Documentation
- Flask: https://flask.palletsprojects.com/
- React: https://react.dev/
- Transformers: https://huggingface.co/docs/transformers/

## Disclaimer

SafeMind is a thesis project and educational tool. It is **not a substitute for professional mental health care**. Users experiencing mental health crises should contact emergency services or qualified mental health professionals.
