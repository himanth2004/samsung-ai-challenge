# AI Sentinel - Behavioral Analysis Application

A Flask-based web application for real-time behavioral pattern analysis and anomaly detection using AI models.

## Features

- **Mouse Dynamics Analysis**: Captures and analyzes mouse movement patterns
- **Typing Pattern Analysis**: Analyzes typing speed and patterns with 250 WPM threshold
- **Real-time Monitoring**: Live behavioral pattern detection
- **AI-powered Classification**: Distinguishes between human and robotic behavior

## Project Structure

```
ai-challenge/src/
├── app.py                 # Main Flask application
├── model_definition.py    # PyTorch model architecture
├── behavior_model.pth     # Trained mouse dynamics model
├── typing_model.pkl       # Trained typing pattern model
├── templates/             # HTML templates
│   ├── index.html         # Landing page
│   ├── main.html          # Command center
│   └── about.html         # About page
├── requirements.txt       # Python dependencies
└── run.py                # Application startup script
```

## Installation & Setup

1. **Navigate to the project directory:**
   ```bash
   cd ai-challenge/src
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python run.py
   ```
   
   Or alternatively:
   ```bash
   python app.py
   ```

## Usage

1. **Start the application** using one of the methods above
2. **Open your browser** and go to `http://localhost:5000`
3. **Navigate to the Command Center** at `http://localhost:5000/main`
4. **Test the features:**
   - Mouse Dynamics: Move your mouse in the monitoring area
   - Typing Analysis: Enter text for pattern analysis

## API Endpoints

- `GET /` - Landing page
- `GET /main` - Command center interface
- `GET /about` - About page
- `POST /predict` - Mouse dynamics prediction
- `POST /predict_typing` - Typing pattern prediction

## Deployment

For deployment on platforms like Render:

1. Ensure all files are in the `src/` directory
2. Set the build command: `pip install -r requirements.txt`
3. Set the start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. The application will be available at the provided URL

## Technical Details

- **Framework**: Flask 2.3.3
- **ML Framework**: PyTorch 2.1.0
- **ML Library**: Scikit-learn 1.3.0
- **Typing Speed Threshold**: 250 WPM (above this = ROBOT classification)
- **Model Types**: 
  - Mouse Dynamics: Transformer-based neural network
  - Typing Patterns: TF-IDF + Machine Learning classifier

## Troubleshooting

- **Model Loading Issues**: Ensure `behavior_model.pth` and `typing_model.pkl` are in the same directory as `app.py`
- **Port Issues**: The application runs on port 5000 by default
- **Dependencies**: Make sure all requirements are installed correctly

## License

This project is part of the AI Sentinel behavioral analysis system.
