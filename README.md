# Stock AI API

This project is a stock analysis API built using Python and Flask. It provides functionalities for fetching stock data, performing sentiment analysis, and making stock recommendations.

## Project Structure

```
stock-ai-api
├── app
│   ├── __init__.py
│   ├── main.py
│   └── goodstockai.py
├── requirements.txt
├── runtime.txt
├── Procfile
├── .gitignore
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd stock-ai-api
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Set the environment variable for Flask:**
   ```
   export FLASK_APP=app/main.py  # On Windows use `set FLASK_APP=app/main.py`
   ```

5. **Run the application locally:**
   ```
   flask run
   ```

## API Endpoints

- **GET /recommendation**: Get stock recommendations based on the provided ticker.
- **GET /sentiment**: Analyze sentiment for a given stock ticker.

## Deployment on Heroku

1. **Create a new Heroku app:**
   ```
   heroku create <app-name>
   ```

2. **Deploy the application:**
   ```
   git push heroku main
   ```

3. **Open the application:**
   ```
   heroku open
   ```

## Usage

You can interact with the API using tools like Postman or curl. Make sure to provide the necessary parameters for the endpoints.

## License

This project is licensed under the MIT License. See the LICENSE file for details.