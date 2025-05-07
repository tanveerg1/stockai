from flask import Flask, request, jsonify
from app.goodstockai import process_query
import os

app = Flask(__name__)
@app.route("/")
def home():
    return "Stock AI API is running!"

@app.route('/api/query', methods=['POST'])
def query_stock():
    data = request.get_json()
    ticker = data.get('ticker', 'TSLA.TO')  # Default to TSLA.TO if no ticker is provided
    query = data.get('query', '')

    ##
    # portfolio = data.get('portfolio', [])  # Portfolio sent from the app

    if not query:
        # return jsonify({"error": "Query is required."}), 400
        # Respond with a friendly message if no query is provided
        return jsonify({"response": "Hi there! How can I assist you with stocks today?"})


    response = process_query(query, ticker)
    return jsonify({"response": response})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000)) 
    app.run(host='0.0.0.0', port=port)