from flask import Flask, request, jsonify
from app.goodstockai import process_query

app = Flask(__name__)
@app.route("/")
def home():
    return "Stock AI API is running!"

@app.route('/api/query', methods=['POST'])
def query_stock():
    data = request.get_json()
    ticker = data.get('ticker', 'TSLA.TO')  # Default to TSLA.TO if no ticker is provided
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required."}), 400

    response = process_query(query, ticker)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)