from flask import Flask, request, jsonify
from goodstockai import process_query
from server_api import client  # Import the MongoDB client
import os
from werkzeug.security import check_password_hash

# MongoDB connection
db_name = os.getenv("MONGO_DB_NAME")
db_collection = os.getenv("MONGO_DB_COLLECTION")

# Database
db = client[db_name]
users_collection = db[db_collection]

app = Flask(__name__)
@app.route("/")
def home():
    return "Stock AI API is running!"

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    # Check if the username already exists
    if users_collection.find_one({"username": username}):
        return jsonify({"error": "Username already exists. Please choose a different one."}), 409

    # Hash the password
    # hashed_password = generate_password_hash(password)

    # Save the user to the database
    users_collection.insert_one({"username": username, "password": password})

    return jsonify({"message": "User registered successfully!"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    # Fetch user from the database
    user = users_collection.find_one({"username": username})
    if not user:
        return jsonify({"error": "Invalid username or password."}), 401

    # Verify the password
    if not check_password_hash(user['password'], password):
        return jsonify({"error": "Invalid username or password."}), 401

    return jsonify({"message": "Login successful!"})

@app.route('/recommend', methods=['POST'])
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