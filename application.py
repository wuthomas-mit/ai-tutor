from flask import Flask, request, jsonify, render_template, session, redirect, g
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv

from helpers import ask, followup, init_clients, load_embeddings, login_required

# Global variables

# Initialize global clients as None
vo = None
client = None

# Load environmental variables
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is not set in the environment!")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY is not set in the environment!")

FLASK_SECRET_KEY = os.getenv("VOYAGE_API_KEY")
if not FLASK_SECRET_KEY:
    raise ValueError("FLASK_SECRET_KEY is not set in the environment!")

####################################################

app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///TA_GPT.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = FLASK_SECRET_KEY

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'  # Can also be 'redis', 'memcached', etc.
app.config['SESSION_FILE_DIR'] = './flask_sessions'  # Where to store session files
app.config['SESSION_PERMANENT'] = False
Session(app)

db = SQLAlchemy(app)

# Define the User model
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)

# Ensure database tables are created
with app.app_context():
    # db.drop_all()
    db.create_all()
    users = User.query.all()
    print(users)  # Should list all users

# Serve the login page
@app.route('/login', methods=['GET', 'POST'])
def login():

    session.clear()

    if request.method == 'POST':
        data = request.get_json()  # Ensure you are receiving JSON data
        user_id = str(data.get('id'))

        if not user_id:
            return jsonify({'message': 'User ID is required!'}), 400

        # Check if the user exists in the database
        user = User.query.get(user_id)
        if user:
            session['user_id'] = user.id  # Save user ID in the session
            session['user_name'] = user.name  # Save user name in the session
            return redirect("/")
        else:
            return jsonify({'message': 'User not found!'}), 404

    # If GET request, serve the login page
    return render_template('login.html')

# Serve the chatbot page
@app.route('/', methods=['GET'])
@login_required
def home():
    # Reset the conversation state when loading the home page
    session['is_first_question'] = True
    user_name = session.get('user_name', 'Guest')
    return render_template('chatbot.html', user_name=user_name)

# Handle chat requests
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_message = request.json.get('message', '')
    
    # Check if this is the first question in the conversation
    is_first_question = session.get('is_first_question', True)
    
    if is_first_question:
        # First question uses ask()
        bot_response = ask(user_message)
        session['is_first_question'] = False  # Update the conversation state
    else:
        # Subsequent questions use followup()
        bot_response = followup(user_message)
    
    return jsonify({'response': bot_response})

# Ensure a user is logged in before any request except /login
@app.before_request
def require_login():
    if not session.get("user_id") and request.endpoint not in ["login", "static"]:
        return redirect("/login")
    
# Logout route
@app.route('/logout', methods=['GET'])
def logout():
    session.clear()  # Clear the session
    return redirect('/login')  # Redirect to the login page

# Initialize the clients and load embeddings
if __name__ == '__main__':
    # Initialize clients and load embeddings
    init_clients()
    document_names, documents, documents_embeddings = load_embeddings()

    app.run(debug=True)
