import os
import voyageai
import anthropic
import json
from embedder import load_embeddings
import os
from dotenv import load_dotenv


# Initialize global clients as None
vo = None
client = None

def init_clients():
    """Initialize the global clients"""
    global vo, client
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize clients using environment variables
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    print("Clients initialized successfully")

if __name__ == "__main__":
    init_clients()
    load_embeddings()