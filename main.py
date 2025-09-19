from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv 
from chatbot import ChatBot, get_storage
from typing import Optional
from uuid import uuid4
import re

load_dotenv()

# FastAPI app
app = FastAPI()

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic models for requests
class ChatRequest(BaseModel):
    message: str
    image_data: Optional[str] = None
    source_type: str = "default"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    problem_set: Optional[str] = None

class LoginRequest(BaseModel):
    email: str

# Initialize chatbot
chatbot = ChatBot()

# Routes
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login_page.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chatbot_clean.html", {"request": request})

@app.post("/login")
async def login(login_request: LoginRequest):
    try:
        # Validate MIT email
        email = login_request.email.lower().strip()
        if not email.endswith('@mit.edu'):
            raise HTTPException(status_code=400, detail="Only MIT email addresses are allowed")
        
        # Get storage instance
        storage = get_storage()
        
        # Create or get user
        user = await storage.create_or_get_user(email)
        if not user:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        # Create session
        session_id = uuid4().hex
        session = await storage.create_session(email, session_id)
        if not session:
            raise HTTPException(status_code=500, detail="Failed to create session")
        
        return {
            "success": True,
            "session_id": session_id,
            "user_email": email,
            "message": "Login successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(chat_request: ChatRequest):
    try:
        # Use your async chatbot
        response = await chatbot.process_message(
            message=chat_request.message,
            user_id=chat_request.user_id,
            session_id=chat_request.session_id,
            image_data=chat_request.image_data,
            source_type=chat_request.source_type,
            problem_set=chat_request.problem_set
        )
        
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback_data: dict):
    # Handle feedback without database for now
    return {"status": "success", "message": "Feedback received"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)