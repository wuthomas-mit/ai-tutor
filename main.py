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
import logging

log = logging.getLogger(__name__)

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

class ABTestChoiceRequest(BaseModel):
    thread_id: str
    chosen_variant: str
    reason: Optional[str] = ""
    ab_test_data: dict
    user_email: Optional[str] = None

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
        result = await chatbot.process_message(
            message=chat_request.message,
            user_id=chat_request.user_id,
            session_id=chat_request.session_id,
            image_data=chat_request.image_data,
            source_type=chat_request.source_type,
            problem_set=chat_request.problem_set
        )
        
        return {
            "response": result["response"],
            "thread_id": result["thread_id"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FeedbackRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str
    feedback: bool
    user_email: Optional[str] = None

@app.post("/feedback")
async def submit_feedback(feedback_data: FeedbackRequest):
    try:
        # Get the storage instance
        storage = get_storage()
        
        # Use thread_id if provided, otherwise generate a default one
        thread_id = feedback_data.thread_id or "unknown"
        
        # Save feedback to database
        result = await storage.save_feedback(
            thread_id=thread_id,
            message_content=feedback_data.message,
            feedback_type=feedback_data.feedback,
            user_email=feedback_data.user_email
        )
        
        if result:
            return {"status": "success", "message": "Feedback saved successfully", "feedback_id": result.get("id")}
        else:
            return {"status": "error", "message": "Failed to save feedback"}
            
    except Exception as e:
        log.error(f"Error in feedback endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

@app.post("/ab-test-choice")
async def submit_ab_test_choice(choice_request: ABTestChoiceRequest):
    try:
        # Create a TutorBot instance to access the save method
        from chatbot import TutorBot
        
        # Create TutorBot instance with the thread_id
        tutor_bot = TutorBot("generic_user", thread_id=choice_request.thread_id)
        
        # Save the A/B test choice
        result = await tutor_bot.save_ab_test_choice(
            choice_request.ab_test_data,
            choice_request.chosen_variant,
            choice_request.reason or "",
            choice_request.user_email
        )
        
        return {
            "status": "success", 
            "message": "A/B test choice recorded successfully",
            "chosen_variant": choice_request.chosen_variant
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save A/B test choice: {str(e)}")

import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)