"""AI Tutor chatbot using open-learning-ai-tutor package with Supabase"""

import logging
import os
import json
from collections.abc import AsyncGenerator
from typing import Optional
from uuid import uuid4
from datetime import datetime, timedelta

from supabase import create_client, Client
# Update this import to use the new package
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from open_learning_ai_tutor.message_tutor import message_tutor
from open_learning_ai_tutor.tools import tutor_tools
from open_learning_ai_tutor.utils import (
    filter_out_system_messages,
    json_to_intent_list,
    json_to_messages,
    tutor_output_to_json,
)

log = logging.getLogger(__name__)


class SupabaseStorage:
    """Supabase storage handler for chat history"""
    
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        environment = os.getenv("ENVIRONMENT", "development")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        
        self.supabase: Client = create_client(url, key)
        self.environment = environment
        
        log.info(f"Connected to Supabase environment: {environment}")
        log.info(f"Supabase URL: {url[:30]}..." if url else "No URL")
    
    async def create_tutorbot_output(self, thread_id: str, chat_json: str, user_email: Optional[str] = None):
        """Save chat output to Supabase"""
        try:
            data = {
                "thread_id": thread_id,
                "chat_json": chat_json,
                "user_email": user_email,
                "created_at": datetime.utcnow().isoformat()
            }
            result = self.supabase.table("tutorbot_outputs").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            log.error(f"Error saving to Supabase: {e}")
            return None
    
    async def get_history(self, thread_id: str):
        """Get latest chat history for a thread from Supabase"""
        try:
            result = self.supabase.table("tutorbot_outputs")\
                .select("*")\
                .eq("thread_id", thread_id)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            log.error(f"Error fetching from Supabase: {e}")
            return None

    # User management functions
    async def create_or_get_user(self, email: str):
        """Create a new user or get existing user by email"""
        try:
            # Try to get existing user
            result = self.supabase.table("users")\
                .select("*")\
                .eq("email", email)\
                .execute()
            
            if result.data:
                # Update last_login for existing user
                user = result.data[0]
                self.supabase.table("users")\
                    .update({"last_login": datetime.utcnow().isoformat()})\
                    .eq("email", email)\
                    .execute()
                return user
            else:
                # Create new user
                new_user = {
                    "email": email,
                    "created_at": datetime.utcnow().isoformat(),
                    "last_login": datetime.utcnow().isoformat()
                }
                result = self.supabase.table("users").insert(new_user).execute()
                return result.data[0] if result.data else None
        except Exception as e:
            log.error(f"Error creating/getting user: {e}")
            return None
    
    async def create_session(self, email: str, session_id: str):
        """Create a new session for the user"""
        try:
            session_data = {
                "session_id": session_id,
                "user_email": email,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat()
            }
            result = self.supabase.table("user_sessions").insert(session_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            log.error(f"Error creating session: {e}")
            return None
    
    async def get_session(self, session_id: str):
        """Get session by session_id"""
        try:
            result = self.supabase.table("user_sessions")\
                .select("*")\
                .eq("session_id", session_id)\
                .gt("expires_at", datetime.utcnow().isoformat())\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            log.error(f"Error getting session: {e}")
            return None

    async def save_feedback(self, thread_id: str, message_content: str, feedback_type: bool, user_email: Optional[str] = None):
        """Save user feedback for a specific message/response"""
        try:
            # First, get the latest tutorbot_output for this thread to establish the relationship
            latest_output = await self.get_history(thread_id)
            tutorbot_output_id = latest_output.get("id") if latest_output else None
            
            feedback_data = {
                "thread_id": thread_id,
                "tutorbot_output_id": tutorbot_output_id,  # Link to the specific conversation
                "message_content": message_content,
                "feedback_type": feedback_type,  # True for thumbs up, False for thumbs down
                "user_email": user_email,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            result = self.supabase.table("feedback").insert(feedback_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            log.error(f"Error saving feedback to Supabase: {e}")
            return None

    async def save_prompt_variant(self, variant_name: str, variant_type: str, prompt_template: str, description: Optional[str] = None):
        """Save or update a prompt variant"""
        try:
            data = {
                "variant_name": variant_name,
                "variant_type": variant_type,
                "prompt_template": prompt_template,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            # Use upsert to handle updates
            result = self.supabase.table("prompt_variants").upsert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            log.error(f"Error saving prompt variant to Supabase: {e}")
            return None

    async def get_prompt_variant(self, variant_name: str):
        """Get a specific prompt variant"""
        try:
            result = self.supabase.table("prompt_variants")\
                .select("*")\
                .eq("variant_name", variant_name)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            log.error(f"Error fetching prompt variant from Supabase: {e}")
            return None

    async def get_all_prompt_variants(self, variant_type: Optional[str] = None):
        """Get all prompt variants, optionally filtered by type"""
        try:
            query = self.supabase.table("prompt_variants").select("*")
            if variant_type:
                query = query.eq("variant_type", variant_type)
            result = query.execute()
            return result.data
        except Exception as e:
            log.error(f"Error fetching prompt variants from Supabase: {e}")
            return []

    async def initialize_prompt_variants(self):
        """Initialize prompt_variants table with current variants from prompts.py"""
        try:
            # Import here to avoid circular imports
            from open_learning_ai_tutor.prompts import (
                PROBLEM_PROMPT_TEMPLATE, 
                PROBLEM_PROMPT_TEMPLATE_V2,
                intent_mapping_variants
            )
            
            # Problem template variants
            problem_variants = [
                {
                    "variant_name": "tutor_problem_v1",
                    "variant_type": "problem_template",
                    "prompt_template": PROBLEM_PROMPT_TEMPLATE,
                    "description": "Original problem template - more directive and structured"
                },
                {
                    "variant_name": "tutor_problem_v2", 
                    "variant_type": "problem_template",
                    "prompt_template": PROBLEM_PROMPT_TEMPLATE_V2,
                    "description": "V2 problem template - more encouraging and discovery-focused"
                }
            ]
            
            # Intent variants
            intent_variants = []
            for variant_name, prompt_text in intent_mapping_variants.items():
                intent_variants.append({
                    "variant_name": variant_name,
                    "variant_type": "intent",
                    "prompt_template": prompt_text,
                    "description": f"Intent variant for {variant_name.split('_')[0]} intent"
                })
            
            # Save all variants
            all_variants = problem_variants + intent_variants
            results = []
            
            for variant in all_variants:
                result = await self.save_prompt_variant(
                    variant["variant_name"],
                    variant["variant_type"], 
                    variant["prompt_template"],
                    variant["description"]
                )
                if result:
                    results.append(result)
                    log.info(f"Saved prompt variant: {variant['variant_name']}")
                else:
                    log.error(f"Failed to save prompt variant: {variant['variant_name']}")
            
            return results
            
        except Exception as e:
            log.error(f"Error initializing prompt variants: {e}")
            return []


# Global storage instance (lazy-loaded)
_storage = None

def get_storage():
    """Get or create the storage instance"""
    global _storage
    if _storage is None:
        _storage = SupabaseStorage()
    return _storage


class TutorBot:
    """
    AI Tutor chatbot using the open-learning-ai-tutor package
    """

    def __init__(
        self,
        user_id: str,
        *,
        name: str = "MIT Open Learning Tutor Chatbot",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        thread_id: Optional[str] = None,
        block_siblings: Optional[list[str]] = None,
        run_readable_id: Optional[str] = None,
        problem_set_title: Optional[str] = None,
        problem: str = "",
        user_email: Optional[str] = None,
    ):
        """Initialize the AI tutor chatbot"""
        self.bot_name = name
        self.model = model or os.getenv("AI_DEFAULT_TUTOR_MODEL", "claude-3-haiku-20240307")
        self.temperature = float(os.getenv("AI_DEFAULT_TEMPERATURE", "0.7")) if temperature is None else temperature
        self.user_id = user_id
        self.user_email = user_email
        self.thread_id = thread_id or uuid4().hex
        
        # Problem set specific attributes
        self.block_siblings = block_siblings
        self.run_readable_id = run_readable_id
        self.problem_set_title = problem_set_title
        self.problem = problem
        self.variant = "canvas"
        
        # Load problem set data
        self.problem_set = get_canvas_problem_set(
            self.run_readable_id or "", self.problem_set_title or ""
        )
        
        # Create LLM for the tutor package
        self.llm = self._create_llm()

    def _create_llm(self):
        """Create and configure the LLM for the tutor package"""
        llm = ChatLiteLLM(model=self.model)
        
        # Set temperature if supported
        unsupported_models = os.getenv('AI_UNSUPPORTED_TEMP_MODELS', '').split(',')
        if self.temperature and self.model not in unsupported_models:
            llm.temperature = self.temperature
            
        return llm

    async def get_tool_metadata(self) -> str:
        """Return metadata for debugging/logging"""
        return json.dumps(
            {
                "block_siblings": self.block_siblings,
                "problem": self.problem,
                "problem_set": self.problem_set,
                "problem_set_title": self.problem_set_title,
                "run_readable_id": self.run_readable_id,
                "model": self.model,
                "thread_id": self.thread_id,
            }
        )

    async def get_completion(
        self,
        message: str,
        *,
        debug: bool = True,  # Enable debug by default to see what's happening
    ) -> AsyncGenerator[str, None]:
        """Get tutor response using the open-learning-ai-tutor package"""
        
        # Load conversation history
        storage = get_storage()
        history = await storage.get_history(self.thread_id)
        
        if history:
            json_history = json.loads(history["chat_json"])
            chat_history = json_to_messages(
                json_history.get("chat_history", [])
            ) + [HumanMessage(content=message)]
            intent_history = json_to_intent_list(json_history["intent_history"])
            assessment_history = json_to_messages(json_history["assessment_history"])
        else:
            chat_history = [HumanMessage(content=message)]
            intent_history = []
            assessment_history = []

        try:
            # Call the tutor package - convert problem_set to string
            problem_set_str = json.dumps(self.problem_set) if isinstance(self.problem_set, dict) else str(self.problem_set)

            result = message_tutor(
                self.problem,
                problem_set_str,
                self.llm,
                [HumanMessage(content=message)],
                chat_history,
                assessment_history,
                intent_history,
                tools=tutor_tools,
                variant=self.variant,
            )
            # Handle A/B testing responses
            if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], dict) and result[0].get("is_ab_test"):
                async for response_chunk in self._handle_ab_test_response(result, message):
                    yield response_chunk
            else:
                # Normal single response
                generator, new_intent_history, new_assessment_history = result
                
                full_response = ""
                new_history = []
                
                async for chunk in generator:
                    if (
                        isinstance(chunk, tuple) and len(chunk) >= 2 and
                        chunk[0] == "messages"
                        and chunk[1]
                        and isinstance(chunk[1][0], AIMessageChunk)
                    ):
                        full_response += chunk[1][0].content
                        yield chunk[1][0].content
                    elif isinstance(chunk, tuple) and len(chunk) >= 2 and chunk[0] == "values":
                        new_history = filter_out_system_messages(chunk[1]["messages"])

                # Save to database
                metadata = {
                    "tutor_model": self.model,
                    "problem_set_title": self.problem_set_title,
                    "run_readable_id": self.run_readable_id,
                }
                json_output = tutor_output_to_json(
                    new_history, new_intent_history, new_assessment_history, metadata
                )
                storage = get_storage()
                await storage.create_tutorbot_output(self.thread_id, json_output, self.user_email)

        except Exception:
            yield '<!-- {"error":{"message":"An error occurred, please try again"}} -->'
            log.exception("Error running AI tutor")

    async def _handle_ab_test_response(self, result, original_message: str) -> AsyncGenerator[str, None]:
        """Handle A/B test responses by collecting both variants and yielding structured data"""
        ab_test_data, new_intent_history, new_assessment_history = result
        
        # Collect both responses completely
        control_response = ""
        treatment_response = ""
        control_history = []
        treatment_history = []
        
        # Process control variant
        control_generator = ab_test_data["responses"][0]["stream"]
        async for chunk in control_generator:
            if (
                chunk[0] == "messages"
                and chunk[1]
                and isinstance(chunk[1][0], AIMessageChunk)
            ):
                control_response += chunk[1][0].content
            elif chunk[0] == "values":
                control_history = filter_out_system_messages(chunk[1]["messages"])
        
        # Process treatment variant
        treatment_generator = ab_test_data["responses"][1]["stream"]
        async for chunk in treatment_generator:
            if (
                chunk[0] == "messages"
                and chunk[1]
                and isinstance(chunk[1][0], AIMessageChunk)
            ):
                treatment_response += chunk[1][0].content
            elif chunk[0] == "values":
                treatment_history = filter_out_system_messages(chunk[1]["messages"])
        
        # Convert message objects to serializable format
        def serialize_messages(messages):
            """Convert message objects to serializable format"""
            serialized = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    serialized.append({
                        "type": msg.__class__.__name__,
                        "content": msg.content
                    })
                else:
                    serialized.append(str(msg))
            return serialized
        
        def serialize_intent_history(intent_history):
            """Convert intent history to serializable format"""
            serialized = []
            for intent_data in intent_history:
                if isinstance(intent_data, dict):
                    serialized_intent = {}
                    for key, value in intent_data.items():
                        if hasattr(value, '__dict__'):
                            serialized_intent[key] = str(value)
                        else:
                            serialized_intent[key] = value
                    serialized.append(serialized_intent)
                else:
                    serialized.append(str(intent_data))
            return serialized
        
        # Create A/B test response structure for frontend
        ab_response = {
            "type": "ab_test_response",
            "control": {
                "content": control_response,
                "variant": "control"
            },
            "treatment": {
                "content": treatment_response,
                "variant": "treatment"
            },
            "metadata": {
                "test_name": "tutor_problem",
                "thread_id": self.thread_id,
                "original_message": original_message,
                "problem_set_title": self.problem_set_title,
                "run_readable_id": self.run_readable_id,
            },
            # Store A/B test variant information
            "ab_test_variants_used": {
                "problem_template": {
                    "control": "tutor_problem_v1",
                    "treatment": "tutor_problem_v2"
                }
                # Intent variants will be added here when they're active
            },
            # Store histories for when user makes choice (serialized)
            "_control_history": serialize_messages(control_history),
            "_treatment_history": serialize_messages(treatment_history),
            "_intent_history": serialize_intent_history(new_intent_history),
            "_assessment_history": serialize_messages(new_assessment_history),
        }
        
        # Yield the structured A/B test response as JSON
        yield f'<!-- {json.dumps(ab_response)} -->'
    
    async def save_ab_test_choice(self, ab_response_data: dict, chosen_variant: str, user_preference_reason: str = "", user_email: Optional[str] = None):
        """Save the user's A/B test choice and update chat history"""
        
        # Get the chosen response data
        chosen_response_data = ab_response_data[chosen_variant]
        chosen_content = chosen_response_data["content"]
        
        # Get the appropriate history based on choice (these are already serialized)
        if chosen_variant == "control":
            new_history = ab_response_data["_control_history"]
        else:
            new_history = ab_response_data["_treatment_history"]
        
        # Get other data (these are already serialized)
        new_intent_history = ab_response_data["_intent_history"]
        new_assessment_history = ab_response_data["_assessment_history"]
        
        # Create metadata including A/B test information
        metadata = {
            "tutor_model": self.model,
            "problem_set_title": self.problem_set_title,
            "run_readable_id": self.run_readable_id,
            "ab_test_chosen_variant": chosen_variant,
            "ab_test_metadata": ab_response_data["metadata"],
            "ab_test_variants_used": ab_response_data.get("ab_test_variants_used", {}),
            "user_preference_reason": user_preference_reason,
        }
        
        # Create the JSON output manually since the histories are already serialized
        json_output = json.dumps({
            "chat_history": new_history,  # Already serialized
            "intent_history": new_intent_history,  # Already serialized
            "assessment_history": new_assessment_history,  # Already serialized
            "metadata": metadata
        })
        
        storage = get_storage()
        await storage.create_tutorbot_output(
            self.thread_id, json_output, user_email
        )
        
        return {
            "success": True,
            "chosen_content": chosen_content,
            "variant": chosen_variant,
        }


class ChatBot:
    """
    Simplified ChatBot wrapper for FastAPI integration
    """
    
    def __init__(self):
        """Initialize the ChatBot"""
        self.default_model = os.getenv("AI_DEFAULT_TUTOR_MODEL", "claude-3-haiku-20240307")
        self.default_temperature = float(os.getenv("AI_DEFAULT_TEMPERATURE", "0.7"))
    
    async def process_message(
        self,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        image_data: Optional[str] = None,
        source_type: str = "default",
        problem_set: Optional[str] = None
    ) -> dict:
        """
        Process a message and return a response
        
        This is the interface expected by main.py
        """
        try:
            # Map problem set IDs to titles
            problem_set_mapping = {
                "pset1": "Problem Set 1: Search",
                "pset2": "Problem Set 2: Games", 
                "pset3": "Problem Set 3: Constraint Satisfaction",
                "pset4": "Problem Set 4: Machine Learning",
                "pset5": "Problem Set 5: Neural Networks"
            }
            
            problem_set_title = None
            if problem_set and problem_set in problem_set_mapping:
                problem_set_title = problem_set_mapping[problem_set]
            
            # Create a TutorBot instance for this conversation
            tutor_bot = TutorBot(
                user_id=user_id or "anonymous",
                thread_id=session_id,
                model=self.default_model,
                temperature=self.default_temperature,
                problem_set_title=problem_set_title,
                problem="",  # Can be set based on context later
                user_email=user_id,  # user_id is the email in this context
            )
            
            # Get response from tutor bot
            response_parts = []
            async for chunk in tutor_bot.get_completion(message):
                response_parts.append(chunk)
            
            return {
                "response": "".join(response_parts),
                "thread_id": tutor_bot.thread_id
            }
            
        except Exception as e:
            log.exception(f"Error in ChatBot.process_message: {e}")
            return {
                "response": "I'm sorry, I encountered an error while processing your message. Please try again.",
                "thread_id": session_id
            }


def get_canvas_problem_set(run_readable_id: str, problem_set_title: str) -> dict:
    """
    Load problem set from local .tex files in ai_chatbots/problem_sets directory
    
    Args:
        run_readable_id: The readable id of the run (currently unused)
        problem_set_title: The title of the problem set
    
    Returns:
        problem_set: Dictionary containing problems and solutions content
    """
    from pathlib import Path
    
    # Get the directory where this file is located
    current_dir = Path(__file__).parent
    problem_sets_dir = current_dir / "problem_sets"
    
    try:
        # Map problem set titles to file prefixes
        title_to_prefix = {
            "Problem Set 1: Search": "PSet1",
            "Problem Set 2: Games": "PSet2",
            # Add more mappings as needed
        }
        
        prefix = title_to_prefix.get(problem_set_title)
        if not prefix:
            return {"error": f"No file mapping found for problem set: {problem_set_title}"}
        
        # Load problems file
        problems_file = problem_sets_dir / f"{prefix}_Problems.tex"
        solutions_file = problem_sets_dir / f"{prefix}_Solutions.tex"
        
        problems_content = ""
        solutions_content = ""
        
        if problems_file.exists():
            with open(problems_file, 'r', encoding='utf-8') as f:
                problems_content = f.read()
        
        if solutions_file.exists():
            with open(solutions_file, 'r', encoding='utf-8') as f:
                solutions_content = f.read()
        
        return {
            "problems": problems_content,
            "solutions": solutions_content,
            "title": problem_set_title,
            "prefix": prefix,
            "metadata": {
                "problems_file": str(problems_file),
                "solutions_file": str(solutions_file),
                "run_readable_id": run_readable_id
            }
        }
        
    except Exception as e:
        return {"error": f"Error loading problem set '{problem_set_title}': {str(e)}"}