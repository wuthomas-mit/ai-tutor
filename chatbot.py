"""AI Tutor chatbot using open-learning-ai-tutor package"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Optional
from uuid import uuid4

from channels.db import database_sync_to_async
from django.conf import settings
from langchain_community.chat_models import ChatLiteLLM
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

from ai_chatbots.models import TutorBotOutput

log = logging.getLogger(__name__)


@database_sync_to_async
def create_tutorbot_output(thread_id, chat_json):
    return TutorBotOutput.objects.create(
        thread_id=thread_id, chat_json=chat_json
    )


@database_sync_to_async
def get_history(thread_id):
    return TutorBotOutput.objects.filter(thread_id=thread_id).last()


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
    ):
        """Initialize the AI tutor chatbot"""
        self.bot_name = name
        self.model = model or settings.AI_DEFAULT_TUTOR_MODEL
        self.temperature = temperature or settings.AI_DEFAULT_TEMPERATURE
        self.user_id = user_id
        self.thread_id = thread_id or uuid4().hex
        
        # Problem set specific attributes
        self.block_siblings = block_siblings
        self.run_readable_id = run_readable_id
        self.problem_set_title = problem_set_title
        self.problem = problem
        self.variant = "canvas"
        
        # Load problem set data
        self.problem_set = get_canvas_problem_set(
            self.run_readable_id, self.problem_set_title
        )
        
        # Create LLM for the tutor package
        self.llm = self._create_llm()

    def _create_llm(self):
        """Create and configure the LLM for the tutor package"""
        llm = ChatLiteLLM(model=self.model)
        
        # Set temperature if supported
        if self.temperature and self.model not in getattr(settings, 'AI_UNSUPPORTED_TEMP_MODELS', []):
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
        debug: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Get tutor response using the open-learning-ai-tutor package"""
        
        # Load conversation history
        history = await get_history(self.thread_id)
        
        if history:
            json_history = json.loads(history.chat_json)
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
            # Call the tutor package
            result = message_tutor(
                self.problem,
                self.problem_set,
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
                        chunk[0] == "messages"
                        and chunk[1]
                        and isinstance(chunk[1][0], AIMessageChunk)
                    ):
                        full_response += chunk[1][0].content
                        yield chunk[1][0].content
                    elif chunk[0] == "values":
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
                await create_tutorbot_output(self.thread_id, json_output)

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
            # Store histories for when user makes choice (serialized)
            "_control_history": serialize_messages(control_history),
            "_treatment_history": serialize_messages(treatment_history),
            "_intent_history": serialize_intent_history(new_intent_history),
            "_assessment_history": serialize_messages(new_assessment_history),
        }
        
        # Yield the structured A/B test response as JSON
        yield f'<!-- {json.dumps(ab_response)} -->'
    
    async def save_ab_test_choice(self, ab_response_data: dict, chosen_variant: str, user_preference_reason: str = ""):
        """Save the user's A/B test choice and update chat history"""
        
        # Get the chosen response data
        chosen_response_data = ab_response_data[chosen_variant]
        chosen_content = chosen_response_data["content"]
        
        # Get the appropriate history based on choice
        if chosen_variant == "control":
            new_history = ab_response_data["_control_history"]
        else:
            new_history = ab_response_data["_treatment_history"]
        
        # Get other data
        new_intent_history = ab_response_data["_intent_history"]
        new_assessment_history = ab_response_data["_assessment_history"]
        
        # Create metadata including A/B test information
        metadata = {
            "tutor_model": self.model,
            "problem_set_title": self.problem_set_title,
            "run_readable_id": self.run_readable_id,
            "ab_test_chosen_variant": chosen_variant,
            "ab_test_metadata": ab_response_data["metadata"],
            "user_preference_reason": user_preference_reason,
        }
        
        # Save to database
        json_output = tutor_output_to_json(
            new_history, new_intent_history, new_assessment_history, metadata
        )
        await create_tutorbot_output(
            self.thread_id, json_output
        )
        
        return {
            "success": True,
            "chosen_content": chosen_content,
            "variant": chosen_variant,
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