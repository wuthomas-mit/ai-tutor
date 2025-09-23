import pytest
import json
from open_learning_ai_tutor.utils import (
    messages_to_json,
    json_to_messages,
    intent_list_to_json,
    json_to_intent_list,
    tutor_output_to_json,
)

from open_learning_ai_tutor.constants import Intent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


def test_messages_to_json():
    """Test messages_to_json function"""
    messages = [
        SystemMessage(content="tutor prompt"),
        HumanMessage(content="what should i try first"),
        AIMessage(content="Let's start by thinking about the problem."),
    ]

    expected_output = [
        {
            "type": "SystemMessage",
            "content": "tutor prompt",
        },
        {
            "type": "HumanMessage",
            "content": "what should i try first",
        },
        {
            "type": "AIMessage",
            "content": "Let's start by thinking about the problem.",
        },
    ]
    assert messages_to_json(messages) == expected_output


def test_json_to_messages():
    """Test json_to_messages function"""
    json_messages = [
        {
            "type": "SystemMessage",
            "content": "tutor prompt",
        },
        {
            "type": "HumanMessage",
            "content": "what should i try first",
        },
        {
            "type": "AIMessage",
            "content": "Let's start by thinking about the problem.",
        },
    ]

    expected_output = [
        SystemMessage(content="tutor prompt"),
        HumanMessage(content="what should i try first"),
        AIMessage(content="Let's start by thinking about the problem."),
    ]

    assert json_to_messages(json_messages) == expected_output


def test_intent_list_to_json():
    intent_lists = [
        [Intent.P_LIMITS],
        [Intent.P_GENERALIZATION, Intent.P_HYPOTHESIS, Intent.P_ARTICULATION],
        [Intent.S_STATE, Intent.S_CORRECTION],
        [Intent.G_REFUSE, Intent.P_ARTICULATION],
        [Intent.G_REFUSE],
    ]

    expected_output = '[["P_LIMITS"], ["P_GENERALIZATION", "P_HYPOTHESIS", "P_ARTICULATION"], ["S_STATE", "S_CORRECTION"], ["G_REFUSE", "P_ARTICULATION"], ["G_REFUSE"]]'

    assert intent_list_to_json(intent_lists) == expected_output


def test_json_to_intent_list():
    json_str = '[["P_LIMITS"], ["P_GENERALIZATION", "P_HYPOTHESIS", "P_ARTICULATION"], ["S_STATE", "S_CORRECTION"], ["G_REFUSE", "P_ARTICULATION"], ["G_REFUSE"]]'

    expected_output = [
        [Intent.P_LIMITS],
        [Intent.P_GENERALIZATION, Intent.P_HYPOTHESIS, Intent.P_ARTICULATION],
        [Intent.S_STATE, Intent.S_CORRECTION],
        [Intent.G_REFUSE, Intent.P_ARTICULATION],
        [Intent.G_REFUSE],
    ]

    assert json_to_intent_list(json_str) == expected_output


def test_tutor_output_to_json():
    chat_history = [
        HumanMessage(content="what should i try first"),
        AIMessage(content="Let's start by thinking about the problem."),
    ]
    intent_history = [[Intent.P_HYPOTHESIS]]
    assessment_history = [
        HumanMessage(content="what do i do next?"),
        AIMessage(content='{"justification": "test", "selection": "g"}'),
    ]
    metadata = {"tutor_model": "test_model"}

    expected_output = {
        "chat_history": messages_to_json(chat_history),
        "intent_history": intent_list_to_json(intent_history),
        "assessment_history": messages_to_json(assessment_history),
        "metadata": metadata,
    }

    expected_output = json.dumps(expected_output)

    assert (
        tutor_output_to_json(chat_history, intent_history, assessment_history, metadata)
        == expected_output
    )
