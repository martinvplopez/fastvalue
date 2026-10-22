"""FastValue multi-agent decision system.

Agents are discovered automatically from ``agents/*_agent.py`` modules that use
``@register_agent``. Import the orchestrator from ``agents.master_agent``.
"""

from agents.base import AgentAssessment, BaseAgent
from agents.registry import agent_metadata, create_agents, get_agent_classes, register_agent

__all__ = [
    "AgentAssessment",
    "BaseAgent",
    "agent_metadata",
    "create_agents",
    "get_agent_classes",
    "register_agent",
]
