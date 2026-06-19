from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.force_offline or not self.langchain_agent:
            return self._reply_offline(thread_id, message)
            
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        
        session = self.sessions[thread_id]
        
        # Add user message
        session.messages.append({"role": "user", "content": message})
        
        # Calculate prompt token usage
        history_text = " ".join([m["content"] for m in session.messages])
        prompt_tokens = estimate_tokens(history_text)
        session.prompt_tokens_processed += prompt_tokens
        
        # Prepare context for langchain
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        langchain_messages = []
        for m in session.messages:
            if m["role"] == "user":
                langchain_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                langchain_messages.append(AIMessage(content=m["content"]))
            else:
                langchain_messages.append(SystemMessage(content=m["content"]))
                
        response = self.langchain_agent.invoke(langchain_messages)
        ai_message = response.content
        
        # Add AI message
        session.messages.append({"role": "assistant", "content": ai_message})
        
        # Calculate generation tokens
        gen_tokens = estimate_tokens(ai_message)
        session.token_usage += gen_tokens
        
        return {
            "response": ai_message,
            "agent_tokens": gen_tokens,
            "prompt_tokens_processed": prompt_tokens
        }

    def token_usage(self, thread_id: str) -> int:
        return self.sessions[thread_id].token_usage if thread_id in self.sessions else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions[thread_id].prompt_tokens_processed if thread_id in self.sessions else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        
        session = self.sessions[thread_id]
        session.messages.append({"role": "user", "content": message})
        
        prompt_tokens = estimate_tokens(" ".join([m["content"] for m in session.messages]))
        session.prompt_tokens_processed += prompt_tokens
        
        reply_content = f"Offline reply to: {message[:20]}..."
        session.messages.append({"role": "assistant", "content": reply_content})
        
        gen_tokens = estimate_tokens(reply_content)
        session.token_usage += gen_tokens
        
        return {
            "response": reply_content,
            "agent_tokens": gen_tokens,
            "prompt_tokens_processed": prompt_tokens
        }

    def _maybe_build_langchain_agent(self):
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
