from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        
        profiles_dir = self.config.state_dir / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profile_store = UserProfileStore(profiles_dir)
        
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.force_offline or not self.langchain_agent:
            return self._reply_offline(user_id, thread_id, message)
            
        if thread_id not in self.thread_tokens:
            self.thread_tokens[thread_id] = 0
        if thread_id not in self.thread_prompt_tokens:
            self.thread_prompt_tokens[thread_id] = 0
            
        updates = extract_profile_updates(message)
        if updates:
            current_profile = self.profile_store.read_text(user_id)
            new_facts = "\n" + "\n".join([f"- {k}: {v}" for k, v in updates.items()])
            self.profile_store.write_text(user_id, current_profile + new_facts)
            
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        prompt_tokens += estimate_tokens(message)
        self.thread_prompt_tokens[thread_id] += prompt_tokens
            
        self.compact_memory.append(thread_id, "user", message)
        
        profile_content = self.profile_store.read_text(user_id)
        cm_ctx = self.compact_memory.context(thread_id)
        
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        langchain_messages = []
        
        system_prompt = f"Profile:\n{profile_content}\n\nSummary of older context:\n{cm_ctx['summary']}"
        langchain_messages.append(SystemMessage(content=system_prompt))
        
        for m in cm_ctx["messages"]:
            if m["role"] == "user":
                langchain_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                langchain_messages.append(AIMessage(content=m["content"]))
                
        response = self.langchain_agent.invoke(langchain_messages)
        ai_message = response.content
        
        self.compact_memory.append(thread_id, "assistant", ai_message)
        
        gen_tokens = estimate_tokens(ai_message)
        self.thread_tokens[thread_id] += gen_tokens
        
        return {
            "response": ai_message,
            "agent_tokens": gen_tokens,
            "prompt_tokens_processed": prompt_tokens
        }

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.thread_tokens:
            self.thread_tokens[thread_id] = 0
            self.thread_prompt_tokens[thread_id] = 0
            
        updates = extract_profile_updates(message)
        if updates:
            current_profile = self.profile_store.read_text(user_id)
            new_facts = "\n" + "\n".join([f"- {k}: {v}" for k, v in updates.items()])
            self.profile_store.write_text(user_id, current_profile + new_facts)
            
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        prompt_tokens += estimate_tokens(message)
        self.thread_prompt_tokens[thread_id] += prompt_tokens
            
        self.compact_memory.append(thread_id, "user", message)
        
        ai_message = self._offline_response(user_id, thread_id, message)
        
        self.compact_memory.append(thread_id, "assistant", ai_message)
        
        gen_tokens = estimate_tokens(ai_message)
        self.thread_tokens[thread_id] += gen_tokens
        
        return {
            "response": ai_message,
            "agent_tokens": gen_tokens,
            "prompt_tokens_processed": prompt_tokens
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_tokens = estimate_tokens(self.profile_store.read_text(user_id))
        cm_ctx = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(cm_ctx["summary"]))
        msg_tokens = estimate_tokens(" ".join([m["content"] for m in cm_ctx["messages"]]))
        return profile_tokens + summary_tokens + msg_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        profile = self.profile_store.read_text(user_id).lower()
        msg = message.lower()
        if "tên gì" in msg or "tên là gì" in msg or "tên tôi là gì" in msg:
            match = re.search(r'- name:\s+(.*)', profile, re.IGNORECASE)
            if match:
                return f"Bạn tên là {match.group(1)}."
            return "Tôi chưa biết tên bạn."
        if "sống ở đâu" in msg or "ở đâu" in msg:
            match = re.search(r'- location:\s+(.*)', profile, re.IGNORECASE)
            if match:
                return f"Bạn sống ở {match.group(1)}."
            return "Tôi chưa biết bạn sống ở đâu."
        if "thích" in msg:
            match = re.search(r'- likes:\s+(.*)', profile, re.IGNORECASE)
            if match:
                return f"Bạn thích {match.group(1)}."
            
        return f"Offline reply to: {message[:20]}..."

    def _maybe_build_langchain_agent(self):
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
