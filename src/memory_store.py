from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.strip()) // 4)


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', user_id)
        return self.root_dir / f"{clean_id}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# Profile for {user_id}\n\nNo facts recorded yet."

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0


def extract_profile_updates(message: str) -> dict[str, str]:
    message_lower = message.lower()
    facts = {}
    if "tên tôi là" in message_lower or "tôi tên là" in message_lower:
        match = re.search(r'tên (tôi )?là (.*?)([\.,]|$)', message_lower)
        if match:
            facts["Name"] = match.group(2).strip().title()
    if "tôi thích" in message_lower:
        match = re.search(r'tôi thích (.*?)([\.,]|$)', message_lower)
        if match:
            facts["Likes"] = match.group(1).strip()
    if "tôi sống ở" in message_lower or "sống tại" in message_lower:
        match = re.search(r'(sống ở|sống tại) (.*?)([\.,]|$)', message_lower)
        if match:
            facts["Location"] = match.group(2).strip().title()
    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    summary = []
    for m in messages[:max_items]:
        role = m.get("role", "user")
        content = m.get("content", "")
        summary.append(f"{role.capitalize()}: {content[:30]}...")
    return "\n".join(summary)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0
            }
        
        thread_state = self.state[thread_id]
        thread_state["messages"].append({"role": role, "content": content})
        
        total_text = str(thread_state["summary"]) + " " + " ".join([m["content"] for m in thread_state["messages"]])
        current_tokens = estimate_tokens(total_text)
        
        if current_tokens > self.threshold_tokens and len(thread_state["messages"]) > self.keep_messages:
            to_summarize = thread_state["messages"][:-self.keep_messages]
            kept_messages = thread_state["messages"][-self.keep_messages:]
            
            new_summary_text = summarize_messages(to_summarize)
            if thread_state["summary"]:
                combined = str(thread_state["summary"]) + "\n" + new_summary_text
                if len(combined) > 200:
                    combined = "..." + combined[-197:]
                thread_state["summary"] = combined
            else:
                thread_state["summary"] = new_summary_text
                
            thread_state["messages"] = kept_messages
            thread_state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return self.state.get(thread_id, {}).get("compactions", 0)
