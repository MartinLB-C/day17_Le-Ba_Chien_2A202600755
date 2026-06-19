from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, CompactMemoryManager


def make_config(tmp_path: Path):
    config = load_config()
    config.state_dir = tmp_path / "state"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.compact_threshold_tokens = 50  # very low for testing
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    store.root_dir.mkdir(exist_ok=True)
    
    # Write
    store.write_text("user_1", "Profile A\n- Name: John")
    assert store.file_size("user_1") > 0
    
    # Read
    content = store.read_text("user_1")
    assert "John" in content
    
    # Edit
    changed = store.edit_text("user_1", "John", "Doe")
    assert changed is True
    content2 = store.read_text("user_1")
    assert "Doe" in content2


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    cm = CompactMemoryManager(
        threshold_tokens=config.compact_threshold_tokens,
        keep_messages=config.compact_keep_messages
    )
    
    # Add messages to trigger compaction
    for i in range(10):
        cm.append("thread_1", "user", f"this is a long message {i} " * 5)
        cm.append("thread_1", "assistant", f"reply {i} " * 5)
        
    assert cm.compaction_count("thread_1") > 0
    ctx = cm.context("thread_1")
    assert len(ctx["messages"]) == config.compact_keep_messages
    assert len(ctx["summary"]) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    
    baseline = BaselineAgent(config, force_offline=True)
    baseline.reply("user_1", "thread_1", "tên tôi là testuser.")
    ans_base = baseline.reply("user_1", "thread_2", "tên tôi là gì?")
    assert "testuser" not in ans_base["response"].lower()
    
    advanced = AdvancedAgent(config, force_offline=True)
    advanced.reply("user_1", "thread_1", "tên tôi là testuser.")
    ans_adv = advanced.reply("user_1", "thread_2", "tên tôi là gì?")
    assert "testuser" in ans_adv["response"].lower()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    
    for i in range(20):
        baseline.reply("user_1", "thread_long", f"dummy message {i} " * 5)
        advanced.reply("user_1", "thread_long", f"dummy message {i} " * 5)
        
    base_prompt = baseline.prompt_token_usage("thread_long")
    adv_prompt = advanced.prompt_token_usage("thread_long")
    
    assert adv_prompt < base_prompt
