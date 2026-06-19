from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from tabulate import tabulate


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for e in expected if e.lower() in ans_lower)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    score = 0.5
    if recall_points(answer, expected) > 0.5:
        score += 0.5
    return min(1.0, score)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    total_recall = 0.0
    total_quality = 0.0
    recall_count = 0
    
    agent_tokens_only = 0
    prompt_tokens_processed = 0
    compactions = 0
    memory_growth = 0
    
    for conv in conversations:
        user_id = conv.get("user_id", "test_user")
        
        # Thread 1: context building
        thread_id = conv.get("thread_id", "thread_1")
        for turn in conv.get("turns", []):
            message = turn["message"] if isinstance(turn, dict) else turn
            agent.reply(user_id, thread_id, message)
            
        agent_tokens_only += agent.token_usage(thread_id)
        prompt_tokens_processed += agent.prompt_token_usage(thread_id)
        
        if hasattr(agent, "compaction_count"):
            compactions += agent.compaction_count(thread_id)
            
        # Thread 2: recall questions
        recall_questions = conv.get("recall_questions", [])
        if recall_questions:
            recall_thread = f"{thread_id}_recall"
            for rq in recall_questions:
                ans = agent.reply(user_id, recall_thread, rq["question"])
                expected = rq.get("expected", [])
                
                points = recall_points(ans["response"], expected)
                total_recall += points
                total_quality += heuristic_quality(ans["response"], expected)
                recall_count += 1
                
            agent_tokens_only += agent.token_usage(recall_thread)
            prompt_tokens_processed += agent.prompt_token_usage(recall_thread)
            
        if hasattr(agent, "memory_file_size"):
            memory_growth += agent.memory_file_size(user_id)
            
    avg_recall = total_recall / max(1, recall_count)
    avg_quality = total_quality / max(1, recall_count)
    
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens_only,
        prompt_tokens_processed=prompt_tokens_processed,
        recall_score=round(avg_recall, 2),
        response_quality=round(avg_quality, 2),
        memory_growth_bytes=memory_growth,
        compactions=compactions
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent", 
        "Agent tokens only", 
        "Prompt tokens processed", 
        "Cross-session recall", 
        "Response quality", 
        "Memory growth (bytes)", 
        "Compactions"
    ]
    
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            r.recall_score,
            r.response_quality,
            r.memory_growth_bytes,
            r.compactions
        ])
        
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    config = load_config()
    
    # Check if files exist or create dummy data
    std_data_path = config.data_dir / "conversations.json"
    long_data_path = config.data_dir / "advanced_long_context.json"
    
    # If no data exists, we create dummy for the sake of demo
    if not std_data_path.exists():
        config.data_dir.mkdir(parents=True, exist_ok=True)
        dummy_std = [{
            "user_id": "u1",
            "thread_id": "t1",
            "turns": ["tên tôi là Dũng"],
            "recall_questions": [{"question": "tên tôi là gì?", "expected": ["Dũng"]}]
        }]
        with open(std_data_path, "w", encoding="utf-8") as f:
            json.dump(dummy_std, f)
            
        dummy_long = [{
            "user_id": "u2",
            "thread_id": "t2",
            "turns": ["sống ở HN"] + [f"spam {i}" for i in range(20)],
            "recall_questions": [{"question": "tôi sống ở đâu?", "expected": ["HN"]}]
        }]
        with open(long_data_path, "w", encoding="utf-8") as f:
            json.dump(dummy_long, f)

    print("Running benchmarks with REAL LIVE LLM API via proxy...\n")
    
    # Standard
    std_convs = load_conversations(std_data_path)
    base_std = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=False), std_convs, config)
    adv_std = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=False), std_convs, config)
    
    print("### Standard Benchmark")
    print(format_rows([base_std, adv_std]))
    print("\n")
    
    # Long Context
    long_convs = load_conversations(long_data_path)
    base_long = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=False), long_convs, config)
    adv_long = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=False), long_convs, config)
    
    print("### Long Context Stress Benchmark")
    print(format_rows([base_long, adv_long]))


if __name__ == "__main__":
    main()
