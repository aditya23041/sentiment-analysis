import time

import requests
from rich.console import Console

console = Console()

# 1. Our System: http://127.0.0.1:8000/api/analyze?model=unified
# 2. Agentic-Semantic-Router: http://127.0.0.1:8001/analyze
OUR_URL = "http://127.0.0.1:8000/api/analyze?model=unified"
THEIR_URL = "http://127.0.0.1:8001/analyze"

TEST_CASES = [
    "I absolutely love it when my computer crashes right before I save my work.",
    "Brilliant deduction, Sherlock. We never would have figured that out.",
    "I'm so glad I spent $50 on a movie where the main character does absolutely nothing.",
    "Fantastic weather we're having, if you enjoy freezing rain and hurricane winds.",
    "Thank you for explaining that concept so clearly that I am now completely confused.",
    "I am so incredibly proud of my daughter for graduating today!",
    "I can't stop crying, my heart is completely broken after losing my dog.",
    "This is the most terrifying horror movie I've ever seen, I loved it!",
    "I just feel so peaceful watching the sunset over the ocean.",
    "I'm laughing so hard but I also want to cry at how relatable this is.",
    "I hate you so much but I can't stop loving you.",
    "Well, that was a spectacular failure.",
]

def analyze_system(system_name, url, payload_key="text", parse_func=None):
    results = []
    latencies = []

    for text in TEST_CASES:
        start_time = time.perf_counter()
        try:
            payload = {payload_key: text, "session_id": "benchmark_session", "user_id": "benchmark_user"}
            res = requests.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
            latency = (time.perf_counter() - start_time) * 1000
            latencies.append(latency)

            parsed_result = parse_func(data) if parse_func else str(data)
            results.append((latency, parsed_result))
        except Exception as e:
            results.append((0, f"ERROR: {e!s}"))
            latencies.append(0)

    return results, latencies

def parse_our_system(data):
    sarcasm = "Yes" if data.get("is_sarcastic") else "No"
    polarity = data.get("polarity", 0.0)
    return f"Pol: {polarity:+.2f} | Sarcasm: {sarcasm}"

def parse_their_system(data):
    # Need to verify their JSON schema
    sentiment = data.get("sentiment", "N/A")
    emotions = data.get("emotions", [])
    sarcasm = "Yes" if data.get("is_sarcastic") else "No"
    return f"Sent: {sentiment} | Emo: {emotions} | Sarcasm: {sarcasm}"

if __name__ == "__main__":
    console.print("[bold cyan]Running Accuracy Comparison Benchmark...[/bold cyan]")

    our_results, our_lats = analyze_system("Our System", OUR_URL, "text", parse_our_system)
    their_results, their_lats = analyze_system("Agentic-Semantic-Router", THEIR_URL, "text", parse_their_system)

    table_markdown = "# Accuracy & Nuance Comparison\n\n"
    table_markdown += "| Query | Our System (Unified) | Their System (LangGraph) | Latency Diff |\n"
    table_markdown += "|---|---|---|---|\n"

    for i, text in enumerate(TEST_CASES):
        our_lat, our_res = our_results[i]
        their_lat, their_res = their_results[i]

        diff = our_lat - their_lat
        diff_str = f"**{abs(diff):.1f}ms faster**" if diff < 0 else f"**{abs(diff):.1f}ms slower**"

        table_markdown += f"| {text[:37]}... | {our_lat:.1f}ms<br>{our_res} | {their_lat:.1f}ms<br>{their_res} | {diff_str} |\n"

    avg_our = sum(our_lats)/len(our_lats)
    avg_their = sum(their_lats)/len(their_lats)

    table_markdown += f"\n**Average Latency Our System:** {avg_our:.2f}ms\n"
    table_markdown += f"**Average Latency Their System:** {avg_their:.2f}ms\n"

    with open("benchmarks/comparison_report.md", "w", encoding="utf-8") as f:
        f.write(table_markdown)

    console.print(table_markdown)
