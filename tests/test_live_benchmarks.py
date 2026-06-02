import time

from rich.console import Console
from rich.table import Table

from sentiment_analysis.core.analyzer import SentimentAnalyzer

console = Console()

# 20 Benchmark Tests designed to test deep sarcasm, mixed emotions, and paradoxical statements
BENCHMARKS = [
    # Heavy Sarcasm
    {"text": "I absolutely love it when my computer crashes right before I save my work.", "expected_sarcasm": True, "expected_emotion": "anger"},
    {"text": "Oh great, another flat tire. Just what I needed today!", "expected_sarcasm": True, "expected_emotion": "annoyance"},
    {"text": "Wow, thanks for the incredibly helpful advice that I already tried an hour ago.", "expected_sarcasm": True, "expected_emotion": "annoyance"},
    {"text": "The service at that restaurant was so fast, we only had to wait two hours for a glass of water.", "expected_sarcasm": True, "expected_emotion": "disappointment"},
    {"text": "I'm so glad I spent $50 on a movie where the main character does absolutely nothing.", "expected_sarcasm": True, "expected_emotion": "disappointment"},
    {"text": "Fantastic weather we're having, if you enjoy freezing rain and hurricane winds.", "expected_sarcasm": True, "expected_emotion": "annoyance"},
    {"text": "Thank you for explaining that concept so clearly that I am now completely confused.", "expected_sarcasm": True, "expected_emotion": "confusion"},

    # Pure Emotion (GoEmotions targets)
    {"text": "I am so incredibly proud of my daughter for graduating today!", "expected_sarcasm": False, "expected_emotion": "pride"},
    {"text": "I can't stop crying, my heart is completely broken after losing my dog.", "expected_sarcasm": False, "expected_emotion": "grief"},
    {"text": "This is the most terrifying horror movie I've ever seen, I'm shaking.", "expected_sarcasm": False, "expected_emotion": "fear"},
    {"text": "I'm really anxious about my upcoming interview, I hope I do well.", "expected_sarcasm": False, "expected_emotion": "nervousness"},
    {"text": "Thank you so much for the beautiful gift, you made my entire week!", "expected_sarcasm": False, "expected_emotion": "gratitude"},
    {"text": "I just feel so peaceful watching the sunset over the ocean.", "expected_sarcasm": False, "expected_emotion": "relief"},
    {"text": "I'm genuinely curious how quantum mechanics actually works.", "expected_sarcasm": False, "expected_emotion": "curiosity"},

    # Paradoxical / Mixed State
    {"text": "I'm laughing so hard but I also want to cry at how relatable this is.", "expected_sarcasm": False, "expected_emotion": "amusement"},
    {"text": "I hate you so much but I can't stop loving you.", "expected_sarcasm": False, "expected_emotion": "love"},
    {"text": "This food is disgustingly good, I can't stop eating it.", "expected_sarcasm": False, "expected_emotion": "joy"},

    # Subtle / Dry
    {"text": "Well, that was a spectacular failure.", "expected_sarcasm": True, "expected_emotion": "disappointment"},
    {"text": "I guess I'll just sit here and wait forever.", "expected_sarcasm": False, "expected_emotion": "annoyance"},
    {"text": "Brilliant deduction, Sherlock. We never would have figured that out.", "expected_sarcasm": True, "expected_emotion": "annoyance"},
]

def run_benchmarks(model_name: str = "vader"):
    console.print(f"[bold cyan]Running 20-Test Live Benchmark Suite using model: [yellow]{model_name}[/yellow][/bold cyan]")

    analyzer = SentimentAnalyzer(model=model_name)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Latency (ms)", style="dim", width=12)
    table.add_column("Text", width=50)
    table.add_column("Polarity", justify="right")
    table.add_column("Sarcastic?", justify="center")
    table.add_column("Top Emotion", justify="center")

    start_total = time.perf_counter()

    results = []
    for test in BENCHMARKS:
        text = test["text"]

        start_time = time.perf_counter()
        # Our current pipeline only returns polarity/sentiment, not full goemotions/sarcasm yet
        result = analyzer.analyze(text)
        latency = (time.perf_counter() - start_time) * 1000

        # Extract emotion from metadata if available
        top_emotion = "None"
        if result.metadata.get("emotions"):
            top_emotion = list(result.metadata["emotions"].keys())[0]

        is_sarcastic_str = "[bold magenta]Yes[/]" if result.is_sarcastic else "No"

        table.add_row(
            f"{latency:.2f}ms",
            text[:47] + "..." if len(text) > 50 else text,
            f"{result.polarity:+.2f}",
            is_sarcastic_str,
            top_emotion
        )

        results.append(latency)

    total_time = (time.perf_counter() - start_total) * 1000
    avg_time = sum(results) / len(results)

    console.print(table)
    console.print("\n[bold green]Baseline Performance Metrics:[/bold green]")
    console.print(f"Total Suite Time: {total_time:.2f}ms")
    console.print(f"Average Latency per Query: {avg_time:.2f}ms")

    return {
        "total_latency_ms": total_time,
        "avg_latency_ms": avg_time,
    }

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "unified"
    run_benchmarks(model)
