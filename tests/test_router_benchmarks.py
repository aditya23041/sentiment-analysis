import statistics
import time

from rich.console import Console
from rich.table import Table

console = Console()

# We need to measure initialization time separately
start_init = time.perf_counter()
from sentiment_analysis.core.router import semantic_router

init_time = (time.perf_counter() - start_init) * 1000

# Benchmark Queries
QUERIES = [
    # Exact Matches (Should hit non_emotional_factual or greeting)
    ("what time is it", True),
    ("hello", True),
    ("who is the president", True),

    # Semantic/Fuzzy Matches (Should hit factual despite phrasing)
    ("could you tell me the current time", True),
    ("can you calculate 5 + 10 for me", True),
    ("hey there", True),

    # Emotional/Sarcastic (Should MISS and pass through to ML model)
    ("I absolutely hate my life right now.", False),
    ("Wow, thanks for nothing.", False),
    ("I am so incredibly happy to see you!", False),
]

def run_router_benchmarks():
    console.print("[bold cyan]SynaptoRoute Integration Benchmarks[/bold cyan]")
    console.print(f"Initialization Time (Model + SQLite Load): {init_time:.2f}ms\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Query", width=40)
    table.add_column("Expected Route", justify="center")
    table.add_column("Actual Route", justify="center")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Status", justify="center")

    latencies = []

    for query, expected_hit in QUERIES:
        start_time = time.perf_counter()
        route_result = semantic_router.triage_intent(query)
        latency = (time.perf_counter() - start_time) * 1000
        latencies.append(latency)

        is_hit = route_result is not None
        status = "Pass" if is_hit == expected_hit else "Fail"
        status_str = f"[green]{status}[/green]" if status == "Pass" else f"[red]{status}[/red]"

        table.add_row(
            query,
            "Triage" if expected_hit else "Pass-Through",
            route_result if route_result else "None",
            f"{latency:.2f}",
            status_str
        )

    console.print(table)

    if latencies:
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else max(latencies)

        console.print("\n[bold green]Latency Metrics:[/bold green]")
        console.print(f"Average Latency: {avg_latency:.2f}ms")
        console.print(f"p95 Latency:     {p95_latency:.2f}ms")
        console.print(f"Min Latency:     {min(latencies):.2f}ms")
        console.print(f"Max Latency:     {max(latencies):.2f}ms")

if __name__ == "__main__":
    run_router_benchmarks()
