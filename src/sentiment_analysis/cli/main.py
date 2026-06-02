"""Click-based CLI for sentiment analysis."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to the Python path to allow running main.py directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import click
from dotenv import load_dotenv

# Load environment variables from .env file for CLI usage
load_dotenv()
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sentiment_analysis import __version__
from sentiment_analysis.core.analyzer import SentimentAnalyzer
from sentiment_analysis.models import list_models

console = Console()

SENTIMENT_EMOJI = {
    "VERY_POSITIVE": "🤩",
    "POSITIVE": "😊",
    "NEUTRAL": "😐",
    "NEGATIVE": "😞",
    "VERY_NEGATIVE": "😠",
}

SENTIMENT_COLOR = {
    "VERY_POSITIVE": "bold green",
    "POSITIVE": "green",
    "NEUTRAL": "yellow",
    "NEGATIVE": "dark_orange",
    "VERY_NEGATIVE": "bold red",
}


@click.group()
@click.version_option(version=__version__, prog_name="sentiment")
def cli() -> None:
    """Sentiment Analysis CLI — Multi-model NLP platform."""
    pass


@cli.command()
@click.argument("text")
@click.option("-m", "--model", default="unified", help="Model to use (unified, vader, textblob, transformer)")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def analyze(text: str, model: str, json_output: bool) -> None:
    """Analyze sentiment of a single text."""
    try:
        analyzer = SentimentAnalyzer(model=model)
        result = analyzer.analyze(text)

        if json_output:
            click.echo(result.model_dump_json(indent=2))
            return

        emoji = SENTIMENT_EMOJI.get(result.sentiment.value, "❓")
        color = SENTIMENT_COLOR.get(result.sentiment.value, "white")

        panel_content = Text()
        panel_content.append(f"\n  {emoji} Sentiment: ", style="bold")
        panel_content.append(f"{result.sentiment.value.replace('_', ' ')}\n", style=color)
        panel_content.append(f"  📊 Polarity:     {result.polarity:+.4f}\n")
        panel_content.append(f"  📐 Subjectivity: {result.subjectivity:.4f}\n")
        panel_content.append(f"  🎯 Confidence:   {result.confidence:.0%}\n")

        if result.is_sarcastic:
            panel_content.append(f"  🤡 [bold magenta]Sarcasm Detected![/bold magenta] ({result.sarcasm_probability:.0%})\n")

        if result.metadata.get("emotions"):
            emotions_str = ", ".join([f"{k} ({v:.0%})" for k, v in result.metadata["emotions"].items()])
            panel_content.append(f"  🎭 Emotions:     {emotions_str}\n")

        panel_content.append(f"  🤖 Model:        {result.model_used}\n")

        console.print(Panel(
            panel_content,
            title="[bold]Sentiment Analysis[/bold]",
            subtitle=f"[dim]{model}[/dim]",
            border_style="blue",
        ))

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "-m", "--model", default="unified",
    help="Model to use (unified, vader, textblob, transformer, llm)",
)
def interactive(model: str) -> None:
    """Start an interactive session to analyze texts line by line."""
    console.print(Panel(
        f"[bold]Interactive Sentiment Analysis[/bold]\n"
        f"Model: [dim]{model}[/dim]\n"
        f"Type your sentence and press Enter. Type [bold red]quit[/bold red] to exit.",
        border_style="green"
    ))
    try:
        analyzer = SentimentAnalyzer(model=model)
        while True:
            text = click.prompt(click.style("Enter text", fg="cyan"), prompt_suffix="> ")
            if text.strip().lower() in ("quit", "exit", "q"):
                break
            if not text.strip():
                continue

            result = analyzer.analyze(text)
            emoji = SENTIMENT_EMOJI.get(result.sentiment.value, "❓")
            color = SENTIMENT_COLOR.get(result.sentiment.value, "white")

            label = result.sentiment.value.replace('_', ' ')
            console.print(
                f"  Result: {emoji} [{color}]{label}[/{color}] "
                f"(Polarity: {result.polarity:+.4f}, "
                f"Confidence: {result.confidence:.0%})\n"
            )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Exiting interactive mode.[/dim]")



@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("-c", "--column", default="text", help="Text column name in CSV")
@click.option("-m", "--model", default="vader", help="Model to use")
@click.option("-o", "--output", type=click.Path(), help="Output CSV path")
def batch(file: str, column: str, model: str, output: str | None) -> None:
    """Analyze sentiment from a CSV file."""
    try:
        analyzer = SentimentAnalyzer(model=model)

        with console.status("[bold blue]Analyzing...[/bold blue]"):
            df = analyzer.analyze_csv(file, text_column=column)

        # Summary table
        table = Table(title=f"Batch Results ({len(df)} rows)", border_style="blue")
        table.add_column("#", style="dim", width=4)
        table.add_column("Text", max_width=50, no_wrap=True)
        table.add_column("Sentiment", justify="center")
        table.add_column("Polarity", justify="right")
        table.add_column("Confidence", justify="right")

        for i, (_, row) in enumerate(df.head(25).iterrows(), 1):
            text_val = str(row[column])[:50]
            sent = row["sentiment"]
            emoji = SENTIMENT_EMOJI.get(sent, "❓")
            color = SENTIMENT_COLOR.get(sent, "white")
            table.add_row(
                str(i),
                text_val,
                Text(f"{emoji} {sent}", style=color),
                f"{row['polarity']:+.4f}",
                f"{row['confidence']:.0%}",
            )

        if len(df) > 25:
            table.add_row("...", f"[dim]({len(df) - 25} more rows)[/dim]", "", "", "")

        console.print(table)

        # Distribution summary
        dist = df["sentiment"].value_counts()
        console.print("\n[bold]Distribution:[/bold]")
        for sent, count in dist.items():
            emoji = SENTIMENT_EMOJI.get(sent, "❓")
            pct = count / len(df) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            console.print(f"  {emoji} {sent:<15} {bar} {count} ({pct:.1f}%)")

        if output:
            df.to_csv(output, index=False)
            console.print(f"\n[green]✓[/green] Results saved to {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("text")
def compare(text: str) -> None:
    """Compare all models on the same text."""
    try:
        analyzer = SentimentAnalyzer()
        result = analyzer.compare_models(text)

        table = Table(title="Model Comparison", border_style="blue")
        table.add_column("Model", style="bold")
        table.add_column("Sentiment", justify="center")
        table.add_column("Polarity", justify="right")
        table.add_column("Confidence", justify="right")

        for model_name, r in result.results.items():
            emoji = SENTIMENT_EMOJI.get(r.sentiment.value, "❓")
            color = SENTIMENT_COLOR.get(r.sentiment.value, "white")
            table.add_row(
                model_name.upper(),
                Text(f"{emoji} {r.sentiment.value}", style=color),
                f"{r.polarity:+.4f}",
                f"{r.confidence:.0%}",
            )

        console.print(table)

        if result.consensus:
            emoji = SENTIMENT_EMOJI.get(result.consensus.value, "❓")
            label = result.consensus.value.replace('_', ' ')
            console.print(f"\n[bold]Consensus:[/bold] {emoji} {label}")
        else:
            console.print("\n[yellow]No consensus[/yellow] — models disagree")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.option("-h", "--host", default="127.0.0.1", help="Host to bind to")
@click.option("-p", "--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the web dashboard and API server."""
    console.print(Panel(
        f"[bold]🚀 Starting Sentiment Analysis Platform[/bold]\n\n"
        f"  Dashboard: [link]http://{host}:{port}[/link]\n"
        f"  API Docs:  [link]http://{host}:{port}/docs[/link]\n"
        f"  Models:    {', '.join(list_models())}",
        border_style="green",
    ))

    import uvicorn
    uvicorn.run(
        "sentiment_analysis.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command(name="models")
def list_available_models() -> None:
    """List available sentiment models."""
    models = list_models()
    table = Table(title="Available Models", border_style="blue")
    table.add_column("Name", style="bold")
    table.add_column("Status")

    for m in ["vader", "textblob", "transformer"]:
        if m in models:
            table.add_row(m.upper(), "[green]✓ Available[/green]")
        else:
            table.add_row(m.upper(), "[dim]✗ Not installed[/dim]")

    console.print(table)


if __name__ == "__main__":
    cli()
