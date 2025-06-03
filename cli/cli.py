import typer
import logging
from pathlib import Path
from .utils import load_json, save_json, save_csv
from .extractor import extract_attributes_batch
from .clustering import cluster_attributes, get_cluster_representatives
from .evaluator import evaluate_from_csv
from collections import Counter

typer_app = typer.Typer()


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


@typer_app.command(name="extract")

def extract(
    input: Path = typer.Option(..., help="Path to input reviews.json"),
    output: Path = typer.Option(..., help="Path to save JSON with extracted attributes"),
    csv_output: Path = typer.Option(..., help="Path to save ranked attributes CSV"),
    eps: float = typer.Option(0.5, help="DBSCAN eps parameter"),
    min_samples: int = typer.Option(1, help="DBSCAN min_samples parameter"),
):
    """
    Extract delight attributes and output JSON + ranked CSV.
    """
    _setup_logging()
    data = load_json(str(input))
    reviews = data.get('reviews', [])
    if not reviews:
        logging.error("No reviews found in input JSON.")
        raise typer.Exit(code=1)

    logging.info("Starting attribute extraction on %d reviews...", len(reviews))
    processed = extract_attributes_batch(reviews)

    save_json({'reviews': processed}, str(output))
    logging.info(f"Saved extracted reviews to {output}")

    all_attrs = []
    for rev in processed:
        all_attrs.extend(rev.get('delight_attributes', []))
    unique_attrs = list(set(all_attrs))

    logging.info("Clustering %d unique attributes...", len(unique_attrs))
    attr_to_cluster = cluster_attributes(unique_attrs, eps=eps, min_samples=min_samples)
    reps = get_cluster_representatives(attr_to_cluster, all_attrs)

    cluster_counts = Counter()
    for rev in processed:
        new_attrs = []
        for attr in rev.get('delight_attributes', []):
            cid = attr_to_cluster.get(attr, -1)
            rep = reps.get(cid, attr)
            new_attrs.append(rep)
            cluster_counts[rep] += 1
        rev['delight_attributes'] = list(set(new_attrs))

    save_json({'reviews': processed}, str(output))
    logging.info(f"Saved clustered reviews to {output}")

    rows = [(attr, count) for attr, count in cluster_counts.most_common()]
    save_csv(rows, ["Delight Attribute", "Frequency"], str(csv_output))
    logging.info(f"Saved ranked attributes to {csv_output}")


@typer_app.command(name="evaluate")

def evaluate(
    csv: Path = typer.Option(..., help="Path to delight-evaluation.csv"),
    report: Path = typer.Option(..., help="Where to write evaluation_report.json"),
    threshold: float = typer.Option(0.80, help="Cosine similarity threshold"),
):
    """Run extractor on CSV’s review bodies and score against expected attributes."""
    _setup_logging()
    result = evaluate_from_csv(str(csv), similarity_threshold=threshold)
    save_json(result, str(report))
    logging.info("Saved evaluation report to %s", report)


def main():
    typer_app()


if __name__ == "__main__":
    main()