import json
import tempfile
from cli.evaluator import evaluate_from_csv
import pandas as pd

def test_eval_scores_tp():
    # create mini csv inline
    df = pd.DataFrame([
        {"review_id": "1", "body": "Great fragrance", "delight_attribute": "Fragrance"},
    ])
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        report = evaluate_from_csv(tmp.name, similarity_threshold=0.6, embedder="local")
        assert report["correct_count"] >= 1