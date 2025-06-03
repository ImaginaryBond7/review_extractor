from cli.extractor import extract_attributes_batch

def test_positive_attr():
    reviews = [{"review_id": "1", "body": "I love the fragrance and long-lasting freshness."}]
    out = extract_attributes_batch(reviews)[0]["delight_attributes"]
    assert any("fragrance" in a.lower() for a in out)

def test_negative_attr():
    reviews = [{"review_id": "2", "body": "Awful smell and stains my shirt."}]
    out = extract_attributes_batch(reviews)[0]["delight_attributes"]
    assert out == []