from amazon_er.candidates.schema import CANDIDATE_SCHEMA_VERSION, validate_candidates


def candidate(**updates):
    row = {"target_entity_id": "t1", "candidate_s1_entity_id": "s1", "target_source": "S2", "country": "US"}
    row.update(updates)
    return row


def test_version_and_valid_minimum():
    assert CANDIDATE_SCHEMA_VERSION == "candidate_v1"
    assert validate_candidates([candidate()]).valid


def test_missing_required_field():
    row = candidate()
    del row["country"]
    result = validate_candidates([row])
    assert not result.valid
    assert "missing required" in result.errors[0]


def test_duplicate_detection():
    result = validate_candidates([candidate(), candidate()])
    assert not result.valid
    assert any("duplicate candidate pair" in error for error in result.errors)


def test_duplicate_country_inconsistency():
    result = validate_candidates([candidate(), candidate(country="FR")])
    assert not result.valid
    assert any("inconsistent country" in error for error in result.errors)


def test_missing_retriever_semantics():
    assert validate_candidates([candidate(dense_present=False, dense_score=None, dense_rank=None)]).valid
    assert not validate_candidates([candidate(dense_present=False, dense_score=0.0)]).valid
    assert validate_candidates([candidate(dense_present=True, dense_score=0.7, dense_rank=1)]).valid
