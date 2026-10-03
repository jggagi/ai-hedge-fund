import json
import hashlib
import io
import gzip
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.backend.services import public_research as research


def _request(tickers=None, provider="ollama"):
    return SimpleNamespace(
        tickers=["AAPL"] if tickers is None else tickers,
        end_date="2024-12-31",
        data_source="sec_filings",
        model_name="qwen3.5:4b",
        model_provider=provider,
        get_agent_model_config=lambda _agent: ("qwen3.5:4b", provider),
    )


def _facts():
    old_accn = "0000320193-23-000010"
    new_accn = "0000320193-24-000010"

    def fact(label, records):
        return {"label": label, "units": {"USD": records}}

    return {
        "facts": {"us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": fact("Revenue", [
                {"form": "10-K", "filed": "2023-11-03", "start": "2023-07-01", "end": "2023-09-30", "accn": old_accn, "fp": "FY", "val": 25},
                {"form": "10-K", "filed": "2023-11-03", "start": "2022-09-25", "end": "2023-09-30", "accn": old_accn, "fp": "FY", "val": 100},
                {"form": "10-K", "filed": "2025-01-10", "end": "2024-09-28", "accn": new_accn, "fp": "FY", "val": 200},
            ]),
            "NetIncomeLoss": fact("Net income", [
                {"form": "10-K", "filed": "2023-11-03", "start": "2022-09-25", "end": "2023-09-30", "accn": old_accn, "fp": "FY", "val": 20},
            ]),
        }},
    }


def test_as_of_uses_last_eligible_10k_and_marks_missing_facts():
    observations, gaps = research._observations(_facts(), research.date(2024, 12, 31), "https://data.sec.gov/facts.json")

    assert [item["metric"] for item in observations] == ["revenue", "net_income"]
    assert all(item["filed"] <= "2024-12-31" for item in observations)
    assert {item["accn"] for item in observations} == {"0000320193-23-000010"}
    assert observations[0]["value"] == 100
    assert observations[0]["source_index"]["json_path"].endswith("USD[1]")
    assert len(observations[0]["source_index"]["sha256"]) == 64
    reconstructed_hash = hashlib.sha256(
        json.dumps(observations[0]["source_record"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert reconstructed_hash == observations[0]["source_index"]["sha256"]
    assert {item["metric"] for item in gaps} >= {"cash_and_equivalents", "total_assets", "long_term_debt", "current_debt"}
    assert observations[0]["source_url"].startswith("https://data.sec.gov/")


def test_invalid_model_citations_are_excluded_and_reported():
    result, error = research._validated_interpretation(
        "fundamentals_analyst",
        research.AnalystOutput(reasoning="Revenue appears resilient.", cited_fact_ids=["invented:fact"]),
        {"real:fact"},
    )

    assert result is None
    assert "citation" in error


def test_exact_supplied_valuation_gap_is_allowed_but_unknown_gap_is_rejected():
    source_gap = "SEC company facts contains no market price or valuation series."
    supplied, supplied_error = research._validated_interpretation(
        "risk_analyst",
        research.AnalystOutput(
            reasoning="Market coverage is incomplete for a full assessment.",
            cited_fact_ids=["fact-revenue"],
            gaps=[source_gap],
        ),
        {"fact-revenue"},
        known_gap_reasons={source_gap},
    )
    unknown, unknown_error = research._validated_interpretation(
        "risk_analyst",
        research.AnalystOutput(
            reasoning="The filing supports caution.",
            cited_fact_ids=["fact-revenue"],
            gaps=["Estimated price target is 140."],
        ),
        {"fact-revenue"},
        known_gap_reasons={source_gap},
    )

    assert supplied is not None and supplied["gaps"] == [source_gap] and supplied_error is None
    assert unknown is None and "token='140'" in unknown_error


@pytest.mark.parametrize(
    "output, message",
    [
        (research.AnalystOutput(reasoning="No citation supplied."), "did not cite"),
        (research.AnalystOutput(reasoning="" , cited_fact_ids=["real:fact"]), "did not provide"),
        (research.AnalystOutput(reasoning="Revenue rose by 12 percent.", cited_fact_ids=["real:fact"]), "numeric"),
        (research.AnalystOutput(reasoning="Buy the shares.", cited_fact_ids=["real:fact"]), "recommendation"),
        (research.AnalystOutput(reasoning="Cautious view.", cited_fact_ids=["real:fact"], gaps=["Revenue fell by 12 percent."]), "numeric"),
    ],
)
def test_empty_or_unsafe_interpretations_are_rejected(output, message):
    result, error = research._validated_interpretation("fundamentals_analyst", output, {"real:fact"})
    assert result is None
    assert message in error


def test_numeric_or_recommendation_claims_are_excluded():
    numeric, numeric_error = research._validated_interpretation(
        "risk_analyst",
        research.AnalystOutput(reasoning="Revenue increased by 20 percent.", cited_fact_ids=["real:fact"]),
        {"real:fact"},
    )
    recommendation, recommendation_error = research._validated_interpretation(
        "risk_analyst",
        research.AnalystOutput(reasoning="Investors should buy the company.", cited_fact_ids=["real:fact"]),
        {"real:fact"},
    )
    assert numeric is None and "numeric" in numeric_error
    assert "token='20'" in numeric_error
    assert recommendation is None and "recommendation" in recommendation_error


def test_10k_filing_label_is_allowed_but_year_claims_are_rejected_with_token():
    filing_label, filing_error = research._validated_interpretation(
        "risk_analyst",
        research.AnalystOutput(
            reasoning="The annual 10-K supports a cautious operating assessment.",
            cited_fact_ids=["fact-revenue"],
        ),
        {"fact-revenue"},
    )
    year_claim, year_error = research._validated_interpretation(
        "risk_analyst",
        research.AnalystOutput(
            reasoning="The 2024 filing supports a cautious operating assessment.",
            cited_fact_ids=["fact-revenue"],
        ),
        {"fact-revenue"},
    )

    assert filing_label is not None and filing_error is None
    assert year_claim is None and "token='2024'" in year_error


def test_prompt_requests_short_reasoning_and_cites_every_numeric_or_date_fact():
    prompt = research._prompt("fundamentals_analyst", [], [])
    assert "two or three concise sentences" in prompt
    assert "two or three relevant observations" in prompt
    assert "every numeric or date fact" in prompt


def test_cited_source_values_and_explicitly_scaled_rounding_are_allowed():
    observations = [
        {"fact_id": "fact-revenue", "value": 416_161_000_000, "unit": "USD"},
        {"fact_id": "fact-net_income", "value": 123_456_789, "unit": "USD"},
    ]
    exact, exact_error = research._validated_interpretation(
        "fundamentals_analyst",
        research.AnalystOutput(
            reasoning="Reported revenue was 416161000000 USD.",
            cited_fact_ids=["fact-revenue"],
        ),
        {"fact-revenue", "fact-net_income"},
        observations=observations,
    )
    scaled, scaled_error = research._validated_interpretation(
        "fundamentals_analyst",
        research.AnalystOutput(
            reasoning="Reported revenue was 416.16 billion.",
            cited_fact_ids=["fact-revenue"],
        ),
        {"fact-revenue", "fact-net_income"},
        observations=observations,
    )
    compact_scaled, compact_error = research._validated_interpretation(
        "fundamentals_analyst",
        research.AnalystOutput(
            reasoning="Reported revenue was 416.16B.",
            cited_fact_ids=["fact-revenue"],
        ),
        {"fact-revenue", "fact-net_income"},
        observations=observations,
    )

    assert exact is not None and exact_error is None
    assert scaled is not None and scaled_error is None
    assert compact_scaled is not None and compact_error is None


@pytest.mark.parametrize(
    "reasoning, cited_ids",
    [
        ("Reported revenue was 416 million.", ["fact-revenue"]),
        ("Reported revenue was 999B.", ["fact-revenue"]),
        ("Reported revenue was 999bn.", ["fact-revenue"]),
        ("Reported revenue was -416 billion.", ["fact-revenue"]),
        ("Reported revenue was 9.99e12 USD.", ["fact-revenue"]),
        ("Net income was 123456789 USD.", ["fact-revenue"]),
        ("Revenue margin was 20%.", ["fact-revenue"]),
    ],
)
def test_wrong_scale_uncited_value_and_derived_percentage_are_rejected(reasoning, cited_ids):
    observations = [
        {"fact_id": "fact-revenue", "value": 416_161_000_000, "unit": "USD"},
        {"fact_id": "fact-net_income", "value": 123_456_789, "unit": "USD"},
    ]
    result, error = research._validated_interpretation(
        "fundamentals_analyst",
        research.AnalystOutput(reasoning=reasoning, cited_fact_ids=cited_ids),
        {"fact-revenue", "fact-net_income"},
        observations=observations,
    )
    assert result is None
    assert "token=" in error and "context=" in error
    assert "cited_fact_ids=" in error


def test_valuation_caveat_is_allowed_but_recommendation_is_still_blocked():
    caveat, caveat_error = research._validated_interpretation(
        "risk_analyst",
        research.AnalystOutput(
            reasoning="Valuation cannot be assessed from these disclosures alone.",
            cited_fact_ids=["fact-revenue"],
        ),
        {"fact-revenue"},
        observations=[{"fact_id": "fact-revenue", "value": 416_161_000_000, "unit": "USD"}],
    )
    recommendation, recommendation_error = research._validated_interpretation(
        "risk_analyst",
        research.AnalystOutput(
            reasoning="Investors should buy the shares.",
            cited_fact_ids=["fact-revenue"],
        ),
        {"fact-revenue"},
        observations=[{"fact_id": "fact-revenue", "value": 416_161_000_000, "unit": "USD"}],
    )

    assert caveat is not None and caveat_error is None
    assert recommendation is None and "recommendation" in recommendation_error


def test_cited_full_date_and_fiscal_year_labels_are_allowed():
    observation = {
        "fact_id": "fact-revenue",
        "value": 416_161_000_000,
        "unit": "USD",
        "end": "2025-09-27",
        "filed": "2025-10-31",
        "source_record": {"start": "2024-09-28", "end": "2025-09-27", "filed": "2025-10-31"},
    }
    for wording in (
        "The annual year ended September 27, 2025 (FY2025) reported.",
        "The annual period ended 2025-09-27 (fiscal year 2025) reported.",
        "The year ended 2025-09-27.",
    ):
        result, error = research._validated_interpretation(
            "fundamentals_analyst",
            research.AnalystOutput(reasoning=wording, cited_fact_ids=["fact-revenue"]),
            {"fact-revenue"},
            observations=[observation],
        )
        assert result is not None and error is None


def test_wrong_source_date_and_future_fiscal_year_are_rejected():
    observation = {
        "fact_id": "fact-revenue",
        "value": 416_161_000_000,
        "unit": "USD",
        "end": "2025-09-27",
        "filed": "2025-10-31",
        "source_record": {"end": "2025-09-27", "filed": "2025-10-31"},
    }
    for wording, expected in (
        ("The period ended September 28, 2025.", "unsupported source date"),
        ("The year ended 2025-09-28.", "unsupported source date"),
        ("The annual report refers to FY2030.", "unsupported fiscal-year reference"),
    ):
        result, error = research._validated_interpretation(
            "fundamentals_analyst",
            research.AnalystOutput(reasoning=wording, cited_fact_ids=["fact-revenue"]),
            {"fact-revenue"},
            observations=[observation],
        )
        assert result is None and expected in error


def test_scaled_year_number_is_checked_as_a_value_not_a_date():
    observation = {"fact_id": "fact-revenue", "value": 416_161_000_000, "unit": "USD", "end": "2025-09-27"}
    result, error = research._validated_interpretation(
        "fundamentals_analyst",
        research.AnalystOutput(reasoning="Revenue was 2025 billion.", cited_fact_ids=["fact-revenue"]),
        {"fact-revenue"},
        observations=[observation],
    )
    assert result is None and "token='2025'" in error and "unsupported numeric" in error


def test_graph_fetches_once_calls_two_local_analysts_and_returns_no_prices():
    payload = json.dumps(_facts()).encode()

    @contextmanager
    def fake_response(*_args, **_kwargs):
        yield SimpleNamespace(read=lambda: payload, status=200)

    llm_result = research.AnalystOutput(
        reasoning="The filing supports a cautious operating assessment.",
        cited_fact_ids=["fact-revenue"],
    )
    with patch.object(research, "urlopen", side_effect=fake_response) as fetch, patch.object(
        research, "call_llm", return_value=llm_result
    ) as llm:
        result = research.create_public_research_graph(_request()).compile().invoke({"metadata": {}})

    assert fetch.call_count == 1
    request_obj = fetch.call_args.args[0]
    assert request_obj.full_url.startswith("https://data.sec.gov/api/xbrl/companyfacts/")
    assert request_obj.get_header("User-agent") == research.SEC_USER_AGENT
    assert llm.call_count == 2
    assert all(call.kwargs["state"]["metadata"]["request"].model_provider == "ollama" for call in llm.call_args_list)
    assert result["data"]["current_prices"] == {}
    assert json.loads(result["messages"][0].content)["AAPL"]["action"] == "hold"
    assert result["data"]["research_report"]["observations"]
    assert result["data"]["analyst_signals"]["fundamentals_analyst"]["AAPL"]["signal"] == "neutral"
    assert result["data"]["analyst_signals"]["risk_analyst"]["AAPL"]["confidence"] == 0


def test_non_local_provider_is_rejected_before_network_or_model_calls():
    with patch.object(research, "urlopen") as fetch, patch.object(research, "call_llm") as llm:
        with pytest.raises(ValueError, match="explicit Ollama"):
            research.create_public_research_graph(_request(provider="openai")).invoke({})
    fetch.assert_not_called()
    llm.assert_not_called()


def test_empty_observations_fail_before_model_call():
    payload = json.dumps({"facts": {"us-gaap": {}}}).encode()

    @contextmanager
    def fake_response(*_args, **_kwargs):
        yield SimpleNamespace(read=lambda: payload, status=200)

    with patch.object(research, "urlopen", side_effect=fake_response), patch.object(research, "call_llm") as llm:
        with pytest.raises(RuntimeError, match="did not provide any supported"):
            research.create_public_research_graph(_request()).invoke({})
    llm.assert_not_called()


def test_unusable_model_output_fails_the_graph_instead_of_succeeding_empty():
    payload = json.dumps(_facts()).encode()

    @contextmanager
    def fake_response(*_args, **_kwargs):
        yield SimpleNamespace(read=lambda: payload, status=200)

    with patch.object(research, "urlopen", side_effect=fake_response), patch.object(
        research, "call_llm", return_value=research.AnalystOutput(reasoning="No supported analysis.")
    ):
        with pytest.raises(RuntimeError, match="analysis was rejected.*did not cite"):
            research.create_public_research_graph(_request()).invoke({})


def test_run_stop_from_sec_fetch_is_preserved():
    stopped = research.RunStopped("cancelled")
    context = SimpleNamespace(remaining_seconds=lambda: 5, capture_source=Mock())
    with patch.object(research, "urlopen", side_effect=stopped):
        with pytest.raises(research.RunStopped, match="cancelled"):
            research._fetch_company_facts("0000320193", context)
    context.capture_source.assert_not_called()


def test_sec_response_is_read_across_bounded_chunks():
    payload = b"x" * (research.SEC_READ_CHUNK_BYTES * 2 + 19)
    response = io.BytesIO(payload)
    response.read1 = Mock(wraps=response.read1)

    actual = research._read_response(response, None, time.monotonic() + 10)

    assert actual == payload
    assert response.read1.call_count >= 3


def test_sec_stream_propagates_cancellation_between_chunks():
    response = io.BytesIO(b"x" * (research.SEC_READ_CHUNK_BYTES * 2))
    checks = 0

    def check_active():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise research.RunStopped("cancelled between chunks")

    context = SimpleNamespace(check_active=check_active)
    with pytest.raises(research.RunStopped, match="cancelled between chunks"):
        research._read_response(response, context, time.monotonic() + 10)
    assert checks == 2


def test_sec_stream_rejects_oversize_body_and_expired_deadline():
    with pytest.raises(ValueError, match="size limit"):
        research._read_response(io.BytesIO(b"12345"), None, time.monotonic() + 10, max_bytes=4)
    with pytest.raises(research._FetchDeadlineExceeded, match="whole-request deadline"):
        research._read_response(io.BytesIO(b"body"), None, time.monotonic() - 1)
    clock = iter((100.0, 100.0, 101.0))
    with patch.object(research.time, "monotonic", side_effect=lambda: next(clock)):
        with pytest.raises(research._FetchDeadlineExceeded, match="whole-request deadline"):
            research._read_response(io.BytesIO(b"first chunk then deadline"), None, 100.5)


def test_sec_fetch_uses_30_second_idle_timeout():
    payload = json.dumps(_facts()).encode()

    @contextmanager
    def fake_response(*_args, **_kwargs):
        yield SimpleNamespace(read=lambda: payload, status=200)

    with patch.object(research, "urlopen", side_effect=fake_response) as fetch:
        research._fetch_company_facts("0000320193", None)

    assert fetch.call_args.kwargs["timeout"] == 30
    assert research.SEC_TOTAL_DEADLINE_SECONDS == 120


def test_sec_fetch_requests_and_decodes_gzip_response():
    payload = json.dumps(_facts()).encode()
    compressed = gzip.compress(payload)

    @contextmanager
    def fake_response(*_args, **_kwargs):
        response = io.BytesIO(compressed)
        response.status = 200
        response.headers = {"Content-Encoding": "gzip"}
        yield response

    with patch.object(research, "urlopen", side_effect=fake_response) as fetch:
        decoded, url = research._fetch_company_facts("0000320193", None)

    assert decoded["facts"]["us-gaap"]
    assert url.endswith("CIK0000320193.json")
    assert fetch.call_args.args[0].get_header("Accept-encoding") == "gzip"


def test_sec_gzip_decompression_is_capped():
    compressed = gzip.compress(b"x" * 1000)
    with pytest.raises(ValueError, match="decompressed size limit"):
        research._decode_response_body(
            compressed,
            "gzip",
            None,
            time.monotonic() + 10,
            max_bytes=100,
        )


@pytest.mark.parametrize("tickers", [["AAPL", "MSFT"], ["TSLA"], []])
def test_only_one_supported_ticker_is_accepted(tickers):
    with patch.object(research, "urlopen") as fetch:
        with pytest.raises(ValueError, match="exactly one ticker"):
            research.create_public_research_graph(_request(tickers=tickers)).invoke({})
    fetch.assert_not_called()
