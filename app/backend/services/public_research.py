"""SEC company-facts research graph for local, disclosure-only analysis."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import re
from typing import Any
from urllib.request import Request, urlopen

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.llm.models import ModelProvider
from src.run_context import RunStopped, get_active_run_context
from src.utils.llm import call_llm


SEC_USER_AGENT = "HomeLabResearch/1.0 personal research"
SEC_TIMEOUT_SECONDS = 12.0
NUMERIC_LITERAL = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"(?=(?:\s*(?:thousand|million|billion|mn|bn|k|m|b)s?\b)|(?:[^\w]|$))",
    re.I,
)
ISO_DATE_LITERAL = re.compile(r"(?<!\w)(\d{4}-\d{2}-\d{2})(?!\w)")
MONTH_DATE_LITERAL = re.compile(
    r"(?<!\w)(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})(?!\w)",
    re.I,
)
FISCAL_YEAR_LITERAL = re.compile(
    r"\b(?:FY\s*(\d{4})|fiscal\s+year\s+(\d{4})|(?:calendar|annual|reporting)\s+year\s+(\d{4})|year\s+ended\s+(\d{4}))\b(?!-\d{2}-\d{2})",
    re.I,
)
SCALE_UNITS = {
    "thousand": Decimal("1000"), "million": Decimal("1000000"), "billion": Decimal("1000000000"),
    "k": Decimal("1000"), "m": Decimal("1000000"), "mn": Decimal("1000000"),
    "b": Decimal("1000000000"), "bn": Decimal("1000000000"),
}
MONTH_NUMBERS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
TICKER_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
}

# Each metric is reported as an individual filing observation. Debt components
# remain separate so the report never invents a sum absent from the filing.
FACT_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "cash_and_equivalents": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "total_assets": ("Assets",),
    "long_term_debt": (
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligations",
        "LongTermDebtNoncurrent",
    ),
    "current_debt": ("LongTermDebtCurrent", "ShortTermBorrowings"),
}


class AnalystOutput(BaseModel):
    reasoning: str = Field(description="Qualitative interpretation only; no new numbers")
    cited_fact_ids: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


def _check_active(context: Any) -> None:
    if context is not None:
        context.check_active()


def _as_of_date(request: Any) -> date:
    raw = getattr(request, "end_date", None)
    if not raw:
        raise ValueError("SEC research requires an end_date")
    return date.fromisoformat(raw)


def _filing_records(company_facts: dict[str, Any], cutoff: date) -> tuple[str | None, str | None]:
    filings: set[tuple[str, str, str]] = set()
    gaap = company_facts.get("facts", {}).get("us-gaap", {})
    for fact in gaap.values():
        for records in fact.get("units", {}).values():
            for record in records:
                filed = record.get("filed")
                accession = record.get("accn")
                end = record.get("end")
                if record.get("form") != "10-K" or not filed or not accession or not end:
                    continue
                try:
                    if date.fromisoformat(filed) <= cutoff:
                        filings.add((filed, end, accession))
                except ValueError:
                    continue
    if not filings:
        return None, None
    filed, end, accession = max(filings)
    return accession, end


def _observations(company_facts: dict[str, Any], cutoff: date, source_url: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    gaap = company_facts.get("facts", {}).get("us-gaap", {})
    accession, filing_end = _filing_records(company_facts, cutoff)
    if accession is None:
        return [], [{"metric": "annual_filing", "reason": "No 10-K filing was available by the requested as-of date."}]

    observations: list[dict[str, Any]] = []
    gaps: list[dict[str, str]] = []
    for metric, candidates in FACT_TAGS.items():
        selected: dict[str, Any] | None = None
        selected_tag: str | None = None
        for tag in candidates:
            fact = gaap.get(tag)
            if not fact:
                continue
            candidates_for_tag: list[dict[str, Any]] = []
            for unit, records in fact.get("units", {}).items():
                if unit != "USD":
                    continue
                for record_index, record in enumerate(records):
                    if not (
                        record.get("accn") == accession
                        and record.get("form") == "10-K"
                        and record.get("filed")
                        and record.get("end") == filing_end
                        and record.get("val") is not None
                    ):
                        continue
                    # FY can label a quarter-end comparative fact too. Income
                    # metrics must span nearly a full fiscal year and end with
                    # the selected annual filing period.
                    if metric in {"revenue", "net_income"}:
                        start = record.get("start")
                        if not start:
                            continue
                        try:
                            duration = (date.fromisoformat(record["end"]) - date.fromisoformat(start)).days
                        except ValueError:
                            continue
                        if not 300 <= duration <= 380:
                            continue
                    candidates_for_tag.append({**record, "_record_index": record_index})
            if candidates_for_tag:
                selected = max(candidates_for_tag, key=lambda r: (r["end"], r["filed"]))
                selected_tag = tag
                break
        if selected is None or selected_tag is None:
            gaps.append({"metric": metric, "reason": "This annual filing did not expose a matching USD fact."})
            continue
        fact_id = f"fact-{metric}"
        raw_record = {key: value for key, value in selected.items() if key != "_record_index"}
        snapshot_hash = hashlib.sha256(
            json.dumps(raw_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        exact_path = f"facts.us-gaap.{selected_tag}.units.USD[{selected['_record_index']}]"
        observations.append({
            "fact_id": fact_id,
            "metric": metric,
            "label": gaap[selected_tag].get("label", selected_tag),
            "source_path": exact_path,
            "source_index": {"source_url": source_url, "json_path": exact_path, "sha256": snapshot_hash},
            "source_record": raw_record,
            "value": selected["val"],
            "unit": "USD",
            "end": selected["end"],
            "filed": selected["filed"],
            "form": selected["form"],
            "accn": selected["accn"],
            "source_url": source_url,
        })
    if filing_end:
        gaps.append({"metric": "price_and_valuation", "reason": "SEC company facts contains no market price or valuation series."})
    return observations, gaps


def _prompt(role: str, observations: list[dict[str, Any]], gaps: list[dict[str, str]]) -> str:
    return (
        f"You are the {role} for a local SEC annual-disclosure research note. Use only the supplied observations. "
        "Return exactly one JSON object with the keys reasoning, cited_fact_ids, and gaps, matching this schema: "
        f"{json.dumps(AnalystOutput.model_json_schema(), ensure_ascii=False)}. "
        "Keep reasoning to two or three concise sentences and choose only two or three relevant observations; do not recap all facts. "
        "Prefer qualitative implications because numeric details are already in the observations table. "
        "For every numeric or date fact in reasoning, include that observation's exact fact_id in cited_fact_ids. "
        "cited_fact_ids must contain only exact supplied fact_id values, "
        "and gaps must either copy supplied source gap reasons exactly or be an empty list. "
        "You may quote supplied numeric facts exactly, or with an explicit thousand/million/billion or k/m/mn/b/bn scale, "
        "only when the corresponding fact is cited. Do not calculate ratios, projections, or other derived numbers, "
        "and do not introduce unsupported numbers; the filing label 10-K is allowed. "
        "You may copy filing or period dates and fiscal-year labels exactly from cited observations. "
        "Do not give valuation, return, or buy/sell advice. "
        "If evidence is insufficient, say so.\nOBSERVATIONS:\n"
        f"{json.dumps(observations, ensure_ascii=False)}\nGAPS:\n{json.dumps(gaps, ensure_ascii=False)}"
    )


def _validated_interpretation(
    role: str,
    result: AnalystOutput,
    valid_ids: set[str],
    known_gap_reasons: set[str] | None = None,
    observations: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    citations = list(result.cited_fact_ids)
    if any(fact_id not in valid_ids for fact_id in citations):
        return None, f"{role}: model output contained a citation that was not present in the SEC observations."
    if not citations:
        return None, f"{role}: model output did not cite any supplied SEC observation."
    reasoning = result.reasoning.strip()
    if not reasoning:
        return None, f"{role}: model output did not provide an interpretation."
    numeric_text = re.sub(r"\b10[-\s]?K\b", "", reasoning, flags=re.I)
    cited_values: list[Decimal] = []
    cited_dates: set[date] = set()
    cited_period_years: set[int] = set()
    if observations is not None:
        cited_observations = [
            item
            for item in observations
            if item.get("fact_id") in citations
        ]
        values_by_id = {item.get("fact_id"): item.get("value") for item in cited_observations if item.get("value") is not None}
        try:
            cited_values = [Decimal(str(value)) for value in values_by_id.values()]
        except InvalidOperation:
            cited_values = []
        for item in cited_observations:
            for value in (item.get("end"), item.get("filed")):
                if value:
                    try:
                        parsed = date.fromisoformat(str(value))
                        cited_dates.add(parsed)
                        if value == item.get("end"):
                            cited_period_years.add(parsed.year)
                    except ValueError:
                        continue
            source_record = item.get("source_record") or {}
            for key in ("start", "end", "filed"):
                value = source_record.get(key)
                if value:
                    try:
                        parsed = date.fromisoformat(str(value))
                        cited_dates.add(parsed)
                        if key == "end":
                            cited_period_years.add(parsed.year)
                    except ValueError:
                        continue

    fiscal_year_text = numeric_text
    for fiscal_match in FISCAL_YEAR_LITERAL.finditer(fiscal_year_text):
        year_text = next(group for group in fiscal_match.groups() if group is not None)
        token_diagnostic = repr(year_text[:48])
        context_start = max(0, fiscal_match.start() - 24)
        context_end = min(len(fiscal_year_text), fiscal_match.end() + 24)
        context_diagnostic = repr(fiscal_year_text[context_start:context_end][:96])
        if int(year_text) not in cited_period_years:
            return None, f"{role}: unsupported fiscal-year reference (token={token_diagnostic}, context={context_diagnostic})."
        fiscal_year_text = fiscal_year_text.replace(fiscal_match.group(0), " " * len(fiscal_match.group(0)), 1)

    # Strip only full date expressions that exactly match cited filing/period
    # provenance. Any other date remains visible to the numeric validator.
    date_text = fiscal_year_text
    for date_match in ISO_DATE_LITERAL.finditer(date_text):
        try:
            parsed_date = date.fromisoformat(date_match.group(1))
        except ValueError:
            parsed_date = None
        if parsed_date not in cited_dates:
            token_diagnostic = repr(date_match.group(1)[:48])
            context_start = max(0, date_match.start() - 24)
            context_end = min(len(date_text), date_match.end() + 24)
            context_diagnostic = repr(date_text[context_start:context_end][:96])
            return None, f"{role}: unsupported source date (token={token_diagnostic}, context={context_diagnostic})."
        date_text = date_text.replace(date_match.group(0), " " * len(date_match.group(0)), 1)
    for date_match in MONTH_DATE_LITERAL.finditer(date_text):
        month = MONTH_NUMBERS[date_match.group(1)[:4].lower() if date_match.group(1).lower().startswith("sept") else date_match.group(1)[:3].lower()]
        try:
            parsed_date = date(int(date_match.group(3)), month, int(date_match.group(2)))
        except ValueError:
            continue
        if parsed_date not in cited_dates:
            token_diagnostic = repr(date_match.group(0)[:48])
            context_start = max(0, date_match.start() - 24)
            context_end = min(len(date_text), date_match.end() + 24)
            context_diagnostic = repr(date_text[context_start:context_end][:96])
            return None, f"{role}: unsupported source date (token={token_diagnostic}, context={context_diagnostic})."
        date_text = date_text.replace(date_match.group(0), " " * len(date_match.group(0)), 1)

    numeric_text = date_text
    for numeric_match in NUMERIC_LITERAL.finditer(numeric_text):
        token_text = numeric_match.group(0)
        context_start = max(0, numeric_match.start() - 24)
        context_end = min(len(numeric_text), numeric_match.end() + 24)
        numeric_context = numeric_text[context_start:context_end]
        token_diagnostic = repr(token_text[:48])
        context_diagnostic = repr(numeric_context[:96])
        following = numeric_text[numeric_match.end():]
        if re.match(r"\s*(?:%|percent\b|per\s+cent\b|times\b|x\b)", following, re.I):
            return None, (
                f"{role}: unsupported derived numeric claim (token={token_diagnostic}, context={context_diagnostic}, "
                f"cited_fact_ids={repr(citations[:8])})."
            )
        try:
            number = Decimal(token_text.replace(",", ""))
        except InvalidOperation:
            return None, f"{role}: invalid numeric claim (token={token_diagnostic}, context={context_diagnostic})."
        # Calendar years are not source values and remain blocked as narrative claims.
        is_scaled_number = re.match(r"\s*(?:thousand|million|billion|mn|bn|k|m|b)s?\b", following, re.I)
        has_currency_context = bool(re.search(r"[$€£¥]\s*$", numeric_text[max(0, numeric_match.start() - 2):numeric_match.start()])) or bool(
            re.match(r"\s*(?:USD|dollars?)\b", following, re.I)
        )
        if (
            number == number.to_integral_value()
            and Decimal("1900") <= number <= Decimal("2100")
            and not is_scaled_number
            and not has_currency_context
        ):
            return None, f"{role}: unsupported year reference (token={token_diagnostic}, context={context_diagnostic})."

        suffix = following.lstrip()
        unit_match = re.match(r"(thousand|million|billion|mn|bn|k|m|b)s?\b", suffix, re.I)
        if unit_match and observations is not None:
            scale_name = unit_match.group(1).lower()
            scale = SCALE_UNITS[scale_name]
            decimal_places = max(0, -number.as_tuple().exponent)
            quantum = scale / (Decimal(10) ** decimal_places)
            matched_fact = any(
                (value / quantum).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * quantum == number * scale
                for value in cited_values
            )
        else:
            matched_fact = any(number == value for value in cited_values)
        if not matched_fact:
            cited_ids_diagnostic = repr(citations[:8])
            return None, (
                f"{role}: unsupported numeric claim (token={token_diagnostic}, context={context_diagnostic}, "
                f"cited_fact_ids={cited_ids_diagnostic})."
            )
    if re.search(r"\b(buy|sell|strong buy|strong sell|price target)\b", reasoning, re.I):
        return None, f"{role}: recommendation language was excluded from the report."
    supplied_gaps = known_gap_reasons or set()
    for gap in result.gaps:
        if gap in supplied_gaps:
            continue
        numeric_gap_match = NUMERIC_LITERAL.search(re.sub(r"\b10[-\s]?K\b", "", gap, flags=re.I))
        unsafe_gap_language = re.search(r"\b(buy|sell|price target)\b", gap, re.I)
        if numeric_gap_match:
            token = repr(numeric_gap_match.group(0)[:48])
            context = repr(gap[max(0, numeric_gap_match.start() - 24):numeric_gap_match.end() + 24][:96])
            return None, f"{role}: model output contained an unsupported numeric gap (token={token}, context={context})."
        if unsafe_gap_language:
            return None, f"{role}: model output contained an unsupported recommendation or valuation gap."
    return {
        "role": role,
        "reasoning": reasoning,
        "cited_fact_ids": citations,
        "gaps": [str(gap) for gap in result.gaps],
    }, None


def _provider_value(provider: Any) -> str:
    return str(getattr(provider, "value", provider)).lower()


def _assert_local_models(request: Any) -> dict[str, dict[str, str]]:
    configuration: dict[str, dict[str, str]] = {}
    for role in ("fundamentals_analyst", "risk_analyst"):
        model_name, provider = request.get_agent_model_config(role)
        if not model_name or _provider_value(provider) != "ollama":
            raise ValueError("SEC public research requires an explicit Ollama model for every analyst.")
        configuration[role] = {"model_name": model_name, "model_provider": "ollama"}
    return configuration


def _fetch_company_facts(cik: str, context: Any) -> tuple[dict[str, Any], str]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    timeout = SEC_TIMEOUT_SECONDS
    if context is not None:
        timeout = min(timeout, context.remaining_seconds())
    req = Request(url, headers={"User-Agent": SEC_USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read()
            payload = json.loads(raw.decode("utf-8"))
            if context is not None:
                context.capture_source("GET", url, status_code=getattr(response, "status", 200), body=payload)
            return payload, url
    except Exception as exc:
        if isinstance(exc, RunStopped):
            raise
        if context is not None:
            _check_active(context)
            context.capture_source("GET", url, error=f"{type(exc).__name__}: SEC company facts request failed")
        raise RuntimeError("SEC company facts could not be retrieved") from exc


class PublicResearchGraph:
    """Small duck-typed graph adapter compatible with the existing runner."""

    def __init__(self, request: Any):
        self.request = request

    def compile(self) -> "PublicResearchGraph":
        return self

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        request = self.request
        if getattr(request, "data_source", None) != "sec_filings":
            raise ValueError("Public research graph only supports the sec_filings data source.")
        tickers = [ticker.strip().upper() for ticker in getattr(request, "tickers", [])]
        if len(tickers) != 1 or tickers[0] not in TICKER_CIK:
            raise ValueError("SEC public research supports exactly one ticker: AAPL, MSFT, or NVDA.")
        model_config = _assert_local_models(request)
        cutoff = _as_of_date(request)
        context = (state.get("metadata") or {}).get("run_context") or get_active_run_context()

        _check_active(context)
        facts, source_url = _fetch_company_facts(TICKER_CIK[tickers[0]], context)
        _check_active(context)
        observations, gaps = _observations(facts, cutoff, source_url)
        if not observations:
            raise RuntimeError("SEC annual filing did not provide any supported research observations.")
        if context is not None:
            context.capture_source("OBSERVATIONPACK", source_url, status_code=200, body={"observations": observations})

        valid_ids = {item["fact_id"] for item in observations}
        llm_state = {
            "messages": [],
            "data": state.get("data", {}),
            "metadata": {
                **(state.get("metadata") or {}),
                "request": request,
                "run_context": context,
                "strict_execution": True,
            },
        }
        interpretations: list[dict[str, Any]] = []
        for role in ("fundamentals_analyst", "risk_analyst"):
            _check_active(context)
            result = call_llm(_prompt(role, observations, gaps), AnalystOutput, agent_name=role, state=llm_state)
            _check_active(context)
            interpretation, error = _validated_interpretation(
                role,
                result,
                valid_ids,
                known_gap_reasons={gap["reason"] for gap in gaps},
                observations=observations,
            )
            if error:
                raise RuntimeError(f"SEC {role} analysis was rejected: {error}")
            interpretations.append(interpretation)
            gaps.extend(
                {"metric": f"{role}_gap", "reason": gap}
                for gap in result.gaps
            )

        rationale = "Disclosure-only research; no market price or valuation data was collected, so no trade is proposed."
        decisions = {tickers[0]: {"action": "hold", "quantity": 0, "reasoning": rationale}}
        data = {
            "analyst_signals": {
                item["role"]: {
                    tickers[0]: {
                        "signal": "neutral",
                        "confidence": 0,
                        "reasoning": item["reasoning"],
                        "cited_fact_ids": item["cited_fact_ids"],
                    }
                }
                for item in interpretations
            },
            "current_prices": {},
            "research_report": {
                "source": "SEC companyfacts API",
                "ticker": tickers[0],
                "as_of": cutoff.isoformat(),
                "observations": observations,
                "interpretations": interpretations,
                "gaps": gaps,
                "model_config": model_config,
            },
        }
        return {"messages": [HumanMessage(content=json.dumps(decisions, ensure_ascii=False))], "data": data}


def create_public_research_graph(request: Any) -> PublicResearchGraph:
    """Create the public SEC research graph for one explicitly local ticker."""
    return PublicResearchGraph(request)
