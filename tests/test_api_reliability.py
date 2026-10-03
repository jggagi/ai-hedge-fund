from unittest.mock import Mock, call, patch

import pytest
import requests

from src.tools.api import _make_api_request, get_company_news, get_insider_trades


def _response(status_code=200, payload=None):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload or {}
    return response


def _news(date, url, title=None):
    return {
        "ticker": "AAPL",
        "title": title or url,
        "author": None,
        "source": "mock",
        "date": date,
        "url": url,
        "sentiment": None,
    }


def _trade(filing_date, name, transaction_shares):
    return {
        "ticker": "AAPL",
        "issuer": "Apple Inc.",
        "name": name,
        "title": None,
        "is_board_director": None,
        "transaction_date": filing_date,
        "transaction_shares": transaction_shares,
        "transaction_price_per_share": None,
        "transaction_value": None,
        "shares_owned_before_transaction": None,
        "shares_owned_after_transaction": None,
        "security_title": None,
        "filing_date": filing_date,
    }


class TestRequestReliability:
    @patch("src.tools.api.requests.get")
    @patch("src.tools.api.time.sleep")
    def test_get_retries_timeout_then_succeeds_with_bounded_timeout(self, sleep, get):
        get.side_effect = [requests.Timeout("read stalled"), _response()]

        result = _make_api_request("https://mock.invalid/data", {})

        assert result.status_code == 200
        assert get.call_args_list == [
            call("https://mock.invalid/data", headers={}, timeout=(5, 30)),
            call("https://mock.invalid/data", headers={}, timeout=(5, 30)),
        ]
        sleep.assert_called_once_with(1)

    @patch("src.tools.api.requests.post")
    @patch("src.tools.api.time.sleep")
    def test_post_uses_bounded_timeout_and_retries_connection_error(self, sleep, post):
        post.side_effect = [requests.ConnectionError("connection reset"), _response()]
        body = {"tickers": ["AAPL"]}

        result = _make_api_request("https://mock.invalid/data", {}, method="POST", json_data=body)

        assert result.status_code == 200
        assert post.call_args_list == [
            call("https://mock.invalid/data", headers={}, json=body, timeout=(5, 30)),
            call("https://mock.invalid/data", headers={}, json=body, timeout=(5, 30)),
        ]
        sleep.assert_called_once_with(1)

    @patch("src.tools.api.requests.get", side_effect=requests.Timeout("read stalled"))
    @patch("src.tools.api.time.sleep")
    def test_transport_retry_budget_is_bounded_and_final_error_propagates(self, sleep, get):
        with pytest.raises(requests.Timeout, match="read stalled"):
            _make_api_request("https://mock.invalid/data", {})

        assert get.call_count == 4
        assert sleep.call_args_list == [call(1), call(2), call(4)]

    @patch("src.tools.api.requests.get", return_value=_response(status_code=401))
    @patch("src.tools.api.time.sleep")
    def test_auth_response_is_not_retried(self, sleep, get):
        result = _make_api_request("https://mock.invalid/data", {})

        assert result.status_code == 401
        get.assert_called_once_with("https://mock.invalid/data", headers={}, timeout=(5, 30))
        sleep.assert_not_called()

    @patch("src.tools.api.requests.get")
    def test_rejects_invalid_retry_budget_before_request(self, get):
        for invalid in (-1, 1.5, True):
            with pytest.raises(ValueError, match="non-negative integer"):
                _make_api_request("https://mock.invalid/data", {}, max_retries=invalid)
        get.assert_not_called()


class TestPaginationReliability:
    @patch("src.tools.api._cache")
    @patch("src.tools.api._make_api_request")
    def test_same_day_full_news_page_stops_and_is_not_cached(self, request, cache, caplog):
        cache.get_company_news.return_value = None
        request.return_value = _response(
            payload={"news": [_news("2024-03-08T12:00:00Z", "https://mock.invalid/story/1")]}
        )

        result = get_company_news("AAPL", "2024-03-08", start_date="2024-03-01", limit=1)

        assert len(result) == 1
        assert request.call_count == 1
        cache.set_company_news.assert_not_called()
        assert "cursor did not advance" in caplog.text

    @patch("src.tools.api._cache")
    @patch("src.tools.api._make_api_request")
    def test_same_day_full_insider_page_stops_preserving_distinct_rows(self, request, cache, caplog):
        cache.get_insider_trades.return_value = None
        request.return_value = _response(
            payload={
                "insider_trades": [
                    _trade("2024-03-08", "Insider One", 10),
                    _trade("2024-03-08", "Insider Two", 20),
                ]
            }
        )

        result = get_insider_trades("AAPL", "2024-03-08", start_date="2024-03-01", limit=2)

        assert [trade.name for trade in result] == ["Insider One", "Insider Two"]
        assert request.call_count == 1
        cache.set_insider_trades.assert_not_called()
        assert "cursor did not advance" in caplog.text

    @pytest.mark.parametrize("kind", ["news", "insider"])
    @patch("src.tools.api._cache")
    @patch("src.tools.api._make_api_request")
    def test_overlapping_page_boundary_is_deduplicated(self, request, cache, kind):
        if kind == "news":
            cache.get_company_news.return_value = None
            request.side_effect = [
                _response(payload={"news": [
                    _news("2024-03-08T12:00:00Z", "https://mock.invalid/8"),
                    _news("2024-03-07T12:00:00Z", "https://mock.invalid/7"),
                ]}),
                _response(payload={"news": [
                    _news("2024-03-07T12:00:00Z", "https://mock.invalid/7"),
                    _news("2024-03-06T12:00:00Z", "https://mock.invalid/6"),
                ]}),
                _response(payload={"news": []}),
            ]

            result = get_company_news("AAPL", "2024-03-08", start_date="2024-03-01", limit=2)

            assert [item.url for item in result] == [
                "https://mock.invalid/8",
                "https://mock.invalid/7",
                "https://mock.invalid/6",
            ]
            cache.set_company_news.assert_called_once()
        else:
            cache.get_insider_trades.return_value = None
            request.side_effect = [
                _response(payload={"insider_trades": [
                    _trade("2024-03-08", "Insider Eight", 80),
                    _trade("2024-03-07", "Insider Seven", 70),
                ]}),
                _response(payload={"insider_trades": [
                    _trade("2024-03-07", "Insider Seven", 70),
                    _trade("2024-03-06", "Insider Six", 60),
                ]}),
                _response(payload={"insider_trades": []}),
            ]

            result = get_insider_trades("AAPL", "2024-03-08", start_date="2024-03-01", limit=2)

            assert [item.name for item in result] == ["Insider Eight", "Insider Seven", "Insider Six"]
            cache.set_insider_trades.assert_called_once()

        assert request.call_count == 3
