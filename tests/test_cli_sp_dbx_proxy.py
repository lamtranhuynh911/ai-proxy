import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

import cli_sp_dbx_proxy as proxy
from cli_sp_dbx_proxy import DEFAULT_MODEL, sanitize_body


class SanitizeBodyTests(unittest.TestCase):
    def test_preserves_client_model_and_forwards_experimental_parameters(self):
        body = {
            "model": "claude-sonnet-next-20261001",
            "metadata": {"user_id": "test-user"},
            "service_tier": "auto",
            "top_k": 20,
            "custom_parameter": {"enabled": True},
        }

        result = sanitize_body(body)

        self.assertEqual(result["model"], "system.ai.claude-sonnet-next-20261001")
        self.assertEqual(result["metadata"], {"user_id": "test-user"})
        self.assertEqual(result["service_tier"], "auto")
        self.assertEqual(result["top_k"], 20)
        self.assertEqual(result["custom_parameter"], {"enabled": True})

    def test_preserves_model_with_databricks_prefix(self):
        body = {"model": "system.ai.custom-model"}

        result = sanitize_body(body)

        self.assertEqual(result["model"], "system.ai.custom-model")

    def test_strips_adaptive_thinking_but_preserves_other_parameters(self):
        body = {
            "model": "system.ai.example",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }

        result = sanitize_body(body)

        self.assertNotIn("thinking", result)
        self.assertEqual(result["output_config"], {"effort": "high"})

    def test_forwards_non_adaptive_thinking(self):
        thinking = {"type": "enabled", "budget_tokens": 2048}
        body = {"thinking": thinking}

        result = sanitize_body(body)

        self.assertEqual(result["thinking"], thinking)
        self.assertEqual(result["model"], DEFAULT_MODEL)

    def test_strips_claude_code_local_fields(self):
        body = {
            "mcp_servers": [{"name": "local-tool"}],
            "context_management": {"edits": []},
        }

        result = sanitize_body(body)

        self.assertNotIn("mcp_servers", result)
        self.assertNotIn("context_management", result)

    def test_strips_auto_mode_server_review_request(self):
        body = {"model": "system.ai.example", "safeguards": {"review": True}}

        result = sanitize_body(body)

        self.assertNotIn("safeguards", result)


def sse_response(content: bytes) -> httpx.Response:
    """A 200 response whose body is an unread stream, like a real upstream."""
    return httpx.Response(
        200,
        stream=httpx.ByteStream(content),
        headers={"content-type": "text/event-stream"},
    )


class StreamingForwardTests(unittest.TestCase):
    """Exercise POST /v1/messages against a mock Databricks upstream."""

    def setUp(self):
        self.calls = []
        self.responses = []
        patches = [
            mock.patch.object(proxy, "DATABRICKS_HOST", "https://dbx.example"),
            mock.patch.object(proxy, "get_token", lambda force_refresh=False: (
                "fresh-token" if force_refresh else "stale-token")),
            mock.patch.object(proxy, "_http", httpx.AsyncClient(
                transport=httpx.MockTransport(self._upstream))),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(proxy.app)

    def _upstream(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self.responses.pop(0)

    def _post(self, **extra):
        return self.client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-5", "stream": True, "messages": [], **extra},
        )

    def test_streamed_upstream_error_keeps_status_and_body(self):
        error = {"error": {"type": "not_found_error", "message": "model not found"}}
        self.responses = [httpx.Response(404, json=error)]

        response = self._post()

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), error)

    def test_streamed_success_is_relayed_unchanged(self):
        sse = b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        self.responses = [sse_response(sse)]

        response = self._post()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, sse)
        self.assertEqual(self.calls[0].headers["authorization"], "Bearer stale-token")

    def test_streamed_401_refreshes_token_and_retries(self):
        self.responses = [
            httpx.Response(401, json={"error": "expired"}),
            sse_response(b"ok"),
        ]

        response = self._post()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[1].headers["authorization"], "Bearer fresh-token")

    def test_safeguards_is_not_forwarded_upstream(self):
        self.responses = [sse_response(b"ok")]

        self._post(safeguards={"review": True})

        self.assertNotIn(b"safeguards", self.calls[0].content)


if __name__ == "__main__":
    unittest.main()
