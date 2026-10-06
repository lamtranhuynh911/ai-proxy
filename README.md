# Databricks Proxies
## Introduction
This repository contains proxies for interacting with Databricks AI Gateway. The
Claude proxy exposes the Anthropic Messages API expected by Claude Code in VS
Code.
## Usage
### gpt-proxy.py
To use the `gpt-proxy.py` proxy, follow these steps:
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Run the proxy by executing `python gpt-proxy.py`.
3. The proxy will start listening on port 8000. You can then use a tool like `curl` to send requests to the proxy.
### cli_sp_dbx_proxy.py
To use the `cli_sp_dbx_proxy.py` proxy with Claude Code, follow these steps:
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Start the proxy with `python cli_sp_dbx_proxy.py --environment d`.
3. Configure Claude Code's VS Code extension to use `http://127.0.0.1:8786` as
	its Anthropic base URL and set its auth token to any non-empty value. The
	proxy replaces that value with the selected Databricks service-principal
	token.

The proxy forwards request-body parameters to Databricks for experimentation,
including parameters such as `metadata`, `service_tier`, and `top_k`. It strips
Claude Code's local `mcp_servers` field and gateway-unsupported
`context_management`, and removes `thinking.type: "adaptive"` because the
Databricks model endpoint does not support adaptive thinking. Other `thinking`
configurations are forwarded, but the selected model may reject unsupported
parameters with an upstream 400.

The model supplied by Claude Code is passed through unchanged, with `system.ai.`
added when it is not already present. If Claude Code omits the model, the proxy
uses `system.ai.claude-haiku-4-5`.

### Auto mode

When `ANTHROPIC_BASE_URL` points at a proxy, Claude Code's auto mode asks the
server to review actions by adding a `safeguards` request field and an
`anthropic-beta` value. Databricks cannot perform that review and the proxy does
not forward `anthropic-beta`, so the proxy strips `safeguards`. The response then
carries no review results and Claude Code falls back to sending its own
classifier requests, which the proxy forwards like any other request. To skip the
server-review attempt entirely, set `CLAUDE_CODE_AUTO_MODE_SERVER=0` in Claude
Code's environment.

Upstream errors on streaming requests are returned with their original HTTP
status and body, because Claude Code's recovery paths (such as falling back when
the classifier model is unavailable) match on them.

The classifier runs on Claude Sonnet 5 by default, and Claude Code validates that
model on the first auto-mode request. The Databricks workspace must serve the
resulting `system.ai.` endpoint, or Claude Code falls back to the session model.

For Claude Code installations configured through environment variables:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8786"
export ANTHROPIC_AUTH_TOKEN="local-proxy"
python cli_sp_dbx_proxy.py --environment d
```
## Configuration
The proxy uses Databricks OAuth machine-to-machine authentication. Define credentials
for each environment using the `D_`, `Q_`, and `P_` prefixes:

```bash
export D_DATABRICKS_HOST="https://<dev-workspace-host>"
export D_DATABRICKS_CLIENT_ID="<dev-service-principal-client-id>"
export D_DATABRICKS_CLIENT_SECRET="<dev-service-principal-client-secret>"
export Q_DATABRICKS_HOST="https://<qa-workspace-host>"
export Q_DATABRICKS_CLIENT_ID="<qa-service-principal-client-id>"
export Q_DATABRICKS_CLIENT_SECRET="<qa-service-principal-client-secret>"
export P_DATABRICKS_HOST="https://<prod-workspace-host>"
export P_DATABRICKS_CLIENT_ID="<prod-service-principal-client-id>"
export P_DATABRICKS_CLIENT_SECRET="<prod-service-principal-client-secret>"
python sp_dbx_proxy.py
```

When the proxy starts, enter `d` for development, `q` for QA, or `p` for production.

Access tokens are refreshed automatically before expiry and retried once when the
gateway returns HTTP 401.
## Troubleshooting
If you encounter any issues while using the proxies, check the logs for token or
gateway errors. The service principal must have permission to use the Databricks
AI Gateway.