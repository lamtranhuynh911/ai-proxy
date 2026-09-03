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
3. Configure Claude Code's VS Code extension to use `http://127.0.0.1:8787` as
	its Anthropic base URL and set its auth token to any non-empty value. The
	proxy replaces that value with the selected Databricks service-principal
	token.

For Claude Code installations configured through environment variables:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
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