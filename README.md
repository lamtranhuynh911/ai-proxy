# Databricks Proxies
## Introduction
This repository contains two proxies for interacting with Databricks AI Gateway: `gpt-proxy.py` and `claude_proxy.py`. These proxies allow you to use the Databricks AI Gateway with different models and configurations.
## Usage
### gpt-proxy.py
To use the `gpt-proxy.py` proxy, follow these steps:
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Run the proxy by executing `python gpt-proxy.py`.
3. The proxy will start listening on port 8000. You can then use a tool like `curl` to send requests to the proxy.
### claude_proxy.py
To use the `claude_proxy.py` proxy, follow these steps:
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Run the proxy by executing `python claude_proxy.py`.
3. The proxy will start listening on port 8787. You can then use a tool like `curl` to send requests to the proxy.
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

`DATABRICKS_HOST` and `TOKEN_TTL_SECONDS` may also be configured as environment
variables when those settings are made configurable in the proxy. Access tokens
are refreshed automatically before expiry and retried once when the gateway
returns HTTP 401.
## Troubleshooting
If you encounter any issues while using the proxies, check the logs for token or
gateway errors. The service principal must have permission to use the Databricks
AI Gateway.