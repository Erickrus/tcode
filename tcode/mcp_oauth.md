MCP OAuth flow (tcode implementation notes)

- start_auth(name):
  - Create transport with auth_provider (if supported).
  - Call transport.connect(); if it raises UnauthorizedError that includes an authorization_url, store transport in pending_oauth[name] and emit an event 'mcp.auth.required' with payload {mcp: name, url: authorization_url}.
  - Return the authorization URL to the caller.

- finish_auth(name, code):
  - Lookup transport in pending_oauth[name]. If not found, error.
  - Call transport.finish_auth(code).
  - Remove from pending_oauth and attempt to add/connect the MCP again.
  - Emit events for success/failure.

Notes:
- This mirrors the opencode pattern of storing pendingOAuthTransports when Unauthorized occurs (opencode-dev/packages/opencode/src/mcp/index.ts:150-153) and startAuth/finishAuth behavior (opencode-dev/packages/opencode/src/mcp/index.ts:709-894).
- We must ensure the transport implementation supports finish_auth / captured authorization URL.
