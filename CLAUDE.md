# Clojure MCP Development Guide

## Build Commands
- Run REPL with MCP server: `clojure -X:mcp` (starts on port 7888)
- Run all tests: `clojure -X:test`
- Run single test: `clojure -X:test :dirs '["test"]' :include '"repl_tools_test"'`
- Run linter: `clojure -M:lint` (checks src directory)
- Build JAR: `clojure -T:build ci`
- Install locally: `clojure -T:build install`

## Code Style Guidelines
- **Imports**: Use `:require` with ns aliases (e.g., `[clojure.string :as string]`)
- **Naming**: Use kebab-case for vars/functions; end predicates with `?` (e.g., `is-top-level-form?`)
- **Error handling**: Use `try/catch` with specific exception handling; atom for tracking errors
- **Formatting**: 2-space indentation; maintain whitespace in edited forms
- **Namespaces**: Align with directory structure (`clojure-mcp.repl-tools`)
- **Testing**: Use `deftest` with descriptive names; `testing` for subsections; `is` for assertions
- **REPL Development**: Prefer REPL-driven development for rapid iteration and feedback
- Don't use the clojure -X:lint tool in the workflow

## MCP Tool Guidelines
- Include clear tool `:description` for LLM guidance
- Validate inputs and provide helpful error messages
- Return structured data with both result and error status
- Maintain atom-based state for consistent service access
