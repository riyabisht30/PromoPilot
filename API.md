# Python API and provider integration

This is a Streamlit application, not a REST service. No API server is required.

## Local calculation

workflow.local_plan(rows, cfg, relationships) returns (result, scenarios, trace, summary) after catalogue inspection, planning, verification and simulation, with zero provider calls. Use CSV examples in data/. Config keys are objective (clearance/profit), budget (advertising currency), margin/max_discount (fractions), days, holiday_factor and mechanisms (percentage/buy2/bundle).

engine.plan(catalogue, config, relationships) selects a heuristic portfolio. engine.verify(result, rows) returns constraint checks/validity. engine.simulate(result, rows) returns sensitivity scenarios against the same selected-products baseline. local_plan provides the complete validated workflow.

## AI and human review

agent.run_agent(key, model, goal, rows, cfg, relationships, deadline_seconds=240, stop_on_rate_limit=False) returns the same four outputs only after all required tools succeed. Fixed empty-argument tools: inspect_catalogue, build_plan, validate_plan, simulate_plan. Model arguments cannot override controls.

agent.extract_evidence(key, text_model, vision_model, data, name, deadline_seconds=120, cache=None, use_cache=True, stop_on_rate_limit=False) returns validated extraction with source/model/schema metadata. Human review is separate. Evaluation must bypass caching; UI/evaluation stop on rate limit.

workflow.review_evidence(evidence, reviewed=True) binds explicit approval to content/source/model/schema. workflow.apply_reviewed(rows, cfg, evidence) checks that binding, then returns effective inputs. Null optional fields retain settings; changed evidence requires review again.

workflow.plan_with_quota_fallback(live_runner, rows, cfg, relationships, enabled=False) invokes the live callable once. If enabled, only rate/quota errors trigger a distinct local workflow. result.fallback records the failed attempt and verified partial work. It does not interpret documents or complete the AI workflow.

## Errors, configuration and exports

ProviderError categories distinguish quota, service, JSON/schema, timeout and network failures. EvidenceError distinguishes schema, field and source-quote validation. ProviderError.partial_run is incomplete verified work, not AI success.

App exports JSON with run and source_evidence. run includes result/scenarios/trace, actual execution_route, requested_mode, execution_status, input/evidence origins and effective catalogue. Evidence retains source/hash, quotes, warnings, observed dates, extraction model/schema and review. Quota fallback has completed local execution with ai_workflow_completed=false and failed live metadata.

Read keys/models from environment or local .streamlit/secrets.toml, using the supplied placeholder template. Never publish real keys. No promotion is launched automatically.
