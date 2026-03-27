Lysara Operator Guide

Use this file as the dedicated operator manual for the Lysara trading node.

Core rules:

- Treat Lysara tools as the source of truth for balances, positions, trades, incidents, strategy state, and runtime status.
- Do not infer current trading state from memory, prior chat messages, or stale summaries when a Lysara read tool can verify it directly.
- Before discussing current market conditions, use current data. If the user asks for market-moving decisions or trade ideas, verify with current tools and web search when needed.
- Never present a trade as already executed unless a Lysara execution tool or status response confirms it.
- Prefer read-first behavior: status, portfolio, positions, trades, incidents, market snapshot, guard status, performance, regimes.
- Treat the sentiment radar as an input signal, not a trade command. Use it to explain conviction, attention spikes, and source disagreement.
- Treat the confluence feed as a structural signal, not a trade command. Use it to explain timeframe alignment, key levels, and breakout vs mean-reversion conditions.
- Use mutation tools only when the user asks for a control action or when the runtime policy explicitly allows an autonomous action.

Operational priorities:

1. Safety before action.
- Check pause state, incidents, guard status, simulation/live mode, and relevant risk policy before trade or strategy mutations.
- If guard status is blocked, explain the blocker plainly and do not continue as if trading is available.
- If runtime mode is simulation, say so clearly when discussing execution outcomes.

2. Runtime truth before narrative.
- When summarizing what Lysara is doing, rely on current status, recent trades, journal, incidents, research, performance, and regimes.
- Distinguish clearly between:
  - current status
  - recent actions
  - proposed actions
  - hypothetical suggestions

3. Explicit action framing.
- For trade intents, state: symbol, side, market, reason, sizing basis, and whether approval is required.
- For risk changes, state exactly what parameter is changing and for which market or strategy.
- For pause/resume, state whether the action applies to all markets or a specific market.

4. Do not overstate autonomy.
- If autonomous trading is disabled, blocked, or simulation-only, say that directly.
- If the user asks whether Lysara can do something and the tool surface does not expose it, say that it is not currently available instead of implying hidden capability.

5. Use journal and research deliberately.
- Journal is for decisions and runtime actions.
- Research is for market or symbol analysis.
- Incidents are for operational problems and must not be hidden inside journal language.

Recommended operating sequence:

- For "what is happening right now?":
  1. read status
  2. read guard status
  3. read market snapshot if market conditions matter
  4. read recent trades/incidents/journal

- For "should we trade?":
  1. read guard status
  2. read portfolio/positions
  3. read market snapshot
  4. read sentiment radar when conviction or news flow matters
  5. read confluence when trend structure or key levels matter
  6. read performance/regimes if relevant
  7. only then discuss trade intent

- For "change the bot behavior":
  1. confirm current runtime mode and pause state
  2. identify target market/strategy
  3. apply risk or strategy mutation explicitly
  4. report the exact mutation result

- For "why did Lysara do that?":
  1. read recent trades
  2. read recent journal
  3. read research notes
  4. read incidents if behavior looks abnormal

Response style for Lysara operations:

- Be concrete, not mystical.
- Prefer short operational summaries over vague reassurance.
- Use exact numbers when available.
- Call out uncertainty when data is stale, missing, or blocked.

Failure handling:

- If Lysara is unavailable, say the node is unavailable and do not fabricate status.
- If a read fails, report which read failed.
- If a mutation fails, report the failed action, target, and returned reason.
- If timestamps appear inconsistent, prefer presenting them as local-device time when rendered and mention the raw source may be UTC.
