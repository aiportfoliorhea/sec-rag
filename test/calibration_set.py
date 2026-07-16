test_set = {
  "corpus": "jpm-10K-small-clean.txt (risk factors only)",
  "embedder": "all-MiniLM-L6-v2",
  "pairs": [
    {
      "id": 1,
      "label": "HIT",
      "kind": "paraphrase",
      "q1": "What credit risks does JPMorganChase face from client collateral?",
      "q2": "How is JPMorgan exposed to losses when collateral value declines?",
      "note": "easy paraphrase"
    },
    {
      "id": 2,
      "label": "HIT",
      "kind": "paraphrase",
      "q1": "Why does the Parent Company depend on its subsidiaries for funding?",
      "q2": "How does JPMorgan Chase & Co. rely on its subsidiaries to make payments on its securities?",
      "note": "easy paraphrase; entity alias Parent Company / JPMorgan Chase & Co."
    },
    {
      "id": 3,
      "label": "HIT",
      "kind": "paraphrase",
      "q1": "What happens to JPMorgan's creditors if the Parent Company enters resolution?",
      "q2": "Who absorbs the losses if JPMorgan Chase & Co. goes into a resolution proceeding?",
      "note": "easy paraphrase"
    },
    {
      "id": 4,
      "label": "HIT",
      "kind": "paraphrase",
      "q1": "What liquidity risks could impair JPMorgan's operations?",
      "q2": "What factors could constrain JPMorganChase's liquidity?",
      "note": "easy paraphrase"
    },
    {
      "id": 5,
      "label": "HIT",
      "kind": "paraphrase",
      "q1": "How could government investigations affect JPMorgan?",
      "q2": "What are the consequences for JPMorgan of resolving a governmental enforcement action?",
      "note": "easy paraphrase"
    },
    {
      "id": 6,
      "label": "HIT",
      "kind": "paraphrase",
      "q1": "How do political developments create risk for JPMorgan?",
      "q2": "What geopolitical factors could negatively affect JPMorganChase's business?",
      "note": "easy paraphrase"
    },
    {
      "id": 7,
      "label": "MISS",
      "kind": "entity_role_swap",
      "q1": "Why is the Parent Company dependent on dividends from JPMorgan Chase Bank, N.A.?",
      "q2": "Why is JPMorgan Chase Bank, N.A. restricted in the dividends it can pay to the Parent Company?",
      "note": "directional swap, near-identical vocabulary; PREDICTED OVERLAP",
      "predicted_overlap": True
    },
    {
      "id": 8,
      "label": "MISS",
      "kind": "entity_role_swap",
      "q1": "What losses do holders of the Parent Company's equity absorb in a resolution?",
      "q2": "What losses do the Parent Company's unsecured creditors absorb in a resolution?",
      "note": "loss-absorption order: equity first, then unsecured LTD; PREDICTED OVERLAP",
      "predicted_overlap": True
    },
    {
      "id": 9,
      "label": "MISS",
      "kind": "risk_type_swap",
      "q1": "What credit risks does JPMorgan face from collateral?",
      "q2": "What liquidity risks does JPMorgan face from collateral and deposit outflows?",
      "note": "shared 'collateral' anchor, different risk type; PREDICTED OVERLAP",
      "predicted_overlap": True
    },
    {
      "id": 10,
      "label": "MISS",
      "kind": "risk_type_swap",
      "q1": "How could investigations by governmental authorities harm JPMorgan?",
      "q2": "How could a deficient resolution plan lead regulators to impose requirements on JPMorgan?",
      "note": "investigations vs resolution-plan"
    },
    {
      "id": 11,
      "label": "MISS",
      "kind": "risk_type_swap",
      "q1": "What market risks affect JPMorgan's trading positions?",
      "q2": "What concentration risks affect JPMorgan's credit exposures?",
      "note": "market vs concentration"
    },
    {
      "id": 12,
      "label": "MISS",
      "kind": "cause_vs_consequence",
      "q1": "What happens to JPMorgan if a clearing client becomes insolvent?",
      "q2": "What happens to JPMorgan if the value of collateral it holds declines?",
      "note": "different trigger events"
    },
    {
      "id": 13,
      "label": "MISS",
      "kind": "scope_shift",
      "q1": "How does JPMorgan face risk in jurisdictions with unpredictable legal frameworks?",
      "q2": "How does JPMorgan face risk from governmental policies that penalize doing business with certain industries?",
      "note": "legal-framework unpredictability vs policy penalties"
    },
    {
      "id": 14,
      "label": "HIT",
      "kind": "hard_paraphrase",
      "q1": "If JPM can't get money from the businesses it owns, what happens to its bond payments?",
      "q2": "Why does the Parent Company depend on its subsidiaries for funding?",
      "note": "HARD HIT: same info need, near-zero lexical overlap. Most likely to score LOW and pull the threshold down. The hit that matters.",
      "predicted_low": True
    }
  ]
}
