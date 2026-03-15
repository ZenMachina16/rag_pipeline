"""
Section Prior — Document Structure Awareness for RAG

Maps query intent patterns to expected document sections so the retriever
can boost chunks from the right part of the paper instead of relying purely
on semantic similarity (which is section-blind).

Addresses:
  Category 1 — Section importance failure
  Category 2 — Numeric / experimental queries
  Category 4 — Reranker bias toward dense conceptual sections
"""

import re

# ---------------------------------------------------------------------------
# Section Prior Rules
# ---------------------------------------------------------------------------
# Each rule: (query_pattern_regex, list_of_preferred_section_substrings)
#
# When a query matches a pattern, chunks whose section label contains any of
# the preferred substrings receive a score bonus.
# Order matters: first match wins.

SECTION_PRIOR_RULES = [
    # ---------- Reward / Training ----------
    (r'reward|penalty|incentive|collision penalty|smooth trajectory|'
     r'reward.*time|reward.*change|reward shaping',
     ["reward", "3.1"]),

    # ---------- Obstacle avoidance mechanism ----------
    (r'obstacle.*(detect|avoid|respond)|depth.*plan|occupancy.*map|'
     r'perception.*loop|moving obstacle',
     ["obstacle avoidance", "2.4", "dynamic obstacle"]),

    # ---------- Global path planning ----------
    (r'a\s*star|global path|heuristic cost',
     ["global path", "2.1"]),

    # ---------- RRT / sampling ----------
    (r'random sampling|rrt|b.?spline|collision constraint',
     ["rrt", "2.2"]),

    # ---------- Problem description ----------
    (r'reinforcement learning.*adapt|traditional.*fail|'
     r'high dimensional.*planning',
     ["problem description", "1.2"]),

    # ---------- Contributions ----------
    (r'key contribution|main contribution|hierarchical.*framework|'
     r'rgb.?d.*integrat',
     ["contribution", "1.4"]),

    # ---------- Experimental setup ----------
    (r'simulation platform|robot model|hardware|what.*used',
     ["experiment", "setup", "4."]),

    # ---------- Image dataset / training data ----------
    (r'dataset.*prepared|images.*collected|how many images|'
     r'dataset.*split|training.*dataset',
     ["dataset", "image", "4.2"]),

    # ---------- Scenario results (numbered scenario sections) ----------
    (r'no obstacle scenario|obstacle.?free',
     ["no obstacle", "1. no obstacle"]),

    (r'single.*obstacle|single static',
     ["single static", "2. single"]),

    (r'multiple.*obstacle|multiple static',
     ["multiple static", "3. multiple"]),

    (r'dynamic.*obstacle.*environment|dynamic.*scenario',
     ["dynamic obstacle", "4. dynamic"]),

    # ---------- Conclusion / findings ----------
    (r'main.*finding|experimental finding|success rate|'
     r'limitation|future.*improve|perception.*error.*perform',
     ["conclusion", "6."]),

    # ---------- Noise annealing ----------
    (r'noise anneal|gaussian noise.*action',
     ["noise anneal", "1. noise"]),

    # ---------- Termination ----------
    (r'training.*terminat|termination.*criter|policy.*sav',
     ["termination", "3. termination"]),

    # ---------- Path matching reward ----------
    (r'path matching.*reward|reward.*guid',
     ["path matching", "2. path matching"]),

    # ---------- Workflow ----------
    (r'system workflow|pipeline|workflow',
     ["workflow", "3.2"]),
]

# Compile all patterns once
_COMPILED_RULES = [
    (re.compile(pattern, re.IGNORECASE), sections)
    for pattern, sections in SECTION_PRIOR_RULES
]


def get_section_prior(query: str) -> list[str]:
    """
    Given a query, return the list of preferred section substrings.
    Returns empty list if no rule matches (no boosting applied).
    """
    for pattern, sections in _COMPILED_RULES:
        if pattern.search(query):
            return sections
    return []


def compute_section_boost(
    chunk_section: str,
    preferred_sections: list[str],
    boost: float = 5.0,
) -> float:
    """
    Returns `boost` if chunk_section matches any preferred section,
    else 0.0.
    """
    if not preferred_sections:
        return 0.0

    section_lower = chunk_section.lower()

    for pref in preferred_sections:
        if pref.lower() in section_lower:
            return boost

    return 0.0
