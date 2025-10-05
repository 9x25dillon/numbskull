from typing import Any, Dict, List
from .entropy_engine import entropy_engine  # type: ignore
from .matrix_processor import matrix_processor  # type: ignore
from .al_uls import al_uls
from .motif_engine import motif_engine  # type: ignore
from .suggestions import SUGGESTIONS  # type: ignore


def _prefix_match(prefix: str, state: str) -> List[str]:
    pre = (prefix or "").upper(); pool = SUGGESTIONS.get(state, [])
    return [t for t in pool if t.startswith(pre)]


def _apply_token_to_qgi(qgi: Dict[str, Any], token_text: str) -> None:
    # lightweight defaults if engines are stubs
    entropy_score = getattr(entropy_engine, "score_token", lambda x: 0.0)(token_text)
    volatility_signal = getattr(entropy_engine, "get_volatility_signal", lambda x: None)(token_text)
    qgi.setdefault("entropy_scores", []).append(entropy_score)
    qgi["volatility"] = volatility_signal
    if al_uls.is_symbolic_call(token_text):
        qgi.setdefault("symbolic_calls", []).append(al_uls.parse_symbolic_call(token_text))
    if hasattr(motif_engine, "detect_tags"):
        for t in motif_engine.detect_tags(token_text):
            if t not in qgi.setdefault("motif_tags", []):
                qgi["motif_tags"].append(t)


async def _apply_token_to_qgi_async(qgi: Dict[str, Any], token_text: str) -> None:
    _apply_token_to_qgi(qgi, token_text)
    if qgi.get("symbolic_calls"):
        last = qgi["symbolic_calls"][ -1]
        res = await al_uls.eval_symbolic_call_async(last)
        qgi.setdefault("symbolic_results", []).append(res)


def api_suggest(prefix: str = "", state: str = "S0", use_semantic: bool = True) -> Dict[str, Any]:
    qgi: Dict[str, Any] = {
        "state": state,
        "prefix": prefix,
        "selects": [],
        "filters": [],
        "group_by": [],
        "order": None,
        "tokens": [],
        "entropy_scores": [],
        "volatility": None,
        "symbolic_calls": [],
        "symbolic_results": [],
        "retrieval_routes": [],
        "motif_tags": []
    }
    qgi["tokens"].append(prefix)
    _apply_token_to_qgi(qgi, prefix)
    suggestions = matrix_processor.semantic_state_suggest(prefix, state) if use_semantic and getattr(matrix_processor, "available", lambda: False)() else _prefix_match(prefix, state)
    return {"suggestions": suggestions, "qgi": qgi}


async def api_suggest_async(prefix: str = "", state: str = "S0", use_semantic: bool = True) -> Dict[str, Any]:
    qgi: Dict[str, Any] = {
        "state": state,
        "prefix": prefix,
        "selects": [],
        "filters": [],
        "group_by": [],
        "order": None,
        "tokens": [],
        "entropy_scores": [],
        "volatility": None,
        "symbolic_calls": [],
        "symbolic_results": [],
        "retrieval_routes": [],
        "motif_tags": []
    }
    qgi["tokens"].append(prefix)
    await _apply_token_to_qgi_async(qgi, prefix)
    suggestions = matrix_processor.semantic_state_suggest(prefix, state) if use_semantic and getattr(matrix_processor, "available", lambda: False)() else _prefix_match(prefix, state)
    return {"suggestions": suggestions, "qgi": qgi}
