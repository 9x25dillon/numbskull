import os
import json
import asyncio
from typing import Dict, Any, List, Tuple
import websockets

JULIA_WS_URL = os.environ.get("JULIA_WS_URL", "ws://localhost:8089")
CACHE_TTL_WS = float(os.environ.get("ALULS_WS_TTL", 30))

class TTLCacheWS:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self._store: Dict[Tuple[str, Tuple[str, ...]], Tuple[float, Dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    def _now(self) -> float:
        return asyncio.get_event_loop().time()

    def _key(self, name: str, args: List[str]) -> Tuple[str, Tuple[str, ...]]:
        return (name.upper(), tuple(args))

    def get(self, name: str, args: List[str]) -> Dict[str, Any] | None:
        k = self._key(name, args)
        v = self._store.get(k)
        if not v:
            self.misses += 1; return None
        ts, data = v
        if self._now() - ts <= self.ttl:
            self.hits += 1; return data
        self._store.pop(k, None)
        self.misses += 1; return None

    def set(self, name: str, args: List[str], value: Dict[str, Any]) -> None:
        self._store[self._key(name, args)] = (self._now(), value)

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses, "ttl": self.ttl}

class ALULSWSClient:
    def __init__(self, ws_url: str | None = None):
        self.ws_url = ws_url or JULIA_WS_URL
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.cache = TTLCacheWS(CACHE_TTL_WS)

    async def connect(self):
        if (self.websocket is None) or self.websocket.closed:
            self.websocket = await websockets.connect(self.ws_url)
        return self.websocket

    async def _roundtrip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ws = await self.connect()
            await ws.send(json.dumps(payload))
            resp = await ws.recv()
cursor/bc-f408c7bd-bc2a-48a4-bc8d-0989f628ad52-ef2e
            # Server may wrap results, standardize here
            data = json.loads(resp)
            if isinstance(data, dict) and "result" in data and "type" in data:
                # unwrap eval_result
                if data.get("type") == "eval_result":
                    return data.get("result", data)
                if data.get("type") == "parse_result":
                    return data
            return data
            return json.loads(resp)

        except Exception as e:
            # Reset socket on error to force reconnect later
            try:
                if self.websocket:
                    await self.websocket.close()
            finally:
                self.websocket = None
            return {"ok": False, "error": str(e)}

    async def parse(self, text: str) -> Dict[str, Any]:
        return await self._roundtrip({"type": "parse", "text": text})

    async def eval(self, name: str, args: List[str]) -> Dict[str, Any]:
        cached = self.cache.get(name, args)
        if cached is not None:
            return {**cached, "_cached": True}
        res = await self._roundtrip({"type": "eval", "name": name, "args": args})
        if isinstance(res, dict) and res.get("ok"):
            self.cache.set(name, args, res)
        return res

    async def batch_eval(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # try a single WS roundtrip; if it fails or invalid, fall back per-call
        res = await self._roundtrip({"type": "batch_eval", "calls": calls})
        if isinstance(res, dict) and "results" in res and isinstance(res["results"], list):
            # populate cache for successes
            out: List[Dict[str, Any]] = []
            for c, r in zip(calls, res["results"]):
                if isinstance(r, dict) and r.get("ok"):
                    self.cache.set(c.get("name", ""), c.get("args", []), r)
                out.append(r if isinstance(r, dict) else {"ok": False, "error": "invalid item"})
            return out
        # fallback: per-call
        return [await self.eval(c.get("name", ""), c.get("args", [])) for c in calls]

al_uls_ws_client = ALULSWSClient()
