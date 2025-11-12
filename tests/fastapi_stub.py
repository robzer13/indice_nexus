"""Minimal FastAPI stub for offline unit tests."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str | Dict[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail if detail is not None else ""


class _Route:
    def __init__(self, method: str, path: str, handler: Callable[..., Any]) -> None:
        self.method = method.upper()
        self.path = path
        self.handler = handler
        self.parts = [part for part in path.strip("/").split("/") if part]


class FastAPI:
    def __init__(self, title: str | None = None, version: str | None = None) -> None:
        self.title = title or "FastAPI Stub"
        self.version = version or "0"
        self._routes: List[_Route] = []
        self._middleware: List[Tuple[Any, Dict[str, Any]]] = []

    def add_middleware(self, middleware: Any, **options: Any) -> None:
        self._middleware.append((middleware, options))

    def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register("GET", path)

    def _register(self, method: str, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes.append(_Route(method, path, func))
            return func

        return decorator

    def resolve(self, method: str, path: str) -> Tuple[Callable[..., Any], Dict[str, str]]:
        candidate_segments = [segment for segment in path.strip("/").split("/") if segment]
        for route in self._routes:
            if route.method != method.upper():
                continue
            if len(route.parts) != len(candidate_segments):
                continue
            params: Dict[str, str] = {}
            matched = True
            for route_part, candidate in zip(route.parts, candidate_segments):
                if route_part.startswith("{") and route_part.endswith("}"):
                    params[route_part.strip("{} ")] = candidate
                elif route_part == candidate:
                    continue
                else:
                    matched = False
                    break
            if matched:
                return route.handler, params
        raise HTTPException(404, detail="Not Found")


class _Response:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = int(status_code)
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class TestClient:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> _Response:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
        query_params: Dict[str, Any] = {}
        if parsed.query:
            for key, values in parse_qs(parsed.query).items():
                if values:
                    query_params[key] = values[0]
        if params:
            query_params.update(params)

        path = parsed.path or "/"
        try:
            handler, path_params = self.app.resolve("GET", path)
        except HTTPException as exc:
            return _Response(exc.status_code, {"detail": exc.detail})

        call_kwargs = {**path_params, **query_params}
        annotations = getattr(handler, "__annotations__", {})
        for key, value in list(call_kwargs.items()):
            annotation = annotations.get(key)
            if annotation is int and isinstance(value, str):
                try:
                    call_kwargs[key] = int(value)
                except ValueError:
                    continue
            elif annotation is float and isinstance(value, str):
                try:
                    call_kwargs[key] = float(value)
                except ValueError:
                    continue
            elif annotation is bool and isinstance(value, str):
                lowered = value.lower()
                if lowered in {"true", "1", "yes", "on"}:
                    call_kwargs[key] = True
                elif lowered in {"false", "0", "no", "off"}:
                    call_kwargs[key] = False

        try:
            result = handler(**call_kwargs)
        except HTTPException as exc:
            return _Response(exc.status_code, {"detail": exc.detail})
        except Exception as exc:  # pragma: no cover - defensive stub fallback
            return _Response(500, {"detail": str(exc)})

        return _Response(200, result)


class _CORSMiddleware:
    def __init__(self, app: FastAPI, **_: Any) -> None:
        self.app = app


middleware = SimpleNamespace(cors=SimpleNamespace(CORSMiddleware=_CORSMiddleware))
testclient = SimpleNamespace(TestClient=TestClient)


__all__ = ["FastAPI", "HTTPException", "middleware", "testclient", "TestClient"]
