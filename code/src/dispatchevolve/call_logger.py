from __future__ import annotations

import fcntl
import gzip
import json
import logging
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_EXCHANGE_RATE_CNY_TO_USD: float = 0.139


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from plain or gzip-compressed JSONL logs."""

    log_path = Path(path)
    opener = gzip.open if log_path.suffix == ".gz" else open
    with opener(log_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_model_prices() -> Dict[str, Any]:
    prices_path = Path(__file__).parent / "assets" / "model_prices.json"
    if not prices_path.exists():
        return {}
    try:
        with open(prices_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("_comment", None)
        global _EXCHANGE_RATE_CNY_TO_USD
        _EXCHANGE_RATE_CNY_TO_USD = data.pop("_exchange_rate_cny_to_usd", 0.139)
        return data
    except Exception as exc:
        logger.error(f"Failed to load model_prices.json: {exc}")
        return {}


MODEL_PRICES: Dict[str, Any] = _load_model_prices()


def _calculate_cost_from_prices(
    model: str, prompt_tokens: int, completion_tokens: int
) -> tuple[float, float, str]:
    entry = MODEL_PRICES.get(model)
    if not entry:
        return 0.0, 0.0, "unknown"

    currency = entry.get("currency", "USD")
    in_per_tok = entry.get("input_cost_per_token", 0)
    out_per_tok = entry.get("output_cost_per_token", 0)

    if currency == "CNY":
        cost_cny = prompt_tokens * in_per_tok + completion_tokens * out_per_tok
        cost_usd = cost_cny * _EXCHANGE_RATE_CNY_TO_USD
        return cost_cny, cost_usd, "CNY"
    else:
        cost_usd = prompt_tokens * in_per_tok + completion_tokens * out_per_tok
        cost_cny = cost_usd / _EXCHANGE_RATE_CNY_TO_USD
        return cost_cny, cost_usd, "USD"


def calculate_cost(response) -> tuple[float, float]:
    import litellm

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    model = getattr(response, "model", "") or ""

    try:
        cost_usd = litellm.completion_cost(completion_response=response)
        cost_cny = cost_usd / _EXCHANGE_RATE_CNY_TO_USD
        return cost_cny, cost_usd
    except Exception:
        pass

    bare_model = model
    for prefix in ("openai/", "gemini/", "vertex_ai/"):
        if bare_model.startswith(prefix):
            bare_model = bare_model[len(prefix):]
            break

    for lookup_name in [bare_model, model]:
        cost_cny, cost_usd, _ = _calculate_cost_from_prices(
            lookup_name, prompt_tokens, completion_tokens
        )
        if cost_cny > 0 or cost_usd > 0:
            return cost_cny, cost_usd

    entry = None
    for candidate in [bare_model, model]:
        if candidate in MODEL_PRICES:
            entry = MODEL_PRICES[candidate]
            break
    if not entry:
        for key, val in MODEL_PRICES.items():
            if bare_model.endswith(key) or key.endswith(bare_model):
                entry = val
                break

    if entry:
        try:
            from litellm.types.utils import CostPerToken
            usd_in = entry.get("cost_usd_per_token", {}).get("input", 0)
            usd_out = entry.get("cost_usd_per_token", {}).get("output", 0)
            if not usd_in and not usd_out:
                usd_in = entry.get("input_cost_per_token", 0)
                usd_out = entry.get("output_cost_per_token", 0)
                currency = entry.get("currency", "USD")
                if currency == "CNY":
                    usd_in = usd_in * _EXCHANGE_RATE_CNY_TO_USD
                    usd_out = usd_out * _EXCHANGE_RATE_CNY_TO_USD

            custom = CostPerToken(
                input_cost_per_token=usd_in,
                output_cost_per_token=usd_out,
            )
            cost_usd = litellm.completion_cost(
                completion_response=response, custom_cost_per_token=custom
            )
            cost_cny = cost_usd / _EXCHANGE_RATE_CNY_TO_USD
            return cost_cny, cost_usd
        except Exception:
            pass

    litellm_prices = getattr(litellm, "model_cost", {})
    for candidate in [model, bare_model, f"vertex_ai/{bare_model}", f"gemini/{bare_model}", f"openai/{bare_model}"]:
        if candidate in litellm_prices:
            info = litellm_prices[candidate]
            usd_in = info.get("input_cost_per_token", 0) or 0
            usd_out = info.get("output_cost_per_token", 0) or 0
            if usd_in or usd_out:
                cost_usd = prompt_tokens * usd_in + completion_tokens * usd_out
                cost_cny = cost_usd / _EXCHANGE_RATE_CNY_TO_USD
                return cost_cny, cost_usd

    return 0.0, 0.0


class CallLogger:
    _instance: Optional["CallLogger"] = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: Optional[str] = None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.log_dir = Path(log_dir or os.getenv("DISPATCHEVOLVE_LOG_DIR", "logs"))
        self.success_log = self.log_dir / "llm_calls.jsonl"
        self.failure_log = self.log_dir / "llm_failures.jsonl.gz"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        for p in (self.success_log, self.failure_log):
            if not p.exists():
                p.touch()

        self.session_calls: list[dict] = []
        self.session_cost_cny: float = 0.0
        self.session_cost_usd: float = 0.0

    def log_success(
        self,
        response,
        model: str,
        provider: str,
        prompt: str,
        response_text: str,
        latency_seconds: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        cost_cny, cost_usd = calculate_cost(response)

        record = {
            "timestamp": datetime.now().isoformat(),
            "event": "success",
            "model": model,
            "provider": provider,
            "prompt": prompt,
            "response": response_text,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            "cost_cny": round(cost_cny, 6),
            "cost_usd": round(cost_usd, 6),
            "latency_seconds": round(latency_seconds, 3),
            "metadata": metadata or {},
        }

        self.session_calls.append(record)
        self.session_cost_cny += cost_cny
        self.session_cost_usd += cost_usd

        self._append(self.success_log, record)

        logger.info(
            f"LLM call: model={model} tokens={prompt_tokens}+{completion_tokens} "
            f"cost=¥{cost_cny:.4f}/${cost_usd:.4f} latency={latency_seconds:.2f}s "
            f"session_total=¥{self.session_cost_cny:.4f}/${self.session_cost_usd:.4f}"
        )

    _MAX_PROMPT_CHARS = 4096
    _MAX_ERROR_CHARS = 4096

    def log_failure(
        self,
        model: str,
        provider: str,
        prompt: str,
        error: Exception,
        latency_seconds: float,
        retry_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        prompt_len = len(prompt)
        if prompt_len > self._MAX_PROMPT_CHARS:
            stored_prompt = (
                prompt[: self._MAX_PROMPT_CHARS]
                + f"...[truncated {prompt_len - self._MAX_PROMPT_CHARS} chars]"
            )
        else:
            stored_prompt = prompt

        error_message = str(error)
        if len(error_message) > self._MAX_ERROR_CHARS:
            error_message = (
                error_message[: self._MAX_ERROR_CHARS]
                + f"...[truncated {len(error_message) - self._MAX_ERROR_CHARS} chars]"
            )

        record = {
            "timestamp": datetime.now().isoformat(),
            "event": "failure",
            "model": model,
            "provider": provider,
            "prompt": stored_prompt,
            "prompt_chars": prompt_len,
            "error_type": type(error).__name__,
            "error_message": error_message,
            "latency_seconds": round(latency_seconds, 3),
            "retry_count": retry_count,
            "metadata": metadata or {},
        }

        self._append_gzip(self.failure_log, record)

        logger.warning(
            f"LLM call failed: model={model} error={type(error).__name__} "
            f"retries={retry_count} latency={latency_seconds:.2f}s"
        )

    def get_session_summary(self) -> Dict[str, Any]:
        if not self.session_calls:
            return {
                "total_calls": 0,
                "total_cost_cny": 0.0,
                "total_cost_usd": 0.0,
                "total_tokens": 0,
                "models_used": {},
            }

        models_used: Dict[str, Dict] = {}
        total_tokens = 0

        for call in self.session_calls:
            m = call["model"]
            tokens = call["tokens"]["total"]
            cost_cny = call["cost_cny"]
            cost_usd = call["cost_usd"]

            if m not in models_used:
                models_used[m] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_cny": 0.0,
                    "cost_usd": 0.0,
                }

            models_used[m]["calls"] += 1
            models_used[m]["tokens"] += tokens
            models_used[m]["cost_cny"] += cost_cny
            models_used[m]["cost_usd"] += cost_usd
            total_tokens += tokens

        return {
            "total_calls": len(self.session_calls),
            "total_cost_cny": round(self.session_cost_cny, 6),
            "total_cost_usd": round(self.session_cost_usd, 6),
            "total_tokens": total_tokens,
            "models_used": models_used,
        }

    def get_total_cost(self) -> Dict[str, float]:
        cost_cny = 0.0
        cost_usd = 0.0
        if self.success_log.exists():
            try:
                with open(self.success_log, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        cost_cny += record.get("cost_cny", 0.0)
                        cost_usd += record.get("cost_usd", 0.0)
            except Exception as exc:
                logger.error(f"Failed to read cost log: {exc}")

        return {"cost_cny": round(cost_cny, 6), "cost_usd": round(cost_usd, 6)}

    def print_session_summary(self) -> None:
        summary = self.get_session_summary()
        total = self.get_total_cost()

        print("\n" + "=" * 60)
        print("API Call Cost Report")
        print("=" * 60)
        print(f"Session:")
        print(f"  Calls:  {summary['total_calls']}")
        print(f"  Tokens: {summary['total_tokens']:,}")
        print(f"  Cost:   ¥{summary['total_cost_cny']:.4f} / ${summary['total_cost_usd']:.4f}")

        if summary["models_used"]:
            print(f"\nBy model:")
            for model, stats in summary["models_used"].items():
                print(
                    f"  {model}: {stats['calls']} calls  "
                    f"{stats['tokens']:,} tokens  "
                    f"¥{stats['cost_cny']:.4f} / ${stats['cost_usd']:.4f}"
                )

        print(f"\nHistorical total: ¥{total['cost_cny']:.4f} / ${total['cost_usd']:.4f}")
        print("=" * 60 + "\n")

    def reset_session(self) -> None:
        self.session_calls = []
        self.session_cost_cny = 0.0
        self.session_cost_usd = 0.0

    def _append(self, path: Path, record: dict) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error(f"Failed to append to {path}: {exc}")

    def _append_gzip(self, path: Path, record: dict) -> None:
        """Append one independently readable gzip member under a process lock."""

        payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        member = gzip.compress(payload, compresslevel=6, mtime=0)
        descriptor = None
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            view = memoryview(member)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("gzip log append made no progress")
                view = view[written:]
        except Exception as exc:
            logger.error(f"Failed to append compressed record to {path}: {exc}")
        finally:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)


_call_logger: Optional[CallLogger] = None


def get_call_logger(log_dir: Optional[str] = None) -> CallLogger:
    global _call_logger
    if _call_logger is None:
        _call_logger = CallLogger(log_dir=log_dir)
    return _call_logger


def track_success(
    response,
    model: str,
    provider: str,
    prompt: str,
    response_text: str,
    latency_seconds: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    get_call_logger().log_success(
        response=response,
        model=model,
        provider=provider,
        prompt=prompt,
        response_text=response_text,
        latency_seconds=latency_seconds,
        metadata=metadata,
    )


def track_failure(
    model: str,
    provider: str,
    prompt: str,
    error: Exception,
    latency_seconds: float,
    retry_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    get_call_logger().log_failure(
        model=model,
        provider=provider,
        prompt=prompt,
        error=error,
        latency_seconds=latency_seconds,
        retry_count=retry_count,
        metadata=metadata,
    )
