"""Subprocess helper: load one model and run a tiny inference request."""

from __future__ import annotations

import sys
import traceback

from recsys_nle.nl_explanations.llm import HuggingFaceLLMClient

_EXPECTED_CLI_ARGS = 2
_SMOKE_MESSAGES = [{"role": "user", "content": "Reply with one short greeting."}]


def main() -> None:
    """Load ``model_id`` and run one tiny generation; exit 0 on success."""
    if len(sys.argv) != _EXPECTED_CLI_ARGS:
        sys.stderr.write("Usage: hf_model_load_smoke_worker.py <model_id>\n")
        sys.exit(2)
    model_id = sys.argv[1]
    client: HuggingFaceLLMClient | None = None
    try:
        client = HuggingFaceLLMClient(model_id=model_id)
        _ = client.generate(
            _SMOKE_MESSAGES,
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
    except Exception:  # noqa: BLE001 — report any load failure to parent process
        traceback.print_exc()
        sys.exit(1)
    finally:
        if client is not None:
            client.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
