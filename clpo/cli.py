import argparse
import pathlib
import sys
from loguru import logger
from rich.logging import RichHandler

from .config import CLPOConfig, check_api_keys
from .orchestrator import CLPO_Orchestrator

# ─────────────────────────── Main ───────────────────────────────────────
def main() -> None:
    """
    Main entry point for the CLPO-B Prompt Evolution tool.

    Parses command-line arguments to configure the prompt evolution process, sets up logging,
    validates API keys, and initializes the orchestrator. Supports resuming from a previous
    state and allows overriding the halving schedule.

    Command-line arguments:
        --sh           : Override halving schedule (format: "1,1;2,2;3,3").
        --log-level    : Set logging level (default: "INFO").
        --resume       : Resume from the specified state file if it exists.
        --state-file   : Path to the JSONL state file (default: "clpo_state.jsonl").
    """
    parser = argparse.ArgumentParser(description="CLPO-B Prompt Evolution")
    parser.add_argument(
        "--sh", type=str, help="Override halving schedule, e.g. 1,1;2,2;3,3"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from clpo_state.jsonl if it exists",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="clpo_state.jsonl",
        help="Path to JSONL state file",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(RichHandler(), level=args.log_level.upper())

    cfg_kwargs: dict[str, object] = {"human_in_the_loop": False}
    if args.sh:
        try:
            cfg_kwargs["sh_levels"] = [
                tuple(map(int, x.split(","))) for x in args.sh.split(";")
            ]
        except ValueError as e:
            logger.error("Invalid --sh format: %s", e)
            sys.exit(1)

    BASE_PROMPT = (
        "You are a meticulous planner. Given any high-level objective, produce a numbered "
        "step-by-step plan (10-25 steps). Each step ≤ 25 words and must state resources, "
        "timeline, and potential risks."
    )

    cfg = CLPOConfig(**cfg_kwargs)
    if not check_api_keys(cfg):
        sys.exit(1)

    orch = CLPO_Orchestrator(
        cfg, BASE_PROMPT, state_file=pathlib.Path(args.state_file), resume=args.resume
    )
    with orch:
        if not (args.resume and orch._load_state()):
            # only prepare when we could not resume
            orch.prepare()
        orch.run()


if __name__ == "__main__":
    main()
