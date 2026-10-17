"""Tiered move confirmation.

Tier 1: the best move clearly beats the runner-up -> accept.
Tier 2: re-read a few more still frames and average the scores.
Tier 3: ask a vision LLM to pick from the top 3 (skipped when offline or without an API key).
Tier 4: ask the player.

Stockfish is never used to decide which move was played.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from enum import IntEnum

import chess
import cv2
import numpy as np

from .matcher import Candidate, MatchResult

logger = logging.getLogger(__name__)


class Tier(IntEnum):
    AUTO = 1
    VOTE = 2
    LLM = 3
    USER = 4


@dataclass
class ConfirmSettings:
    min_score: float = 500.0         # best move must explain at least this much change
    t1_margin: float = 1500.0        # best minus runner-up for auto-accept
    t2_margin: float = 800.0         # same, after averaging the re-reads
    max_unexplained: float = 0.35    # share of the change in piece-sized blobs the best move leaves unexplained
    max_off_board: float = 3000.0    # more change than this in blobs beyond the board edge = a hand is there
    vote_frames: int = 3             # extra still frames to read in tier 2


def tier1_ok(result: MatchResult, cfg: ConfirmSettings) -> bool:
    best = result.best
    return (best is not None and best.score >= cfg.min_score and result.margin >= cfg.t1_margin
            and result.unexplained <= cfg.max_unexplained and result.off_board <= cfg.max_off_board)


def average_results(results: list[MatchResult]) -> MatchResult:
    """Combine several reads of the same position change by averaging each move's score."""
    totals: dict[chess.Move, list[Candidate]] = {}
    for result in results:
        for cand in result.candidates:
            totals.setdefault(cand.move, []).append(cand)
    merged = []
    for move, cands in totals.items():
        merged.append(Candidate(
            move=move,
            score=float(np.mean([c.score for c in cands])),
            mask_px=float(np.mean([c.mask_px for c in cands])),
            explained=float(np.mean([c.explained for c in cands])),
        ))
    merged.sort(key=lambda c: c.score, reverse=True)
    total = float(np.mean([r.total_change for r in results]))
    out = MatchResult(merged, total, notes=[f"averaged {len(results)} reads"])
    if merged:
        out.margin = merged[0].score - (merged[1].score if len(merged) > 1 else 0.0)
        out.unexplained = float(np.mean([r.unexplained for r in results]))
        out.off_board = float(np.mean([r.off_board for r in results]))
        out.is_promotion = merged[0].move.promotion is not None
    return out


def tier2_ok(result: MatchResult, cfg: ConfirmSettings) -> bool:
    best = result.best
    return (best is not None and best.score >= cfg.min_score and result.margin >= cfg.t2_margin
            and result.unexplained <= cfg.max_unexplained and result.off_board <= cfg.max_off_board)


class LlmJudge:
    """Tier 3: show before/after photos and the top candidates to a vision model."""

    PROMPT = "Which of these moves was played? Reply with exactly one UCI move from the list."

    def __init__(self, model: str, enabled: bool = True, timeout_s: float = 30.0) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.client = None
        if not enabled:
            return
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            logger.info("Tier 3 (LLM) off: no ANTHROPIC_API_KEY in the environment or .env")
            return
        try:
            import anthropic

            self.client = anthropic.Anthropic(timeout=timeout_s, max_retries=1)
        except Exception as exc:
            logger.warning("Tier 3 (LLM) off: %s", exc)

    @property
    def available(self) -> bool:
        return self.client is not None

    @staticmethod
    def _jpeg_b64(image: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("couldn't encode image")
        return base64.standard_b64encode(buf.tobytes()).decode("ascii")

    def choose(self, board: chess.Board, before: np.ndarray, after: np.ndarray,
               candidates: list[chess.Move], orientation_note: str) -> chess.Move | None:
        if self.client is None or not candidates:
            return None
        import anthropic

        listed = "\n".join(f"- {m.uci()} ({board.san(m)})" for m in candidates)
        context = (
            f"A camera watches a physical chess board. {orientation_note}\n"
            f"Position before the move (FEN): {board.fen()}\n"
            f"{'White' if board.turn else 'Black'} was to move. Candidate moves:\n{listed}\n\n"
            f"{self.PROMPT}"
        )
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2048,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "BEFORE the move:"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": self._jpeg_b64(before)}},
                        {"type": "text", "text": "AFTER the move:"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": self._jpeg_b64(after)}},
                        {"type": "text", "text": context},
                    ],
                }],
            )
        except anthropic.APIConnectionError:
            logger.warning("Tier 3: can't reach the Anthropic API (offline?)")
            return None
        except anthropic.APIStatusError as exc:
            logger.warning("Tier 3: API error %s: %s", exc.status_code, exc.message)
            return None
        if response.stop_reason == "refusal":
            logger.warning("Tier 3: model declined")
            return None
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        by_uci = {m.uci(): m for m in candidates}
        answer = text.split()[0].strip(".,`'\"").lower() if text else ""
        if answer not in by_uci:
            logger.warning("Tier 3: answer %r isn't one of %s; ignoring it", text[:40], list(by_uci))
            return None
        logger.info("Tier 3: model picked %s", answer)
        return by_uci[answer]
