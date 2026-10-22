"""Names vulture reports as unused that are used: by a framework, or kept on purpose.

.venv/bin/vulture chessreader tests tools tools/vulture_whitelist.py --min-confidence 60
"""
# ruff: noqa: B018, F821

from chessreader.openings import OpeningBook
from chessreader.settings import CalibrationSettings

CalibrationSettings.divisible_by_8  # pydantic field validator: pydantic calls it
OpeningBook.positions  # public: the book's size, quoted in the docs
_.comment  # python-chess GameNode.comment: written into the PGN
_.trust_env  # requests.Session.trust_env = False keeps camera traffic off campus proxies
