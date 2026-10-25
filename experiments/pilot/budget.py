"""
Pilot Budget Guard
==================

Hard cap on live spend. Before each uncached call the guard reserves that call's worst-case
cost; if the reservation would push spend past the cap, the call is refused and the run
stops cleanly.
"""

from __future__ import annotations

import asyncio


class BudgetExceeded(RuntimeError):
    """Raised when a call would push spend past the cap."""


class BudgetGuard:
    """Tracks committed spend plus in-flight worst-case reservations against a cap."""

    def __init__(self, max_usd: float, already_spent: float = 0.0) -> None:
        """
        Create a guard.

        Args:
            max_usd: Hard cap on spend in USD
            already_spent: Spend already recorded (e.g. earlier runs in the same cache)
        """
        if max_usd < 0:
            raise ValueError("max_usd must be non-negative")
        self.max_usd = max_usd
        self.spent = already_spent
        self.reserved = 0.0
        self.exhausted = False
        self._lock = asyncio.Lock()

    async def reserve(self, worst_case_usd: float) -> None:
        """
        Reserve a call's worst-case cost.

        Raises:
            BudgetExceeded: If spent + reservations + this call would exceed the cap
        """
        async with self._lock:
            if self.spent + self.reserved + worst_case_usd > self.max_usd:
                self.exhausted = True
                raise BudgetExceeded(
                    f"Call with worst-case cost ${worst_case_usd:.4f} would exceed the "
                    f"${self.max_usd:.2f} budget (spent ${self.spent:.4f}, "
                    f"in flight ${self.reserved:.4f})"
                )
            self.reserved += worst_case_usd

    async def settle(self, worst_case_usd: float, actual_usd: float) -> None:
        """Release a reservation and record the call's actual cost."""
        async with self._lock:
            self.reserved = max(0.0, self.reserved - worst_case_usd)
            self.spent += actual_usd

    async def release(self, worst_case_usd: float) -> None:
        """Release a reservation for a call that failed without being billed."""
        async with self._lock:
            self.reserved = max(0.0, self.reserved - worst_case_usd)

    @property
    def remaining(self) -> float:
        """Budget left after committed spend and reservations."""
        return self.max_usd - self.spent - self.reserved
