# Adapted for DispatchEvolve; see THIRD_PARTY_NOTICES.md for origin and changes.
"""
LLM base interface — re-exported from dispatchevolve.provider.

This module exists for backward compatibility within the genetic optimizer.
All LLM interface definitions are consolidated in dispatchevolve.provider.
"""

from dispatchevolve.provider import LLMInterface

__all__ = ["LLMInterface"]
