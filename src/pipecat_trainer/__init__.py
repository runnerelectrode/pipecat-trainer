"""pipecat-trainer: make any Pipecat bot trainable.

Two lines in a bot (the LLM slot and the observer), a sandbox that runs Pipecat's own eval scenarios as
rollouts, and a scenario converter. The training loop itself is polyloop; polyvoice wires this package's
outputs into it. See README.md.
"""
from pipecat_trainer.llm import PolyvoiceLLMService, polyvoice_llm
from pipecat_trainer.observer import PolyvoiceObserver

__all__ = ["PolyvoiceLLMService", "PolyvoiceObserver", "polyvoice_llm"]
__version__ = "0.1.0"
