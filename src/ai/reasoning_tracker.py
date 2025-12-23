"""
AI Reasoning Tracker
====================

Logs and stores all AI decision rationale for:
1. Transparency and audit trail
2. Model improvement through analysis
3. Dashboard display
4. Debugging and backtesting

Each trading decision is recorded with:
- Input data (market context, indicators)
- AI analysis and reasoning
- Final decision and confidence
- Outcome tracking (for learning)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import threading


logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Types of trading decisions."""
    MARKET_ANALYSIS = "market_analysis"
    SIGNAL_GENERATION = "signal_generation"
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    RISK_ASSESSMENT = "risk_assessment"
    POSITION_SIZING = "position_sizing"


class DecisionOutcome(str, Enum):
    """Outcome of a trading decision."""
    PENDING = "pending"
    EXECUTED = "executed"
    SKIPPED = "skipped"
    PROFITABLE = "profitable"
    UNPROFITABLE = "unprofitable"
    BREAKEVEN = "breakeven"


@dataclass
class ReasoningEntry:
    """
    A single AI reasoning/decision entry.
    
    Records all context and rationale for a trading decision.
    """
    # Identification
    entry_id: str
    decision_type: DecisionType
    market_id: str
    market_name: str = ""
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Input context
    market_context: Dict[str, Any] = field(default_factory=dict)
    technical_indicators: Dict[str, Any] = field(default_factory=dict)
    portfolio_context: Dict[str, Any] = field(default_factory=dict)
    
    # AI analysis
    model_used: str = ""
    prompt_summary: str = ""
    ai_response: Dict[str, Any] = field(default_factory=dict)
    ai_confidence: float = 0.0
    ai_reasoning: str = ""
    
    # Component signals
    component_signals: List[Dict[str, Any]] = field(default_factory=list)
    
    # Final decision
    final_action: str = "HOLD"  # BUY, SELL, HOLD
    final_confidence: float = 0.0
    position_size: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Monte Carlo results (if applicable)
    monte_carlo_results: Dict[str, Any] = field(default_factory=dict)
    
    # Outcome tracking (updated later)
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    actual_pnl: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    outcome_timestamp: Optional[datetime] = None
    
    # Token usage tracking
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = {
            "entry_id": self.entry_id,
            "decision_type": self.decision_type.value,
            "market_id": self.market_id,
            "market_name": self.market_name,
            "timestamp": self.timestamp.isoformat(),
            "market_context": self.market_context,
            "technical_indicators": self.technical_indicators,
            "portfolio_context": self.portfolio_context,
            "model_used": self.model_used,
            "prompt_summary": self.prompt_summary,
            "ai_response": self.ai_response,
            "ai_confidence": self.ai_confidence,
            "ai_reasoning": self.ai_reasoning,
            "component_signals": self.component_signals,
            "final_action": self.final_action,
            "final_confidence": self.final_confidence,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "monte_carlo_results": self.monte_carlo_results,
            "outcome": self.outcome.value,
            "actual_pnl": self.actual_pnl,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "outcome_timestamp": self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ReasoningEntry":
        """Create from dictionary."""
        data = data.copy()
        data["decision_type"] = DecisionType(data["decision_type"])
        data["outcome"] = DecisionOutcome(data["outcome"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("outcome_timestamp"):
            data["outcome_timestamp"] = datetime.fromisoformat(data["outcome_timestamp"])
        return cls(**data)
    
    def get_summary(self) -> str:
        """Get a brief summary of this decision."""
        return (
            f"[{self.timestamp.strftime('%H:%M:%S')}] "
            f"{self.decision_type.value}: {self.final_action} "
            f"{self.market_name[:30]} @ {self.entry_price:.3f} "
            f"(conf: {self.final_confidence:.0%}, AI: {self.ai_reasoning[:50]}...)"
        )


class ReasoningTracker:
    """
    Tracks and persists AI decision reasoning.
    
    Provides:
    - In-memory ring buffer for recent entries
    - Daily log files for historical analysis
    - Query interface for dashboard
    - Outcome tracking for learning
    
    Usage:
        tracker = ReasoningTracker()
        entry = tracker.create_entry(
            decision_type=DecisionType.SIGNAL_GENERATION,
            market_id="market123",
            market_name="Bitcoin $100K",
        )
        entry.ai_reasoning = "Strong bullish signals..."
        tracker.save_entry(entry)
    """
    
    def __init__(
        self,
        log_dir: str = "data/ai_reasoning",
        max_memory_entries: int = 100,
    ):
        """
        Initialize the reasoning tracker.
        
        Args:
            log_dir: Directory for daily log files
            max_memory_entries: Maximum entries to keep in memory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_entries = max_memory_entries
        self._entries: List[ReasoningEntry] = []
        self._entries_by_id: Dict[str, ReasoningEntry] = {}
        self._lock = threading.Lock()
        
        self._entry_counter = 0
        
        # Load today's entries
        self._load_today()
    
    def _load_today(self):
        """Load today's entries from disk."""
        today_file = self.log_dir / f"{date.today().isoformat()}.jsonl"
        
        if today_file.exists():
            try:
                with open(today_file, "r") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            entry = ReasoningEntry.from_dict(data)
                            self._add_to_memory(entry)
                            
                logger.info(f"Loaded {len(self._entries)} reasoning entries from {today_file}")
            except Exception as e:
                logger.error(f"Failed to load reasoning log: {e}")
    
    def _add_to_memory(self, entry: ReasoningEntry):
        """Add entry to in-memory buffer."""
        with self._lock:
            self._entries.append(entry)
            self._entries_by_id[entry.entry_id] = entry
            
            # Trim if over limit
            while len(self._entries) > self.max_memory_entries:
                old = self._entries.pop(0)
                self._entries_by_id.pop(old.entry_id, None)
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID."""
        self._entry_counter += 1
        return f"ai_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self._entry_counter:04d}"
    
    def create_entry(
        self,
        decision_type: DecisionType,
        market_id: str,
        market_name: str = "",
    ) -> ReasoningEntry:
        """
        Create a new reasoning entry.
        
        Args:
            decision_type: Type of decision being made
            market_id: Market identifier
            market_name: Human-readable market name
            
        Returns:
            New ReasoningEntry (not yet saved)
        """
        entry = ReasoningEntry(
            entry_id=self._generate_entry_id(),
            decision_type=decision_type,
            market_id=market_id,
            market_name=market_name,
        )
        return entry
    
    def save_entry(self, entry: ReasoningEntry):
        """
        Save a reasoning entry to memory and disk.
        
        Args:
            entry: The entry to save
        """
        # Add to memory
        self._add_to_memory(entry)
        
        # Append to daily log file
        today_file = self.log_dir / f"{date.today().isoformat()}.jsonl"
        
        try:
            with open(today_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to save reasoning entry: {e}")
    
    def update_outcome(
        self,
        entry_id: str,
        outcome: DecisionOutcome,
        actual_pnl: float = 0.0,
        exit_price: float = 0.0,
        exit_reason: str = "",
    ):
        """
        Update the outcome of a previous decision.
        
        Called when a trade closes to record results.
        """
        with self._lock:
            if entry_id in self._entries_by_id:
                entry = self._entries_by_id[entry_id]
                entry.outcome = outcome
                entry.actual_pnl = actual_pnl
                entry.exit_price = exit_price
                entry.exit_reason = exit_reason
                entry.outcome_timestamp = datetime.utcnow()
                
                # Re-save to disk (append update)
                self._save_outcome_update(entry)
                
                logger.debug(f"Updated outcome for {entry_id}: {outcome.value}, P&L: {actual_pnl}")
    
    def _save_outcome_update(self, entry: ReasoningEntry):
        """Save outcome update to disk."""
        outcome_file = self.log_dir / "outcomes.jsonl"
        
        try:
            with open(outcome_file, "a") as f:
                update = {
                    "entry_id": entry.entry_id,
                    "outcome": entry.outcome.value,
                    "actual_pnl": entry.actual_pnl,
                    "exit_price": entry.exit_price,
                    "exit_reason": entry.exit_reason,
                    "outcome_timestamp": entry.outcome_timestamp.isoformat(),
                }
                f.write(json.dumps(update) + "\n")
        except Exception as e:
            logger.error(f"Failed to save outcome update: {e}")
    
    def get_recent(self, limit: int = 10) -> List[ReasoningEntry]:
        """Get most recent reasoning entries."""
        with self._lock:
            return list(self._entries[-limit:])
    
    def get_by_market(self, market_id: str, limit: int = 5) -> List[ReasoningEntry]:
        """Get recent entries for a specific market."""
        with self._lock:
            entries = [e for e in self._entries if e.market_id == market_id]
            return entries[-limit:]
    
    def get_by_id(self, entry_id: str) -> Optional[ReasoningEntry]:
        """Get specific entry by ID."""
        with self._lock:
            return self._entries_by_id.get(entry_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about AI decisions."""
        with self._lock:
            if not self._entries:
                return {
                    "total_entries": 0,
                    "decisions_today": 0,
                }
            
            today = date.today()
            today_entries = [e for e in self._entries if e.timestamp.date() == today]
            
            # Calculate outcome statistics
            executed = [e for e in self._entries if e.outcome == DecisionOutcome.EXECUTED]
            profitable = [e for e in self._entries if e.outcome == DecisionOutcome.PROFITABLE]
            unprofitable = [e for e in self._entries if e.outcome == DecisionOutcome.UNPROFITABLE]
            
            # Average confidence for different outcomes
            profitable_conf = sum(e.final_confidence for e in profitable) / len(profitable) if profitable else 0
            unprofitable_conf = sum(e.final_confidence for e in unprofitable) / len(unprofitable) if unprofitable else 0
            
            # Token usage
            total_tokens = sum(e.input_tokens + e.output_tokens for e in self._entries)
            
            return {
                "total_entries": len(self._entries),
                "decisions_today": len(today_entries),
                "executed_trades": len(executed),
                "profitable_trades": len(profitable),
                "unprofitable_trades": len(unprofitable),
                "win_rate": len(profitable) / len(executed) if executed else 0,
                "avg_profitable_confidence": profitable_conf,
                "avg_unprofitable_confidence": unprofitable_conf,
                "total_tokens_used": total_tokens,
                "avg_latency_ms": sum(e.latency_ms for e in self._entries) / len(self._entries) if self._entries else 0,
            }
    
    def get_for_dashboard(self, limit: int = None) -> List[Dict]:
        """
        Get entries formatted for dashboard display.
        
        Args:
            limit: Maximum entries to return (None = all entries)
        
        Returns:
            List of formatted entries with FULL reasoning text
        """
        with self._lock:
            entries = self._entries if limit is None else self._entries[-limit:]
        
        return [
            {
                "entry_id": e.entry_id,
                "time": e.timestamp.strftime("%H:%M:%S"),
                "date": e.timestamp.strftime("%Y-%m-%d"),
                "type": e.decision_type.value,
                "market": e.market_name,  # FULL market name
                "market_id": e.market_id,
                "action": e.final_action,
                "confidence": f"{e.final_confidence:.0%}",
                "ai_confidence": f"{e.ai_confidence:.0%}",
                "reasoning": e.ai_reasoning,  # FULL reasoning text - no truncation
                "prompt_summary": e.prompt_summary,
                "technical_indicators": e.technical_indicators,
                "component_signals": e.component_signals,
                "entry_price": e.entry_price,
                "stop_loss": e.stop_loss,
                "take_profit": e.take_profit,
                "position_size": e.position_size,
                "outcome": e.outcome.value,
                "actual_pnl": e.actual_pnl,
                "pnl": f"${e.actual_pnl:+.2f}" if e.outcome != DecisionOutcome.PENDING else "-",
                "exit_reason": e.exit_reason,
                "exit_price": e.exit_price,
                "input_tokens": e.input_tokens,
                "output_tokens": e.output_tokens,
                "latency_ms": e.latency_ms,
            }
            for e in reversed(entries)
        ]
    
    def export_for_analysis(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Export entries for ML analysis/backtesting.
        
        Reads from log files for the date range.
        """
        entries = []
        
        current = start_date
        while current <= end_date:
            log_file = self.log_dir / f"{current.isoformat()}.jsonl"
            
            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        if line.strip():
                            entries.append(json.loads(line))
            
            current = date.fromordinal(current.toordinal() + 1)
        
        return entries


# Singleton instance
_tracker_instance: Optional[ReasoningTracker] = None


def _get_project_root() -> Path:
    """Get the project root directory (where data/ folder lives)."""
    # Try to find project root by looking for data/ directory
    current = Path(__file__).resolve()
    
    # Go up from src/ai/reasoning_tracker.py to project root
    for _ in range(5):  # Max 5 levels up
        if (current / "data").exists():
            return current
        if (current / "config").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


def get_reasoning_tracker(log_dir: str = None) -> ReasoningTracker:
    """Get or create the global reasoning tracker instance.
    
    Uses absolute path resolution to ensure the tracker works
    regardless of which directory the code is run from.
    """
    global _tracker_instance
    if _tracker_instance is None:
        if log_dir is None:
            # Use absolute path based on project root
            project_root = _get_project_root()
            log_dir = str(project_root / "data" / "ai_reasoning")
        
        _tracker_instance = ReasoningTracker(log_dir=log_dir)
    return _tracker_instance


if __name__ == "__main__":
    # Test the reasoning tracker
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("REASONING TRACKER TEST")
    print("=" * 60)
    
    tracker = ReasoningTracker(log_dir="data/test_reasoning")
    
    # Create a test entry
    entry = tracker.create_entry(
        decision_type=DecisionType.SIGNAL_GENERATION,
        market_id="test_market_123",
        market_name="Will Bitcoin reach $100K by Dec 31?",
    )
    
    # Populate entry
    entry.market_context = {
        "current_price": 0.65,
        "volume_24h": 125000,
    }
    entry.ai_reasoning = "Strong bullish momentum detected. CUSUM shows positive regime shift, price above EWMA upper band. AI confidence high due to consistent technical signals and favorable sentiment."
    entry.ai_confidence = 0.75
    entry.final_action = "BUY"
    entry.final_confidence = 0.72
    entry.position_size = 2.0
    entry.entry_price = 0.65
    entry.stop_loss = 0.58
    entry.take_profit = 0.75
    entry.input_tokens = 850
    entry.output_tokens = 320
    entry.latency_ms = 1250
    
    # Save
    tracker.save_entry(entry)
    print(f"\nCreated entry: {entry.entry_id}")
    print(f"Summary: {entry.get_summary()}")
    
    # Update outcome
    tracker.update_outcome(
        entry_id=entry.entry_id,
        outcome=DecisionOutcome.PROFITABLE,
        actual_pnl=0.35,
        exit_price=0.71,
        exit_reason="take_profit",
    )
    
    print(f"\nUpdated outcome: {entry.outcome.value}, P&L: ${entry.actual_pnl:.2f}")
    
    # Get stats
    print("\nTracker Stats:")
    stats = tracker.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")