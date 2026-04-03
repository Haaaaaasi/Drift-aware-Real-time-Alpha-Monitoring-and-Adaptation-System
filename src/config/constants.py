"""System-wide constants for DARAMS."""

from enum import Enum


class BarType(str, Enum):
    DAILY = "daily"
    MIN_30 = "30min"
    MIN_5 = "5min"
    MIN_1 = "1min"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


class ModelStatus(str, Enum):
    SHADOW = "shadow"
    PRODUCTION = "production"
    RETIRED = "retired"


class AlertSeverity(str, Enum):
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class MonitorType(str, Enum):
    DATA = "data"
    ALPHA = "alpha"
    MODEL = "model"
    STRATEGY = "strategy"


class MetaSignalMethod(str, Enum):
    RULE_BASED = "rule_based"
    ML_META = "ml_meta"
    REGIME_ENSEMBLE = "regime_ensemble"


class AdaptationPolicy(str, Enum):
    SCHEDULED = "scheduled"
    PERFORMANCE_TRIGGERED = "performance_triggered"
    RECURRING_CONCEPT = "recurring_concept"


MVP_V1_ALPHA_IDS: list[str] = [
    "wq001", "wq002", "wq003", "wq004", "wq006",
    "wq008", "wq009", "wq012", "wq014", "wq018",
    "wq020", "wq023", "wq026", "wq028", "wq041",
]

LABEL_HORIZONS: list[int] = [1, 5, 10, 20]

DEFAULT_LOOKBACK_WINDOW: int = 252
IC_ROLLING_WINDOW: int = 20
SHARPE_ROLLING_WINDOW: int = 60
