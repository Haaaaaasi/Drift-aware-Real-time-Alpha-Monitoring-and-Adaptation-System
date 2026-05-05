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


# 正式研究預設使用 TEJ survivorship-correct 資料；csv/yfinance 僅保留作為 demo 對照。
DEFAULT_DATA_SOURCE: str = "tej"
DATA_SOURCE_DEFAULT_PATHS: dict[str, str] = {
    "csv": "data/tw_stocks_ohlcv.csv",
    "tej": "data/tw_stocks_tej.parquet",
}

MVP_V1_ALPHA_IDS: list[str] = [
    "wq001", "wq002", "wq003", "wq004", "wq006",
    "wq008", "wq009", "wq012", "wq014", "wq018",
    "wq020", "wq023", "wq026", "wq028", "wq041",
]

# 全部 101 個 WQ101 alpha ID（wq001..wq101）— MVP v3 擴充使用
WQ101_ALL_ALPHA_IDS: list[str] = [f"wq{n:03d}" for n in range(1, 102)]

# 需要 indclass panel 的 alpha 子集合（產業分類；目前以 hash-based proxy 取代真實產業碼）
WQ101_INDCLASS_ALPHA_IDS: list[str] = [
    "wq048", "wq058", "wq059", "wq063", "wq067", "wq069", "wq070",
    "wq076", "wq079", "wq080", "wq082", "wq087", "wq089", "wq090",
    "wq091", "wq093", "wq097", "wq100",
]

# 純量價 alpha（不需產業碼）— 83 個
WQ101_PURE_PRICE_ALPHA_IDS: list[str] = [
    aid for aid in WQ101_ALL_ALPHA_IDS if aid not in set(WQ101_INDCLASS_ALPHA_IDS)
]

# Deprecated: 舊 DolphinDB/yfinance 時代的 45-alpha 常數，只保留給歷史相容。
# 正式研究請一律透過 src.config.alpha_selection.load_effective_alpha_ids()
# 讀取 reports/alpha_ic_analysis/effective_alphas.json（TEJ IS-only 64 alphas）。
V3_EFFECTIVE_ALPHA_IDS: list[str] = [
    "wq040", "wq061", "wq026", "wq073", "wq044", "wq014", "wq024", "wq006",
    "wq043", "wq064", "wq021", "wq084", "wq055", "wq007", "wq071", "wq001",
    "wq087", "wq012", "wq035", "wq101", "wq016", "wq038", "wq041", "wq057",
    "wq062", "wq054", "wq009", "wq015", "wq003", "wq079", "wq013", "wq050",
    "wq017", "wq074", "wq042", "wq083", "wq045", "wq018", "wq023", "wq002",
    "wq072", "wq095", "wq099", "wq081", "wq004",
]

LABEL_HORIZONS: list[int] = [1, 5, 10, 20]

DEFAULT_LOOKBACK_WINDOW: int = 252
IC_ROLLING_WINDOW: int = 20
SHARPE_ROLLING_WINDOW: int = 60
