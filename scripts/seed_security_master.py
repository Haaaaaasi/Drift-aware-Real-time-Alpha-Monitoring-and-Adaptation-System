"""Seed security_master table with sample TWSE stocks."""

from __future__ import annotations

import psycopg2.extras

from src.common.db import get_pg_connection
from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("seed_security_master")

SAMPLE_STOCKS = [
    ("2330", "台積電", "TWSE", 26, "1994-09-05"),
    ("2317", "鴻海", "TWSE", 26, "1991-06-18"),
    ("2454", "聯發科", "TWSE", 26, "2001-07-23"),
    ("2308", "台達電", "TWSE", 26, "1988-12-19"),
    ("2881", "富邦金", "TWSE", 14, "2001-12-19"),
    ("2882", "國泰金", "TWSE", 14, "2001-12-19"),
    ("2891", "中信金", "TWSE", 14, "2003-01-02"),
    ("2303", "聯電", "TWSE", 26, "1985-07-26"),
    ("1301", "台塑", "TWSE", 13, "1964-10-28"),
    ("1303", "南亞", "TWSE", 13, "1972-01-10"),
    ("2412", "中華電", "TWSE", 27, "2000-10-27"),
    ("3711", "日月光投控", "TWSE", 26, "2018-04-30"),
    ("2886", "兆豐金", "TWSE", 14, "2002-02-04"),
    ("2884", "玉山金", "TWSE", 14, "2002-02-04"),
    ("2357", "華碩", "TWSE", 26, "1996-09-28"),
    ("2382", "廣達", "TWSE", 26, "1999-11-08"),
    ("2345", "智邦", "TWSE", 26, "1997-01-08"),
    ("3008", "大立光", "TWSE", 26, "2002-02-07"),
    ("2002", "中鋼", "TWSE", 10, "1971-12-23"),
    ("1326", "台化", "TWSE", 13, "1984-08-13"),
]


def seed():
    conn = get_pg_connection()
    try:
        sql = """
            INSERT INTO security_master
                (security_id, name, exchange, industry_code, listing_date, is_active)
            VALUES (%s, %s, %s, %s, %s, TRUE)
            ON CONFLICT (security_id) DO UPDATE SET
                name = EXCLUDED.name,
                industry_code = EXCLUDED.industry_code,
                updated_at = now()
        """
        records = [(s[0], s[1], s[2], s[3], s[4]) for s in SAMPLE_STOCKS]
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records)
        conn.commit()
        logger.info("security_master_seeded", count=len(records))
    finally:
        conn.close()


if __name__ == "__main__":
    seed()
    print(f"Seeded {len(SAMPLE_STOCKS)} securities")
