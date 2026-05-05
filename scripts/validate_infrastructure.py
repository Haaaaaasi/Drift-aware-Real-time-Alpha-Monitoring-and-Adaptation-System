"""Validate DARAMS infrastructure and auto-apply migrations where possible.

Checks connectivity for PostgreSQL, Redis, and DolphinDB, then:
  - Runs 001_init_tables.sql on PostgreSQL if tables don't exist yet.
  - Seeds security_master if empty.
  - Prints a per-service status table and the next steps.

Usage:
    python scripts/validate_infrastructure.py
    python scripts/validate_infrastructure.py --skip-dolphindb
    python scripts/validate_infrastructure.py --run-migrations
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

# ── add project root to sys.path ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Connectivity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tcp_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_postgres(host: str, port: int, db: str, user: str, password: str) -> dict:
    result = {"name": "PostgreSQL", "host": f"{host}:{port}", "status": "FAIL", "detail": ""}
    if not _tcp_reachable(host, port):
        result["detail"] = "TCP connection refused — is docker-compose up?"
        return result
    try:
        import psycopg2
        conn = psycopg2.connect(host=host, port=port, dbname=db, user=user, password=password,
                                connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        ver = cur.fetchone()[0].split(",")[0]
        cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'")
        n_tables = cur.fetchone()[0]
        conn.close()
        result["status"] = "OK"
        result["detail"] = f"{ver} | {n_tables} tables in public schema"
        result["n_tables"] = n_tables
        result["conn"] = True
    except Exception as exc:
        result["detail"] = str(exc)
    return result


def check_redis(host: str, port: int) -> dict:
    result = {"name": "Redis", "host": f"{host}:{port}", "status": "FAIL", "detail": ""}
    if not _tcp_reachable(host, port):
        result["detail"] = "TCP connection refused — is docker-compose up?"
        return result
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host=host, port=port, socket_connect_timeout=5)
        info = r.ping()
        result["status"] = "OK" if info else "FAIL"
        result["detail"] = "PONG received"
    except Exception as exc:
        result["detail"] = str(exc)
    return result


def check_dolphindb(host: str, port: int) -> dict:
    result = {"name": "DolphinDB", "host": f"{host}:{port}", "status": "FAIL", "detail": ""}
    if not _tcp_reachable(host, port):
        result["detail"] = "TCP connection refused — is docker-compose up?"
        return result
    try:
        import dolphindb as ddb
        s = ddb.session()
        s.connect(host, port, "admin", "123456")
        ver = s.run("version()")
        s.close()
        result["status"] = "OK"
        result["detail"] = f"Server version: {ver}"
    except Exception as exc:
        result["detail"] = str(exc)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Migration helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_pg_migration(host: str, port: int, db: str, user: str, password: str,
                     sql_path: Path) -> bool:
    """Apply SQL migration file to PostgreSQL."""
    try:
        import psycopg2
        conn = psycopg2.connect(host=host, port=port, dbname=db, user=user,
                                password=password, connect_timeout=5)
        conn.autocommit = True
        cur = conn.cursor()
        sql = sql_path.read_text(encoding="utf-8")
        cur.execute(sql)
        conn.close()
        return True
    except Exception as exc:
        print(f"  [MIGRATION ERROR] {exc}")
        return False


def seed_security_master(host: str, port: int, db: str, user: str, password: str) -> None:
    """Insert placeholder rows into security_master if empty."""
    try:
        import psycopg2
        conn = psycopg2.connect(host=host, port=port, dbname=db, user=user,
                                password=password, connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM security_master")
        if cur.fetchone()[0] > 0:
            conn.close()
            return

        # 20 placeholder Taiwan stocks
        symbols = [
            ("2330", "TSMC", "Semiconductors", 1),
            ("2317", "Hon Hai", "Electronics", 2),
            ("2454", "MediaTek", "Semiconductors", 1),
            ("2412", "Chunghwa Telecom", "Telecoms", 3),
            ("2308", "Delta Electronics", "Electronics", 2),
            ("2382", "Quanta Computer", "Electronics", 2),
            ("2357", "ASUSTeK", "Electronics", 2),
            ("2303", "UMC", "Semiconductors", 1),
            ("2002", "China Steel", "Materials", 4),
            ("1301", "Formosa Plastics", "Materials", 4),
            ("2881", "Fubon Financial", "Financials", 5),
            ("2882", "Cathay Financial", "Financials", 5),
            ("2886", "Mega Financial", "Financials", 5),
            ("1303", "Nan Ya Plastics", "Materials", 4),
            ("2912", "President Chain Store", "Consumer", 6),
            ("4904", "Far EasTone", "Telecoms", 3),
            ("6505", "Formosa Petrochemical", "Materials", 4),
            ("1101", "Taiwan Cement", "Materials", 4),
            ("2474", "Catcher Technology", "Electronics", 2),
            ("2049", "Hiwin Technologies", "Industrials", 7),
        ]
        for sid, name, sector, icode in symbols:
            cur.execute(
                """
                INSERT INTO security_master
                    (security_id, exchange, currency, industry_code, is_active)
                VALUES (%s, 'TWSE', 'TWD', %s, TRUE)
                ON CONFLICT (security_id) DO NOTHING
                """,
                (sid, icode),
            )
        conn.commit()
        conn.close()
        print(f"  [OK] Seeded {len(symbols)} rows into security_master")
    except Exception as exc:
        print(f"  [SEED ERROR] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _print_status_table(results: list[dict]) -> None:
    w = 60
    print(f"\n{'='*w}")
    print(f"  {'Service':<15} {'Host':<20} {'Status':<8} Detail")
    print(f"  {'-'*13} {'-'*18} {'-'*6} {'-'*28}")
    for r in results:
        icon = "[OK]" if r["status"] == "OK" else ("[SKIP]" if r["status"] == "SKIP" else "[FAIL]")
        print(f"  {r['name']:<15} {r['host']:<20} {icon:<8} {r['detail'][:45]}")
    print(f"{'='*w}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DARAMS infrastructure")
    parser.add_argument("--pg-host", default="127.0.0.1")
    parser.add_argument("--pg-port", type=int, default=5433)
    parser.add_argument("--pg-db", default="darams")
    parser.add_argument("--pg-user", default="darams")
    parser.add_argument("--pg-password", default="darams_dev")
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--ddb-host", default="127.0.0.1")
    parser.add_argument("--ddb-port", type=int, default=8848)
    parser.add_argument("--skip-dolphindb", action="store_true",
                        help="Skip DolphinDB check (if not yet deployed)")
    parser.add_argument("--run-migrations", action="store_true",
                        help="Auto-apply PostgreSQL migration if tables are missing")
    args = parser.parse_args()

    print("\nDARAMS Infrastructure Validator")
    print("Checking services …")

    results = []

    # PostgreSQL
    pg = check_postgres(args.pg_host, args.pg_port, args.pg_db, args.pg_user, args.pg_password)
    results.append(pg)

    # Redis
    rd = check_redis(args.redis_host, args.redis_port)
    results.append(rd)

    # DolphinDB (optional)
    if not args.skip_dolphindb:
        ddb = check_dolphindb(args.ddb_host, args.ddb_port)
        results.append(ddb)
    else:
        results.append({"name": "DolphinDB", "host": f"{args.ddb_host}:{args.ddb_port}",
                        "status": "SKIP", "detail": "skipped via --skip-dolphindb"})

    _print_status_table(results)

    # ── Auto-migration ───────────────────────────────────────────────────────
    pg_ok = pg.get("status") == "OK"

    if pg_ok and args.run_migrations:
        n_tables = pg.get("n_tables", -1)
        migration_path = Path(__file__).resolve().parents[1] / "migrations" / "001_init_tables.sql"
        if n_tables == 0:
            print(f"[MIGRATION] Applying {migration_path.name} …")
            ok = run_pg_migration(
                args.pg_host, args.pg_port, args.pg_db, args.pg_user, args.pg_password,
                migration_path,
            )
            if ok:
                print("  [OK] Migration applied successfully.")
                seed_security_master(args.pg_host, args.pg_port, args.pg_db,
                                     args.pg_user, args.pg_password)
            else:
                print("  [FAIL] Migration failed — check the error above.")
        else:
            print(f"[MIGRATION] {n_tables} tables already exist — skipping.")
            seed_security_master(args.pg_host, args.pg_port, args.pg_db,
                                 args.pg_user, args.pg_password)

    elif pg_ok and not args.run_migrations:
        n_tables = pg.get("n_tables", -1)
        if n_tables == 0:
            print("[HINT] PostgreSQL is up but has 0 tables.")
            print("       Re-run with --run-migrations to apply 001_init_tables.sql automatically.")

    # ── Next steps ───────────────────────────────────────────────────────────
    all_ok = all(r["status"] in ("OK", "SKIP") for r in results)

    print("Next steps:")
    if not pg_ok or not rd.get("status") == "OK":
        print("  1. Start Docker services:")
        print("       docker-compose up -d")
        print("  2. Wait ~10 seconds then re-run this script:")
        print("       python scripts/validate_infrastructure.py --run-migrations")
    elif all_ok:
        print("  All services are up!")
        print()
        print("  Run the real data pipeline:")
        print("    a) Ensure TEJ parquet exists:")
        print("         python scripts/ingest_tej_csv.py")
        print("    b) Run pipeline with TEJ + Python alphas:")
        print("         python -m pipelines.daily_batch_pipeline --data-source tej")
        print("    c) Generate backtest report:")
        print("         python scripts/generate_report.py --csv data/tw_stocks_tej.parquet")
        print()
        print("  Or run with synthetic data (no dependencies):")
        print("    python -m pipelines.daily_batch_pipeline --synthetic")
        print("    python scripts/generate_report.py")

    if not args.skip_dolphindb and results[-1]["status"] != "OK":
        print()
        print("  DolphinDB setup (once PG+Redis are up):")
        print("    • docker exec -it darams-dolphindb ddb_script /scripts/setup_database.dos")
        print("    • docker exec -it darams-dolphindb ddb_script /scripts/load_wq101alpha.dos")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
