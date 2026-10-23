import sqlite3
import os
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

PKT = timezone(timedelta(hours=5))
DB_PATH = "hospital_analytics.db"

analytics_router = APIRouter()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ==============================================================================
# DATABASE INITIALIZATION & 30-DAY TIMELINE SEEDER
# ==============================================================================
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patient_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        date_str TEXT,
        day_name TEXT,
        time_str TEXT,
        hour_int INTEGER,
        total_patients INTEGER,
        seated INTEGER,
        standing INTEGER,
        empty_chairs INTEGER
    )
    """)
    conn.commit()

    # Check if database has less than 7 days of history. If so, seed full 30 days!
    cursor.execute("SELECT COUNT(DISTINCT date_str) as day_count FROM patient_snapshots")
    row = cursor.fetchone()
    day_count = row["day_count"] if row else 0

    if day_count < 7:
        print("[Analytics DB] Seeding 30 days of historical shift analytics into SQLite...")
        now = datetime.now(PKT)
        records = []

        # Shift templates
        weekday_pattern = [
            ("08:00 AM", 8, 14, 11, 3),
            ("09:00 AM", 9, 27, 20, 7),  # Morning Surge
            ("11:00 AM", 11, 23, 17, 6),
            ("01:00 PM", 13, 12, 10, 2),  # Lunch dip
            ("03:00 PM", 15, 24, 18, 6),  # Afternoon Surge
            ("06:00 PM", 18, 15, 12, 3),
            ("09:00 PM", 21, 8, 6, 2)
        ]
        weekend_pattern = [
            ("09:00 AM", 9, 10, 8, 2),
            ("12:00 PM", 12, 13, 10, 3),
            ("03:00 PM", 15, 11, 9, 2),
            ("07:00 PM", 19, 7, 6, 1)
        ]

        # Generate 30 past days from today backwards
        for days_back in range(30, -1, -1):
            past_date = now - timedelta(days=days_back)
            date_str = past_date.strftime("%Y-%m-%d")
            day_name = past_date.strftime("%A")
            is_weekend = day_name in ["Saturday", "Sunday"]
            pattern = weekend_pattern if is_weekend else weekday_pattern

            for t_str, h_int, tot, s, st in pattern:
                # Add natural fluctuation: Mondays and Fridays have higher surges
                surge_adj = 3 if day_name in ["Monday", "Friday"] else 0
                final_tot = max(2, tot + surge_adj - (days_back % 3))
                final_s = int(final_tot * 0.75)
                final_st = final_tot - final_s

                records.append((
                    f"{date_str} {t_str}",
                    date_str,
                    day_name,
                    t_str,
                    h_int,
                    final_tot,
                    final_s,
                    final_st,
                    max(0, 67 - final_tot)
                ))

        cursor.executemany("""
        INSERT INTO patient_snapshots 
        (timestamp, date_str, day_name, time_str, hour_int, total_patients, seated, standing, empty_chairs)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()

    conn.close()

init_db()

# ==============================================================================
# REAL-TIME LOGGING (CALLED LIVE BY YOLO STREAM)
# ==============================================================================
last_logged_time_str = ""

def record_live_snapshot(total: int, seated: int, standing: int):
    global last_logged_time_str
    now = datetime.now(PKT)
    cur_time_str = now.strftime("%I:%M %p")
    
    if cur_time_str == last_logged_time_str:
        return
    last_logged_time_str = cur_time_str

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO patient_snapshots 
    (timestamp, date_str, day_name, time_str, hour_int, total_patients, seated, standing, empty_chairs)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now.strftime("%Y-%m-%d %H:%M:%S"),
        now.strftime("%Y-%m-%d"),
        now.strftime("%A"),
        cur_time_str,
        now.hour,
        total,
        seated,
        standing,
        max(0, 67 - total)
    ))
    conn.commit()
    conn.close()

# ==============================================================================
# API 1: RECENT TIMELINE BAR CHART
# ==============================================================================
@analytics_router.get("/api/analytics_history")
async def get_analytics_history():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT day_name, time_str, total_patients, seated, standing
    FROM patient_snapshots
    ORDER BY id DESC
    LIMIT 16
    """)
    recent_rows = cursor.fetchall()
    recent_rows = list(reversed(recent_rows))

    categories = [f"{r['day_name'][:3]} {r['time_str']}" for r in recent_rows]
    totals = [r["total_patients"] for r in recent_rows]
    seated_arr = [r["seated"] for r in recent_rows]
    standing_arr = [r["standing"] for r in recent_rows]

    cursor.execute("SELECT day_name, time_str, MAX(total_patients) as peak FROM patient_snapshots")
    peak_row = cursor.fetchone()
    peak_info = f"{peak_row['day_name']} at {peak_row['time_str']} ({peak_row['peak']} Patients)" if peak_row and peak_row["peak"] else "Normal Flow"

    conn.close()
    return JSONResponse({
        "categories": categories,
        "totals": totals,
        "seated": seated_arr,
        "standing": standing_arr,
        "peak_info": peak_info
    })

# ==============================================================================
# API 2: MULTI-DAY / 30-DAY CONTINUOUS TREND LINE CHART
# ==============================================================================
@analytics_router.get("/api/trend_data")
async def get_trend_data(range_type: str = Query("30d")):
    conn = get_db_connection()
    cursor = conn.cursor()

    if range_type == "24h":
        # Hourly breakdown for the last 24 records
        cursor.execute("""
        SELECT time_str as label, total_patients, seated, standing
        FROM patient_snapshots
        ORDER BY id DESC
        LIMIT 24
        """)
        rows = list(reversed(cursor.fetchall()))
        categories = [r["label"] for r in rows]
        totals = [r["total_patients"] for r in rows]
        seated = [r["seated"] for r in rows]

    elif range_type == "7d":
        # Day-by-day aggregates for past 7 days
        cursor.execute("""
        SELECT date_str, day_name, MAX(total_patients) as peak_val, ROUND(AVG(total_patients), 1) as avg_val
        FROM patient_snapshots
        GROUP BY date_str
        ORDER BY date_str DESC
        LIMIT 7
        """)
        rows = list(reversed(cursor.fetchall()))
        categories = [f"{r['day_name'][:3]} ({r['date_str'][-5:]})" for r in rows]
        totals = [r["peak_val"] for r in rows]
        seated = [r["avg_val"] for r in rows]

    else:  # "30d" (Default: Past 1 Month)
        cursor.execute("""
        SELECT date_str, day_name, MAX(total_patients) as peak_val, ROUND(AVG(total_patients), 1) as avg_val
        FROM patient_snapshots
        GROUP BY date_str
        ORDER BY date_str DESC
        LIMIT 30
        """)
        rows = list(reversed(cursor.fetchall()))
        categories = [f"{r['date_str'][-5:]} ({r['day_name'][:3]})" for r in rows]
        totals = [r["peak_val"] for r in rows]
        seated = [r["avg_val"] for r in rows]

    conn.close()

    return JSONResponse({
        "categories": categories,
        "totals": totals,
        "seated": seated,
        "range": range_type
    })
