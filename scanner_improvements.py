"""
Scanner Improvements — 6 upgrades for better market reading
1. News Filter — auto-suppress signals during red folder events
2. Session Quality Score — boost London/NY open signals
3. Entry Validation — skip auto-grade if entry already missed
4. 1H Candle Confirmation — verify direction before alerting
5. Spread Check — block signals during wide spread conditions
6. Consecutive Loss Protection — warn after 2 losses in a row
"""

import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ─── PIP SPECIFICATION SYSTEM ────────────────────────────────────────────────
# Single source of truth for pip size, minimum SL distance, and ATR threshold.
# All functions that need pair-specific tolerances should call get_pip_spec().

PIP_SPECS = {
    "EURUSD": {"pip": 0.0001, "min_sl": 0.0012, "min_atr": 0.00035},  # 12 pips
    "GBPUSD": {"pip": 0.0001, "min_sl": 0.0015, "min_atr": 0.00042},  # 15 pips — most volatile forex
    "AUDUSD": {"pip": 0.0001, "min_sl": 0.0010, "min_atr": 0.00035},  # 10 pips
    "NZDUSD": {"pip": 0.0001, "min_sl": 0.0010, "min_atr": 0.00035},  # 10 pips
    "USDCAD": {"pip": 0.0001, "min_sl": 0.0012, "min_atr": 0.00049},  # 12 pips
    "USDCHF": {"pip": 0.0001, "min_sl": 0.0010, "min_atr": 0.00049},  # 10 pips
    "USDJPY": {"pip": 0.01,   "min_sl": 0.08,   "min_atr": 0.035, "max_lots": 0.50},    # 8 pips JPY
    "EURJPY": {"pip": 0.01,   "min_sl": 0.15,   "min_atr": 0.08},
    "GBPJPY": {"pip": 0.01,   "min_sl": 0.15,   "min_atr": 0.10},
    "XAUUSD": {"pip": 0.01,   "min_sl": 12.0,   "min_atr": 15.0},    # 12 points — gold needs room;
                                                                     # min_atr raised 2.0->15.0, see
                                                                     # scanner.py _futures_thr comment
                                                                     # for the research this matches
    "XAGUSD": {"pip": 0.001,  "min_sl": 0.05,   "min_atr": 0.03},
    "US100":  {"pip": 1.0, "min_sl": 80.0,  "min_atr": 20.0, "pip_size": 1.0, "pip_value": 1.0, "min_sl_pips": 80,  "max_sl_pips": 300, "digits": 2, "unit": "pts"},
    "US30":   {"pip": 1.0, "min_sl": 60.0,  "min_atr": 17.0, "pip_size": 1.0, "pip_value": 1.0, "min_sl_pips": 60,  "max_sl_pips": 250, "digits": 2, "unit": "pts"},
    "US500":  {"pip": 0.1, "min_sl":  8.0,  "min_atr":  0.7, "pip_size": 0.1, "pip_value": 0.5, "min_sl_pips": 20,  "max_sl_pips": 150, "digits": 2, "unit": "pts"},
    "USOIL":  {"pip": 0.01, "min_sl": 0.40, "min_atr": 0.20},
}

_FOREX_PIP_SPEC_PAIRS = {k for k, v in PIP_SPECS.items() if v["pip"] <= 0.01 and k not in ("XAUUSD", "XAGUSD")}

# Minimum wick pierce required to count as a genuine sweep (raw price units).
# Scaled proportionally to each pair's real volatility (≈ 20% of min_sl reference).
# Previously flat 1 pip for all forex, 0 for XAUUSD → false sweeps on GBPUSD/XAUUSD/USOIL.
_SWEEP_PIERCE_BUFFER: dict[str, float] = {
    "EURUSD": 0.0002,  # 2 pips  — 17% of 12-pip min SL
    "GBPUSD": 0.0003,  # 3 pips  — 20% of 15-pip min SL; previously 1 pip caused false sweeps
    "AUDUSD": 0.0002,  # 2 pips  — 20% of 10-pip min SL
    "NZDUSD": 0.0002,  # 2 pips  — 20% of 10-pip min SL
    "USDCAD": 0.0002,  # 2 pips  — 17% of 12-pip min SL
    "USDCHF": 0.0002,  # 2 pips  — 20% of 10-pip min SL
    "USDJPY": 0.02,    # 2 pips  — 25% of 8-pip min SL
    "EURJPY": 0.02,    # 2 pips  — JPY pairs same scale
    "GBPJPY": 0.02,    # 2 pips  — JPY pairs same scale
    "XAUUSD": 3.0,     # 3 pts   — 10% of 30-pt min SL; previously 0, any wick qualified
    "USOIL":  0.05,    # 5 cents — 12.5% of $0.40 min SL; previously $0.01 (too small at $65-85)
}


# ─── TP1 R-MULTIPLE TARGETS ───────────────────────────────────────────────────
# Single source of truth for per-pair TP1 targets. validate_risk_reward() reads
# this directly so the safety floor always matches the actual construction target.
# Sub-threshold pairs (walk-forward FX study 2003-2025, 23 rolling windows —
# Sharpe below 0.5): reverted to 1.5R pending live fills. Not permanent judgments.
TP1_MULTIPLIER: dict[str, float] = {
    "default": 2.0,
    "GBPUSD": 1.5,   # Sharpe 0.31; widest SL band (15-25 pip) compounds the misfit
    "AUDUSD": 1.5,   # Sharpe 0.09 — weakest pair tested, "weak across every strategy"
    "USDCAD": 1.5,   # Sharpe 0.14 — carry-cost-eroded, sub-threshold all windows
}


def _tp1_mult(symbol: str) -> float:
    """Return the TP1 R-multiple for the given symbol."""
    return TP1_MULTIPLIER.get(symbol.upper(), TP1_MULTIPLIER["default"])


def get_pip_spec(symbol: str) -> dict:
    """Return pip spec for symbol, defaulting to standard 4dp forex if unknown."""
    return PIP_SPECS.get(symbol.upper(), {"pip": 0.0001, "min_sl": 0.0010, "min_atr": 0.0007})


# Max Asian range sizes (in pip units) above which the Judas Swing edge degrades.
# Wide ranges produce noisy sweeps with wide SLs and poor RR.
_MAX_ASIA_RANGE_PIPS = {
    "EURUSD": 60,
    "GBPUSD": 70,
    "XAUUSD": 1200,
    "USDJPY": 60,
    "AUDUSD": 50,
    "NZDUSD": 50,
    "USDCAD": 55,
    "USDCHF": 50,
    "US100":  2000,  # indices have wide overnight ranges by nature — don't filter on this
    "US30":   1500,  # previous day H/L sweep is the reference, not Asia range tightness
    "US500":  500,
}
_DEFAULT_MAX_ASIA_RANGE_PIPS = 60


def is_asia_range_tight(symbol: str, asia_high: float, asia_low: float) -> tuple[bool, str]:
    """
    Return (is_tight, reason) for a given Asia session range.
    Tight ranges produce the highest-probability Judas Swing setups.
    Wide ranges (> max threshold for the pair) produce noisy sweeps with
    poor RR — ATR is wide, SL must be wide, and TP1 becomes hard to reach.
    """
    if asia_high <= asia_low:
        return True, "no range data"
    sym = symbol.upper()
    spec = PIP_SPECS.get(sym, {"pip": 0.0001})
    pip_size = spec["pip"]
    range_pips = (asia_high - asia_low) / pip_size
    max_pips = _MAX_ASIA_RANGE_PIPS.get(sym, _DEFAULT_MAX_ASIA_RANGE_PIPS)
    if range_pips > max_pips:
        return False, f"Asia range {range_pips:.0f}p > max {max_pips}p — wide range reduces Judas edge"
    return True, f"Asia range {range_pips:.0f}p — tight"


# ─── 1. NEWS FILTER ───────────────────────────────────────────────────────────

# Shrunk from 45/30 (75 min total self-imposed buffer). FTMO's Challenge/Verification
# phase has zero news-trading restriction; the funded Standard-account rule (2 min
# each side) is the only compliance requirement that will ever apply. 5/5 covers real
# execution/slippage risk around the release itself without eating legitimate
# news-driven setups the old window blocked. Revisit if funded on Standard.
NEWS_BLOCK_MINUTES_BEFORE = 5
NEWS_BLOCK_MINUTES_AFTER  = 5

# Minimal fallback — only used when ALL live feeds fail
FALLBACK_NEWS = []

# 60-minute cache for today's event list
_FF_CACHE: dict = {"events": None, "fetched_at": 0.0}
_FF_CACHE_TTL = 3600


def _et_to_utc_minutes(hour: int, minute: int) -> int:
    """Convert Eastern Time (EDT=UTC-4, EST=UTC-5) to UTC minutes-since-midnight."""
    month = datetime.now(timezone.utc).month
    offset = 4 if 3 <= month <= 11 else 5   # EDT Mar-Nov, EST Dec-Feb
    utc = hour * 60 + minute + offset * 60
    return utc % (24 * 60)


def is_nfp_friday() -> bool:
    """Return True if today is the first Friday of the month (NFP release day)."""
    from datetime import date, timedelta
    today = date.today()
    if today.weekday() != 4:  # 4 = Friday
        return False
    first_day = today.replace(day=1)
    days_to_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_to_friday)
    return today == first_friday


# Known FOMC announcement dates (second day of each meeting, 2:00 PM ET = 18:00/19:00 UTC)
_FOMC_DATES_2026 = {
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-10",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
}

# Finnhub returns country codes — map to currency for pair matching
_COUNTRY_TO_CURRENCY = {
    'US': 'USD',
    'EU': 'EUR', 'DE': 'EUR', 'FR': 'EUR', 'IT': 'EUR', 'ES': 'EUR',
    'GB': 'GBP',
    'JP': 'JPY',
    'AU': 'AUD',
    'NZ': 'NZD',
    'CA': 'CAD',
    'CH': 'CHF',
}

_TRADEABLE_CURRENCIES = {'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF'}

# Which currencies are relevant when scanning a given symbol
_SYMBOL_CURRENCIES: dict = {
    'EURUSD': {'EUR', 'USD'}, 'GBPUSD': {'GBP', 'USD'},
    'USDJPY': {'USD', 'JPY'}, 'AUDUSD': {'AUD', 'USD'},
    'NZDUSD': {'NZD', 'USD'}, 'USDCAD': {'USD', 'CAD'},
    'USDCHF': {'USD', 'CHF'}, 'EURJPY': {'EUR', 'JPY'},
    'GBPJPY': {'GBP', 'JPY'}, 'XAUUSD': {'USD'},
    'XAGUSD': {'USD'}, 'GC': {'USD'}, 'MGC': {'USD'},
    'ES': {'USD'}, 'MES': {'USD'}, 'NQ': {'USD'}, 'MNQ': {'USD'},
    'RTY': {'USD'}, 'YM': {'USD'}, 'CL': {'USD'}, 'MCL': {'USD'},
    'US100': {'USD'}, 'US30': {'USD'},
    'US500': {'USD'},
}


def _fetch_forexfactory_json() -> list:
    """
    Fetch today's high/medium impact events from ForexFactory weekly JSON feed.
    Returns list of dicts: {time_utc, currency, event, impact}.
    Feed returns dates as full ISO 8601 datetimes in Eastern Time (e.g. 2026-06-25T08:30:00-04:00).
    Parse the full datetime and convert to UTC directly — no separate date/time fields.
    """
    from datetime import timezone as _tz
    now = datetime.now(timezone.utc)
    events: list = []
    # Cache for 2 hours to avoid rate limiting (429) from repeated hourly hits
    _ff_cache_key = now.strftime("%Y-%m-%d-%H") if now.hour % 2 == 0 else now.replace(hour=now.hour - 1).strftime("%Y-%m-%d-%H")
    if hasattr(_fetch_forexfactory_json, '_cache') and _fetch_forexfactory_json._cache.get('key') == _ff_cache_key:
        logger.info(f"[news] ForexFactory cache hit — {len(_fetch_forexfactory_json._cache['data'])} events")
        return _fetch_forexfactory_json._cache['data']
    try:
        resp = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        for item in resp.json():
            impact_raw = (item.get("impact") or "").lower()
            if impact_raw not in ("high", "medium"):
                continue
            currency = (item.get("currency") or item.get("country") or "").upper()
            if currency not in _TRADEABLE_CURRENCIES:
                continue
            date_str = (item.get("date") or "").strip()
            if not date_str:
                continue
            try:
                # Feed returns full ISO 8601 datetime with ET offset e.g. 2026-06-25T08:30:00-04:00
                from datetime import datetime as _dt
                event_dt_local = _dt.fromisoformat(date_str)
                # Convert to UTC
                event_dt_utc = event_dt_local.astimezone(_tz.utc)
                # Only include today's events (UTC date)
                if event_dt_utc.date() != now.date():
                    continue
                time_utc = event_dt_utc.replace(second=0, microsecond=0)
            except Exception:
                continue
            events.append({
                "time_utc": time_utc,
                "currency": currency,
                "impact": impact_raw,
                "event": item.get("title", ""),
            })
        logger.info(f"[news] ForexFactory backup: {len(events)} events today")
    except Exception as e:
        logger.warning(f"[news] ForexFactory JSON failed: {e}")
    _fetch_forexfactory_json._cache = {'key': _ff_cache_key, 'data': events}
    return events


def _inject_monday_block(events: list, now: datetime) -> list:
    """Add static Monday 8:30 AM ET block if today is Monday and not already present."""
    if now.weekday() != 0:
        return events
    if any(e.get("event") == "Monday US/CAD Data Window" for e in events):
        return events
    et_offset = 4 if 3 <= now.month <= 11 else 5
    utc_h = (8 + et_offset) % 24  # 12 UTC in EDT, 13 UTC in EST
    events.append({
        "time_utc": now.replace(hour=utc_h, minute=30, second=0, microsecond=0),
        "currency": "USD",
        "event": "Monday US/CAD Data Window",
        "impact": "high",
    })
    logger.info(f"[news] Static Monday block: {utc_h:02d}:30 UTC — US/CAD data window added")
    return events


def _smart_hardcoded_fallback() -> list:
    """Try ForexFactory, then fall back to NFP/FOMC hardcoded events."""
    import time as _t
    now = datetime.now(timezone.utc)

    events = _fetch_forexfactory_json()

    if not events:
        today_iso = now.date().isoformat()
        if is_nfp_friday():
            nfp_utc_h = 12 if (3 <= now.month <= 11) else 13
            events.append({
                "time_utc": now.replace(hour=nfp_utc_h, minute=30, second=0, microsecond=0),
                "currency": "USD",
                "event": "Non-Farm Payrolls",
                "impact": "high",
            })
            logger.info(f"[news] Hardcoded: NFP Friday detected — blocking {nfp_utc_h}:30 UTC")
        if today_iso in _FOMC_DATES_2026:
            events.append({
                "time_utc": now.replace(hour=18, minute=0, second=0, microsecond=0),
                "currency": "USD",
                "event": "FOMC Rate Decision",
                "impact": "high",
            })
            logger.info("[news] Hardcoded: FOMC date detected — blocking 18:00 UTC")

        # Block all 8:30 AM ET (12:30 UTC) releases on known high-impact days
        # Core PCE is last business day of month, GDP is end of quarter
        # Rather than hardcoding dates, block 12:30 UTC on ANY day where
        # Finnhub AND ForexFactory both failed — conservative safety net
        # This prevents trading during the most common USD release window
        day_of_week = now.weekday()  # 0=Monday, 4=Friday
        # Live ForexFactory + Finnhub feeds handle news blocking — no hardcoded day blocks

    # Monday block removed — ForexFactory calendar handles this dynamically
    # events = _inject_monday_block(events, now)
    _FF_CACHE.update({"events": events, "fetched_at": _t.time()})
    return events


def fetch_forexfactory_today() -> list:
    """
    Fetch today's high-impact events from Finnhub economic calendar. Caches result for 60 minutes.
    Returns list of dicts: {time_utc, currency, event, impact}.
    Falls back to smart hardcoded fallback (NFP/FOMC only) when Finnhub is unavailable.
    """
    import time as _t
    now = datetime.now(timezone.utc)

    if _FF_CACHE["events"] is not None and (_t.time() - _FF_CACHE["fetched_at"]) < _FF_CACHE_TTL:
        return _FF_CACHE["events"]

    try:
        today = now.strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        url = (
            f"https://finnhub.io/api/v1/calendar/economic"
            f"?from={today}&to={tomorrow}"
            f"&token=d8k8lehr01qjgd6soecgd8k8lehr01qjgd6soed0"
        )

        response = requests.get(url, timeout=5)
        data = response.json()

        events: list = []
        for event in data.get('economicCalendar', []):
            raw_impact = event.get('impact', '')
            if raw_impact in ('high', '3'):
                impact = 'high'
            elif raw_impact in ('medium', '2'):
                impact = 'medium'
            else:
                continue

            country = event.get('country', '').upper()
            currency = _COUNTRY_TO_CURRENCY.get(country)
            if not currency or currency not in _TRADEABLE_CURRENCIES:
                continue

            time_str = event.get('time', '')
            try:
                if 'T' in time_str:
                    time_utc = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                elif time_str:
                    time_utc = datetime.strptime(time_str[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                else:
                    continue
                if time_utc.date() != now.date():
                    continue
            except Exception:
                continue
            events.append({
                'time_utc': time_utc,
                'currency': currency,
                'impact': impact,
                'event': event.get('event', ''),
            })
            logger.info(f"[news] {impact.capitalize()} impact event: {event.get('event')} at {time_str} ({currency})")

        high_count = sum(1 for e in events if e['impact'] == 'high')
        med_count = sum(1 for e in events if e['impact'] == 'medium')
        logger.info(f"[news] Finnhub calendar: {high_count} high, {med_count} medium impact events today")

        if high_count == 0:
            logger.info("[news] Finnhub returned 0 high-impact events — trying ForexFactory backup")
            ff_events = _fetch_forexfactory_json()
            if ff_events:
                existing_keys = {(e['currency'], e['event']) for e in events}
                for fe in ff_events:
                    if (fe['currency'], fe['event']) not in existing_keys:
                        events.append(fe)
                        existing_keys.add((fe['currency'], fe['event']))

        # Monday block removed — ForexFactory calendar handles this dynamically
    # events = _inject_monday_block(events, now)
        _FF_CACHE.update({"events": events, "fetched_at": _t.time()})
        return events

    except Exception as e:
        logger.warning(f"[news] Finnhub failed: {e} — using ForexFactory/hardcoded fallback")
        return _smart_hardcoded_fallback()


def is_news_window(symbol: str = "") -> tuple[bool, str, str]:
    """
    Check if current time is within a news window for the given symbol.
    Returns (is_blocked, block_reason, medium_warning).
    HIGH impact → block signal entirely (is_blocked=True).
    MEDIUM impact → warn only (is_blocked=False, medium_warning set).
    Filters to currencies relevant to symbol; no filtering when symbol is empty.
    """
    now = datetime.now(timezone.utc)
    cur = now.hour * 60 + now.minute
    relevant = _SYMBOL_CURRENCIES.get(symbol.upper(), set()) if symbol else set()
    medium_warning = ""
    try:
        for e in fetch_forexfactory_today():
            if relevant and e['currency'] not in relevant:
                continue
            ev = e["time_utc"].hour * 60 + e["time_utc"].minute
            if e['impact'] == 'high':
                if -NEWS_BLOCK_MINUTES_BEFORE <= (cur - ev) <= NEWS_BLOCK_MINUTES_AFTER:
                    reason = (
                        f"High impact news: {e['currency']} {e['event']} "
                        f"at {e['time_utc'].strftime('%H:%M')} UTC"
                    )
                    return True, reason, ""
            elif e['impact'] == 'medium' and not medium_warning:
                if -30 <= (cur - ev) <= 30:
                    medium_warning = (
                        f"Medium impact news within 30 min — "
                        f"{e['currency']} {e['event']} "
                        f"at {e['time_utc'].strftime('%H:%M')} UTC"
                    )
    except Exception:
        pass
    return False, "", medium_warning


def get_next_news_event() -> str:
    """Return description of next high-impact event today."""
    now = datetime.now(timezone.utc)
    cur = now.hour * 60 + now.minute
    try:
        upcoming = sorted(
            (e for e in fetch_forexfactory_today()
             if e["time_utc"].hour * 60 + e["time_utc"].minute > cur),
            key=lambda e: e["time_utc"].hour * 60 + e["time_utc"].minute,
        )
        if upcoming:
            e = upcoming[0]
            ev_min = e["time_utc"].hour * 60 + e["time_utc"].minute
            return (
                f"{e['currency']} {e['event']} in {ev_min - cur} minutes "
                f"({e['time_utc'].strftime('%H:%M')} UTC)"
            )
    except Exception:
        pass
    return "No more high-impact events today"


def check_upcoming_news(lookahead_minutes: int = 45) -> tuple[bool, str, int]:
    """
    Check if high-impact news is within lookahead_minutes in the future.
    Returns (news_approaching, reason, minutes_until).
    """
    now = datetime.now(timezone.utc)
    cur = now.hour * 60 + now.minute
    try:
        for e in fetch_forexfactory_today():
            ev = e["time_utc"].hour * 60 + e["time_utc"].minute
            mins_until = ev - cur
            if 0 < mins_until <= lookahead_minutes:
                return (
                    True,
                    f"High impact news: {e['currency']} {e['event']} "
                    f"at {e['time_utc'].strftime('%H:%M')} UTC",
                    mins_until,
                )
    except Exception:
        pass
    return False, "", 0


# ─── 2. SESSION QUALITY SCORE ─────────────────────────────────────────────────

SESSION_MULTIPLIERS = {
    "London Open": 1.5,   # 7-10 UTC — best session
    "NY Open": 1.3,       # 13-16 UTC — second best
    "London": 1.1,        # 10-13 UTC — decent
    "NY": 1.0,            # 16-21 UTC — normal
    "Asian": 0.7,         # 0-7 UTC — avoid
    "Off-Session": 0.5,   # Outside hours
}

SESSION_SCORE_BONUS = {
    "London Open": 1,
    "NY Open": 1,
    "London": 0,
    "NY": 0,
    "Asian": -1,
    "Off-Session": -2,
}

def get_session_score_bonus(session: str) -> int:
    """Return score bonus/penalty based on session quality."""
    return SESSION_SCORE_BONUS.get(session, 0)

def get_session_quality(session: str) -> str:
    """Return human readable session quality."""
    multiplier = SESSION_MULTIPLIERS.get(session, 1.0)
    if multiplier >= 1.3:
        return "PRIME"
    elif multiplier >= 1.0:
        return "GOOD"
    elif multiplier >= 0.7:
        return "SLOW"
    else:
        return "AVOID"


# ─── 3. ENTRY VALIDATION ──────────────────────────────────────────────────────

ENTRY_MAX_PIPS_FOREX = 25      # 25 pips max deviation for forex
# Raised from 15 — ICT OB zones are 10-20 pips wide on 15M.
# Distance measured from OB MID not edge — 25 pip tolerance covers zone edge entries.
# 150+ pip misses still blocked correctly. Catches 20-25 pip timing gaps.
ENTRY_MAX_POINTS_GOLD = 50     # 50 points max deviation for gold
# Gold OB zones are 30-80 points wide on 15M — 50pt tolerance covers zone edge entries
ENTRY_MAX_POINTS_FUTURES = 200  # fallback flat tolerance (used when sl_dist not available)

def validate_entry(symbol: str, entry_price: float, current_price: float, direction: str = "BUY", sl_dist: float | None = None) -> tuple[bool, float]:
    """
    Check if price has moved past the entry in the wrong direction beyond max deviation.
    For BUY: only block if price dropped BELOW entry (OB broken to downside).
    For SELL: only block if price rallied ABOVE entry (OB broken to upside).
    Price above entry on BUY and price below entry on SELL are valid limit-order states.
    Returns (is_valid, deviation).
    """
    deviation = abs(current_price - entry_price)
    sym = symbol.upper()

    if sym in ("XAUUSD", "GC", "MGC", "XAGUSD"):
        max_dev = ENTRY_MAX_POINTS_GOLD
    elif sym in ("ES", "MES", "NQ", "MNQ", "RTY", "YM", "CL", "MCL", "NG", "US100", "US30", "US500"):
        # Scale tolerance to the trade's own SL distance — a 60pt-SL trade and a
        # 300pt-SL trade should not share a flat 200pt ceiling. 50% of SL distance
        # with a 50pt floor prevents unreasonably tight blocks on very tight SLs.
        max_dev = max(sl_dist * 0.5, 50) if sl_dist else ENTRY_MAX_POINTS_FUTURES
    else:
        # Forex (standard and JPY) — use pip spec so JPY pairs get correct scaling
        max_dev = get_pip_spec(sym)["pip"] * ENTRY_MAX_PIPS_FOREX

    if direction == "BUY":
        # Only invalid if price is below the entry by more than max_dev
        broken = current_price < entry_price - max_dev
    else:
        # Only invalid if price is above the entry by more than max_dev
        broken = current_price > entry_price + max_dev

    return not broken, round(deviation, 5)


# ─── 4. 1H CANDLE CONFIRMATION ────────────────────────────────────────────────

def check_1h_candle_confirmation(candles_1h: list, direction: str) -> tuple[bool, str]:
    """
    Verify the most recent 1H candle closes in the signal direction.
    Returns (confirmed, reason).
    """
    if not candles_1h or len(candles_1h) < 2:
        return True, "Insufficient 1H data — proceeding"

    latest = candles_1h[0]
    candle_direction = "bullish" if latest["close"] > latest["open"] else "bearish"
    expected = "bullish" if direction == "BUY" else "bearish"

    if candle_direction == expected:
        body_size = abs(latest["close"] - latest["open"])
        total_range = latest["high"] - latest["low"]
        body_ratio = body_size / total_range if total_range > 0 else 0

        if body_ratio >= 0.5:
            return True, f"Strong 1H {candle_direction} confirmation ({body_ratio:.0%} body)"
        else:
            return True, f"Weak 1H {candle_direction} confirmation ({body_ratio:.0%} body)"
    else:
        return False, f"1H candle is {candle_direction} — conflicts with {direction} signal"


# ─── 5. SPREAD CHECK ──────────────────────────────────────────────────────────

MAX_SPREADS = {
    "XAUUSD": 0.50,    # 50 cents on gold
    "EURUSD": 0.00015, # 1.5 pips
    "GBPUSD": 0.00020, # 2 pips
    "USDJPY": 0.020,   # 2 pips
    "AUDUSD": 0.00020, # 2 pips
    "ES": 0.25,        # 1 tick
    "NQ": 0.25,        # 1 tick
    "CL": 0.02,        # 2 cents crude
    "GC": 0.50,        # 50 cents gold futures
}

def check_spread(symbol: str, bid: float, ask: float) -> tuple[bool, float]:
    """
    Check if spread is within acceptable range.
    Returns (is_acceptable, spread).
    """
    spread = abs(ask - bid)
    sym = symbol.upper()
    # Pip-spec pairs: max spread = 2 pips in that pair's pip units
    if sym in PIP_SPECS:
        max_spread = PIP_SPECS[sym]["pip"] * 2
    else:
        max_spread = MAX_SPREADS.get(sym, 0.001)
    is_ok = spread <= max_spread
    return is_ok, round(spread, 5)


def estimate_spread_from_candles(candles: list) -> float:
    """Estimate current spread from recent candle data."""
    if not candles:
        return 0.0
    # Use smallest recent candle body as spread estimate
    recent = candles[:5]
    bodies = [abs(c["close"] - c["open"]) for c in recent]
    return min(bodies) if bodies else 0.0


# ─── 6. CONSECUTIVE LOSS PROTECTION ──────────────────────────────────────────

def check_consecutive_losses(user_id: int, max_losses: int = 2) -> tuple[bool, int]:
    """
    Check if user has hit consecutive loss limit.
    Returns (should_warn, consecutive_count).
    """
    try:
        import psycopg2, os
        from dotenv import load_dotenv
        load_dotenv('/home/ubuntu/apfee/.env')
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cur = conn.cursor()

        cur.execute("""
            SELECT result FROM trade_insights
            WHERE user_id = %s
            AND created_at > NOW() - INTERVAL '24 hours'
            AND result IN ('WIN', 'LOSS')
            ORDER BY created_at DESC
            LIMIT 5
        """, (user_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return False, 0

        consecutive = 0
        for row in rows:
            if row[0] == 'LOSS':
                consecutive += 1
            else:
                break

        return consecutive >= max_losses, consecutive

    except Exception as e:
        logger.error(f"Consecutive loss check error: {e}")
        return False, 0


def get_loss_warning_message(consecutive: int) -> str:
    """Generate warning message for consecutive losses."""
    if consecutive >= 3:
        return (
            f"🚨 *3 consecutive losses today*\n\n"
            f"Stop trading for the day. Protect your challenge buffer.\n"
            f"Come back tomorrow with fresh eyes.\n\n"
            f"This is the system protecting you. Trust it."
        )
    elif consecutive >= 2:
        return (
            f"⚠️ *2 consecutive losses*\n\n"
            f"Take a 30-minute break before next trade.\n"
            f"Only take 10/10 A+ signals for the rest of today.\n\n"
            f"Consider calling it a day — protecting your buffer is more important than chasing losses."
        )
    return ""


# ─── COMBINED PRE-SCAN VALIDATION ─────────────────────────────────────────────

def run_pre_scan_checks(symbol: str, entry_price: float,
                         current_price: float, direction: str,
                         session: str, candles_1h: list = None) -> dict:
    """
    Run all pre-scan validation checks.
    Returns dict with pass/fail for each check.
    """
    results = {
        "passed": True,
        "blocks": [],
        "warnings": [],
        "session_bonus": 0,
        "news_blocked": False,
        "entry_valid": True,
        "candle_confirmed": True,
    }

    # 1. News filter
    news_blocked, news_reason, news_warning = is_news_window(symbol)
    if news_blocked:
        results["news_blocked"] = True
        results["passed"] = False
        results["blocks"].append(f"🚨 {news_reason}")
    elif news_warning:
        results["warnings"].append(f"⚠️ {news_warning}")

    # 2. Session quality bonus
    session_bonus = get_session_score_bonus(session)
    results["session_bonus"] = session_bonus
    quality = get_session_quality(session)
    if quality == "AVOID":
        results["warnings"].append(f"⚠️ Asian/off-session — lower probability setup")
    elif quality == "PRIME":
        results["warnings"].append(f"✅ Prime session — highest probability window")

    # 3. Entry validation
    entry_valid, deviation = validate_entry(symbol, entry_price, current_price, direction)
    results["entry_valid"] = entry_valid
    if not entry_valid:
        results["passed"] = False
        results["blocks"].append(f"❌ Entry missed — price moved {deviation} from zone")

    # 4. 1H candle confirmation
    if candles_1h:
        confirmed, reason = check_1h_candle_confirmation(candles_1h, direction)
        results["candle_confirmed"] = confirmed
        if not confirmed:
            results["warnings"].append(f"⚠️ {reason}")

    return results


# ─── 7. LIQUIDITY SWEEP DETECTION ────────────────────────────────────────────

def detect_liquidity_sweep(candles: list, direction: str, symbol: str = "") -> tuple[bool, float]:
    """
    Check last 12 candles for a liquidity sweep.
    BUY: candle low pierced below previous 7-candle low but closed back above it.
    SELL: candle high pierced above previous 7-candle high but closed back below it.
    Also checks 20-candle swing high/low as sweep targets.
    Returns (sweep_detected, swept_level).
    """
    if not candles or len(candles) < 8:
        return False, 0.0

    # Minimum pierce distance — scaled per pair (see _SWEEP_PIERCE_BUFFER).
    # Falls back to pip size for known forex pairs, 0 for unlisted instruments.
    sym = symbol.upper() if symbol else ""
    _fallback = PIP_SPECS[sym]["pip"] if sym in _FOREX_PIP_SPEC_PAIRS else 0.0
    min_pierce = _SWEEP_PIERCE_BUFFER.get(sym, _fallback)

    recent = candles[:12]

    if direction == "BUY":
        # Check rolling 7-candle low sweep
        for i, c in enumerate(recent):
            if i + 7 >= len(candles):
                break
            prev_low = min(candles[i+1]["low"], candles[i+2]["low"], candles[i+3]["low"],
                           candles[i+4]["low"], candles[i+5]["low"],
                           candles[i+6]["low"], candles[i+7]["low"])
            if c["low"] < prev_low - min_pierce and c["close"] > prev_low:
                return True, round(prev_low, 5)
        # Check 20-candle swing low as sweep target
        if len(candles) >= 20:
            swing_low = min(c["low"] for c in candles[1:21])
            for c in recent:
                if c["low"] < swing_low - min_pierce and c["close"] > swing_low:
                    return True, round(swing_low, 5)
    else:  # SELL
        # Check rolling 7-candle high sweep
        for i, c in enumerate(recent):
            if i + 7 >= len(candles):
                break
            prev_high = max(candles[i+1]["high"], candles[i+2]["high"], candles[i+3]["high"],
                            candles[i+4]["high"], candles[i+5]["high"],
                            candles[i+6]["high"], candles[i+7]["high"])
            if c["high"] > prev_high + min_pierce and c["close"] < prev_high:
                return True, round(prev_high, 5)
        # Check 20-candle swing high as sweep target
        if len(candles) >= 20:
            swing_high = max(c["high"] for c in candles[1:21])
            for c in recent:
                if c["high"] > swing_high + min_pierce and c["close"] < swing_high:
                    return True, round(swing_high, 5)

    return False, 0.0


def detect_liquidity_run(candles: list, direction: str, symbol: str = "") -> tuple[bool, float]:
    """
    Detect a liquidity run (trend continuation) in the last 12 candles.
    BUY: candle closes ABOVE prev 7-candle swing high with a strong bullish body.
    SELL: candle closes BELOW prev 7-candle swing low with a strong bearish body.
    Unlike a sweep, price does NOT close back — it continues through the level.
    Returns (run_detected, level).
    """
    if not candles or len(candles) < 8:
        return False, 0.0

    sym = symbol.upper() if symbol else ""
    _fallback = PIP_SPECS[sym]["pip"] if sym in _FOREX_PIP_SPEC_PAIRS else 0.0
    min_pierce = _SWEEP_PIERCE_BUFFER.get(sym, _fallback)

    recent = candles[:12]

    def _is_strong_candle(c: dict) -> bool:
        total_range = c["high"] - c["low"]
        if total_range == 0:
            return False
        body = abs(c["close"] - c["open"])
        return body / total_range >= 0.5

    if direction == "BUY":
        for i, c in enumerate(recent):
            if i + 7 >= len(candles):
                break
            prev_high = max(
                candles[i+1]["high"], candles[i+2]["high"], candles[i+3]["high"],
                candles[i+4]["high"], candles[i+5]["high"],
                candles[i+6]["high"], candles[i+7]["high"],
            )
            if (c["close"] > prev_high + min_pierce and
                    c["close"] > c["open"] and _is_strong_candle(c)):
                return True, round(prev_high, 5)
        if len(candles) >= 20:
            swing_high = max(c["high"] for c in candles[1:21])
            for c in recent:
                if (c["close"] > swing_high + min_pierce and
                        c["close"] > c["open"] and _is_strong_candle(c)):
                    return True, round(swing_high, 5)
    else:  # SELL
        for i, c in enumerate(recent):
            if i + 7 >= len(candles):
                break
            prev_low = min(
                candles[i+1]["low"], candles[i+2]["low"], candles[i+3]["low"],
                candles[i+4]["low"], candles[i+5]["low"],
                candles[i+6]["low"], candles[i+7]["low"],
            )
            if (c["close"] < prev_low - min_pierce and
                    c["close"] < c["open"] and _is_strong_candle(c)):
                return True, round(prev_low, 5)
        if len(candles) >= 20:
            swing_low = min(c["low"] for c in candles[1:21])
            for c in recent:
                if (c["close"] < swing_low - min_pierce and
                        c["close"] < c["open"] and _is_strong_candle(c)):
                    return True, round(swing_low, 5)

    return False, 0.0


# ─── 8. REJECTION CANDLE DETECTION ───────────────────────────────────────────

def detect_rejection_candle(candles: list, direction: str, ob_zone_mid: float, symbol: str = "") -> tuple[bool, str]:
    """
    Check most recent 3 candles for rejection patterns near an OB zone.
    BUY: hammer, bullish engulfing, pin bar (lower wick >60% of range).
    SELL: shooting star, bearish engulfing, pin bar (upper wick >60% of range).
    Must be within 15 pips of ob_zone_mid.
    Returns (found, candle_type).
    """
    if not candles or len(candles) < 2:
        return False, ""

    # Proximity threshold: 15 pips via pip spec for forex pairs; price-level fallback for gold/futures
    sym = symbol.upper() if symbol else ""
    if sym in _FOREX_PIP_SPEC_PAIRS:
        threshold = PIP_SPECS[sym]["pip"] * 15
    else:
        threshold = 0.0015 if ob_zone_mid < 100 else 15.0

    for i in range(min(3, len(candles))):
        c = candles[i]
        candle_mid = (c["high"] + c["low"]) / 2
        if abs(candle_mid - ob_zone_mid) > threshold:
            continue

        body = abs(c["close"] - c["open"])
        total_range = c["high"] - c["low"]
        if total_range == 0:
            continue

        upper_wick = c["high"] - max(c["close"], c["open"])
        lower_wick = min(c["close"], c["open"]) - c["low"]

        if direction == "BUY":
            if body > 0 and lower_wick >= 2 * body:
                return True, "hammer"
            if lower_wick / total_range > 0.6:
                return True, "pin bar"
            if i + 1 < len(candles):
                prev = candles[i + 1]
                if (c["close"] > c["open"] and prev["close"] < prev["open"] and
                        c["close"] >= prev["open"] and c["open"] <= prev["close"]):
                    return True, "bullish engulfing"
        else:  # SELL
            if body > 0 and upper_wick >= 2 * body:
                return True, "shooting star"
            if upper_wick / total_range > 0.6:
                return True, "pin bar"
            if i + 1 < len(candles):
                prev = candles[i + 1]
                if (c["close"] < c["open"] and prev["close"] > prev["open"] and
                        c["close"] <= prev["low"] and c["open"] >= prev["high"]):
                    return True, "bearish engulfing"

    return False, ""


# ─── 9. RANGE FILTER ──────────────────────────────────────────────────────────

def is_ranging_market(candles: list) -> bool:
    """
    Returns True only when ALL three conditions are met simultaneously:
    1. 4-candle range < 0.15% of price
    2. 10-candle range < 0.5% of price
    3. No liquidity sweep in the last 5 candles

    A liquidity sweep is a candle whose wick pierced beyond the prior 3-candle
    high/low but whose body closed back inside — a sign of institutional activity
    that never occurs in a truly ranging market.
    """
    if not candles or len(candles) < 4:
        return False

    current_price = candles[0]["close"]
    if current_price == 0:
        return False

    # Early exit: 20-candle range > 0.5% means the market is NOT ranging
    if len(candles) >= 20:
        lookback20 = candles[:20]
        range20 = (max(c["high"] for c in lookback20) - min(c["low"] for c in lookback20)) / current_price
        if range20 > 0.005:
            return False

    # Condition 1: 4-candle range < 0.15%
    recent4 = candles[:4]
    range4 = (max(c["high"] for c in recent4) - min(c["low"] for c in recent4)) / current_price
    if range4 >= 0.0015:
        return False

    # Condition 2: 10-candle range < 0.5%
    if len(candles) >= 10:
        lookback10 = candles[:10]
        range10 = (max(c["high"] for c in lookback10) - min(c["low"] for c in lookback10)) / current_price
        if range10 >= 0.005:
            return False

    # Condition 3: no liquidity sweep in the last 5 candles
    # A sweep: wick breaks the prior-3-candle high or low, but candle closes back inside
    if len(candles) >= 5:
        for i in range(5):
            candle = candles[i]
            # need at least 3 candles before this one for the reference window
            if i + 3 >= len(candles):
                break
            prior3 = candles[i + 1: i + 4]
            prior_high = max(c["high"] for c in prior3)
            prior_low  = min(c["low"]  for c in prior3)
            close = candle["close"]
            # Bullish sweep: wick below prior low but close back above it
            if candle["low"] < prior_low and close > prior_low:
                return False
            # Bearish sweep: wick above prior high but close back below it
            if candle["high"] > prior_high and close < prior_high:
                return False

    return True


# ─── 10. PREVIOUS DAY HIGH/LOW ────────────────────────────────────────────────

def get_previous_day_levels(candles: list) -> dict:
    """
    Find PDH/PDL across last 5 trading days from candles list (newest first).
    Returns dict with pdh, pdl, current_price.
    """
    if not candles:
        return {"pdh": None, "pdl": None, "current_price": None}

    current_price = candles[0]["close"]
    current_day = datetime.now(timezone.utc).date()
    past_days: dict = {}

    for c in candles:
        dt_str = c.get("datetime", "")
        if not dt_str:
            continue
        try:
            if "T" in str(dt_str):
                c_dt = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
            else:
                c_dt = datetime.strptime(str(dt_str)[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            c_day = c_dt.date()
            if c_day < current_day:
                past_days.setdefault(c_day, []).append(c)
        except Exception:
            continue

    # Use up to the 5 most recent past trading days
    sorted_days = sorted(past_days.keys(), reverse=True)[:5]
    multi_day_candles = [c for day in sorted_days for c in past_days[day]]

    if multi_day_candles:
        pdh = max(c["high"] for c in multi_day_candles)
        pdl = min(c["low"] for c in multi_day_candles)
        return {"pdh": round(pdh, 5), "pdl": round(pdl, 5), "current_price": round(current_price, 5)}

    # Fallback: older half of available candles
    mid = max(len(candles) // 2, 2)
    older = candles[mid:]
    if not older:
        return {"pdh": None, "pdl": None, "current_price": round(current_price, 5)}

    pdh = max(c["high"] for c in older)
    pdl = min(c["low"] for c in older)
    return {"pdh": round(pdh, 5), "pdl": round(pdl, 5), "current_price": round(current_price, 5)}


# ─── 11. TIME OF DAY FILTER PER PAIR ─────────────────────────────────────────

PAIR_OPTIMAL_HOURS = {
    "EURUSD": [(7, 21)],            # London open through NY close
    "GBPUSD": [(7, 21)],            # London open through NY close
    "XAUUSD": [(7, 21)],            # London open through NY close
    "USDJPY": [(0, 3), (7, 21)],    # Asian session + London/NY close
    "AUDUSD": [(7, 17)],            # London open through mid NY
    "NZDUSD": [(7, 17)],            # London open through mid NY
    "USDCAD": [(12, 21)],           # NY session close (most active for CAD)
    "USDCHF": [(7, 21)],            # London open through NY close
    "EURJPY": [(0, 3), (7, 16)],
    "GBPJPY": [(0, 3), (7, 16)],
    "ES":     [(12, 16)],           # NY open 8 AM EDT = 12:00 UTC
    "MES":    [(12, 16)],
    "NQ":     [(12, 16)],
    "MNQ":    [(12, 16)],
    "CL":     [(12, 16)],
    "MCL":    [(12, 16)],
}

def is_optimal_time_for_pair(symbol: str) -> tuple[bool, str]:
    """
    Check if current UTC hour is within optimal trading window for the pair.
    Returns (is_optimal, reason_string).
    """
    hour = datetime.now(timezone.utc).hour
    sym = symbol.upper()
    windows = PAIR_OPTIMAL_HOURS.get(sym)

    if not windows:
        return True, ""

    for start, end in windows:
        if start <= hour < end:
            return True, f"{sym} in optimal window {start:02d}:00–{end:02d}:00 UTC"

    window_strs = [f"{s:02d}:00–{e:02d}:00" for s, e in windows]
    return False, f"{sym} outside optimal hours — optimal: {', '.join(window_strs)} UTC"


# ─── 12. RISK REWARD MINIMUM FILTER ──────────────────────────────────────────

def validate_risk_reward(entry: float, sl: float, tp1: float, symbol: str = "", min_rr: float | None = None) -> tuple[bool, float]:
    """
    Validate actual risk/reward ratio meets the pair's TP1 target floor.
    min_rr defaults to the TP1_MULTIPLIER value for symbol (2.0 for most pairs,
    1.5 for sub-threshold pairs). Pass symbol at every call site so the floor
    always matches the actual construction target — one source of truth.
    Returns (is_valid, actual_rr).
    """
    if min_rr is None:
        min_rr = _tp1_mult(symbol) if symbol else TP1_MULTIPLIER["default"]

    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return False, 0.0

    actual_rr = round(abs(tp1 - entry) / sl_dist, 2)
    passes = round(actual_rr, 1) >= min_rr
    if not passes:
        logger.warning(
            f"[RR_check] FAILED — entry={entry} sl={sl} tp1={tp1} "
            f"sl_dist={round(sl_dist, 5)} actual_rr={actual_rr} min={min_rr}"
        )
    return passes, actual_rr


# ─── 13. CORRELATION FILTER ───────────────────────────────────────────────────

CORRELATED_SAME_DIRECTION = {
    "EURUSD": ["GBPUSD", "AUDUSD", "NZDUSD"],
    "GBPUSD": ["EURUSD", "AUDUSD", "NZDUSD"],
    "AUDUSD": ["EURUSD", "GBPUSD", "NZDUSD"],
    "NZDUSD": ["EURUSD", "GBPUSD", "AUDUSD"],
    "USDJPY": ["USDCAD", "USDCHF"],
    "USDCAD": ["USDJPY", "USDCHF"],
    "USDCHF": ["USDJPY", "USDCAD"],
    "ES": ["NQ", "YM"],
    "NQ": ["ES", "YM"],
    "YM": ["ES", "NQ"],
    "XAUUSD": ["GC", "MGC"],
    "GC": ["XAUUSD"],
}

INVERSE_CORRELATED = {
    "EURUSD": ["USDJPY", "USDCAD", "USDCHF"],
    "GBPUSD": ["USDJPY", "USDCAD", "USDCHF"],
    "AUDUSD": ["USDJPY", "USDCAD", "USDCHF"],
    "NZDUSD": ["USDJPY", "USDCAD", "USDCHF"],
    "USDJPY": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
    "USDCAD": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
    "USDCHF": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
}

def check_pair_correlation(symbol: str, direction: str, active_signals: list) -> tuple[bool, str, str]:
    """
    Block signals that double USD exposure via same-direction correlation.
    Warn (but allow) on inverse correlation — USD theme conflict.

    Same-direction: e.g. EURUSD SELL + GBPUSD SELL — both express EUR/GBP weakness vs USD. Hard block.
    Inverse: e.g. EURUSD BUY active, USDJPY BUY fires — conflicting USD themes. Warn, don't block.

    active_signals: list of dicts with 'symbol' and 'direction' keys.
    Returns (is_ok_to_trade, block_reason, correlation_warning).
    """
    if not active_signals:
        return True, "", ""

    sym = symbol.upper()
    new_dir = direction.upper()

    active_map = {
        s.get("symbol", "").upper(): s.get("direction", "").upper()
        for s in active_signals
    }

    # Same-direction block — hard stop, doubling exposure
    for corr_sym in CORRELATED_SAME_DIRECTION.get(sym, []):
        if corr_sym in active_map and active_map[corr_sym] == new_dir:
            return False, f"Correlated pair {corr_sym} already has a {new_dir} signal — skip to avoid doubling exposure", ""

    # Inverse-correlation: same USD theme via opposite pair directions — warn but allow
    for inv_sym in INVERSE_CORRELATED.get(sym, []):
        if inv_sym in active_map:
            active_dir = active_map[inv_sym]
            if active_dir != new_dir:
                usd_theme = "USD strengthening" if new_dir == "BUY" and "JPY" in sym else "USD conflict"
                warning = (
                    f"⚠️ {usd_theme} — {sym} {new_dir} signal while {inv_sym} {active_dir} active. "
                    f"USD theme conflict detected."
                )
                return True, "", warning

    return True, "", ""


# ─── 14. MULTI-TIMEFRAME OB CONFLUENCE ───────────────────────────────────────

# Mirrors YFINANCE_FUTURES_MAP from scanner.py — kept in sync manually
_MTF_FUTURES_MAP = {
    'ES': 'ES=F', 'MES': 'MES=F', 'NQ': 'NQ=F', 'MNQ': 'MNQ=F',
    'RTY': 'RTY=F', 'YM': 'YM=F', 'CL': 'CL=F', 'MCL': 'MCL=F',
    'GC': 'GC=F', 'MGC': 'MGC=F', 'NG': 'NG=F',
    'US100': 'NQ=F', 'US30': 'YM=F',
    'US500': 'ES=F',
}


def _fetch_htf_candles(symbol: str, interval_yf: str, period_yf: str,
                       interval_td: str, outputsize: int) -> list | None:
    """Fetch candles for MTF analysis — yFinance for futures, Twelve Data for forex."""
    sym = symbol.upper()
    ticker = _MTF_FUTURES_MAP.get(sym)
    if ticker:
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period=period_yf, interval=interval_yf)
            if hist.empty:
                return None
            result = []
            for _, row in hist.iloc[::-1].iterrows():
                result.append({
                    "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]), "close": float(row["Close"]),
                })
            return result[:outputsize]
        except Exception as e:
            logger.debug(f"MTF yFinance fetch error {symbol} {interval_yf}: {e}")
            return None
    else:
        # TD primary for forex HTF candles
        try:
            from config import TWELVE_DATA_API_KEY
            from scanner import _td_available, _mark_td_exhausted
            if TWELVE_DATA_API_KEY and _td_available():
                from market import normalize_symbol as _norm_sym
                td_symbol = _norm_sym(sym)
                resp = requests.get("https://api.twelvedata.com/time_series",
                    params={"symbol": td_symbol, "interval": interval_td,
                            "outputsize": outputsize, "apikey": TWELVE_DATA_API_KEY}, timeout=8)
                data = resp.json()
                if data.get("status") != "error" and "values" in data:
                    return [{"open": float(v["open"]), "high": float(v["high"]),
                             "low": float(v["low"]), "close": float(v["close"])}
                            for v in data["values"]][:outputsize]
                _msg = data.get("message", "")
                if "credits" in _msg.lower():
                    _mark_td_exhausted()
        except Exception as e:
            logger.debug(f"MTF TD fetch error {symbol} {interval_td}: {e}")

        # yFinance fallback
        try:
            import yfinance as yf
            _MTF_FOREX_MAP = {
                'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X',
                'AUDUSD': 'AUDUSD=X', 'USDCAD': 'USDCAD=X', 'NZDUSD': 'NZDUSD=X',
                'USDCHF': 'USDCHF=X', 'EURGBP': 'EURGBP=X', 'EURJPY': 'EURJPY=X',
                'GBPJPY': 'GBPJPY=X', 'XAUUSD': 'GC=F',
            }
            yf_ticker = _MTF_FOREX_MAP.get(sym)
            if not yf_ticker:
                return None
            hist = yf.Ticker(yf_ticker).history(period=period_yf, interval=interval_yf)
            if hist.empty:
                return None
            result = []
            for _, row in hist.iloc[::-1].iterrows():
                result.append({
                    "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]), "close": float(row["Close"]),
                })
            return result[:outputsize]
        except Exception as e:
            logger.debug(f"MTF yFinance forex fetch error {symbol} {interval_yf}: {e}")
            return None


def _detect_ob_htf(candles: list, trend: str) -> dict | None:
    """Minimal OB detection for MTF confluence — same logic as detect_order_block in scanner.py."""
    if not candles or len(candles) < 5:
        return None
    if trend == "bullish":
        for i in range(1, min(15, len(candles))):
            c = candles[i]
            if c["close"] < c["open"] and candles[i - 1]["close"] > c["high"]:
                return {"high": c["high"], "low": c["low"],
                        "mid": round((c["high"] + c["low"]) / 2, 5)}
    elif trend == "bearish":
        for i in range(1, min(15, len(candles))):
            c = candles[i]
            if c["close"] > c["open"] and candles[i - 1]["close"] < c["low"]:
                return {"high": c["high"], "low": c["low"],
                        "mid": round((c["high"] + c["low"]) / 2, 5)}
    return None


def detect_mtf_ob_confluence(symbol: str, current_ob: dict, direction: str,
                              candles_1h: list = None, candles_4h: list = None) -> tuple[bool, str]:
    """
    Check if the 15M order block also aligns with a 1H or 4H order block at the same level.
    Triple timeframe OB confluence is the highest probability SMC setup.
    Returns (confluence_found, description).
    Scoring applied by caller: triple→+3, 4H only→+2, 1H only→+1.
    Pass candles_1h/candles_4h (newest-first) to skip API fetches.
    """
    if not current_ob:
        return False, ""

    ob_low = current_ob["low"]
    ob_high = current_ob["high"]
    trend = "bullish" if direction == "BUY" else "bearish"
    mid_price = current_ob.get("mid", ob_high)

    # Tolerance: 10 pips for forex, 10 points for gold/futures
    tol_1h = 0.001 if mid_price < 100 else 10.0
    # Tolerance: 20 pips for forex, 20 points for gold/futures
    tol_4h = 0.002 if mid_price < 100 else 20.0

    def _overlaps(low_a, high_a, low_b, high_b, tol):
        return low_a <= high_b + tol and low_b <= high_a + tol

    try:
        # Use pre-fetched candles if provided, else fetch from API
        _c1h = candles_1h if candles_1h else _fetch_htf_candles(symbol, "1h", "14d", "1h", 100)
        ob_1h = _detect_ob_htf(_c1h, trend) if _c1h else None

        _c4h = candles_4h if candles_4h else _fetch_htf_candles(symbol, "4h", "30d", "4h", 60)
        ob_4h = _detect_ob_htf(_c4h, trend) if _c4h else None

        h1_conf = bool(ob_1h and _overlaps(ob_low, ob_high, ob_1h["low"], ob_1h["high"], tol_1h))
        h4_conf = bool(ob_4h and _overlaps(ob_low, ob_high, ob_4h["low"], ob_4h["high"], tol_4h))

        if h1_conf and h4_conf:
            return True, "Triple timeframe OB confluence (15M+1H+4H) — highest probability setup"
        elif h4_conf:
            return True, "4H OB confluence confirmed — strong institutional level"
        elif h1_conf:
            return True, "1H OB confluence confirmed — solid structural level"

    except Exception as e:
        logger.debug(f"MTF OB confluence error for {symbol}: {e}")

    return False, ""


# ─── 15. MOMENTUM DETECTION ──────────────────────────────────────────────────

def detect_momentum(candles: list, direction: str) -> tuple[bool, str]:
    """
    Detect if price is moving with strong momentum in signal direction.
    Strong momentum = 3+ consecutive candles closing in same direction
    with increasing body sizes.
    Returns (momentum_detected, description).
    """
    if not candles or len(candles) < 3:
        return False, ""

    expected_bullish = direction.upper() == "BUY"
    consecutive = 0
    prev_body = None
    bodies_increasing = True

    for c in candles[:5]:
        is_bullish = c["close"] > c["open"]
        if is_bullish == expected_bullish:
            body = abs(c["close"] - c["open"])
            if prev_body is not None and body <= prev_body:
                bodies_increasing = False
            prev_body = body
            consecutive += 1
        else:
            break

    if consecutive >= 3:
        dir_label = "bullish" if expected_bullish else "bearish"
        if bodies_increasing:
            return True, f"Strong momentum — 3 consecutive {dir_label} candles with increasing volume"
        return True, f"Momentum confirmed — consecutive {dir_label} candles"

    return False, ""


# ─── 16. DAILY BIAS ───────────────────────────────────────────────────────────

_DAILY_BIAS_FUTURES_MAP = {
    'ES': 'ES=F', 'MES': 'MES=F', 'NQ': 'NQ=F', 'MNQ': 'MNQ=F',
    'RTY': 'RTY=F', 'YM': 'YM=F', 'CL': 'CL=F', 'MCL': 'MCL=F',
    'GC': 'GC=F', 'MGC': 'MGC=F', 'NG': 'NG=F', 'XAUUSD': 'GC=F',
    'US100': 'NQ=F', 'US30': 'YM=F',
    'US500': 'ES=F', 'USOIL': 'CL=F',
}

_FOREX_YFINANCE_MAP = {
    'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X', 'USDCAD': 'USDCAD=X', 'NZDUSD': 'NZDUSD=X',
    'USDCHF': 'USDCHF=X',
}

_previous_bias: dict = {}    # {symbol: bias_string} — last known bias per symbol
_last_bias_shift: dict = {}  # {symbol: timestamp}   — last bias-shift alert per symbol


def get_daily_bias(symbol: str, candles: list = None) -> dict:
    """
    Get the daily candle bias for a symbol.
    Pass candles (newest-first) to skip the API fetch and use pre-fetched data.
    Returns dict with bias, strength, confirmed flag, and reason.
    """
    _default = {"bias": "unknown", "strength": "weak", "today_candle": "neutral",
                "confirmed": False, "reason": "Insufficient data",
                "intraday_override": False, "intraday_move_pct": 0.0}
    sym = symbol.upper()
    try:
        if candles and len(candles) >= 3:
            # Pre-fetched candles are newest-first; reverse to oldest→newest for computation
            _daily_candles = list(reversed(candles))
        else:
            _daily_candles = None
            # DIAGNOSTIC — this fast path silently falls through to a fresh TD/yfinance
            # fetch whenever pre-fetched candles are missing OR too few (< 3), even if
            # get_htf_bias() already succeeded with the SAME candles for its own,
            # looser 5-candle-window need. Confirmed live: USDJPY was showing
            # "No bias data available" in htf_bias_diag while d1_trend (from the same
            # candles_daily bundle, via get_htf_bias) was populated fine — logging
            # exactly what was actually passed in, to see if this is the cause.
            logger.info(
                f"[daily_bias_diag] {symbol} fast path NOT used — "
                f"candles={'None' if candles is None else f'{len(candles)} items'} (need >= 3)"
            )

        # Build raw_candles (oldest→newest) either from pre-fetched data or API
        raw_candles = []
        ticker = _DAILY_BIAS_FUTURES_MAP.get(sym)
        if _daily_candles is not None:
            raw_candles = _daily_candles
        elif ticker:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period="30d", interval="1d")
            if hist.empty or len(hist) < 3:
                return _default
            for _, row in hist.iloc[-20:].iterrows():
                raw_candles.append({
                    "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]),  "close": float(row["Close"]),
                })
        else:
            yf_ticker = _FOREX_YFINANCE_MAP.get(sym)
            td_success = False
            try:
                from config import TWELVE_DATA_API_KEY
                from scanner import _td_available, _mark_td_exhausted
                if TWELVE_DATA_API_KEY and _td_available():
                    from market import normalize_symbol as _norm_sym
                    td_sym = _norm_sym(sym)
                    resp = requests.get(
                        "https://api.twelvedata.com/time_series",
                        params={"symbol": td_sym, "interval": "1day",
                                "outputsize": 20, "apikey": TWELVE_DATA_API_KEY},
                        timeout=8
                    )
                    data = resp.json()
                    if data.get("status") != "error" and "values" in data:
                        for v in reversed(data["values"]):
                            raw_candles.append({
                                "open": float(v["open"]), "high": float(v["high"]),
                                "low":  float(v["low"]),  "close": float(v["close"]),
                            })
                        td_success = True
                    else:
                        _msg = data.get("message", "")
                        if "credits" in _msg.lower():
                            _mark_td_exhausted()
                        logger.info(f"[daily_bias_diag] {symbol} TD returned non-success: {_msg!r}")
            except Exception as _td_exc:
                logger.info(f"[daily_bias_diag] {symbol} TD fetch raised: {_td_exc!r}")
            if not td_success:
                if not yf_ticker:
                    logger.info(f"[daily_bias_diag] {symbol} no yFinance ticker mapping — giving up")
                    return _default
                try:
                    import yfinance as yf
                    hist = yf.Ticker(yf_ticker).history(period="30d", interval="1d")
                    if hist.empty or len(hist) < 3:
                        logger.info(f"[daily_bias_diag] {symbol} yFinance {yf_ticker} returned {len(hist)} rows (need >= 3)")
                        return _default
                    for _, row in hist.iloc[-20:].iterrows():
                        raw_candles.append({
                            "open": float(row["Open"]), "high": float(row["High"]),
                            "low":  float(row["Low"]),  "close": float(row["Close"]),
                        })
                except Exception as _yf_exc:
                    logger.info(f"[daily_bias_diag] {symbol} yFinance {yf_ticker} raised: {_yf_exc!r}")
                    return _default

        if len(raw_candles) < 5:
            logger.info(f"[daily_bias_diag] {symbol} raw_candles={len(raw_candles)} after all fetch attempts (need >= 5)")
            return _default

        def _candle_dir(c):
            return "bullish" if c["close"] > c["open"] else "bearish"

        def _body_ratio(c):
            rng = c["high"] - c["low"]
            return abs(c["close"] - c["open"]) / rng if rng else 0

        def _score(c):
            return 1 if c["close"] > c["open"] else -1

        # raw_candles are oldest→newest; last entry is today
        today = raw_candles[-1]
        d1    = raw_candles[-2]
        d2    = raw_candles[-3]
        d3    = raw_candles[-4]
        d4    = raw_candles[-5]

        today_dir   = _candle_dir(today)
        today_ratio = _body_ratio(today)

        intraday_move_pct = ((today["close"] - today["open"]) / today["open"] * 100) if today["open"] else 0.0
        intraday_override = today_ratio > 0.6

        # Today's weight is SCALED by how decisive its body actually is (reusing the
        # same 0.6 body-ratio threshold already used for intraday_override just above),
        # instead of a flat +-5 regardless of how tiny/undeveloped the candle currently
        # is. Confirmed live via htf_bias_diag: with the old flat weighting, a barely-
        # formed today candle sitting right around its own open could flip the
        # "confirmed" verdict (bearish -> neutral) within 32 seconds on otherwise
        # completely stable underlying data — today's incomplete move, weighted at full
        # strength (the single largest weight in the whole score), was acting as pure
        # noise rather than genuine signal. A candle that's already moved decisively
        # (body_ratio >= 0.6, the SAME bar this function already uses elsewhere to call
        # a move "strong") still gets its full original weight; anything less scales
        # down proportionally rather than voting at full strength on a coin flip.
        # round() keeps this an int (required — three f-strings downstream use the
        # ':+d' format spec on `weighted`, which strictly rejects a float; the earlier
        # version of this fix produced a float here, which broke get_daily_bias() with
        # "Unknown format code 'd' for object of type 'float'" on every real call since
        # deploy — confirmed live, this was the actual cause of the "No bias data
        # available" symptom being investigated, not a separate fetch-path issue).
        _today_weight = round(5 * min(today_ratio / 0.6, 1.0)) if today_ratio else 0

        # Weighted score: today (scaled, max 5), d1×4, d2×3, d3×2, d4×1 — max ±15
        weighted = (
            _score(today) * _today_weight +
            _score(d1)    * 4 +
            _score(d2)    * 3 +
            _score(d3)    * 2 +
            _score(d4)    * 1
        )

        if weighted >= 2:
            bias = "bullish"
        elif weighted <= -2:
            bias = "bearish"
        else:
            bias = "neutral"

        # Threshold lowered 7→5: requiring 7/15 misses first day of new trend moves.
        # 5/15 means at least today + yesterday aligned — sufficient for intraday ICT.
        confirmed = abs(weighted) >= 5

        # Strength
        if today_ratio > 0.6 and _candle_dir(today) == bias:
            strength = "strong"
        elif abs(weighted) >= 5:
            strength = "moderate"
        else:
            strength = "weak"

        # Strong today candle override — only confirm if history agrees
        if today_ratio > 0.7 and weighted >= 3:
            # Institutional conviction: strong candle + history aligned
            bias      = today_dir
            strength  = "strong"
            confirmed = True
            reason    = (
                f"Strong today candle ({today_dir}, body {today_ratio:.0%}) confirmed by history "
                f"(weighted {weighted:+d}/15)"
            )
            return {
                "bias": bias, "strength": strength,
                "today_candle": today_dir, "confirmed": confirmed, "reason": reason,
                "intraday_override": True, "intraday_move_pct": round(intraday_move_pct, 2),
            }
        elif today_ratio > 0.7 and weighted < 0:
            # Strong candle against recent history — potential reversal, not confirmation
            bias      = today_dir
            strength  = "moderate"
            confirmed = False
            reason    = (
                f"Strong today candle ({today_dir}, body {today_ratio:.0%}) AGAINST history "
                f"(weighted {weighted:+d}/15) — reversal signal"
            )
            return {
                "bias": bias, "strength": strength,
                "today_candle": today_dir, "confirmed": confirmed, "reason": reason,
                "intraday_override": False, "intraday_move_pct": round(intraday_move_pct, 2),
            }

        reason = (
            f"Weighted score {weighted:+d}/15 "
            f"(today {_candle_dir(today)}, d1 {_candle_dir(d1)}, "
            f"d2 {_candle_dir(d2)}, d3 {_candle_dir(d3)}, d4 {_candle_dir(d4)})"
        )

        return {
            "bias": bias,
            "strength": strength,
            "today_candle": today_dir,
            "confirmed": confirmed,
            "reason": reason,
            "intraday_override": intraday_override,
            "intraday_move_pct": round(intraday_move_pct, 2),
        }

    except Exception as e:
        logger.error(f"[daily_bias] {symbol}: {e}")
        return _default


def check_daily_bias_alignment(symbol: str, direction: str, _prefetched: dict = None) -> tuple[bool, str]:
    """
    Check if trade direction aligns with the daily bias.
    Returns (aligned, message).
    Pass _prefetched to reuse an already-fetched get_daily_bias() result.
    """
    bias = _prefetched if _prefetched is not None else get_daily_bias(symbol)
    b = bias["bias"]
    confirmed = bias["confirmed"]

    if b == "unknown":
        return False, "⚠️ No bias data available — proceed with extra caution"

    if not confirmed:
        return False, f"⚠️ Daily bias {b} not confirmed — gate fails"

    if direction.upper() == "BUY" and b == "bearish":
        return False, "⚠️ DAILY BIAS CONFLICT — Trading BUY against confirmed bearish daily bias. High risk."
    if direction.upper() == "SELL" and b == "bullish":
        return False, "⚠️ DAILY BIAS CONFLICT — Trading SELL against confirmed bullish daily bias. High risk."

    if b not in ("neutral", "unknown"):
        return True, f"✅ Daily bias {b} confirms {direction.upper()} direction"

    return True, ""


# ─── 17. EQUAL HIGHS/LOWS DETECTION ──────────────────────────────────────────

def detect_equal_highs_lows(candles: list, direction: str, symbol: str = "") -> tuple[bool, str]:
    """
    Detect equal highs (SELL) or equal lows (BUY) liquidity pools in last 20 candles.
    Swing high/low: candle is higher/lower than 2 candles on each side.
    Equal = within 3 pips/points of each other.
    Returns (detected, description).
    """
    if not candles or len(candles) < 5:
        return False, ""

    recent = candles[:20]
    n = len(recent)
    if n < 5:
        return False, ""

    current_price = recent[0]["close"]
    if current_price > 500:
        tol = 3.0       # gold/indices: 3 points
    elif current_price > 20:
        tol = 0.03      # JPY pairs: 3 pips
    else:
        tol = 0.0003    # forex: 3 pips

    # Offset futures prices to spot domain for display (avoids circular import from scanner)
    _SPOT_DISPLAY_OFFSETS = {"XAUUSD": -30, "XAGUSD": -0.30}
    _disp_off = _SPOT_DISPLAY_OFFSETS.get(symbol.upper(), 0) if symbol else 0

    if direction.upper() == "SELL":
        swing_highs = []
        for i in range(2, n - 2):
            h = recent[i]["high"]
            if (h > recent[i-1]["high"] and h > recent[i-2]["high"] and
                    h > recent[i+1]["high"] and h > recent[i+2]["high"]):
                swing_highs.append(h)
        for j in range(len(swing_highs)):
            cluster = [swing_highs[j]]
            for k in range(j + 1, len(swing_highs)):
                if abs(swing_highs[k] - swing_highs[j]) <= tol:
                    cluster.append(swing_highs[k])
            if len(cluster) >= 2:
                level = sum(cluster) / len(cluster)
                display_level = round(level + _disp_off, 3)
                return True, f"Equal highs liquidity pool at {display_level} — banks will sweep this"
    else:  # BUY
        swing_lows = []
        for i in range(2, n - 2):
            l = recent[i]["low"]
            if (l < recent[i-1]["low"] and l < recent[i-2]["low"] and
                    l < recent[i+1]["low"] and l < recent[i+2]["low"]):
                swing_lows.append(l)
        for j in range(len(swing_lows)):
            cluster = [swing_lows[j]]
            for k in range(j + 1, len(swing_lows)):
                if abs(swing_lows[k] - swing_lows[j]) <= tol:
                    cluster.append(swing_lows[k])
            if len(cluster) >= 2:
                level = sum(cluster) / len(cluster)
                display_level = round(level + _disp_off, 3)
                return True, f"Equal lows liquidity pool at {display_level} — banks will sweep this"

    return False, ""


# ─── 18. MARKET STRUCTURE SHIFT DETECTION ────────────────────────────────────

def detect_market_structure_shift(candles: list, direction: str) -> tuple[bool, str]:
    """
    Detect a full market structure shift (MSS) — stronger than BOS, confirms reversal.
    Bullish MSS: prior lower highs + lower lows, current close breaks above most recent swing high.
    Bearish MSS: prior higher highs + higher lows, current close breaks below most recent swing low.
    Returns (detected_in_direction, description).
    If detected against direction: returns (False, 'MSS against signal direction').
    """
    if not candles or len(candles) < 10:
        return False, ""

    recent = candles[:20]
    n = len(recent)
    if n < 8:
        return False, ""

    current_close = recent[0]["close"]

    # Analyse structure in candles[2:] — skip the 2 most recent (possible MSS candles)
    structure = recent[2:]
    sc_n = len(structure)
    if sc_n < 6:
        return False, ""

    swing_highs = []  # (index, price) — index within structure[], newest-first
    swing_lows = []

    for i in range(2, sc_n - 2):
        h = structure[i]["high"]
        l = structure[i]["low"]
        if (h > structure[i-1]["high"] and h > structure[i-2]["high"] and
                h > structure[i+1]["high"] and h > structure[i+2]["high"]):
            swing_highs.append((i, h))
        if (l < structure[i-1]["low"] and l < structure[i-2]["low"] and
                l < structure[i+1]["low"] and l < structure[i+2]["low"]):
            swing_lows.append((i, l))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, ""

    # Smallest index = most recent; largest index = oldest (newest-first array)
    most_recent_sh = min(swing_highs, key=lambda x: x[0])[1]
    oldest_sh = max(swing_highs, key=lambda x: x[0])[1]
    most_recent_sl = min(swing_lows, key=lambda x: x[0])[1]
    oldest_sl = max(swing_lows, key=lambda x: x[0])[1]

    lower_highs = most_recent_sh < oldest_sh
    lower_lows = most_recent_sl < oldest_sl
    higher_highs = most_recent_sh > oldest_sh
    higher_lows = most_recent_sl > oldest_sl

    if direction.upper() == "BUY":
        if lower_highs and lower_lows and current_close > most_recent_sh:
            return True, "Market structure shift confirmed — full BUY reversal"
        if higher_highs and higher_lows and current_close < most_recent_sl:
            return False, "MSS against signal direction"
    else:  # SELL
        if higher_highs and higher_lows and current_close < most_recent_sl:
            return True, "Market structure shift confirmed — full SELL reversal"
        if lower_highs and lower_lows and current_close > most_recent_sh:
            return False, "MSS against signal direction"

    return False, ""


# ─── 19. PREMIUM/DISCOUNT ZONE FILTER ────────────────────────────────────────

def check_premium_discount_zone(candles: list, entry: float, direction: str) -> tuple[bool, str]:
    """
    Filter entries by whether they sit in a premium or discount zone.
    D1 range split at equilibrium with ±10% neutral buffer around midpoint.
    BUY clearly in discount (below lower_mid) → labelled ok.
    SELL clearly in premium (above upper_mid) → labelled ok.
    Neutral zone → silent (no label, no warning).
    Returns (is_favourable, description). Empty description = neutral or no data.
    """
    if not candles or len(candles) < 5:
        return True, ""

    recent = candles[:20]
    d1_high = max(c["high"] for c in recent)
    d1_low  = min(c["low"]  for c in recent)

    if d1_high == d1_low:
        return True, ""

    range_size = d1_high - d1_low
    midpoint   = (d1_high + d1_low) / 2
    buffer     = range_size * 0.10
    upper_mid  = midpoint + buffer
    lower_mid  = midpoint - buffer

    if direction.upper() == "BUY":
        if entry < lower_mid:
            return True, "Entry in discount zone — buying at value"
        return True, ""
    else:  # SELL
        if entry > upper_mid:
            return True, "Entry in premium zone — selling at premium"
        return True, ""


# ─── 20. KILL ZONE TIMING BONUS ──────────────────────────────────────────────

# ── DST DISCLOSURE ────────────────────────────────────────────────────────────
# All windows below are hardcoded UTC. No DST adjustment is applied here.
# A _et_to_utc_minutes() helper (line ~98) exists and is used for news/FOMC
# timing, but is NOT wired to this kill zone dict.
# Consequence: every ET-anchored window below drifts by 1 hour between seasons:
#   Summer (EDT, UTC-4): values listed are correct.
#   Winter (EST, UTC-5): every ET-based window is 1 hour too early in UTC.
# This is a known gap; proper DST-aware kill zones would require either
# calling _et_to_utc_minutes() dynamically or maintaining two window sets.
# Flagged as a future improvement — do not silently assume these are always right.
# ──────────────────────────────────────────────────────────────────────────────
_KILL_ZONES = {
    'asian':               (23,  2),  # 23:00-02:00 UTC wraps midnight — JPY, AUD, NZD only
    'london':              (6,  10),  # 06:00-10:00 UTC covers summer (6-9) and winter (7-10)
    'ny_open':             (12, 15),  # 12:00-15:00 UTC — all pairs
    'london_close':        (15, 16),  # 15:00-16:00 UTC — EUR, GBP bonus window
    'evening':             (19, 22),  # 19:00-22:00 UTC — post-FOMC Asia-correlated pairs
    # ICT Silver Bullet windows — 3 daily sessions (ICT confirmed: London Open, NY AM, NY PM)
    # Current UTC values are correct for EDT (summer, UTC-4). EST winter shift: each +1h.
    #   silver_bullet_london: winter correct window = 08:00-09:00 UTC
    #   silver_bullet_ny:     winter correct window = 15:00-16:00 UTC
    #   silver_bullet_ny_pm:  winter correct window = 19:00-20:00 UTC
    'silver_bullet_london': (7,  8),  # 3:00-4:00 AM EDT = 07:00-08:00 UTC (summer) ✓
    'silver_bullet_ny':    (14, 15),  # 10:00-11:00 AM EDT = 14:00-15:00 UTC (summer) ✓
    'silver_bullet_ny_pm': (18, 19),  # 2:00-3:00 PM EDT = 18:00-19:00 UTC (summer) ✓; EST winter: 19:00-20:00 UTC
    # Gold-specific windows: log analysis confirmed ALL 22 XAUUSD 5M BOS events fell in the
    # 10:00-12:00 and 16:00-17:00 UTC gaps, causing 100% kill_zone gate rejection (Jul 24-26).
    # Root cause: standard london window ends 10:00 UTC; ny_open starts 12:00; london_close
    # ends 16:00. These two windows cover the institutionally active London→NY transition and
    # the Gold PM session where real 5M displacement is appearing in market data.
    'xau_pre_ny':          (10, 12),  # 10:00-12:00 UTC — London close/Pre-NY transition, Gold only
    'xau_pm':              (16, 17),  # 16:00-17:00 UTC — PM session continuation, Gold only
    # LBMA Gold Fix — set twice daily at 10:30 AM and 3:00 PM London time (source: LBMA + ICE).
    # AM fix: 10:30 London BST = 09:30 UTC (summer); 10:30 London GMT = 10:30 UTC (winter, caught by xau_pre_ny).
    # PM fix: 3:00 PM London BST = 14:00 UTC (summer); 3:00 PM London GMT = 15:00 UTC (winter, caught by london_close).
    # These windows are correct for EDT/summer. In GMT/winter they drift 1h (see DST disclosure above).
    'london_fix':          (9,  10),  # LBMA AM Gold Fix: 10:30 AM London = 09:30 UTC (BST/summer) ✓
    'london_fix_pm':       (14, 15),  # LBMA PM Gold Fix: 3:00 PM London = 14:00 UTC (BST/summer) ✓; GMT winter: 15:00 UTC
    'ny_indices':          (12, 16),  # 12:00-16:00 UTC — covers pre-market sweep + NYSE open
    # ICT research: indices kill zone starts at 8:30AM EST (12:30 UTC) not 9:00AM
    # Pre-market 12:30-13:30 UTC is where the Judas Swing sweep happens on indices
}

_PAIR_KILL_ZONES = {
    # Silver Bullet windows (london, ny_am, ny_pm) placed before generic ny_open/london_close
    # so the specific label wins priority in is_kill_zone() which returns on first match.
    'EURUSD':  ['london', 'silver_bullet_london', 'silver_bullet_ny', 'silver_bullet_ny_pm', 'ny_open', 'london_close'],
    'GBPUSD':  ['london', 'silver_bullet_london', 'silver_bullet_ny', 'silver_bullet_ny_pm', 'ny_open', 'london_close'],
    # XAUUSD: gold-specific Fix windows listed first so they win label over generic London/NY Open.
    # london_fix (09-10) and london_fix_pm (14-15) must precede london (06-10) and ny_open (12-15)
    # respectively, otherwise those broader windows shadow the Fix labels.
    'XAUUSD':  ['london_fix', 'london_fix_pm', 'london', 'xau_pre_ny', 'ny_open', 'london_close',
                'silver_bullet_london', 'silver_bullet_ny', 'silver_bullet_ny_pm', 'xau_pm'],
    'USDJPY':  ['asian',  'london', 'ny_open', 'evening'],
    'AUDUSD':  ['asian',  'london', 'ny_open', 'evening'],
    'NZDUSD':  ['asian',  'london', 'ny_open', 'evening'],
    'USDCAD':  ['london', 'ny_open'],
    'USDCHF':  ['london', 'ny_open'],
    'US100':   ['ny_open', 'ny_indices'],
    'US30':    ['ny_open', 'ny_indices'],
    'US500':   ['ny_open', 'ny_indices'],
    'USOIL':   ['london', 'ny_open'],
}

_KILL_ZONE_LABELS = {
    'asian':               "Asian Open",
    'london':              "London Open",
    'ny_open':             "NY Open",
    'london_close':        "London Close",
    'evening':             "Evening Session",
    'silver_bullet_london': "⭐ London Silver Bullet",
    'silver_bullet_ny':    "⭐ NY AM Silver Bullet",
    'silver_bullet_ny_pm': "⭐ NY PM Silver Bullet",
    'london_fix':          "🏅 Gold AM Fix",
    'london_fix_pm':       "🏅 Gold PM Fix",
    'ny_indices':          "NY Indices Open",
    'xau_pre_ny':          "Gold Pre-NY Transition",
    'xau_pm':              "Gold PM Session",
}

# Maps each news-adjacent kill zone to its documented ICT follow-through window.
# Source: ICT NY Killzone guide — "If you missed the 8:30 AM move, the 9:30 AM
# open often provides a re-entry or second leg into a newly formed FVG."
# This is purely informational — no gate logic, thresholds, or R:R changes.
SECOND_LEG_WINDOWS = {
    # After NY 8:30am EST news block (12:30-13:00 UTC), the 9:30am EST
    # NYSE cash open (13:30-14:00 UTC) often gives the re-entry leg.
    'ny_open': (13, 14),  # 13:00-14:00 UTC follow-through window
    # London second-leg already covered by existing 'london' window (06-10 UTC).
}

def is_kill_zone(symbol: str) -> tuple[bool, str]:
    """
    Check if current UTC time falls within the pair's ICT kill zone window.
    Returns (in_kill_zone, label_string).
    """
    hour = datetime.now(timezone.utc).hour
    valid_zones = _PAIR_KILL_ZONES.get(symbol.upper(), ['london', 'ny_open'])
    for zone in valid_zones:
        start, end = _KILL_ZONES[zone]
        if start > end:  # wraps midnight (e.g. asian: 23-02)
            in_zone = hour >= start or hour < end
        else:
            in_zone = start <= hour < end
        if in_zone:
            label = _KILL_ZONE_LABELS[zone]
            return True, f"Kill zone active — {label} — peak institutional activity"
    return False, ""


# ─── 21. MARKET STRUCTURE ANALYSIS ───────────────────────────────────────────

def analyze_market_structure(candles: list) -> dict:
    """
    Identify market structure using swing highs/lows on the last 20 candles.
    Candles must be newest-first.

    Returns dict with keys:
      structure: 'uptrend' | 'downtrend' | 'ranging'
      choch:     True if Change of Character (reversal) detected
      bos:       True if Break of Structure (continuation) detected
      last_swing_high: float | None
      last_swing_low:  float | None
    """
    _default = {
        "structure": "ranging", "choch": False, "bos": False,
        "last_swing_high": None, "last_swing_low": None,
    }

    if len(candles) < 20:
        return _default

    # MOMENTUM OVERRIDE — strong directional moves bypass swing structure check
    recent = candles[:10]  # newest first
    bullish_count = sum(1 for c in recent if c['close'] > c['open'])
    bearish_count = sum(1 for c in recent if c['close'] < c['open'])

    if bullish_count >= 7:
        return {
            "structure": "uptrend",
            "choch": False,
            "bos": True,
            "last_swing_high": candles[0]['high'],
            "last_swing_low": candles[9]['low'],
        }
    elif bearish_count >= 7:
        return {
            "structure": "downtrend",
            "choch": False,
            "bos": True,
            "last_swing_high": candles[9]['high'],
            "last_swing_low": candles[0]['low'],
        }

    swing_highs = []
    swing_lows = []
    chron = list(reversed(candles[:20]))

    for i in range(2, len(chron) - 2):
        if (chron[i]['high'] > chron[i-1]['high'] and
                chron[i]['high'] > chron[i-2]['high'] and
                chron[i]['high'] > chron[i+1]['high'] and
                chron[i]['high'] > chron[i+2]['high']):
            swing_highs.append(chron[i]['high'])

        if (chron[i]['low'] < chron[i-1]['low'] and
                chron[i]['low'] < chron[i-2]['low'] and
                chron[i]['low'] < chron[i+1]['low'] and
                chron[i]['low'] < chron[i+2]['low']):
            swing_lows.append(chron[i]['low'])

    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return _default

    if len(swing_highs) >= 2:
        last_highs = swing_highs[-2:]
    else:
        last_highs = [swing_highs[0], swing_highs[0]]

    if len(swing_lows) >= 2:
        last_lows = swing_lows[-2:]
    else:
        last_lows = [swing_lows[0], swing_lows[0]]
    current_close = chron[-1]['close']

    higher_highs = last_highs[1] > last_highs[0]
    higher_lows = last_lows[1] > last_lows[0]
    lower_highs = last_highs[1] < last_highs[0]
    lower_lows = last_lows[1] < last_lows[0]

    if higher_highs and higher_lows:
        structure = "uptrend"
    elif lower_highs and lower_lows:
        structure = "downtrend"
    else:
        structure = "ranging"

    # CHoCH — Change of Character (trend reversal: price breaks last swing extreme)
    choch = False
    if structure == "uptrend" and current_close < last_lows[-1]:
        choch = True
        structure = "downtrend"
    elif structure == "downtrend" and current_close > last_highs[-1]:
        choch = True
        structure = "uptrend"

    # BOS — Break of Structure (trend continuation: price extends beyond last swing)
    # Requires candle BODY to close beyond swing — wick pokes don't count
    bos = False
    if structure == "uptrend":
        current_body_high = max(candles[0]['close'], candles[0]['open'])
        if current_body_high > last_highs[-1]:
            bos = True
        else:
            for c in candles[1:10]:
                if max(c['close'], c['open']) > last_highs[-1]:
                    bos = True
                    break
    elif structure == "downtrend":
        current_body_low = min(candles[0]['close'], candles[0]['open'])
        if current_body_low < last_lows[-1]:
            bos = True
        else:
            for c in candles[1:10]:
                if min(c['close'], c['open']) < last_lows[-1]:
                    bos = True
                    break

    return {
        "structure": structure,
        "choch": choch,
        "bos": bos,
        "last_swing_high": last_highs[-1],
        "last_swing_low": last_lows[-1],
    }


# ─── 22. UNIFIED TRADE DIRECTION ─────────────────────────────────────────────

def get_trade_direction(symbol: str, candles_15m: list, htf_bias_override: dict = None) -> tuple:
    """
    Single source of truth for trade direction — used by scanner and /bias.
    Combines daily bias with 15M market structure (swing high/low analysis).

    htf_bias_override: optional pre-computed result from get_htf_bias() (Daily/4H/1H
    aligned). When provided, its properly-timeframe-checked 'bias' field is used
    instead of the simpler Daily-only get_daily_bias() score. This matters because
    ICT/SMC top-down methodology (confirmed across multiple current sources) treats
    a 15M structure read that disagrees with Daily as ambiguous UNLESS the
    intermediate 4H/1H timeframes have ALSO turned — otherwise it's very likely
    just a normal retracement within an intact higher-timeframe trend, not a real
    reversal ("a move that looks like a reversal on the 15M is often just a
    retracement on the 4H... check two timeframes above your entry timeframe").
    Confirmed live: this exact gap left USDJPY blocked on 'htf_bias' for ~16 hours
    overnight and ~28 minutes on a separate signal, in both cases while Daily bias
    was genuinely correct — the old check only ever compared Daily directly
    against 15M, with no way to tell a real reversal from a normal pullback.
    get_htf_bias()'s own 'bias' field already implements the right hierarchy:
    fully aligned -> that direction; 1H/4H agree but Daily doesn't -> trust 1H/4H;
    1H/4H disagree with each other -> 'mixed' (mapped to neutral here, same
    treatment as an unclear daily bias).

    Returns (direction, strength):
      ('BUY',  'strong')  — bias bullish  + market structure uptrend
      ('SELL', 'strong')  — bias bearish  + market structure downtrend
      ('BUY',  'weak')    — bias neutral  + market structure uptrend
      ('SELL', 'weak')    — bias neutral  + market structure downtrend
      (None,   'ranging') — market structure ranging → skip
      (None,   'conflict')— bias conflicts with market structure → skip
    """
    if htf_bias_override and htf_bias_override.get('bias') not in (None, 'unclear'):
        _htf_b = htf_bias_override['bias']
        bias = 'neutral' if _htf_b == 'mixed' else _htf_b
        _bias_detail = htf_bias_override
    else:
        _daily_bias_fallback = get_daily_bias(symbol)
        bias = _daily_bias_fallback.get('bias', 'neutral')
        _bias_detail = _daily_bias_fallback
        htf_bias_override = None  # no h4/h1 detail available for the retracement check below
    ms = analyze_market_structure(candles_15m)
    structure = ms["structure"]

    if structure == 'ranging':
        return None, 'ranging'
    if bias == 'bullish' and structure == 'uptrend':
        return 'BUY', 'strong'
    elif bias == 'bearish' and structure == 'downtrend':
        return 'SELL', 'strong'
    elif bias == 'neutral' and structure == 'uptrend':
        return 'BUY', 'weak'
    elif bias == 'neutral' and structure == 'downtrend':
        return 'SELL', 'weak'
    elif bias == 'bullish' and structure == 'downtrend':
        # RETRACEMENT CHECK — before blocking, check whether 4H AND 1H (not just the
        # collapsed bias field) still BOTH confirm the original bullish direction.
        # If so, this 15M downtrend is very likely a normal retracement inside an
        # intact higher-timeframe uptrend, not a genuine reversal — exactly the
        # principle multiple current ICT/SMC sources describe: "a move that looks
        # like a reversal on the 15M is often just a retracement on the 4H... check
        # two timeframes above your entry timeframe; if structure on both remains
        # intact and pointing in the original direction, treat the move as a
        # retracement." Confirmed live: without this, USDJPY sat blocked on
        # 'htf_bias' for ~16 hours overnight and ~28 minutes on a separate signal,
        # in both cases very plausibly because 4H/1H never actually turned — only
        # the noisy 15M read kept flipping.
        if htf_bias_override and htf_bias_override.get('h4_trend') == 'bullish' and htf_bias_override.get('h1_trend') == 'bullish':
            logger.info(f"[scanner] {symbol} 15M downtrend vs bullish bias, but 4H+1H both still bullish — retracement, not conflict")
            return 'BUY', 'strong'
        logger.info(f"[scanner] {symbol} Gate 2 fail — HTF bias conflicts with 15M structure — no trade (bias={bias}, structure={structure}, daily_bias_detail={_bias_detail})")
        return None, 'conflict'
    elif bias == 'bearish' and structure == 'uptrend':
        # RETRACEMENT CHECK — same logic mirrored for SELL — see the BUY branch above.
        if htf_bias_override and htf_bias_override.get('h4_trend') == 'bearish' and htf_bias_override.get('h1_trend') == 'bearish':
            logger.info(f"[scanner] {symbol} 15M uptrend vs bearish bias, but 4H+1H both still bearish — retracement, not conflict")
            return 'SELL', 'strong'
        logger.info(f"[scanner] {symbol} Gate 2 fail — HTF bias conflicts with 15M structure — no trade (bias={bias}, structure={structure}, daily_bias_detail={_bias_detail})")
        return None, 'conflict'
    else:
        return None, 'ranging'


# ─── 23. DISPLACEMENT DETECTION ──────────────────────────────────────────────

def detect_displacement(candles: list, direction: str, symbol: str = "") -> dict | None:
    """
    Detect a strong displacement move in the last 20 candles.
    Displacement = 5+ consecutive same-direction candles, leaving an FVG behind.

    Returns displacement dict with fvg_top, fvg_bottom, fvg_mid (CE level),
    start_price, end_price, candle_count, direction. Returns None if not found.
    """
    if not candles or len(candles) < 10:
        return None

    for start_idx in range(0, min(15, len(candles) - 5)):
        consecutive = 0
        for i in range(start_idx, min(start_idx + 20, len(candles))):
            c = candles[i]
            if direction == 'BUY' and c['close'] > c['open']:
                consecutive += 1
            elif direction == 'SELL' and c['close'] < c['open']:
                consecutive += 1
            else:
                break

        if consecutive >= 5:
            disp_candles = candles[start_idx:start_idx + consecutive]

            if direction == 'BUY':
                fvg_bottom = disp_candles[-1]['low']
                fvg_top = disp_candles[0]['high']
                if fvg_top > fvg_bottom:
                    start_price = disp_candles[-1]['open']
                    end_price   = disp_candles[0]['close']
                    disp_range  = end_price - start_price
                    ote_high = end_price - (disp_range * 0.62)
                    ote_low  = end_price - (disp_range * 0.79)
                    return {
                        'direction': 'BUY',
                        'start_price': start_price,
                        'end_price': end_price,
                        'fvg_top': fvg_top,
                        'fvg_bottom': fvg_bottom,
                        'fvg_mid': (fvg_top + fvg_bottom) / 2,
                        'ote_high': ote_high,
                        'ote_low': ote_low,
                        'ote_mid': (ote_high + ote_low) / 2,
                        'candle_count': consecutive,
                    }
            else:
                fvg_top = disp_candles[-1]['high']
                fvg_bottom = disp_candles[0]['low']
                if fvg_top > fvg_bottom:
                    start_price = disp_candles[-1]['open']
                    end_price   = disp_candles[0]['close']
                    disp_range  = start_price - end_price
                    ote_high = end_price + (disp_range * 0.79)
                    ote_low  = end_price + (disp_range * 0.62)
                    return {
                        'direction': 'SELL',
                        'start_price': start_price,
                        'end_price': end_price,
                        'fvg_top': fvg_top,
                        'fvg_bottom': fvg_bottom,
                        'fvg_mid': (fvg_top + fvg_bottom) / 2,
                        'ote_high': ote_high,
                        'ote_low': ote_low,
                        'ote_mid': (ote_high + ote_low) / 2,
                        'candle_count': consecutive,
                    }

    return None


def is_price_in_displacement_fvg(current_price: float, displacement: dict) -> bool:
    """Check if price has retraced into the displacement FVG zone."""
    if not displacement:
        return False
    return displacement['fvg_bottom'] <= current_price <= displacement['fvg_top']


# ─── 24. DRAW ON LIQUIDITY ────────────────────────────────────────────────────

def get_draw_on_liquidity(symbol: str, candles: list, direction: str, asia_levels: dict) -> dict:
    """
    Identify the next draw on liquidity — where price is being pulled to.

    For BUY: next liquidity above current price
    - Asia High (if not yet swept)
    - Previous session high
    - Equal highs in last 20 candles

    For SELL: next liquidity below current price
    - Asia Low (if not yet swept)
    - Previous session low
    - Equal lows in last 20 candles

    Returns dict with:
    - level: price level of draw on liquidity
    - type: 'asia_high', 'asia_low', 'equal_highs', 'equal_lows', 'session_high', 'session_low'
    - distance_pips: how far price needs to travel
    """
    if not candles or not direction:
        return {}

    current_price = candles[0]['close']
    pip_spec = PIP_SPECS.get(symbol.upper(), {})
    if symbol.upper() in ('XAUUSD', 'US100', 'US30', 'US500'):
        pip_size = 1.0  # gold and index CFDs measured in points
    elif 'JPY' in symbol.upper():
        pip_size = 0.01
    else:
        pip_size = pip_spec.get('pip', 0.0001)

    # Minimum distance — draws closer than this are noise, not meaningful liquidity targets
    _MIN_DRAW_PIPS = {'USDJPY': 20, 'XAUUSD': 50, 'US100': 150, 'US30': 100, 'US500': 30}
    _min_draw_pips = _MIN_DRAW_PIPS.get(symbol.upper(), 15)

    draws = []

    if direction == 'BUY':
        # Asia High as draw
        if asia_levels and asia_levels.get('high', 0) > current_price:
            _dist = (asia_levels['high'] - current_price) / pip_size
            if _dist >= _min_draw_pips:
                draws.append({
                    'level': asia_levels['high'],
                    'type': 'asia_high',
                    'distance_pips': _dist,
                })

        # Equal highs in last 20 candles
        highs = [c['high'] for c in candles[:20]]
        for i in range(len(highs) - 1):
            for j in range(i + 1, len(highs)):
                if abs(highs[i] - highs[j]) < pip_size * 3:
                    level = max(highs[i], highs[j])
                    if level > current_price:
                        _dist = (level - current_price) / pip_size
                        if _dist >= _min_draw_pips:
                            draws.append({
                                'level': level,
                                'type': 'equal_highs',
                                'distance_pips': _dist,
                            })
                        break

        # Previous session high (last 96 candles = 24 hours on 15M)
        session_high = max(c['high'] for c in candles[1:97]) if len(candles) > 97 else 0
        if session_high > current_price:
            _dist = (session_high - current_price) / pip_size
            if _dist >= _min_draw_pips:
                draws.append({
                    'level': session_high,
                    'type': 'session_high',
                    'distance_pips': _dist,
                })

    else:  # SELL
        # Asia Low as draw
        if asia_levels and 0 < asia_levels.get('low', 0) < current_price:
            _dist = (current_price - asia_levels['low']) / pip_size
            if _dist >= _min_draw_pips:
                draws.append({
                    'level': asia_levels['low'],
                    'type': 'asia_low',
                    'distance_pips': _dist,
                })

        # Equal lows in last 20 candles
        lows = [c['low'] for c in candles[:20]]
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                if abs(lows[i] - lows[j]) < pip_size * 3:
                    level = min(lows[i], lows[j])
                    if level < current_price:
                        _dist = (current_price - level) / pip_size
                        if _dist >= _min_draw_pips:
                            draws.append({
                                'level': level,
                                'type': 'equal_lows',
                                'distance_pips': _dist,
                            })
                        break

        # Previous session low
        session_low = min(c['low'] for c in candles[1:97]) if len(candles) > 97 else 0
        if 0 < session_low < current_price:
            _dist = (current_price - session_low) / pip_size
            if _dist >= _min_draw_pips:
                draws.append({
                    'level': session_low,
                    'type': 'session_low',
                    'distance_pips': _dist,
                })

    if draws:
        nearest = min(draws, key=lambda x: x['distance_pips'])
        return nearest

    return {}


# ─── 25. WEEKLY HIGH/LOW SWEEP LEVELS ────────────────────────────────────────

_weekly_levels: dict = {}  # {symbol: {"high": float, "low": float, "fetched_at": float}}
_WEEKLY_LEVELS_TTL = 3600 * 6  # refresh every 6 hours

_WEEKLY_FUTURES_MAP = {
    'US100': 'NQ=F', 'US30': 'YM=F', 'US500': 'ES=F',
    'XAUUSD': 'GC=F', 'XAGUSD': 'SI=F',
}
_WEEKLY_FOREX_MAP = {
    'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X', 'NZDUSD': 'NZDUSD=X', 'USDCAD': 'USDCAD=X',
    'USDCHF': 'USDCHF=X', 'EURJPY': 'EURJPY=X', 'GBPJPY': 'GBPJPY=X',
}


def get_weekly_levels(symbol: str) -> dict:
    """
    Fetch previous week's high/low via yFinance weekly candles.
    Returns {"high": float, "low": float} or empty dict on failure.
    Cached for 6 hours.
    """
    import time as _t
    sym = symbol.upper()
    cached = _weekly_levels.get(sym)
    if cached and (_t.time() - cached.get("fetched_at", 0)) < _WEEKLY_LEVELS_TTL:
        return cached

    try:
        import yfinance as yf
        ticker = _WEEKLY_FUTURES_MAP.get(sym) or _WEEKLY_FOREX_MAP.get(sym)
        if not ticker:
            return {}
        hist = yf.Ticker(ticker).history(period="1mo", interval="1wk")
        if hist is None or len(hist) < 2:
            return {}
        prev_week = hist.iloc[-2]
        result = {
            "high": float(prev_week["High"]),
            "low": float(prev_week["Low"]),
            "fetched_at": _t.time(),
        }
        _weekly_levels[sym] = result
        logger.debug(f"[weekly_levels] {sym} prev week H={result['high']} L={result['low']}")
        return result
    except Exception as e:
        logger.debug(f"[weekly_levels] {symbol} fetch error: {e}")
        return {}


def detect_weekly_level_sweep(candles: list, direction: str, symbol: str = "") -> tuple[bool, float, str]:
    """
    Check if a liquidity sweep has occurred against the previous week's high or low.
    BUY: wick pierced below prev week low but closed back above it.
    SELL: wick pierced above prev week high but closed back below it.
    Higher-timeframe sweep = higher probability, larger stops triggered.
    Returns (detected, level, description).
    """
    if not candles or len(candles) < 3:
        return False, 0.0, ""

    sym = symbol.upper() if symbol else ""
    weekly = get_weekly_levels(sym)
    if not weekly:
        return False, 0.0, ""

    _fallback = PIP_SPECS[sym]["pip"] if sym in _FOREX_PIP_SPEC_PAIRS else 0.0
    min_pierce = _SWEEP_PIERCE_BUFFER.get(sym, _fallback)
    recent = candles[:12]

    if direction == "BUY":
        level = weekly["low"]
        for c in recent:
            if c["low"] < level - min_pierce and c["close"] > level:
                return True, round(level, 5), f"Weekly low sweep at {round(level, 5)}"
    else:  # SELL
        level = weekly["high"]
        for c in recent:
            if c["high"] > level + min_pierce and c["close"] < level:
                return True, round(level, 5), f"Weekly high sweep at {round(level, 5)}"

    return False, 0.0, ""


# ─── 26. ROUND NUMBER SWEEP ───────────────────────────────────────────────────

_ROUND_NUMBER_INTERVALS = {
    "EURUSD": 0.0050,   # 50 pip intervals: 1.1400, 1.1450, 1.1500
    "GBPUSD": 0.0050,
    "AUDUSD": 0.0050,
    "NZDUSD": 0.0050,
    "USDCAD": 0.0050,
    "USDCHF": 0.0050,
    "USDJPY": 0.50,     # 50 pip intervals: 161.00, 161.50
    "EURJPY": 0.50,
    "GBPJPY": 0.50,
    "XAUUSD": 50.0,     # 50 point intervals: 4150, 4200, 4250
    "XAGUSD": 0.50,
    "US100":  50.0,     # 50 point intervals: 21000, 21050
    "US30":   50.0,
    "US500":  50.0,
}


def get_nearest_round_numbers(symbol: str, price: float, count: int = 5) -> list:
    """Return round number levels above and below price for a symbol."""
    sym = symbol.upper()
    interval = _ROUND_NUMBER_INTERVALS.get(sym, 0.0050)
    base = round(price / interval) * interval
    return [round(base + interval * i, 8) for i in range(-count, count + 1)]


def detect_round_number_sweep(candles: list, direction: str, symbol: str = "") -> tuple[bool, float, str]:
    """
    Check if a sweep has occurred at a round number level.
    BUY: wick pierces below a round number but candle closes back above it.
    SELL: wick pierces above a round number but candle closes back below it.
    Round number touch = bonus sweep confirmation, higher signal strength.
    Returns (detected, round_level, description).
    """
    if not candles or len(candles) < 3:
        return False, 0.0, ""

    sym = symbol.upper() if symbol else ""
    interval = _ROUND_NUMBER_INTERVALS.get(sym, 0.0050)
    tolerance = interval * 0.10  # within 10% of interval counts as "touching"
    _fallback = PIP_SPECS[sym]["pip"] if sym in _FOREX_PIP_SPEC_PAIRS else 0.0
    min_pierce = _SWEEP_PIERCE_BUFFER.get(sym, _fallback)

    current_price = candles[0]["close"]
    round_levels = get_nearest_round_numbers(sym, current_price, count=5)
    recent = candles[:12]

    for c in recent:
        if direction == "BUY":
            for level in round_levels:
                if (c["low"] <= level + tolerance and
                        c["low"] >= level - interval * 0.5 - min_pierce and
                        c["close"] > level):
                    return True, round(level, 8), f"Round number sweep at {round(level, 5)}"
        else:  # SELL
            for level in round_levels:
                if (c["high"] >= level - tolerance and
                        c["high"] <= level + interval * 0.5 + min_pierce and
                        c["close"] < level):
                    return True, round(level, 8), f"Round number sweep at {round(level, 5)}"

    return False, 0.0, ""


# ─── 27. BOS DISPLACEMENT QUALITY SCORE ──────────────────────────────────────

# ─── 28. OB QUALITY SCORING ──────────────────────────────────────────────────

_mitigated_fvgs: dict = {}  # key: f"{symbol}_{fvg_low:.5f}_{fvg_high:.5f}"
_mitigated_obs:  dict = {}  # key: f"{symbol}_{ob_low:.5f}_{ob_high:.5f}"


def score_ob_quality(candle: dict, symbol: str = "") -> tuple[int, str]:
    """
    Score OB candle quality on candle anatomy (0-7 points).
    Returns (score, tier) where tier is 'S-tier'/'A-tier'/'B-tier'/'C-tier'.
    S=6-7, A=4-5, B=2-3, C=0-1
    """
    body  = abs(candle['close'] - candle['open'])
    total = candle['high'] - candle['low']
    if total == 0:
        return 0, "C-tier"
    score = 0

    body_ratio = body / total
    if body_ratio > 0.7:   score += 3
    elif body_ratio > 0.5: score += 2
    elif body_ratio > 0.3: score += 1

    if candle['close'] > candle['open']:
        close_pos = (candle['close'] - candle['low']) / total
    else:
        close_pos = (candle['high'] - candle['close']) / total
    if close_pos > 0.8:   score += 2
    elif close_pos > 0.6: score += 1

    if candle['close'] > candle['open']:
        opp_wick = (candle['high'] - candle['close']) / total
    else:
        opp_wick = (candle['close'] - candle['low']) / total
    if opp_wick < 0.1:   score += 2
    elif opp_wick < 0.2: score += 1

    if score >= 6:   tier = "S-tier"
    elif score >= 4: tier = "A-tier"
    elif score >= 2: tier = "B-tier"
    else:            tier = "C-tier"

    return score, tier


def is_fvg_mitigated(symbol: str, fvg_low: float, fvg_high: float) -> bool:
    """Return True if this exact FVG zone was already mitigated today."""
    # Round to 4dp to avoid float precision mismatches while still
    # distinguishing zones that are close but not identical
    key = f"{symbol}_{round(fvg_low, 4):.4f}_{round(fvg_high, 4):.4f}"
    return key in _mitigated_fvgs


def mark_fvg_mitigated(symbol: str, fvg_low: float, fvg_high: float) -> None:
    """Record that price has entered this FVG zone — treat as consumed."""
    key = f"{symbol}_{round(fvg_low, 4):.4f}_{round(fvg_high, 4):.4f}"
    _mitigated_fvgs[key] = True
    logger.info(f"[fvg] {symbol} FVG mitigated — removing")


def is_ob_mitigated(symbol: str, ob_low: float, ob_high: float) -> bool:
    """Return True if this OB zone was already mitigated (price entered it)."""
    key = f"{symbol}_{ob_low:.5f}_{ob_high:.5f}"
    return key in _mitigated_obs


def mark_ob_mitigated(symbol: str, ob_low: float, ob_high: float) -> None:
    """Record that price has entered this OB zone — treat as consumed."""
    key = f"{symbol}_{ob_low:.5f}_{ob_high:.5f}"
    _mitigated_obs[key] = True
    logger.info(f"[ob] {symbol} OB mitigated — removing")


def clear_daily_mitigation_state() -> None:
    """Clear FVG and OB mitigation dicts at start of a new day."""
    _mitigated_fvgs.clear()
    _mitigated_obs.clear()
    logger.info("[mitigation] Daily state cleared — fresh start")


def score_bos_quality(candles: list, direction: str, timeframe: str = "15M") -> tuple[str, int, str]:
    """
    Assess BOS displacement quality by counting consecutive candles in the BOS direction.
    Window tightened to 6 candles (90 min on 15M) — a BOS older than that is stale;
    the entry window has closed and any retrace to OB/FVG is likely to fail.
    Thresholds: 15M requires 3+ consecutive for "strong"; 5M requires 4+ because
    three consecutive 5M candles (15 min of one-way price) is producible from noise
    whereas three 15M candles (45 min) implies genuine institutional displacement.
    Returns (quality, count, signal_label) where signal_label is Telegram-ready.
    """
    if not candles or len(candles) < 2:
        return "weak", 0, "⚠️ BOS: confirmed (weak displacement)"

    expected_bullish = direction.upper() == "BUY"
    consecutive = 0
    from datetime import datetime, timezone as _tz
    _hour = datetime.now(_tz.utc).hour
    # Asian session (23-06 UTC): BOS takes longer due to lower liquidity
    # London/NY (06-20 UTC): 6-candle window sufficient for institutional moves
    _bos_window = 8 if (23 <= _hour or _hour < 6) else 6
    for c in candles[:_bos_window]:
        is_bull = c["close"] > c["open"]
        if is_bull == expected_bullish:
            consecutive += 1
        else:
            break

    # 5M noise is higher — require one extra candle before calling "strong"
    _strong_threshold = 4 if timeframe == "5M" else 3

    if consecutive >= _strong_threshold:
        return "strong", consecutive, "✅ BOS: confirmed (strong displacement)"
    if consecutive >= 2:
        return "moderate", consecutive, "⚠️ BOS: confirmed (moderate displacement)"
    return "weak", consecutive, "⚠️ BOS: confirmed (weak displacement — gate fail)"


# ─── 29. BREAKER BLOCK DETECTION (ICT Unicorn Model) ─────────────────────────

def detect_breaker_block(candles_15m: list, direction: str) -> dict | None:
    """
    Detect the most recent valid breaker block per ICT Unicorn Model spec.
    candles are newest-first.

    BULLISH BREAKER (BUY): Last down-close candle before a swing high that was
    subsequently swept, where price then broke ABOVE structure with displacement.
    That candle's body range becomes support on retest.

    BEARISH BREAKER (SELL): Last up-close candle before a swing low that was
    subsequently swept, where price then broke BELOW structure with displacement.
    That candle's body range becomes resistance on retest.

    Returns {'low': float, 'high': float, 'type': 'bullish_breaker'|'bearish_breaker',
             'swept_level': float} or None.
    """
    if not candles_15m or len(candles_15m) < 10:
        return None

    lookback = min(20, len(candles_15m))
    recent = candles_15m[:lookback]

    def _is_displacement(c: dict) -> bool:
        total_range = c["high"] - c["low"]
        if total_range == 0:
            return False
        body = abs(c["close"] - c["open"])
        return body / total_range >= 0.5

    if direction == "BUY":
        # Scan for swing highs: candle higher than 2 neighbours on each side
        for i in range(2, lookback - 2):
            sh = recent[i]["high"]
            if not (sh > recent[i-1]["high"] and sh > recent[i-2]["high"] and
                    sh > recent[i+1]["high"] and sh > recent[i+2]["high"]):
                continue

            # Swing high must have been swept by a newer candle (lower index)
            if not any(recent[j]["high"] > sh for j in range(0, i)):
                continue

            # Displacement BOS: strong bullish candle closing above the swing high
            if not any(
                recent[j]["close"] > sh and _is_displacement(recent[j])
                for j in range(0, i)
            ):
                continue

            # Find the last down-close candle older than the swing high (higher index)
            for k in range(i + 1, lookback):
                c = recent[k]
                if c["close"] < c["open"]:
                    return {
                        "low":  round(min(c["open"], c["close"]), 5),
                        "high": round(max(c["open"], c["close"]), 5),
                        "type": "bullish_breaker",
                        "swept_level": round(sh, 5),
                    }

    else:  # SELL
        # Scan for swing lows: candle lower than 2 neighbours on each side
        for i in range(2, lookback - 2):
            sl = recent[i]["low"]
            if not (sl < recent[i-1]["low"] and sl < recent[i-2]["low"] and
                    sl < recent[i+1]["low"] and sl < recent[i+2]["low"]):
                continue

            # Swing low must have been swept by a newer candle (lower index)
            if not any(recent[j]["low"] < sl for j in range(0, i)):
                continue

            # Displacement BOS: strong bearish candle closing below the swing low
            if not any(
                recent[j]["close"] < sl and _is_displacement(recent[j])
                for j in range(0, i)
            ):
                continue

            # Find the last up-close candle older than the swing low (higher index)
            for k in range(i + 1, lookback):
                c = recent[k]
                if c["close"] > c["open"]:
                    return {
                        "low":  round(min(c["open"], c["close"]), 5),
                        "high": round(max(c["open"], c["close"]), 5),
                        "type": "bearish_breaker",
                        "swept_level": round(sl, 5),
                    }

    return None


# ─── 30. ORB (OPENING RANGE BREAKOUT) DETECTION ──────────────────────────────
# Research-backed rules (tradethatswing.com, litefinance.org, crosstrade.io,
# buildalpha.com, damnpropfirms.com): 40-60% base win rate, 55%+ with HTF filter.
# "ORB's job is to catch trend days and skip the chop" — enters ON breakout,
# not on a pullback. Complements OB Retracement by catching displacement moves.

ORB_INSTRUMENTS = {"US100", "US30", "US500", "XAUUSD", "USDJPY"}

# Per-instrument kill zones to check for ORB (subset of _PAIR_KILL_ZONES).
# Indices are closed at London open — NY open only. Gold and JPY active in both.
ORB_KILL_ZONES_FOR_SYMBOL = {
    "US100":  ["ny_open"],
    "US30":   ["ny_open"],
    "US500":  ["ny_open"],
    "XAUUSD": ["london", "ny_open"],
    "USDJPY": ["london", "ny_open"],
}

# Session state — cleared organically when a new session's date+kz_name key is created.
# Key: (symbol_upper, date_str, kz_name)
# Value: {"high": float|None, "low": float|None, "long_fired": bool, "short_fired": bool}
_orb_state: dict = {}


def _parse_candle_dt(dt_str: str) -> "datetime | None":
    """Parse candle datetime string to timezone-aware UTC datetime."""
    try:
        dt = datetime.fromisoformat(str(dt_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def detect_orb_breakout(
    symbol: str,
    candles_15m: list,
    candles_5m: list,
    kz_name: str,
    daily_bias: dict,
) -> dict | None:
    """
    Detect an Opening Range Breakout (ORB) continuation signal.

    Rules (research-consistent):
    1. Opening range = first 15 min of the kill zone (kz_start_hour:00–:15 UTC).
    2. Entry trigger = 5M or 15M candle CLOSE beyond range high (BUY) or range low (SELL).
       Wick-only breaks do NOT count — confirmed close required.
    3. SL = opposite side of the opening range.
    4. TP = entry ± 2× range width (2R default, consistent with codebase convention).
    5. One trade per side per session per symbol (long_fired / short_fired flags).
    6. 75-minute cutoff after range close — no signal after deadline.
    7. HTF bias gate: only BUY when daily bias is confirmed bullish,
                      only SELL when daily bias is confirmed bearish.

    Returns signal dict or None.
    Log format: [orb] {symbol} {direction} breakout confirmed — range {high}-{low},
                entry={entry}, sl={sl}, tp={tp}
    """
    sym = symbol.upper()
    now = datetime.now(timezone.utc)
    today_str = str(now.date())
    state_key = (sym, today_str, kz_name)

    kz_bounds = _KILL_ZONES.get(kz_name)
    if kz_bounds is None:
        return None
    kz_start_hour = kz_bounds[0]

    range_open_dt  = now.replace(hour=kz_start_hour, minute=0,  second=0, microsecond=0)
    range_close_dt = now.replace(hour=kz_start_hour, minute=15, second=0, microsecond=0)
    deadline       = range_close_dt + timedelta(minutes=75)

    # Must be after the range has closed and before the deadline
    if now < range_close_dt or now > deadline:
        return None

    # HTF bias gate — check before building range to fail fast
    bias      = daily_bias.get("bias", "neutral")
    confirmed = daily_bias.get("confirmed", False)
    if not confirmed or bias == "neutral":
        return None

    # Init session state
    if state_key not in _orb_state:
        _orb_state[state_key] = {
            "high": None, "low": None,
            "long_fired": False, "short_fired": False,
        }
    state = _orb_state[state_key]

    # Both sides already traded — nothing left to do this session
    if state["long_fired"] and state["short_fired"]:
        return None

    # Build opening range from 15M candles in the range window
    range_candles = [
        c for c in candles_15m
        if (dt := _parse_candle_dt(c.get("datetime", ""))) is not None
        and range_open_dt <= dt < range_close_dt
    ]
    if range_candles:
        orb_high = max(c["high"] for c in range_candles)
        orb_low  = min(c["low"]  for c in range_candles)
        state["high"] = orb_high
        state["low"]  = orb_low
    elif state["high"] is not None and state["low"] is not None:
        orb_high = state["high"]
        orb_low  = state["low"]
    else:
        return None  # range not yet established

    if orb_high <= orb_low:
        return None

    range_width = orb_high - orb_low

    # Find the first confirming breakout close in the post-range window
    def _find_breakout(candles: list) -> dict | None:
        for c in candles:
            dt = _parse_candle_dt(c.get("datetime", ""))
            if dt is None:
                continue
            if not (range_close_dt <= dt < deadline):
                continue
            close_val = c["close"]
            if close_val > orb_high and not state["long_fired"] and bias == "bullish":
                return {"direction": "BUY",  "entry": close_val}
            if close_val < orb_low  and not state["short_fired"] and bias == "bearish":
                return {"direction": "SELL", "entry": close_val}
        return None

    # 5M first (faster confirmation), fall back to 15M
    raw = _find_breakout(candles_5m) or _find_breakout(candles_15m)
    if raw is None:
        return None

    direction = raw["direction"]
    entry_raw = raw["entry"]

    # Import scanner-level helpers via lazy import (avoids circular at module load)
    try:
        from scanner import FUTURES_SPOT_OFFSET as _FSO
        from scanner import _min_sl_dist, _max_sl_dist
    except Exception:
        _FSO = {}
        def _min_sl_dist(s): return 0.0   # noqa
        def _max_sl_dist(s): return 999.0  # noqa

    spot_off = _FSO.get(sym, 0)
    _is_pts  = sym in ("XAUUSD", "US100", "US30", "US500")
    _dp      = 3 if _is_pts else (3 if "JPY" in sym else 5)

    if direction == "BUY":
        sl_dist  = max(abs(entry_raw - orb_low), _min_sl_dist(sym))
        sl_dist  = min(sl_dist, _max_sl_dist(sym))
        sl_raw   = entry_raw - sl_dist
        tp1_raw  = entry_raw + range_width * 2.0
        tp2_raw  = entry_raw + range_width * 3.0
        state["long_fired"]  = True
    else:
        sl_dist  = max(abs(orb_high - entry_raw), _min_sl_dist(sym))
        sl_dist  = min(sl_dist, _max_sl_dist(sym))
        sl_raw   = entry_raw + sl_dist
        tp1_raw  = entry_raw - range_width * 2.0
        tp2_raw  = entry_raw - range_width * 3.0
        state["short_fired"] = True

    logger.info(
        f"[orb] {sym} {direction} breakout confirmed — "
        f"range {round(orb_high + spot_off, _dp)}-{round(orb_low + spot_off, _dp)}, "
        f"entry={round(entry_raw + spot_off, _dp)}, "
        f"sl={round(sl_raw + spot_off, _dp)}, "
        f"tp={round(tp1_raw + spot_off, _dp)}"
    )

    return {
        "direction":   direction,
        "entry":       round(entry_raw + spot_off, _dp),
        "sl":          round(sl_raw    + spot_off, _dp),
        "tp1":         round(tp1_raw   + spot_off, _dp),
        "tp2":         round(tp2_raw   + spot_off, _dp),
        "range_high":  round(orb_high  + spot_off, _dp),
        "range_low":   round(orb_low   + spot_off, _dp),
        "range_width": round(range_width, _dp),
        "kz_name":     kz_name,
    }
