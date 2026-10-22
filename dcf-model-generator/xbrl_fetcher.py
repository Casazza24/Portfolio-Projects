"""Fetch standardized financial data from SEC EDGAR XBRL API."""

import pandas as pd
from edgar import lookup_cik, sec_get

CONCEPT_MAP = {
    # Income Statement
    'Revenues': [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'SalesRevenueNet',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
    ],
    'CostOfRevenue': ['CostOfRevenue', 'CostOfGoodsAndServicesSold'],
    'GrossProfit': ['GrossProfit'],
    'OperatingIncome': ['OperatingIncomeLoss'],
    'NetIncome': ['NetIncomeLoss'],
    'InterestExpense': ['InterestExpense', 'InterestExpenseDebt'],

    # Balance Sheet
    'TotalAssets': ['Assets'],
    'TotalDebt': ['LongTermDebt', 'LongTermDebtAndCapitalLeaseObligations'],
    'Cash': ['CashAndCashEquivalentsAtCarryingValue'],
    'TotalLiabilities': ['Liabilities'],

    # Cash Flow
    'OperatingCashFlow': ['NetCashProvidedByOperatingActivities'],
    'CapEx': ['PaymentsToAcquirePropertyPlantAndEquipment'],
    'DepreciationAmortization': ['DepreciationDepletionAndAmortization'],
}

# Balance-sheet concepts are point-in-time ("instant") facts with no start date.
INSTANT_FIELDS = {'TotalAssets', 'TotalDebt', 'Cash', 'TotalLiabilities'}

ANNUAL_FORMS = {'10-K', '10-K/A'}
ANNUAL_DAYS = (350, 380)  # 52/53-week fiscal years land in this band

# XBRL field -> (hist_df label, compare absolute values). TotalDebt is left
# out on purpose: XBRL LongTermDebt excludes current debt and leases, so it
# is not comparable to yfinance "Total Debt".
FIELD_TO_YF_LABEL = {
    'Revenues': ('Total Revenue', False),
    'CostOfRevenue': ('Cost Of Revenue', False),
    'GrossProfit': ('Gross Profit', False),
    'OperatingIncome': ('Operating Income', False),
    'NetIncome': ('Net Income', False),
    'Cash': ('Cash And Cash Equivalents', False),
    'CapEx': ('Capital Expenditure', True),
    'DepreciationAmortization': ('Depreciation And Amortization', False),
}

PERIOD_MATCH_DAYS = 10


def fetch_xbrl_facts(ticker):
    """Pull all XBRL facts for a ticker from SEC EDGAR."""
    cik, _ = lookup_cik(ticker)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return sec_get(url).json()


def _is_annual_observation(entry, instant):
    """True if the fact covers one full fiscal year (or is a year-end instant)."""
    if entry.get('form') not in ANNUAL_FORMS or 'end' not in entry:
        return False
    if instant:
        return 'start' not in entry
    if 'start' not in entry:
        return False
    days = (pd.Timestamp(entry['end']) - pd.Timestamp(entry['start'])).days
    return ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]


def extract_annual_financials(facts_json, n_years=4):
    """Extract annual values for key line items, keyed by fiscal period end.

    The `fy` tag on a fact is the fiscal year of the *filing*, so one 10-K
    tags its current year and its comparative prior years with the same fy.
    Observations are therefore keyed by the reporting period's end date.
    When several filings report the same period (comparatives in later
    10-Ks, 10-K/A amendments, restatements), the most recently filed value
    wins. Candidate concepts are merged, earlier candidates taking priority
    for a given period, since companies switch tags over time.

    Returns {field_name: {period_end (Timestamp): value}} for the most recent
    n_years periods.
    """
    us_gaap = facts_json.get('facts', {}).get('us-gaap', {})
    results = {}

    for field_name, concept_candidates in CONCEPT_MAP.items():
        instant = field_name in INSTANT_FIELDS
        by_period = {}  # end -> (priority, entry)
        for priority, concept_key in enumerate(concept_candidates):
            usd_entries = us_gaap.get(concept_key, {}).get('units', {}).get('USD', [])
            for entry in usd_entries:
                if not _is_annual_observation(entry, instant):
                    continue
                end = pd.Timestamp(entry['end'])
                current = by_period.get(end)
                if current is None:
                    by_period[end] = (priority, entry)
                    continue
                cur_priority, cur_entry = current
                if priority < cur_priority:
                    by_period[end] = (priority, entry)
                elif priority == cur_priority and entry['filed'] > cur_entry['filed']:
                    by_period[end] = (priority, entry)

        if by_period:
            recent = sorted(by_period, reverse=True)[:n_years]
            results[field_name] = {end: by_period[end][1]['val'] for end in recent}

    return results


def _match_period(xbrl_end, yf_cols):
    """Find the yfinance column whose period end is within PERIOD_MATCH_DAYS."""
    best = None
    for col in yf_cols:
        gap = abs((pd.Timestamp(col) - xbrl_end).days)
        if gap <= PERIOD_MATCH_DAYS and (best is None or gap < best[0]):
            best = (gap, col)
    return None if best is None else best[1]


def validate_against_yfinance(ticker, xbrl_data, hist_df, tolerance=0.05):
    """Compare XBRL values against yfinance hist_df. Return discrepancy report.

    Statuses: MATCH, MISMATCH, XBRL_MISSING (field or period absent from SEC
    data), YF_MISSING (absent from yfinance), NONCOMPARABLE (e.g. XBRL value
    of zero, so a relative difference is undefined).

    Args:
        tolerance: fractional difference threshold (0.05 = 5%)

    Returns:
        list of dicts with field, year, period_end, xbrl_val, yf_val, pct_diff, status
    """
    yf_cols = [c for c in hist_df.columns if hasattr(c, 'year')]

    def entry(field, period_end, xbrl_val, yf_val, pct_diff, status):
        return {
            'field': field,
            'year': None if period_end is None else period_end.year,
            'period_end': period_end,
            'xbrl_val': xbrl_val, 'yf_val': yf_val,
            'pct_diff': pct_diff, 'status': status,
        }

    report = []
    for field, (yf_label, use_abs) in FIELD_TO_YF_LABEL.items():
        field_data = xbrl_data.get(field, {})
        matched_cols = set()

        for period_end, xbrl_val in field_data.items():
            col = _match_period(pd.Timestamp(period_end), yf_cols)
            if col is None:
                continue  # XBRL period outside the yfinance window; not expected
            matched_cols.add(col)

            yf_val = hist_df.loc[yf_label, col] if yf_label in hist_df.index else None
            if yf_val is None or pd.isna(yf_val):
                report.append(entry(field, pd.Timestamp(col), xbrl_val, None, None, 'YF_MISSING'))
                continue

            x, y = (abs(xbrl_val), abs(yf_val)) if use_abs else (xbrl_val, yf_val)
            if x == 0:
                report.append(entry(field, pd.Timestamp(col), xbrl_val, yf_val, None, 'NONCOMPARABLE'))
                continue
            pct_diff = abs(x - y) / abs(x)
            report.append(entry(
                field, pd.Timestamp(col), xbrl_val, yf_val, pct_diff,
                'MATCH' if pct_diff <= tolerance else 'MISMATCH',
            ))

        for col in yf_cols:
            if col not in matched_cols:
                yf_val = hist_df.loc[yf_label, col] if yf_label in hist_df.index else None
                report.append(entry(field, pd.Timestamp(col), None, yf_val, None, 'XBRL_MISSING'))

    return report


def summarize_xbrl_report(report, min_coverage=0.5):
    """Classify a cross-check as VERIFIED, DISCREPANCIES, INCONCLUSIVE or UNAVAILABLE.

    "No mismatches" is not enough to call the data verified: at least
    min_coverage of the expected field-years must actually be compared, and
    the latest year's revenue must be among the matches.
    """
    if not report:
        return {'status': 'UNAVAILABLE', 'n_match': 0, 'n_mismatch': 0,
                'n_missing': 0, 'n_noncomparable': 0, 'coverage': 0.0,
                'reason': 'No XBRL data to compare.'}

    counts = {s: sum(1 for r in report if r['status'] == s)
              for s in ('MATCH', 'MISMATCH', 'XBRL_MISSING', 'YF_MISSING', 'NONCOMPARABLE')}
    n_compared = counts['MATCH'] + counts['MISMATCH']
    coverage = n_compared / len(report)

    latest_end = max((r['period_end'] for r in report if r['period_end'] is not None), default=None)
    latest_revenue_matched = any(
        r['field'] == 'Revenues' and r['period_end'] == latest_end and r['status'] == 'MATCH'
        for r in report
    )

    if counts['MISMATCH']:
        status = 'DISCREPANCIES'
        reason = f"{counts['MISMATCH']} value(s) differ by more than tolerance."
    elif n_compared == 0:
        status = 'UNAVAILABLE'
        reason = 'No overlapping field-years between SEC XBRL and Yahoo Finance.'
    elif coverage < min_coverage or not latest_revenue_matched:
        status = 'INCONCLUSIVE'
        reason = (f"Only {n_compared}/{len(report)} field-years compared ({coverage:.0%})"
                  + ("" if latest_revenue_matched else "; latest-year revenue not verified") + ".")
    else:
        status = 'VERIFIED'
        reason = f"{n_compared}/{len(report)} field-years compared ({coverage:.0%}), all within tolerance."

    return {'status': status, 'n_match': counts['MATCH'], 'n_mismatch': counts['MISMATCH'],
            'n_missing': counts['XBRL_MISSING'] + counts['YF_MISSING'],
            'n_noncomparable': counts['NONCOMPARABLE'], 'coverage': coverage, 'reason': reason}
