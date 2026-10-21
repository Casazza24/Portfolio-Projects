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


def fetch_xbrl_facts(ticker):
    """Pull all XBRL facts for a ticker from SEC EDGAR."""
    cik, _ = lookup_cik(ticker)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return sec_get(url).json()


def extract_annual_financials(facts_json, n_years=4):
    """Extract annual (10-K) values for key financial line items.

    Returns a dict of {field_name: {year: value}} for the most recent n_years.
    """
    us_gaap = facts_json.get('facts', {}).get('us-gaap', {})
    results = {}

    for field_name, concept_candidates in CONCEPT_MAP.items():
        for concept_key in concept_candidates:
            if concept_key not in us_gaap:
                continue

            units = us_gaap[concept_key].get('units', {})
            usd_entries = units.get('USD', [])
            if not usd_entries:
                continue

            annual = [
                e for e in usd_entries
                if e.get('form') == '10-K'
                and 'fy' in e
                and e.get('fp') == 'FY'
            ]
            if not annual:
                continue

            by_year = {}
            for entry in annual:
                fy = entry['fy']
                if fy not in by_year or entry['filed'] > by_year[fy]['filed']:
                    by_year[fy] = entry

            sorted_years = sorted(by_year.keys(), reverse=True)[:n_years]
            results[field_name] = {y: by_year[y]['val'] for y in sorted_years}
            break

    return results


def validate_against_yfinance(ticker, xbrl_data, hist_df, tolerance=0.05):
    """Compare XBRL values against yfinance hist_df. Return discrepancy report.

    Args:
        tolerance: fractional difference threshold (0.05 = 5%)

    Returns:
        list of dicts with field, year, xbrl_val, yf_val, pct_diff, status
    """
    field_to_yf_label = {
        'Revenues': 'Total Revenue',
        'GrossProfit': 'Gross Profit',
        'NetIncome': 'Net Income',
        'TotalAssets': None,
        'Cash': 'Cash And Cash Equivalents',
        'CostOfRevenue': 'Cost Of Revenue',
    }

    yf_years = {}
    for col in hist_df.columns:
        if hasattr(col, 'year'):
            yf_years[col.year] = col

    report = []
    for field, yf_label in field_to_yf_label.items():
        if yf_label is None:
            continue
        if field not in xbrl_data:
            report.append({'field': field, 'year': None, 'xbrl_val': None,
                           'yf_val': None, 'pct_diff': None, 'status': 'XBRL_MISSING'})
            continue

        for year, xbrl_val in xbrl_data[field].items():
            if year not in yf_years:
                report.append({'field': field, 'year': year, 'xbrl_val': xbrl_val,
                               'yf_val': None, 'pct_diff': None, 'status': 'YF_MISSING'})
                continue

            col = yf_years[year]
            try:
                yf_val = hist_df.loc[yf_label, col]
                if pd.isna(yf_val):
                    yf_val = None
            except KeyError:
                yf_val = None

            if yf_val is None:
                report.append({'field': field, 'year': year, 'xbrl_val': xbrl_val,
                               'yf_val': None, 'pct_diff': None, 'status': 'YF_MISSING'})
                continue

            if xbrl_val == 0 and yf_val == 0:
                pct_diff = 0.0
            elif xbrl_val == 0:
                pct_diff = float('inf')
            else:
                pct_diff = abs(xbrl_val - yf_val) / abs(xbrl_val)

            report.append({
                'field': field,
                'year': year,
                'xbrl_val': xbrl_val,
                'yf_val': yf_val,
                'pct_diff': pct_diff,
                'status': 'MATCH' if pct_diff <= tolerance else 'MISMATCH',
            })

    return report
