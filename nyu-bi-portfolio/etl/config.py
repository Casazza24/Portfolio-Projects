"""Pipeline configuration. Change DB_URL for your local setup."""

DB_URL = "postgresql://localhost/nonprofit_analytics"

# GivingTuesday 990 Data Marts - research-ready CSVs
# https://nonprofitecosystem.givingtuesday.org/datamarts/
DATA_SOURCE_URL = "https://nonprofitecosystem.givingtuesday.org/datamarts/"

RAW_DIR = "data/raw"
