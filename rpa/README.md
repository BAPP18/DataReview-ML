# RPA Layer — File-Based Enterprise Integration

## Overview

This RPA layer demonstrates enterprise automation without API integration.
It uses Power Automate Desktop or browser/UI automation for file-export workflows.

## Workflow

```
open browser
  -> login to demo/sandbox environment
  -> run saved report
  -> set filter
  -> export CSV
  -> save file to watched folder
```

Then:

```
watched folder
    -> ingestion job
    -> validation
```

## Implementation

### Option A: Power Automate Desktop

1. Open Power Automate Desktop
2. Create new flow: "Solar Data Export"
3. Add actions:
   - Launch new Chrome/Edge
   - Go to URL: [demo environment]
   - Enter credentials
   - Click "Run Report"
   - Set filter: [date range]
   - Click "Export CSV"
   - Save to: `D:\data viewer\data\watched\`
4. Schedule: Daily at 06:00

### Option B: Python + Selenium

See `rpa/selenium_export.py` for a Python-based alternative.

## Watched Folder Structure

```
data/watched/
├── crm/
├── erp/
├── partner/
└── document/
```

## Ingestion Trigger

When a new file appears in the watched folder:
1. Validate file format
2. Convert to canonical schema
3. Run data quality checks
4. Insert into PostgreSQL
5. Update review queue

## Notes

- No vendor API integration required
- Demonstrates realistic enterprise automation
- Suitable for portfolio demonstration
