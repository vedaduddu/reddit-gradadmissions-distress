# Cleaning pipeline

Agam Goyal's filtration implementation is preserved in `bot_filter/` and
`ai_detector/`. Generated JSON reports, the trained detector artifact, and
filtered Reddit data are intentionally stored outside Git.

`verify_filtered_data.py` is a separate read-only verification step added for
this study. It does not reproduce or replace Agam's filtration logic.
