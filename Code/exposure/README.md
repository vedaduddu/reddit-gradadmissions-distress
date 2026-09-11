# Exposure and candidate-control extraction

This stage uses the current Qwen v6 post labels and final filtered comments.
It does not modify its inputs.

Run on the server:

```bash
tmux new-session -d -s exposure \
"bash Code/exposure/run_exposure.sh"
```

Outputs are written to `Data/exposure/`:

- `anchor_comment_events.jsonl.gz`: all linked September--November comments in anchor threads;
- `non_anchor_comment_events.jsonl.gz`: all linked September--November comments in non-anchor threads;
- `treated_user_cycles.csv`: one treated observation per user-cycle, indexed at the first qualifying anchor comment;
- `candidate_control_user_community_cycles.csv`: unexposed non-anchor commenters available within each community-cycle matching pool;
- `excluded_anchor_author_user_cycles.csv`: otherwise relevant commenters excluded because they authored an anchor during that cycle; and
- `cohort_flow.json`: extraction, exclusion, and cohort counts.

Anchor authors are excluded from both treated and control cohorts within the
cycle in which they authored an anchor. A user exposed anywhere in the three
communities during a cycle cannot be a control in any community that cycle.
Repeated anchor comments remain as descriptive exposure counts but do not
create duplicate treated observations.
