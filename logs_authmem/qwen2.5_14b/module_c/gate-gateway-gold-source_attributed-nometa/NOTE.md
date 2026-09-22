# Not dataset gold

In Module C there is no dataset gold label. The `gold` in this directory's name (and the
`"label_source": "gold"` inside `episodes.jsonl`) is an alias of `reference`: every label here was
produced by the judge model running the paper's source-first attribution prompt
(`new_src/bench/module_c.py`, `_to_records`). Do not cite these numbers as a gate running on
ground-truth labels.

Runs made after this note record `"label_source": "reference"` and print the same warning in
their summary; the directory name is kept only so existing plots still find the data.
