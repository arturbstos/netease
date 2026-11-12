"""Feed position on one scale (written by notebook 01, read by every later step).

The raw impressPosition field mixes three numbering schemes, found in the audit of notebook 01:
  from_0   the first page of the feed is numbered 0,1,2,3 (top left, top right, bottom left, bottom right)
  from_1   the first page is numbered 1,2,3,4, as the data documentation describes
  doubled  two copies of the first page are logged at the same moment, each numbered 1,2,3,4
Pages further down the feed keep the scheme of the session they belong to. The scheme belongs to the page, not to the
user: most users with several pages see more than one scheme.

build() rewrites the impressions to {P}/impressions_place/dDD.parquet, one file per day, with
  impressPosition   the card's place in the feed counted from 0 on every page (raw number minus the page's offset)
  impressTime       unchanged, except that the second copy of a doubled page gets impressTime + 1 ms, so that every
                    step that identifies a page by (userId, impressTime) treats the two copies as two pages
  raw_position      the original impressPosition
  position_offset   0 or 1, the offset subtracted
  offset_source     'top'     : the page itself starts at 0 or 1
                    'session' : a deeper page, offset of the user's latest top page within the previous 30 minutes
                    'user'    : otherwise the offset the user's top pages have most often over the month
                    'default' : otherwise 1, the documented convention
Numbers that are skipped (a place taken by something that is not logged) stay skipped.
"""
import json
from pathlib import Path

SESSION_MS = 30 * 60 * 1000
OUT = "impressions_place"


def build(con, P, O, deadline=None):
    """Writes {P}/impressions_place/dDD.parquet for every day and returns a summary. Resumable: days already written
    are kept, so an interrupted run continues where it stopped. With deadline (a time.time() value) it returns None
    when time runs out; call it again to continue."""
    import time
    src = f"read_parquet('{P}/impressions/*.parquet')"
    out = Path(P) / OUT
    (out / "_top").mkdir(parents=True, exist_ok=True)
    for f in list(out.glob("tmp_*")) + list((out / "_top").glob("tmp_*")):
        f.unlink()                                     # a file left half-written by an interrupted run
    late = lambda: deadline is not None and time.time() > deadline
    # 1. top-of-feed pages per user and scheme, day by day (a page never spans two days, so memory stays small)
    for d in range(1, 31):
        f = out / "_top" / f"d{d:02d}.parquet"
        if f.exists():
            continue
        if late():
            return None
        con.execute(f"""COPY (SELECT userId, scheme, count(*) AS pages FROM (
            SELECT userId, CASE WHEN max(c) > 1 THEN 'doubled' WHEN min(impressPosition) = 0 THEN 'from_0' ELSE 'from_1' END AS scheme
            FROM (SELECT userId, impressTime, impressPosition, count(*) AS c FROM {src} WHERE dt = {d} GROUP BY 1, 2, 3)
            GROUP BY userId, impressTime HAVING min(impressPosition) <= 1) GROUP BY 1, 2) TO '{f}' (FORMAT parquet)""")
    top_pages = f"read_parquet('{out}/_top/d*.parquet')"
    # each user's usual offset: the offset of most of the user's top pages over the month (ties go to 1)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE user_offset AS
      SELECT userId, CASE WHEN sum(pages) FILTER (WHERE scheme = 'from_0') > coalesce(sum(pages) FILTER (WHERE scheme <> 'from_0'), 0)
                          THEN 0 ELSE 1 END AS off FROM {top_pages} GROUP BY userId""")
    # 2. rewrite the impressions day by day
    for d in range(1, 31):
        f = out / f"d{d:02d}.parquet"
        if f.exists():
            continue
        if late():
            return None
        con.execute(f"""COPY (
          WITH x AS (SELECT *, row_number() OVER (PARTITION BY userId, impressTime, impressPosition ORDER BY row_id) AS page_copy
                     FROM {src} WHERE dt = {d}),
          pg AS (SELECT userId, impressTime, page_copy, min(impressPosition) AS first_pos FROM x GROUP BY 1, 2, 3),
          top AS (SELECT userId, impressTime, (first_pos <> 0)::INT AS off FROM pg WHERE first_pos <= 1 AND page_copy = 1),
          o AS (SELECT p.userId, p.impressTime, p.page_copy,
                  CASE WHEN p.first_pos <= 1 THEN (p.first_pos <> 0)::INT
                       WHEN t.off IS NOT NULL AND p.impressTime - t.impressTime <= {SESSION_MS} THEN t.off
                       WHEN u.off IS NOT NULL THEN u.off ELSE 1 END AS off,
                  CASE WHEN p.first_pos <= 1 THEN 'top'
                       WHEN t.off IS NOT NULL AND p.impressTime - t.impressTime <= {SESSION_MS} THEN 'session'
                       WHEN u.off IS NOT NULL THEN 'user' ELSE 'default' END AS offset_source
                FROM pg p ASOF LEFT JOIN top t ON p.userId = t.userId AND p.impressTime >= t.impressTime
                LEFT JOIN user_offset u ON u.userId = p.userId)
          SELECT x.* EXCLUDE (page_copy, impressPosition, impressTime),
                 x.impressTime + x.page_copy - 1 AS impressTime,
                 greatest(x.impressPosition - o.off, 0)::INT AS impressPosition,
                 x.impressPosition AS raw_position, o.off::TINYINT AS position_offset, o.offset_source
          FROM x JOIN o USING (userId, impressTime, page_copy)
        ) TO '{f}' (FORMAT parquet)""")
    if late():
        return None
    # 3. a summary for the thesis: impressions by offset source and top-of-feed pages by scheme
    new = f"read_parquet('{out}/d*.parquet')"
    s = {"impressions_by_offset_source": dict(con.execute(f"SELECT offset_source, count(*) FROM {new} GROUP BY 1").fetchall()),
         "impressions": con.execute(f"SELECT count(*) FROM {new}").fetchone()[0],
         "impressions_raw": con.execute(f"SELECT count(*) FROM {src}").fetchone()[0],
         "top_pages_by_scheme": dict(con.execute(f"SELECT scheme, sum(pages)::BIGINT FROM {top_pages} GROUP BY 1").fetchall())}
    assert s["impressions"] == s["impressions_raw"], s
    (Path(O) / "01_position_fix.json").write_text(json.dumps(s, indent=1))
    (out / "_done").write_text("ok")
    return s
