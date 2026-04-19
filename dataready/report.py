# dataready/report.py

from datetime import datetime


class ReportGenerator:
    """Generates a standalone, self-contained HTML quality report."""

    def generate(self, df, scan_result, score_result,
                 output_path: str = "dataready_report.html") -> str:
        html = self._build_html(df, scan_result, score_result)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Report saved → {output_path}")
        return output_path

    # ── HTML builders ─────────────────────────────────────────────────────────

    def _build_html(self, df, scan, score) -> str:
        score_val  = score["score"]
        grade      = score["grade"]
        label      = score["label"]
        rows, cols = scan.shape
        now        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        score_color = (
            "#2cb67d" if score_val >= 90 else
            "#7c6af7" if score_val >= 75 else
            "#f7b731" if score_val >= 60 else
            "#f75757"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DataReady Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"Segoe UI",system-ui,sans-serif;background:#0f1117;color:#e0e0e0;padding:32px 24px}}
  .container{{max-width:980px;margin:0 auto}}
  h1{{font-size:24px;font-weight:600;margin-bottom:4px}}
  h2{{font-size:15px;font-weight:500;margin-bottom:14px;color:#aaa;
      border-bottom:1px solid #2e3150;padding-bottom:8px;margin-top:4px}}
  .meta{{font-size:12px;color:#666;margin-bottom:28px}}
  .card{{background:#1a1d2e;border:1px solid #2e3150;border-radius:10px;
         padding:20px;margin-bottom:18px}}
  /* score */
  .score-row{{display:flex;align-items:center;gap:24px;flex-wrap:wrap}}
  .score-big{{font-size:52px;font-weight:700;color:{score_color}}}
  .grade{{font-size:32px;font-weight:600;color:{score_color};margin-left:8px}}
  .score-label{{font-size:13px;color:#aaa;margin-top:4px}}
  .bar-wrap{{flex:1;min-width:120px;background:#252840;border-radius:4px;height:14px;overflow:hidden}}
  .bar-fill{{height:100%;border-radius:4px;background:{score_color};width:{score_val}%}}
  /* grids */
  .grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
  .grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
  @media(max-width:600px){{.grid3,.grid4{{grid-template-columns:repeat(2,1fr)}}}}
  .stat{{background:#252840;border-radius:8px;padding:12px 14px;text-align:center}}
  .stat .val{{font-size:22px;font-weight:600;color:#7c6af7}}
  .stat .lbl{{font-size:11px;color:#666;margin-top:2px}}
  /* issues */
  .issue{{border-left:3px solid;padding:12px 14px;margin-bottom:10px;
          border-radius:0 6px 6px 0;background:#13151f}}
  .issue.critical{{border-color:#f75757}}
  .issue.warning{{border-color:#f7b731}}
  .issue.info{{border-color:#7c6af7}}
  .badge{{display:inline-block;font-size:10px;padding:2px 8px;
          border-radius:10px;font-weight:600;margin-right:6px}}
  .badge-critical{{background:rgba(247,87,87,.15);color:#f75757}}
  .badge-warning{{background:rgba(247,183,49,.15);color:#f7b731}}
  .badge-info{{background:rgba(124,106,247,.15);color:#7c6af7}}
  .issue-desc{{font-size:13px;margin:4px 0}}
  .issue-cols{{font-size:11px;color:#666;margin-bottom:4px}}
  .issue-fix{{font-size:12px;color:#2cb67d}}
  .issue-note{{font-size:12px;color:#f7b731}}
  /* table */
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{background:#252840;padding:8px 10px;text-align:left;color:#aaa;font-weight:500}}
  td{{padding:7px 10px;border-bottom:1px solid #1e2130;color:#ccc;word-break:break-word}}
  tr:last-child td{{border-bottom:none}}
  /* breakdown */
  .bd-item{{display:flex;justify-content:space-between;font-size:13px;
            padding:6px 0;border-bottom:1px solid #1e2130}}
  .bd-item:last-child{{border-bottom:none}}
  footer{{text-align:center;font-size:11px;color:#444;margin-top:28px}}
</style>
</head>
<body>
<div class="container">
  <h1>DataReady — Quality Report</h1>
  <div class="meta">
    Generated: {now} &nbsp;|&nbsp; Shape: {rows:,} rows × {cols} columns
    &nbsp;|&nbsp; Issues: {len(scan.issues)}
  </div>

  <div class="card">
    <h2>Quality Score</h2>
    <div class="score-row">
      <div>
        <span class="score-big">{score_val}</span>
        <span class="grade">{grade}</span>
        <div class="score-label">{label}</div>
      </div>
      <div class="bar-wrap"><div class="bar-fill"></div></div>
    </div>
  </div>

  <div class="card">
    <h2>Overview</h2>
    {self._overview_html(scan)}
  </div>

  <div class="card">
    <h2>Score Breakdown</h2>
    {self._breakdown_html(score)}
  </div>

  <div class="card">
    <h2>Issues ({len(scan.issues)} found)</h2>
    {self._issues_html(scan)}
  </div>

  <div class="card">
    <h2>Column Details</h2>
    {self._columns_html(df, scan)}
  </div>

  <footer>Generated by DataReady &mdash; open-source ML data quality library</footer>
</div>
</body>
</html>"""

    def _overview_html(self, scan) -> str:
        stats = [
            ("Null cols",        len(scan.null_counts)),
            ("Duplicate rows",   f"{scan.duplicate_row_count:,}"),
            ("High skew cols",   len(scan.high_skew_cols)),
            ("Outlier cols",     len(scan.outlier_counts_iqr)),
            ("High cardinality", len(scan.high_cardinality_cols)),
            ("ID columns",       len(scan.id_cols)),
            ("Corr. pairs",      len(scan.high_corr_pairs)),
            ("Leakage suspects", len(scan.leakage_suspects)),
            ("Encoding issues",  len(scan.encoding_issue_cols)),
            ("Bool-as-string",   len(scan.boolean_as_string)),
            ("Num-as-string",    len(scan.numeric_as_string)),
            ("Whitespace cols",  len(scan.whitespace_cols)),
        ]
        items = "".join(
            f'<div class="stat">'
            f'<div class="val">{v}</div>'
            f'<div class="lbl">{l}</div>'
            f'</div>'
            for l, v in stats
        )
        return f'<div class="grid4">{items}</div>'

    def _issues_html(self, scan) -> str:
        if not scan.issues:
            return '<p style="color:#2cb67d">✓ No issues found.</p>'

        grouped: dict[str, list] = {"critical": [], "warning": [], "info": []}
        for issue in scan.issues:
            grouped[issue.severity].append(issue)

        html = ""
        for sev in ("critical", "warning", "info"):
            for issue in grouped[sev]:
                cols_str = (
                    ", ".join(f"<code>{c}</code>" for c in issue.columns)
                    if issue.columns else "dataset-wide"
                )
                fix_html = (
                    f'<div class="issue-fix">✓ Fix: {issue.fix_description}</div>'
                    if issue.fix_available else
                    f'<div class="issue-note">⚠ Note: {issue.fix_description}</div>'
                )
                html += f"""<div class="issue {sev}">
  <div><span class="badge badge-{sev}">{sev.upper()}</span>
    <strong style="font-size:12px">{issue.issue_type}</strong></div>
  <div class="issue-desc">{issue.description}</div>
  <div class="issue-cols">Columns: {cols_str}</div>
  {fix_html}
</div>"""
        return html

    def _columns_html(self, df, scan) -> str:
        rows = ""
        for col in df.columns:
            dtype    = str(df[col].dtype)
            inferred = scan.inferred_types.get(col, "unknown")
            nulls    = scan.null_pcts.get(col, 0)
            card     = scan.cardinality.get(col, "?")
            skew_val = scan.skew_values.get(col, "")
            skew_str = f"{skew_val:.2f}" if isinstance(skew_val, float) else ""
            flags = []
            if col in scan.id_cols:              flags.append("ID")
            if col in scan.high_cardinality_cols:flags.append("high-card")
            if col in scan.numeric_as_string:    flags.append("num-as-str")
            if col in scan.boolean_as_string:    flags.append("bool-as-str")
            if col in scan.datetime_as_string:   flags.append("dt-as-str")
            if col in scan.leakage_suspects:     flags.append("leakage")
            if col in scan.encoding_issue_cols:  flags.append("encoding")
            flag_str = " ".join(
                f'<span style="background:#252840;padding:1px 5px;border-radius:4px;'
                f'font-size:10px;color:#aaa">{f}</span>'
                for f in flags
            )
            rows += f"""<tr>
  <td>{col}</td><td>{dtype}</td><td>{inferred}</td>
  <td>{nulls}%</td><td>{f"{card:,}" if isinstance(card, int) else card}</td><td>{skew_str}</td><td>{flag_str}</td>
</tr>"""
        return f"""<table>
  <thead><tr>
    <th>Column</th><th>dtype</th><th>Inferred</th>
    <th>Null %</th><th>Unique</th><th>Skew</th><th>Flags</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>"""

    def _breakdown_html(self, score) -> str:
        if not score["breakdown"]:
            return '<p style="color:#2cb67d">No deductions — perfect score!</p>'
        html = ""
        for reason, ded in sorted(score["breakdown"].items(), key=lambda x: -x[1]):
            html += (
                f'<div class="bd-item">'
                f'<span>{reason}</span>'
                f'<span style="color:#f75757">-{ded:.1f}</span>'
                f'</div>'
            )
        html += (
            f'<div class="bd-item" style="font-weight:600;margin-top:4px">'
            f'<span>Final score</span>'
            f'<span style="color:#2cb67d">{score["score"]}/100</span>'
            f'</div>'
        )
        return html