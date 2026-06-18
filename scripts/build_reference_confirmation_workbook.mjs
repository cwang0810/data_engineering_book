import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const dataPath = path.join(root, "publishing", "final_review", "reference_integrity_audit.json");
const outDir = path.join(root, "publishing", "final_review");
const outFile = path.join(outDir, "springer_reference_confirmation.xlsx");

const payload = JSON.parse(await fs.readFile(dataPath, "utf8"));
const references = payload.references || [];
const checks = payload.external_checks || [];
const uncited = payload.uncited_references || [];
const missingSameChapter = payload.missing_same_chapter_references || [];

const checkByRef = new Map(
  checks.map((row) => [`${row.file}:${row.line}:${row.entry_no}`, row]),
);
const uncitedKeys = new Set(
  uncited.map((row) => `${row.file}:${row.line}:${row.entry_no}`),
);

const labels = {
  review_id: "Review ID",
  part: "Part / Section",
  unit: "Unit",
  source_markdown: "Source Markdown",
  line: "Line",
  entry_no: "Reference No.",
  first_author: "First Author",
  year: "Year",
  title: "Parsed Title",
  entry: "Reference Entry",
  doi: "DOI",
  arxiv: "arXiv",
  url: "URL",
  citation_key: "Citation Key",
  external_status: "Machine Check Status",
  external_source: "Machine Check Source",
  external_identifier: "Matched Identifier",
  url_status: "URL Status",
  format_issues: "Format Issues",
  citation_issue: "Citation Issue",
  priority: "Priority",
  confirmation_status: "Confirmation Status",
  reviewer: "Reviewer",
  action_needed: "Action Needed",
  notes: "Notes / Evidence",
};

const columns = Object.keys(labels);

function columnName(n) {
  let name = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    name = String.fromCharCode(65 + rem) + name;
    n = Math.floor((n - 1) / 26);
  }
  return name;
}

function writeSheet(sheet, matrix, widths = []) {
  if (!matrix.length) return;
  const endCol = columnName(matrix[0].length);
  const used = sheet.getRange(`A1:${endCol}${matrix.length}`);
  used.values = matrix;
  used.format = {
    font: { name: "Arial", size: 10, color: "#111827" },
    borders: { preset: "all", style: "thin", color: "#D1D5DB" },
    verticalAlignment: "top",
    wrapText: true,
  };
  sheet.getRange(`A1:${endCol}1`).format = {
    fill: "#E2F0D9",
    font: { name: "Arial", size: 10, bold: true, color: "#111827" },
    borders: { preset: "all", style: "thin", color: "#9CA3AF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  for (let i = 0; i < widths.length; i += 1) {
    if (widths[i]) {
      sheet.getRange(`${columnName(i + 1)}:${columnName(i + 1)}`).format.columnWidth = widths[i];
    }
  }
}

function classifyFile(file) {
  const parts = file.split("/");
  const part = parts.find((item) => /^part\d+$/.test(item)) || "";
  const base = path.basename(file);
  const ch = base.match(/^ch(\d{2})_/);
  const project = base.match(/^p(\d{2})_/);
  const appendix = base.match(/^appendix_([a-z])_/);
  if (ch) return { part: part || "chapters", unit: `Ch${ch[1]}` };
  if (project) return { part: part || "projects", unit: `P${project[1]}` };
  if (appendix) return { part: "appendices", unit: `Appendix ${appendix[1].toUpperCase()}` };
  if (base === "afterword.md") return { part: "back matter", unit: "Afterword" };
  return { part: part || "front/back matter", unit: base.replace(/\.md$/, "") };
}

function priorityFor(ref, check, citationIssue) {
  const formatIssues = ref.format_issues || [];
  if (citationIssue.includes("missing same-chapter reference")) return "P0";
  if ((check?.status || "") === "metadata-mismatch" || (check?.status || "") === "url-problem") return "P1";
  if (formatIssues.includes("missing-doi-arxiv-url")) return "P1";
  if (formatIssues.includes("missing-year") || formatIssues.includes("missing-first-author")) return "P1";
  if (citationIssue || formatIssues.length) return "P2";
  return "P3";
}

function actionFor(ref, check, citationIssue) {
  const actions = [];
  const formatIssues = ref.format_issues || [];
  if (formatIssues.includes("missing-doi-arxiv-url")) actions.push("add DOI/arXiv/stable URL or record exception");
  if (formatIssues.includes("missing-year")) actions.push("add publication year");
  if (formatIssues.includes("missing-first-author")) actions.push("fix first author / organization name");
  if (formatIssues.includes("missing-terminal-period")) actions.push("fix terminal period");
  if (formatIssues.includes("url-trailing-punctuation")) actions.push("clean URL punctuation");
  if (citationIssue.includes("uncited")) actions.push("cite in body, remove, or mark as retained background reference");
  if (citationIssue.includes("missing same-chapter reference")) actions.push("add corresponding reference entry or revise body citation");
  if (check?.status === "metadata-mismatch") actions.push("verify title/year against matched record");
  if (check?.status === "url-problem") actions.push("replace or update unreachable URL");
  if (!actions.length) actions.push("confirm bibliographic details and mark Reviewed");
  return actions.join("; ");
}

function statusFor(priority) {
  if (priority === "P3") return "Ready for spot check";
  return "Needs human review";
}

const rows = references.map((ref, index) => {
  const key = `${ref.file}:${ref.line}:${ref.entry_no}`;
  const check = checkByRef.get(key) || {};
  const meta = classifyFile(ref.file);
  const citationIssues = [];
  if (uncitedKeys.has(key)) citationIssues.push("uncited in same chapter");
  const priority = priorityFor(ref, check, citationIssues.join("; "));
  return {
    review_id: `REF-${String(index + 1).padStart(4, "0")}`,
    part: meta.part,
    unit: meta.unit,
    source_markdown: ref.file,
    line: ref.line,
    entry_no: ref.entry_no,
    first_author: ref.first_author || "",
    year: ref.year || "",
    title: ref.title || "",
    entry: ref.entry || "",
    doi: ref.doi || "",
    arxiv: ref.arxiv || "",
    url: ref.url || "",
    citation_key: ref.key || "",
    external_status: check.status || "",
    external_source: check.source || "",
    external_identifier: check.identifier || "",
    url_status: check.url_status || "",
    format_issues: (ref.format_issues || []).join("; "),
    citation_issue: citationIssues.join("; "),
    priority,
    confirmation_status: statusFor(priority),
    reviewer: "",
    action_needed: actionFor(ref, check, citationIssues.join("; ")),
    notes: "",
  };
});

const byPriority = new Map();
const byUnit = new Map();
const issueCounts = new Map();
for (const row of rows) {
  byPriority.set(row.priority, (byPriority.get(row.priority) || 0) + 1);
  byUnit.set(row.unit, (byUnit.get(row.unit) || 0) + 1);
  for (const issue of `${row.format_issues}; ${row.citation_issue}`.split(";").map((item) => item.trim()).filter(Boolean)) {
    issueCounts.set(issue, (issueCounts.get(issue) || 0) + 1);
  }
}
for (const row of missingSameChapter) {
  issueCounts.set("body citation missing same-chapter reference", (issueCounts.get("body citation missing same-chapter reference") || 0) + 1);
}

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const confirm = workbook.worksheets.add("Reference Confirmation");
const issues = workbook.worksheets.add("Issue Counts");
const guidance = workbook.worksheets.add("Guidance");

writeSheet(summary, [
  ["Metric", "Value"],
  ["Generated at UTC", payload.generated_at_utc || ""],
  ["Scope", (payload.scope_roots || []).join("; ")],
  ["Reference rows", rows.length],
  ["Body author-year citations", payload.summary?.body_author_year_citations || 0],
  ["Missing same-chapter body citations", payload.summary?.missing_same_chapter_references || 0],
  ["Uncited same-chapter references", payload.summary?.uncited_references || 0],
  ["Duplicate reference groups", payload.summary?.duplicate_reference_groups || 0],
  ["Machine check status counts", JSON.stringify(payload.summary?.external_status_counts || {})],
  [],
  ["Priority", "Rows"],
  ...[...byPriority.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => [k, v]),
  [],
  ["Unit", "Rows"],
  ...[...byUnit.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => [k, v]),
], [36, 100]);

writeSheet(
  confirm,
  [
    columns.map((key) => labels[key]),
    ...rows.map((row) => columns.map((key) => row[key] ?? "")),
  ],
  [14, 16, 14, 42, 8, 12, 20, 10, 44, 78, 28, 20, 42, 22, 18, 20, 38, 12, 34, 28, 10, 24, 18, 52, 48],
);

writeSheet(issues, [
  ["Issue", "Rows"],
  ...[...issueCounts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])),
], [44, 12]);

writeSheet(guidance, [
  ["Field", "How to use"],
  ["Confirmation Status", "Change to Reviewed after the assigned editor confirms authors, title, year, venue, DOI/arXiv/URL, and same-chapter citation use."],
  ["Action Needed", "Suggested next action from the automatic audit. It is a work queue, not a final judgment."],
  ["Priority", "P0 blocks consistency; P1 needs bibliographic or traceability review; P2 is cleanup; P3 is ready for spot check."],
  ["Notes / Evidence", "Record DOI lookup, official publisher page, reason for keeping an uncited background reference, or exception rationale."],
], [30, 110]);

await fs.mkdir(outDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outFile);
console.log(outFile);
