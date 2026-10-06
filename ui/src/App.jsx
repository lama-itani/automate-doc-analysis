import React, { useState, useRef, useEffect, useCallback } from "react";

/* ---------- API ---------- */
// Relative URLs: works at the Application root and under a path prefix.
const POLL_MS = 5000;
const ACCEPT = ".pdf,.png,.jpg,.jpeg,.tif,.tiff";

async function api(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch (e) {
    throw new Error(`Network error: ${e.message}`);
  }
  const text = await res.text();
  let body = null;
  if (text) {
    try { body = JSON.parse(text); } catch { body = null; }
  }
  if (!res.ok) {
    const detail = body && body.detail
      ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail))
      : text || res.statusText;
    throw new Error(`${res.status}: ${detail}`);
  }
  return body;
}

const casePath = (id) => `api/cases/${encodeURIComponent(id)}`;

/* ---------- Bilingual label helper ---------- */
// lang: "both" | "es" | "en". `t` is {es, en}.
function L({ t, lang }) {
  if (!t) return null;
  if (lang === "es") return <>{t.es}</>;
  if (lang === "en") return <>{t.en}</>;
  if (!t.en || t.en === t.es) return <>{t.es}</>;
  return (
    <>
      {t.es} <span className="sub">({t.en})</span>
    </>
  );
}

const UI = {
  upload: { es: "Cargar expediente", en: "Upload case file" },
  process: { es: "Procesar expediente", en: "Process case file" },
  caseId: { es: "Número de expediente", en: "Case number" },
  cases: { es: "Expedientes", en: "Cases" },
  progress: { es: "Progreso", en: "Progress" },
  noVerdict: { es: "Sin veredicto", en: "No verdict" },
  failedDocs: { es: "Documentos con error", en: "Documents with errors" },
  verdict: { es: "Veredicto", en: "Verdict" },
  issues: { es: "Problemas", en: "Issues" },
  inventory: { es: "Inventario de documentos", en: "Document inventory" },
  docs: { es: "Documentos", en: "Documents" },
  lineage: { es: "Linaje", en: "Lineage" },
  present: { es: "Presente", en: "Present" },
  missing: { es: "Faltante", en: "Missing" },
  cause: { es: "Causa", en: "Cause" },
  hint: { es: "Posible origen", en: "Possible origin" },
  evidence: { es: "Documentos implicados", en: "Documents involved" },
  undetermined: { es: "No determinado", en: "Not determined" },
  techToggle: { es: "Detalles técnicos", en: "Technical details" },
  retry: { es: "Reintentar", en: "Retry" },
};

const VERDICT = {
  VERDE: { cls: "green", es: "Aprobado", en: "Approved" },
  AMARILLO: { cls: "amber", es: "Revisión manual", en: "Manual review" },
  ROJO: { cls: "red", es: "Rechazado", en: "Rejected" },
};

const TYPE_LABEL = {
  ID_DOCUMENT: { es: "Identificación", en: "ID" },
  APPLICATION: { es: "Solicitud", en: "Application" },
  BIRTH_CERT: { es: "Cert. nacimiento", en: "Birth cert." },
  OTHER: { es: "Otro", en: "Other" },
  EMPTY: { es: "Vacío", en: "Empty" },
};

const stagePill = (stage) => (stage === "OCR_DONE" ? "ok" : stage === "FAILED" ? "bad" : "");

/* ---------- Top bar ---------- */
function TopBar({ tech, setTech }) {
  return (
    <header className="topbar">
      <div className="brand">PS-06 <span>Case Review</span></div>
      <div className="controls">
        <label className="switch">
          <input type="checkbox" checked={tech} onChange={(e) => setTech(e.target.checked)} />
          <L t={UI.techToggle} lang="both" />
        </label>
      </div>
    </header>
  );
}

/* ---------- Error box ---------- */
function ErrorBox({ message, onRetry, lang }) {
  if (!message) return null;
  return (
    <section className="banner red">
      <div className="lamp" />
      <div>
        <p>{message}</p>
        {onRetry && (
          <button className="primary" onClick={onRetry}><L t={UI.retry} lang={lang} /></button>
        )}
      </div>
    </section>
  );
}

/* ---------- Upload ---------- */
function UploadCard({ lang, onStarted }) {
  const [caseId, setCaseId] = useState("");
  const [files, setFiles] = useState([]);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const input = useRef(null);

  const pick = (list) => { setFiles(Array.from(list)); setError(null); };

  const submit = async () => {
    const id = caseId.trim();
    setSending(true);
    setError(null);
    const form = new FormData();
    form.append("case_id", id);
    files.forEach((f) => form.append("files", f, f.name));
    try {
      await api("api/cases", { method: "POST", body: form });
      setFiles([]);
      setCaseId("");
      onStarted(id);
    } catch (e) {
      setError(e.message);
    } finally {
      setSending(false);
    }
  };

  return (
    <section className="card">
      <h2><L t={UI.upload} lang={lang} /></h2>
      <label style={{ display: "block", marginBottom: 10 }}>
        <L t={UI.caseId} lang={lang} />
        <input
          type="text"
          value={caseId}
          onChange={(e) => setCaseId(e.target.value)}
          placeholder="985356"
          style={{ display: "block", width: "100%", marginTop: 4, padding: "8px 10px", border: "1px solid var(--line)", borderRadius: 8, font: "inherit" }}
        />
      </label>
      <div
        className="drop"
        onClick={() => input.current.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); pick(e.dataTransfer.files); }}
      >
        <input ref={input} type="file" multiple hidden accept={ACCEPT} onChange={(e) => pick(e.target.files)} />
        {files.length === 0
          ? <span>Drop PDFs or images here or click to browse <span className="sub">(Arrastre PDFs o imágenes, o haga clic)</span></span>
          : <span>{files.length} file(s) selected</span>}
      </div>
      {files.length > 0 && (
        <ul className="filelist">
          {files.map((f) => (
            <li key={f.name}><span>{f.name}</span><span className="pill">READY</span></li>
          ))}
        </ul>
      )}
      <button className="primary" disabled={!files.length || !caseId.trim() || sending} onClick={submit}>
        {sending ? "Uploading…" : <L t={UI.process} lang={lang} />}
      </button>
      {error && <p className="note" style={{ color: "var(--red)" }}>{error}</p>}
    </section>
  );
}

/* ---------- Case list (survives Application restarts) ---------- */
function CasesCard({ cases, current, onOpen, lang }) {
  if (!cases.length) return null;
  return (
    <section className="card">
      <h2><L t={UI.cases} lang={lang} /> <span className="count">{cases.length}</span></h2>
      <ul className="issues">
        {cases.map((c) => (
          <li key={c.id}>
            <button className={current === c.id ? "issue on" : "issue"} onClick={() => onOpen(c.id)}>
              <span className={"dot " + (c.failed || c.interrupted ? "rojo" : c.ocr_done === c.total ? "verde" : "amarillo")} />
              <span><b>{c.id}</b> · <L t={c.case_label} lang={lang} /></span>
              <span className="muted">{c.ocr_done}/{c.total}</span>
              <span className="chev">›</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

/* ---------- Per-document progress ---------- */
function ProgressCard({ s, lang, tech }) {
  const finished = s.ocr_done + s.failed;
  return (
    <section className="card">
      <h2>
        <L t={UI.progress} lang={lang} /> · {s.id}
        <span className="count">{finished}/{s.total}</span>
        <span className={"pill " + (s.case_status === "FAILED" || s.interrupted ? "bad" : s.ocr_done === s.total ? "ok" : "")}>
          <L t={s.case_label} lang={lang} />
        </span>
      </h2>
      <div className="meter"><div style={{ width: `${(s.ocr_done / s.total) * 100}%` }} /></div>
      {s.message && <p className="hint"><L t={s.message} lang={lang} /></p>}
      <ul className="filelist">
        {s.docs.map((d) => (
          <li key={d.name} style={{ flexWrap: "wrap" }}>
            <span>{d.name}</span>
            <span className={"pill " + stagePill(d.stage)}><L t={d.stage_label} lang={lang} /></span>
            {d.error_detail && (
              <span className="note" style={{ flexBasis: "100%", color: "var(--red)" }}>{d.error_detail}</span>
            )}
            {tech && (
              <span className="note" style={{ flexBasis: "100%" }}>
                stage {d.stage} · attempts {d.tries} · updated {d.last_updated}
              </span>
            )}
          </li>
        ))}
      </ul>
      {s.running && <p className="note">Each document runs as its own job; start-up takes about 2-3 minutes.</p>}
    </section>
  );
}

/* ---------- Failed case: no verdict ---------- */
function FailedCard({ v, lang }) {
  return (
    <section className="banner red">
      <div className="lamp" />
      <div>
        <div className="banner-top">
          <strong><L t={UI.noVerdict} lang={lang} /></strong>
          <span className="case-id">Case {v.id}</span>
        </div>
        <h3><L t={UI.failedDocs} lang={lang} /></h3>
        <ul className="fe">
          {v.failed.map((d) => (
            <li key={d.name}><b>{d.name}</b> (attempt {d.tries}): {d.error_detail}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}

/* ---------- Verdict banner ---------- */
function VerdictBanner({ c, lang }) {
  const v = VERDICT[c.verdict];
  return (
    <section className={"banner " + v.cls}>
      <div className="lamp" />
      <div>
        <div className="banner-top">
          <strong>{c.verdict}</strong> · <L t={{ es: v.es, en: v.en }} lang={lang} />
          <span className="case-id">Case {c.id}</span>
        </div>
        <p><L t={c.justification} lang={lang} /></p>
      </div>
    </section>
  );
}

/* ---------- Issues with drill-down ---------- */
function Issues({ c, lang, onSelect, selected }) {
  return (
    <section className="card">
      <h2><L t={UI.issues} lang={lang} /> <span className="count">{c.issues.length}</span></h2>
      {c.issues.length === 0 && <p className="muted">Ninguno <span className="sub">(None)</span></p>}
      <ul className="issues">
        {c.issues.map((i) => (
          <li key={i.code}>
            <button className={selected === i.code ? "issue on" : "issue"} onClick={() => onSelect(i.code)}>
              <span className={"dot " + i.severity.toLowerCase()} />
              <span><L t={i.title} lang={lang} /></span>
              <span className="chev">›</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function IssueDetail({ c, code, lang, tech, onOpenDoc }) {
  const i = c.issues.find((x) => x.code === code);
  if (!i) {
    return (
      <section className="card empty">
        <p>Select an issue to see its details. <span className="sub">(Seleccione un problema para ver su detalle.)</span></p>
      </section>
    );
  }
  return (
    <section className="card detail">
      <h2><L t={i.title} lang={lang} /></h2>
      {tech && <code className="code">{i.code} · {i.severity}</code>}
      {i.cause && (<><h3><L t={UI.cause} lang={lang} /></h3><p><L t={i.cause} lang={lang} /></p></>)}
      {i.hint && (<><h3><L t={UI.hint} lang={lang} /></h3><p className="hint"><L t={i.hint} lang={lang} /></p></>)}
      {i.docs.length > 0 && (
        <>
          <h3><L t={UI.evidence} lang={lang} /></h3>
          <div className="chips">
            {i.docs.map((d) => (
              <button key={d} className="chip" onClick={() => onOpenDoc(d)}>{d}</button>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

/* ---------- Inventory (S1) ---------- */
function Inventory({ c, lang }) {
  const have = c.s1.filter((r) => r.present).length;
  return (
    <section className="card">
      <h2><L t={UI.inventory} lang={lang} /> <span className="count">{have}/{c.s1.length}</span></h2>
      <div className="meter"><div style={{ width: `${(have / c.s1.length) * 100}%` }} /></div>
      <ul className="inv">
        {c.s1.map((r) => (
          <li key={r.es}>
            <span className={"tick " + (r.present ? "ok" : "no")}>{r.present ? "✓" : "✕"}</span>
            <span><L t={r} lang={lang} /></span>
            <span className={"pill " + (r.present ? "ok" : "bad")}><L t={r.present ? UI.present : UI.missing} lang="es" /></span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/* ---------- Documents (S2) ---------- */
function Documents({ c, lang, tech, open, setOpen }) {
  return (
    <section className="card">
      <h2><L t={UI.docs} lang={lang} /> <span className="count">{c.docs.length}</span></h2>
      <ul className="docs">
        {c.docs.map((d) => {
          const isOpen = open === d.name;
          const ok = d.missing.length === 0 && d.flags.length === 0;
          return (
            <li key={d.name} className={isOpen ? "open" : ""}>
              <button className="docrow" onClick={() => setOpen(isOpen ? null : d.name)}>
                <span className={"dot " + (ok ? "verde" : "amarillo")} />
                <span className="docname">{d.name}</span>
                <span className="tag"><L t={TYPE_LABEL[d.type]} lang="en" /></span>
                <span className="muted">{ok ? "No remarks" : `${d.missing.length} fields missing`}</span>
                <span className="chev">{isOpen ? "⌄" : "›"}</span>
              </button>
              {isOpen && (
                <div className="docbody">
                  <div className="cols">
                    <div>
                      <h4>Extracted ({d.fields.length})</h4>
                      <div className="chips">{d.fields.length ? d.fields.map((f) => <span key={f} className="chip ok">{f}</span>) : <span className="muted">None</span>}</div>
                    </div>
                    <div>
                      <h4>Missing ({d.missing.length})</h4>
                      <div className="chips">{d.missing.length ? d.missing.map((f) => <span key={f} className="chip bad">{f}</span>) : <span className="muted">None</span>}</div>
                    </div>
                  </div>
                  {d.flags.length > 0 && (
                    <>
                      <h4 style={{ marginTop: 12 }}>Flags ({d.flags.length})</h4>
                      <div className="chips">{d.flags.map((f) => <span key={f} className="chip bad">{f}</span>)}</div>
                    </>
                  )}
                  {tech && (
                    <div className="tech">
                      stage {d.stage} · attempts {d.tries} · text pages {d.pagesText} · VLM pages {d.pagesVlm} · {d.seconds.toFixed(1)}s
                    </div>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
      {tech && <p className="note">Model: {c.meta.model} · total {c.meta.totalSeconds}s</p>}
    </section>
  );
}

/* ---------- Technical run details (mirrors Results_*.md) ---------- */
function TechPanel({ c }) {
  const done = c.docs.filter((d) => d.stage === "OCR_DONE").length;
  const failed = c.docs.filter((d) => d.stage === "FAILED").length;
  const extracted = c.docs.filter((d) => d.type === "ID_DOCUMENT" || d.type === "BIRTH_CERT");
  return (
    <section className="card">
      <h2>Technical details <span className="sub">(Detalles técnicos)</span></h2>
      <div className="stats">
        <div><b>{done}/{c.docs.length}</b><span>OCR done</span></div>
        <div><b>{failed}</b><span>failed</span></div>
        <div><b>{c.meta.totalSeconds}s</b><span>total processing time</span></div>
        <div><b>{c.meta.model}</b><span>model</span></div>
      </div>
      <div className="tablewrap">
        <table className="ttable">
          <thead>
            <tr>
              <th>Document</th><th>Stage</th><th>Try</th><th>Type</th>
              <th title="Pages read from the PDF text layer">Text layer (pages)</th>
              <th title="Pages read by the vision model">Visual layer (pages)</th>
              <th>Time (s)</th>
            </tr>
          </thead>
          <tbody>
            {c.docs.map((d) => (
              <tr key={d.name}>
                <td>{d.name}</td>
                <td><span className={"pill " + stagePill(d.stage)}>{d.stage}</span></td>
                <td>{d.tries}</td>
                <td>{d.type}</td>
                <td>{d.pagesText}</td>
                <td>{d.pagesVlm}</td>
                <td>{d.seconds.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">Total is the sum over documents. Real elapsed time is shorter when documents run in parallel.</p>
      <h3>Field extraction (ID and birth certificates)</h3>
      <ul className="fe">
        {extracted.map((d) => (
          <li key={d.name}>
            <b>{d.name}</b>: {d.fields.length} fields
            <div className="chips">{d.fields.map((f) => <span key={f} className="chip">{f}</span>)}</div>
          </li>
        ))}
      </ul>
      {c.snapshot && c.snapshot.s3_footnote && <p className="note">S3: {c.snapshot.s3_footnote}</p>}
    </section>
  );
}

/* ---------- Lineage (S3) ---------- */
function Lineage({ c, lang }) {
  return (
    <section className="card">
      <h2><L t={UI.lineage} lang={lang} /></h2>
      <ol className="tree">
        {c.lineage.map((p) => (
          <li key={p.gen} className={p.name ? "known" : "unknown"}>
            <span className="gen">{p.gen}</span>
            <div>
              <div className="role"><L t={p.role} lang={lang} /></div>
              <div className="pname">{p.name || <span className="muted"><L t={UI.undetermined} lang={lang} /></span>}</div>
              {(p.birth || p.place) && <div className="muted">{[p.birth, p.place].filter(Boolean).join(" · ")}</div>}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ---------- Verdict view ---------- */
function VerdictView({ result, lang, tech }) {
  const [issue, setIssue] = useState(null);
  const [openDoc, setOpenDoc] = useState(null);
  const docsRef = useRef(null);

  const jumpToDoc = (name) => {
    setOpenDoc(name);
    setTimeout(() => docsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  };

  return (
    <>
      <VerdictBanner c={result} lang={lang} />
      <div className="grid2">
        <Issues c={result} lang={lang} onSelect={setIssue} selected={issue} />
        <IssueDetail c={result} code={issue} lang={lang} tech={tech} onOpenDoc={jumpToDoc} />
      </div>
      <div className="grid2">
        <Inventory c={result} lang={lang} />
        <Lineage c={result} lang={lang} />
      </div>
      <div ref={docsRef}>
        <Documents c={result} lang={lang} tech={tech} open={openDoc} setOpen={setOpenDoc} />
      </div>
      {tech && <TechPanel c={result} />}
    </>
  );
}

/* ---------- App ---------- */
export default function App() {
  const lang = "both"; // ES + EN shown together for now
  const [tech, setTech] = useState(false);
  const [cases, setCases] = useState([]);
  const [caseId, setCaseId] = useState(null);
  const [status, setStatus] = useState(null);
  const [verdict, setVerdict] = useState(null);
  const [error, setError] = useState(null);
  const [reload, setReload] = useState(0);

  const loadCases = useCallback(async () => {
    try {
      setCases(await api("api/cases"));
    } catch (e) {
      setError(`Could not load cases: ${e.message}`);
    }
  }, []);

  useEffect(() => { loadCases(); }, [loadCases]);

  // Poll the open case until every document is finished, then fetch the verdict once.
  useEffect(() => {
    if (!caseId) return undefined;
    let stopped = false;
    let timer = null;
    setStatus(null);
    setVerdict(null);
    setError(null);

    const tick = async () => {
      let s;
      try {
        s = await api(casePath(caseId));
      } catch (e) {
        if (stopped) return;
        setError(`Status check failed (retrying): ${e.message}`);
        timer = setTimeout(tick, POLL_MS);
        return;
      }
      if (stopped) return;
      setStatus(s);
      setError(null);

      if (s.docs.some((d) => d.stage === "RECEIVED")) {
        if (!s.interrupted) timer = setTimeout(tick, POLL_MS);
        return; // interrupted: nothing more will happen, stop polling
      }
      try {
        const v = await api(`${casePath(caseId)}/verdict`);
        if (stopped) return;
        setVerdict(v);
        // Rules just ran: refresh the case label (e.g. "Evaluado").
        const after = await api(casePath(caseId));
        if (!stopped) setStatus(after);
      } catch (e) {
        if (!stopped) setError(`Verdict failed: ${e.message}`);
      }
      if (!stopped) loadCases();
    };

    tick();
    return () => { stopped = true; clearTimeout(timer); };
  }, [caseId, reload, loadCases]);

  const openCase = (id) => {
    if (id === caseId) setReload((n) => n + 1);
    else setCaseId(id);
  };

  const started = (id) => {
    loadCases();
    openCase(id);
  };

  return (
    <>
      <TopBar tech={tech} setTech={setTech} />
      <main className="page">
        <div className="grid2">
          <UploadCard lang={lang} onStarted={started} />
          <CasesCard cases={cases} current={caseId} onOpen={openCase} lang={lang} />
        </div>
        <ErrorBox message={error} onRetry={caseId ? () => setReload((n) => n + 1) : null} lang={lang} />
        {status && <ProgressCard s={status} lang={lang} tech={tech} />}
        {verdict && verdict.result === "failed" && <FailedCard v={verdict} lang={lang} />}
        {verdict && verdict.result === "verdict" && <VerdictView key={verdict.id} result={verdict} lang={lang} tech={tech} />}
      </main>
    </>
  );
}
