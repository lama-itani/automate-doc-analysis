import React, { useState, useRef } from "react";
import { SAMPLE_CASE } from "./data.js";

/* ---------- Bilingual label helper ---------- */
// lang: "both" | "es" | "en". `t` is {es, en}.
function L({ t, lang }) {
  if (!t) return null;
  if (lang === "es") return <>{t.es}</>;
  if (lang === "en") return <>{t.en}</>;
  return (
    <>
      {t.es} <span className="sub">({t.en})</span>
    </>
  );
}

const UI = {
  upload: { es: "Cargar expediente", en: "Upload case file" },
  process: { es: "Procesar expediente", en: "Process case file" },
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

/* ---------- Upload + process ---------- */
function UploadCard({ lang, onDone }) {
  const [files, setFiles] = useState([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({});
  const input = useRef(null);

  const pick = (list) => setFiles(Array.from(list));

  // Prototype only: simulates RECEIVED -> OCR_DONE per document.
  const process = () => {
    setRunning(true);
    const start = {};
    files.forEach((f) => (start[f.name] = "RECEIVED"));
    setProgress(start);
    files.forEach((f, i) => {
      setTimeout(() => setProgress((p) => ({ ...p, [f.name]: "OCR_DONE" })), 900 + i * 700);
    });
    setTimeout(() => { setRunning(false); onDone(); }, 900 + files.length * 700 + 400);
  };

  return (
    <section className="card">
      <h2><L t={UI.upload} lang={lang} /></h2>
      <div
        className="drop"
        onClick={() => input.current.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); pick(e.dataTransfer.files); }}
      >
        <input ref={input} type="file" multiple hidden onChange={(e) => pick(e.target.files)} />
        {files.length === 0
          ? <span>Drop PDFs here or click to browse <span className="sub">(Arrastre PDFs o haga clic)</span></span>
          : <span>{files.length} file(s) selected</span>}
      </div>
      {files.length > 0 && (
        <ul className="filelist">
          {files.map((f) => (
            <li key={f.name}>
              <span>{f.name}</span>
              <span className={"pill " + (progress[f.name] === "OCR_DONE" ? "ok" : "")}>
                {progress[f.name] || "READY"}
              </span>
            </li>
          ))}
        </ul>
      )}
      <button className="primary" disabled={!files.length || running} onClick={process}>
        {running ? "Processing…" : <L t={UI.process} lang={lang} />}
      </button>
      <p className="note">Prototype: processing is simulated and shows a sample case.</p>
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
        <p>Select an issue to see its cause. <span className="sub">(Seleccione un problema para ver su causa.)</span></p>
      </section>
    );
  }
  return (
    <section className="card detail">
      <h2><L t={i.title} lang={lang} /></h2>
      {tech && <code className="code">{i.code}</code>}
      <h3><L t={UI.cause} lang={lang} /></h3>
      <p><L t={i.cause} lang={lang} /></p>
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
          const ok = d.missing.length === 0;
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
                  {tech && (
                    <div className="tech">
                      stage {d.stage} · attempts {d.tries} · text pages {d.pagesText} · VLM pages {d.pagesVlm} · {d.seconds}s
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
  const total = c.docs.reduce((s, d) => s + d.seconds, 0).toFixed(1);
  const extracted = c.docs.filter((d) => d.type === "ID_DOCUMENT" || d.type === "BIRTH_CERT");
  return (
    <section className="card">
      <h2>Technical details <span className="sub">(Detalles técnicos)</span></h2>
      <div className="stats">
        <div><b>{done}/{c.docs.length}</b><span>OCR done</span></div>
        <div><b>{failed}</b><span>failed</span></div>
        <div><b>{total}s</b><span>total processing time</span></div>
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
                <td><span className={"pill " + (d.stage === "OCR_DONE" ? "ok" : "bad")}>{d.stage}</span></td>
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
              {p.birth && <div className="muted">{p.birth}</div>}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ---------- App ---------- */
export default function App() {
const lang = "both"; // ES + EN shown together for now
  const [tech, setTech] = useState(false);
  const [result, setResult] = useState(SAMPLE_CASE); // preloaded so the page is never empty
  const [issue, setIssue] = useState(null);
  const [openDoc, setOpenDoc] = useState(null);
  const docsRef = useRef(null);

  const jumpToDoc = (name) => {
    setOpenDoc(name);
    setTimeout(() => docsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  };

  return (
    <>
      <TopBar tech={tech} setTech={setTech} />
      <main className="page">
        <UploadCard lang={lang} onDone={() => setResult({ ...SAMPLE_CASE })} />
        {result && (
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
        )}
      </main>
    </>
  );
}
