/* Build presentation/NMT_Presentation.pptx from presentation/data.js (same numbers as the website).
 *   npm install pptxgenjs@3.12.0      (once, in any folder; set NODE_PATH to its node_modules if elsewhere)
 *   node presentation/build_pptx.js
 * Theme colours/fonts live in THEME; every slide uses the two layouts defined below.
 * Needs Node.js and pptxgenjs only (jszip comes with pptxgenjs).
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const raw = fs.readFileSync(path.join(HERE, "data.js"), "utf8");
const D = JSON.parse(raw.slice(raw.indexOf("=") + 1).trim().replace(/;\s*$/, ""));
const OUT = process.argv[2] || path.join(HERE, "NMT_Presentation.pptx");

/* ---------------- theme (mirrors the website) ---------------- */
const THEME = {
  name: "Indic NMT Night",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B1222", lt1: "EEF2F8", dk2: "141E33", lt2: "A9B4C7",
    accent1: "4B93EF", accent2: "F27C48", accent3: "2FC493", accent4: "E8C27A",
    accent5: "9B8CFF", accent6: "F0B429", hlink: "4B93EF", folHlink: "9B8CFF",
  },
};
const HEX = { bg: "0B1222", panel: "141E33", panel2: "1B2740", ink: "EEF2F8", muted: "A9B4C7", dim: "7B8699",
  grid: "26324A", gold: "E8C27A", base: "4B93EF", bt: "F27C48", nllb: "2FC493" };
const INDIC = "Nirmala UI";
const TEAM = [["Amulya Jain", "500122439"], ["Himanshu Pokhriyal", "500121200"], ["Anirvan Ranjan", "500119330"], ["Narind Verma", "500121898"], ["Somansh Goel", "500123667"]];
const SUBJECT = "Computational Linguistics and Natural Language Processing Lab", FACULTY = "Dr Sahinur Rehman Laskar", INSTITUTION = "University of Petroleum and Energy Studies (UPES)";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.333 x 7.5 in
pres.title = "Low-Resource Neural Machine Translation - WMT26 Indic MT";
pres.subject = "NLP project presentation";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;              // tx1=dk1 (navy bg), bg1=lt1 (ink), tx2=dk2 (panel), bg2=lt2 (muted)
const W = 13.333, M = 0.6;

const MODELS = ["baseline", "backtranslation", "nllb_finetune"];
const MLABEL = { baseline: "Transformer baseline", backtranslation: "+ Back-translation", nllb_finetune: "NLLB-200 fine-tuned" };
const MCOL = { baseline: C.accent1, backtranslation: C.accent2, nllb_finetune: C.accent3 };
const MHEX = { baseline: HEX.base, backtranslation: HEX.bt, nllb_finetune: HEX.nllb };
const LANG = { as: "Assamese", mni: "Manipuri", lus: "Mizo", bodo: "Bodo", en: "English" };
const dirs = D.directions.filter(d => D.scores.some(r => r.direction === d));
const score = (m, d) => D.scores.find(r => r.model === m && r.direction === d);
const sig = (m, d, k) => D.significance.find(r => r.model === m && r.direction === d && r.metric === k);
const avg = (m, k) => { const v = dirs.map(d => score(m, d)?.[k]).filter(x => x != null); return v.reduce((a, b) => a + b, 0) / v.length; };
const f1 = x => (+x).toFixed(1), f2 = x => (+x).toFixed(2);
const sgn = x => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(2);

/* ---------------- layouts ---------------- */
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.55, w: W - 1.6, h: 1.5, fontSize: 46,
      color: C.background1, align: "center", valign: "middle", fontFace: THEME.headFontFace }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.text1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 0.32, w: 9, h: 0.34, fontSize: 12, bold: true,
      color: C.accent4, charSpacing: 3, fontFace: THEME.bodyFontFace, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.66, w: W - 2 * M, h: 0.8, fontSize: 34, align: "left",
      color: C.background1, fontFace: THEME.headFontFace, margin: 0, valign: "top" }, text: "" } },
    { text: { text: "WMT26 Indic MT  ·  Low-resource NMT", options: { x: M, y: 7.05, w: 6, h: 0.3, fontSize: 10,
      color: C.background2, margin: 0, isTextBox: true } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});

/* ---------------- helpers ---------------- */
let N = 0;
function content(section, kicker, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  s.addText(title, { placeholder: "title", align: "left" });
  return s;
}
function card(s, x, y, w, h, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.text2 },
    line: { color: C.text2 }, objectName: name || "card" });
}
function badge(s, x, y, glyph, color) {         // motif: outlined circle with a glyph
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: C.text1 }, line: { color: color || C.accent4, width: 1.5 }, objectName: "badge" });
  s.addText(glyph, { x, y, w: 0.5, h: 0.5, align: "center", valign: "middle", fontSize: 14, bold: true,
    color: color || C.accent4, margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
}
function stat(s, x, y, w, big, label, color) {
  card(s, x, y, w, 1.25, "stat");
  s.addText(big, { x: x + 0.25, y: y + 0.12, w: w - 0.5, h: 0.7, fontSize: 36, fontFace: THEME.headFontFace,
    color: color || C.background1, margin: 0, isTextBox: true });
  s.addText(label, { x: x + 0.25, y: y + 0.82, w: w - 0.5, h: 0.32, fontSize: 11, color: C.background2, margin: 0,
    charSpacing: 1, isTextBox: true });
}
const darkChart = (o) => Object.assign({
  catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false, legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.ink,
  titleColor: HEX.ink, titleFontFace: "+mn-lt", titleFontSize: 13, dataLabelColor: HEX.ink, dataLabelFontFace: "+mn-lt",
}, o);

/* =============== 1. TITLE =============== */
{
  pres.addSection({ title: "Title" });
  const s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Title" });
  s.addText("NATURAL LANGUAGE PROCESSING  ·  PROJECT PRESENTATION", { x: M, y: 0.95, w: W - 2 * M, h: 0.4, align: "center",
    fontSize: 13, bold: true, charSpacing: 4, color: C.accent4, isTextBox: true });
  s.addText("Low-Resource Neural Machine Translation", { placeholder: "title" });
  s.addText("for North-East Indian languages  ·  WMT26 Indic MT Shared Task", { x: M, y: 3.0, w: W - 2 * M, h: 0.45,
    align: "center", fontSize: 18, color: C.background2, isTextBox: true });
  const chips = [["English", null], ["অসমীয়া · Assamese", INDIC], ["মৈতৈলোন্ · Manipuri", INDIC], ["Mizo ṭawng", null], ["बर' · Bodo", INDIC]];
  const cw = 2.05, gap = 0.18, x0 = (W - (chips.length * cw + (chips.length - 1) * gap)) / 2;
  chips.forEach(([t, ff], i) => {
    const x = x0 + i * (cw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.7, w: cw, h: 0.44, rectRadius: 0.22, fill: { color: C.text2 }, line: { color: C.accent4, width: 0.75 }, objectName: "language chip" });
    s.addText(t, { x, y: 3.7, w: cw, h: 0.44, align: "center", valign: "middle", fontSize: 13, color: C.background1,
      fontFace: ff || THEME.bodyFontFace, margin: 0, isTextBox: true });
  });
  const tw = 2.2, tg = 0.2, tx0 = (W - (5 * tw + 4 * tg)) / 2;
  for (let i = 0; i < 5; i++) {
    const x = tx0 + i * (tw + tg);
    card(s, x, 4.5, tw, 0.95, "team member");
    s.addText([{ text: TEAM[i][0], options: { fontSize: 14, bold: true, color: C.background1, breakLine: true } },
      { text: "SAP ID " + TEAM[i][1], options: { fontSize: 12, color: C.background2 } }],
      { x, y: 4.5, w: tw, h: 0.95, align: "center", valign: "middle", margin: 0, isTextBox: true });
  }
  [["COURSE", SUBJECT, 5.2], ["FACULTY", FACULTY, 3.0], ["INSTITUTION", INSTITUTION, 3.6]].reduce((x, [lab, val, w]) => {
    s.addText([{ text: lab, options: { bold: true, color: C.accent4, fontSize: 11, charSpacing: 2, breakLine: true } },
      { text: val, options: { color: C.background1, fontSize: 14 } }],
      { x, y: 5.75, w, h: 0.75, align: "center", valign: "top", margin: 0, isTextBox: true });
    return x + w + 0.15;
  }, (W - (5.2 + 3.0 + 3.6 + 0.3)) / 2);
  s.addNotes("Introduce the team and the project: low-resource machine translation between English and four North-East Indian languages, from the WMT26 Indic MT shared task.");
}

/* =============== 2. INTRODUCTION =============== */
{
  pres.addSection({ title: "Introduction" });
  const s = content("Introduction", "01 · Introduction", "Translating languages that barely have data");
  s.addText([
    { text: "Neural MT needs millions of sentence pairs. North-East India has dozens of languages with only a few thousand. The " },
    { text: "WMT26 Indic MT shared task", options: { bold: true, color: C.background1 } },
    { text: " releases parallel data for 11 of them. We asked how far careful engineering can take a model in this regime." },
  ], { x: M, y: 1.55, w: 10.5, h: 0.95, fontSize: 16, color: C.background2, margin: 0, isTextBox: true });
  const cards = [
    ["?", "The problem", "English ↔ Assamese, Manipuri, Mizo and Bodo, with 11k–49k training pairs each: 100–1000× less than high-resource pairs."],
    ["!", "Why it is hard", "Four scripts, rich morphology, noisy religious and web-domain data, and one language (Bodo) that no large pretrained MT model has seen."],
    ["→", "Our goal", "Compare a Transformer baseline with two improvements, back-translation and NLLB-200 transfer, using confidence intervals and significance tests."],
  ];
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  cards.forEach(([g, h, t], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.75, cw, 2.75);
    badge(s, x + 0.3, 3.0, g);
    s.addText(h, { x: x + 0.95, y: 3.0, w: cw - 1.2, h: 0.5, fontSize: 20, bold: true, fontFace: THEME.headFontFace, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.3, y: 3.7, w: cw - 0.6, h: 1.6, fontSize: 15, color: C.background2, valign: "top", margin: 0, isTextBox: true });
  });
  const chips = ["8 translation directions", "3 systems", "30+ training runs", `${D.gpu_hours} GPU-hours`, "5-member pipeline"];
  let x = M;
  chips.forEach(t => {
    const w = 0.35 + t.length * 0.095;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.85, w, h: 0.42, rectRadius: 0.21, fill: { color: C.text1 }, line: { color: C.background2, width: 0.75 }, objectName: "chip" });
    s.addText(t, { x, y: 5.85, w, h: 0.42, align: "center", valign: "middle", fontSize: 12, color: C.background1, margin: 0, isTextBox: true });
    x += w + 0.18;
  });
  s.addNotes("Motivation: NMT quality depends heavily on parallel data. These languages have very little. Three challenges: scripts and morphology, noisy data, and an unseen language for pretrained models (Bodo). We compare one baseline with two improvements and evaluate them rigorously.");
}

/* =============== 3. DATASET =============== */
{
  pres.addSection({ title: "Dataset" });
  const s = content("Dataset", "02 · Dataset description", "WMT26 Indic MT, 2026 release");
  const hdr = ["Pair", "Script", "Train", "Dev", "Test", "Removed"].map((t, i) => ({ text: t, options: { bold: true, color: C.background2, fontSize: 12, align: i > 1 ? "right" : "left", fill: { color: C.text2 } } }));
  const SCRIPT = { as: "Bengali-Assamese", mni: "Bengali", lus: "Latin", bodo: "Devanagari" };
  const rows = D.pairs.map(p => {
    const x = p.pair.split("-")[1], rem = Object.values(p.removed).reduce((a, b) => a + b, 0);
    const c = (t, al) => ({ text: t, options: { color: C.background1, fontSize: 14, align: al || "right" } });
    return [c(`en ↔ ${x}  ${LANG[x]}`, "left"), { text: SCRIPT[x], options: { color: C.background2, fontSize: 13 } },
      c(p.splits.train.toLocaleString("en-US")), c(p.splits.dev.toLocaleString("en-US")), c(p.splits.test.toLocaleString("en-US")),
      { text: rem.toLocaleString("en-US"), options: { color: C.background2, fontSize: 14, align: "right" } }];
  });
  card(s, M, 1.65, 7.4, 4.1);
  s.addTable([hdr, ...rows], { x: M + 0.25, y: 1.85, w: 6.9, colW: [1.95, 1.55, 0.95, 0.75, 0.75, 0.95], rowH: 0.48,
    border: { type: "solid", pt: 0.75, color: HEX.grid }, fill: { color: C.text2 }, valign: "middle", margin: [0.03, 0.08, 0.03, 0.08] });
  s.addText([
    { text: "Cleaning (train only): ", options: { bold: true, color: C.background1 } },
    { text: "NFC normalisation, removal of empty, duplicate and wrong-script pairs, length-ratio filtering, and removal of every pair that overlaps dev/test. Dev and test are never filtered.", options: { breakLine: true } },
    { text: "Tokeniser: ", options: { bold: true, color: C.background1 } },
    { text: "one joint SentencePiece unigram model per pair (8k vocab; 6k for Bodo)." },
  ], { x: M + 0.25, y: 4.45, w: 6.9, h: 1.95, fontSize: 13, color: C.background2, paraSpaceAfter: 6, valign: "top", margin: 0, isTextBox: true });
  const tot = D.pairs.reduce((a, p) => a + p.splits.train, 0);
  stat(s, 8.35, 1.65, 2.05, `${Math.round(tot / 1000)}k`, "TRAIN PAIRS");
  stat(s, 10.63, 1.65, 2.1, "11", "PAIRS RELEASED");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.35, y: 3.1, w: 4.38, h: 2.2, rectRadius: 0.1, fill: { color: "FFFFFF" }, line: { color: C.text2 }, objectName: "figure frame" });
  s.addImage({ path: path.join(HERE, "figures", "split_sizes.png"), x: 8.45, y: 3.2, w: 4.18, h: 2.0,
    sizing: { type: "contain", w: 4.18, h: 2.0 }, altText: "Train, dev and test sizes for all 11 released pairs, log scale" });
  s.addText("The official test sets are password-protected for registered teams, so dev and test are held out from the training release (seed 42).",
    { x: 8.35, y: 5.45, w: 4.38, h: 1.1, fontSize: 12, color: C.background2, valign: "top", margin: 0, isTextBox: true });
  s.addNotes("We model 4 of the 11 pairs in both directions: 3 Category I pairs with three different scripts, plus Bodo from Category II. Cleaning is applied only to train, so every system is scored on identical dev and test sentences. Point out the assumption: our own held-out test split, because the official one is password-protected.");
}

/* =============== 4. METHODOLOGY =============== */
{
  pres.addSection({ title: "Methodology" });
  const s = content("Methodology", "03 · Methodology", "One pipeline, three systems");
  const steps = [["M1", "Data", "Download, checksums, held-out splits"], ["M2", "Preprocess", "NFC, filters, SentencePiece, EDA"],
    ["M3", "Baseline", "Transformer from scratch"], ["M4", "Advanced", "Back-translation, NLLB-200"], ["M5", "Evaluate", "chrF++, BLEU, COMET, bootstrap"]];
  const sw = 2.15, sg = 0.33;
  steps.forEach(([m, h, t], i) => {
    const x = M + i * (sw + sg);
    card(s, x, 1.65, sw, 1.35, "pipeline step");
    s.addText([{ text: m, options: { fontSize: 11, bold: true, color: C.accent4, charSpacing: 2, breakLine: true } },
      { text: h, options: { fontSize: 18, bold: true, color: C.background1, fontFace: THEME.headFontFace, breakLine: true } },
      { text: t, options: { fontSize: 12, color: C.background2 } }],
      { x: x + 0.2, y: 1.72, w: sw - 0.4, h: 1.21, valign: "middle", margin: 0, isTextBox: true });
    if (i < steps.length - 1) s.addText("›", { x: x + sw, y: 1.65, w: sg, h: 1.35, align: "center", valign: "middle", fontSize: 26, color: C.accent4, margin: 0, isTextBox: true });
  });
  const cards = [
    ["baseline", "A small pre-norm encoder–decoder sized for limited data; the reference point.",
      [["size", "d 256 · 4 heads · 4+4 layers · 9.4M params"], ["regularise", "dropout 0.3 · label smoothing 0.1"], ["train", "AdamW, ≤20k steps, early stop"], ["decode", "beam 5"]]],
    ["backtranslation", "The baseline translates 150k+ English sentences; tagged synthetic pairs are mixed 1:1 with the real data.",
      [["X→en", "classic back-translation"], ["en→X", "forward translation"], ["added", D.pairs.map(p => Math.round(p.synthetic / 1000) + "k").join(" / ") + " pairs"], ["quality", "round-trip chrF++ 20–27"]]],
    ["nllb_finetune", "Transfer from Meta's 200-language model. Bodo is unseen, so it borrows the Hindi tag.",
      [["model", "nllb-200-distilled-600M"], ["tags", "asm_Beng · mni_Beng · lus_Latn · hin_Deva"], ["train", "LR 5e-5, batch 32, ≤3 epochs"], ["decode", "beam 4"]]],
  ];
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  cards.forEach(([m, desc, kv], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 3.3, cw, 3.45);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 3.55, w: 0.22, h: 0.22, fill: { color: MCOL[m] }, line: { color: MCOL[m] }, objectName: "model colour" });
    s.addText(MLABEL[m], { x: x + 0.65, y: 3.43, w: cw - 0.9, h: 0.45, fontSize: 18, bold: true, fontFace: THEME.headFontFace, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(desc, { x: x + 0.3, y: 3.98, w: cw - 0.6, h: 0.95, fontSize: 13, color: C.background2, valign: "top", margin: 0, isTextBox: true });
    s.addText(kv.map(([k, v], j) => [{ text: k + "   ", options: { bold: true, color: C.accent4, fontSize: 11 } },
      { text: v, options: { color: C.background1, fontSize: 12, breakLine: j < kv.length - 1 } }]).flat(),
      { x: x + 0.3, y: 4.98, w: cw - 0.6, h: 1.6, valign: "top", paraSpaceAfter: 5, margin: 0, isTextBox: true });
  });
  s.addNotes("Five stages, one owner each (M1–M5), with fixed file formats between them. Three systems: a from-scratch Transformer baseline; the same model trained with back-translated synthetic data; and fine-tuning of the pretrained NLLB-200 model. Bodo is handled with a Hindi proxy language tag because NLLB does not know it.");
}

/* =============== 5. EXPERIMENTS =============== */
{
  pres.addSection({ title: "Experiments" });
  const s = content("Experiments", "04 · Experimental setup", "Measured carefully, not just once");
  const sw = (W - 2 * M - 3 * 0.3) / 4;
  [["chrF++", "PRIMARY METRIC"], ["1000×", "BOOTSTRAP RESAMPLES"], ["p < 0.05", "PAIRED SIGNIFICANCE TEST"], [`${D.gpu_hours} h`, "GPU TIME (A100, T4, L4)"]]
    .forEach(([b, l], i) => stat(s, M + i * (sw + 0.3), 1.65, sw, b, l, i === 0 ? C.accent4 : null));
  const cols = [
    ["Protocol", ["Fixed dev set for checkpoint selection and early stopping", "Test set decoded once per final model", "Train ∩ test overlap removed; tokeniser sees train only", "Seed 42; every run logs config, hardware and time"]],
    ["Runs", ["3 systems × 8 directions = 24 test outputs", "Learning curve: baseline on 10 / 25 / 50 / 100% of en↔as", "NLLB zero-shot score logged before fine-tuning"]],
    ["Analysis", ["chrF++, BLEU and COMET with signatures", "64 worst sentences per model, categorised", "Sentence length vs quality, attention maps"]],
  ];
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  cols.forEach(([h, items], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 3.2, cw, 3.45);
    badge(s, x + 0.3, 3.45, String(i + 1));
    s.addText(h, { x: x + 0.95, y: 3.45, w: cw - 1.2, h: 0.5, fontSize: 20, bold: true, fontFace: THEME.headFontFace, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })),
      { x: x + 0.3, y: 4.15, w: cw - 0.6, h: 2.35, fontSize: 14, color: C.background2, paraSpaceAfter: 8, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("chrF++ is the primary metric because it handles rich morphology better than BLEU. Every score has a 95% bootstrap confidence interval, and every model is compared with the baseline using a paired bootstrap significance test. The test set is used once, never for tuning.");
}

/* =============== 6. RESULTS: bar chart =============== */
{
  pres.addSection({ title: "Results" });
  const s = content("Results", "05 · Results", "Transfer learning wins in every direction");
  const base = avg("baseline", "chrf++");
  const tw = (W - 2 * M - 2 * 0.3) / 3;
  MODELS.forEach((m, i) => {
    const x = M + i * (tw + 0.3), a = avg(m, "chrf++");
    card(s, x, 1.6, tw, 1.0);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.98, w: 0.22, h: 0.22, fill: { color: MCOL[m] }, line: { color: MCOL[m] }, objectName: "model colour" });
    s.addText(f1(a), { x: x + 0.7, y: 1.68, w: 1.5, h: 0.82, fontSize: 36, fontFace: THEME.headFontFace, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    s.addText([{ text: MLABEL[m], options: { bold: true, color: C.background1, breakLine: true } },
      { text: "avg chrF++" + (m === "baseline" ? "" : `  ·  ${a - base >= 0 ? "+" : "−"}${Math.abs(a - base).toFixed(1)} vs baseline`), options: { color: C.background2 } }],
      { x: x + 2.2, y: 1.68, w: tw - 2.35, h: 0.82, fontSize: 12, valign: "middle", margin: 0, isTextBox: true });
  });
  card(s, M, 2.85, W - 2 * M, 3.95);
  s.addChart(pres.charts.BAR, MODELS.map(m => ({ name: MLABEL[m], labels: dirs, values: dirs.map(d => score(m, d)["chrf++"]) })),
    darkChart({ x: M + 0.2, y: 2.95, w: W - 2 * M - 0.4, h: 3.35, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
      chartColors: [HEX.base, HEX.bt, HEX.nllb], showLegend: true, legendPos: "t", valAxisMinVal: 0, valAxisMaxVal: 70, valAxisMajorUnit: 10,
      showValue: false, showTitle: false, objectName: "Test chrF++ per direction" }));
  s.addText("Test chrF++ on held-out sets (1,000-sample bootstrap; CIs ±0.6–1.3). NLLB is significantly better than the baseline in all 8 directions (p < 0.05). Back-translation is significantly better only for en-bodo and bodo-en.",
    { x: M + 0.25, y: 6.3, w: W - 2 * M - 0.5, h: 0.42, fontSize: 11, color: C.background2, margin: 0, isTextBox: true });
  s.addNotes(`Averages over all 8 directions: baseline ${f2(avg("baseline", "chrf++"))}, back-translation ${f2(avg("backtranslation", "chrf++"))}, NLLB ${f2(avg("nllb_finetune", "chrf++"))} chrF++. The NLLB gain ranges from +4.3 (en-bodo) to +19.1 (mni-en). Back-translation helps only Bodo, the smallest pair, and slightly hurts four of the other directions.`);
}

/* =============== 7. COMPARISON: tables =============== */
{
  const s = content("Results", "05 · Results · model comparison", "Every metric tells the same story");
  card(s, M, 1.6, 7.35, 5.2);
  s.addText("Gain over the baseline (test chrF++)", { x: M + 0.3, y: 1.72, w: 6.8, h: 0.4, fontSize: 16, bold: true, fontFace: THEME.headFontFace, color: C.background1, margin: 0, isTextBox: true });
  const hdr = ["Direction", "Baseline", "+ BT  Δ", "NLLB  Δ"].map((t, i) => ({ text: t, options: { bold: true, fontSize: 12, color: C.background2, align: i ? "right" : "left" } }));
  const cell = (m, d) => {
    const s0 = sig(m, d, "chrf++"); if (!s0) return { text: "n/a", options: { color: C.background2, align: "right" } };
    const col = s0.sig ? (s0.delta > 0 ? C.accent3 : C.accent2) : C.background2;
    return { text: sgn(s0.delta) + (s0.sig ? (s0.delta > 0 ? " *" : " †") : ""), options: { color: col, bold: s0.sig, align: "right" } };
  };
  const rows = dirs.map(d => [{ text: d, options: { color: C.background1 } }, { text: f2(score("baseline", d)["chrf++"]), options: { color: C.background1, align: "right" } }, cell("backtranslation", d), cell("nllb_finetune", d)]);
  s.addTable([hdr, ...rows], { x: M + 0.3, y: 2.2, w: 6.75, colW: [1.95, 1.6, 1.6, 1.6], rowH: 0.43, fontSize: 13,
    border: { type: "solid", pt: 0.75, color: HEX.grid }, fill: { color: C.text2 }, valign: "middle", margin: [0.03, 0.08, 0.03, 0.08] });
  s.addText("* / † = significantly better / worse than the baseline (paired bootstrap, p < 0.05)",
    { x: M + 0.3, y: 6.2, w: 6.75, h: 0.4, fontSize: 11, color: C.background2, margin: 0, isTextBox: true });

  const x2 = M + 7.65, w2 = W - M - x2;
  card(s, x2, 1.6, w2, 5.2);
  s.addText("Average over 8 directions", { x: x2 + 0.3, y: 1.72, w: w2 - 0.6, h: 0.4, fontSize: 16, bold: true, fontFace: THEME.headFontFace, color: C.background1, margin: 0, isTextBox: true });
  const mets = [["chrF++", "chrf++"], ["BLEU", "bleu"], ["COMET", "comet"]];
  mets.forEach(([lab, k], i) => {
    const y = 2.3 + i * 1.45;
    s.addText(lab, { x: x2 + 0.3, y, w: 1.5, h: 0.35, fontSize: 13, bold: true, color: C.accent4, margin: 0, isTextBox: true });
    MODELS.forEach((m, j) => {
      const xx = x2 + 0.3 + j * ((w2 - 0.6) / 3);
      s.addText([{ text: f1(avg(m, k)), options: { fontSize: 26, fontFace: THEME.headFontFace, color: m === "nllb_finetune" ? C.accent3 : C.background1, breakLine: true } },
        { text: ["Baseline", "+ BT", "NLLB"][j], options: { fontSize: 11, color: C.background2 } }],
        { x: xx, y: y + 0.35, w: (w2 - 0.6) / 3, h: 0.95, valign: "top", margin: 0, isTextBox: true });
    });
  });
  s.addNotes("Left: per-direction gains over the baseline with significance. NLLB wins everywhere; back-translation is a mixed result, helping only Bodo. Right: averages over all 8 directions for all three metrics. COMET (wmt22-comet-da, ×100) agrees with chrF++ and BLEU.");
}

/* =============== 8. WHY: learning curve + zero-shot =============== */
{
  const s = content("Results", "05 · Results · why", "Data size is the bottleneck");
  const hw = (W - 2 * M - 0.3) / 2;
  card(s, M, 1.6, hw, 5.2);
  const lcDirs = [...new Set(D.learning_curve.map(r => r.direction))];
  const lcRows = lcDirs.map(d => D.learning_curve.filter(r => r.direction === d).sort((a, b) => a.n - b.n));
  s.addChart(pres.charts.LINE, lcRows.map((rr, k) => ({ name: lcDirs[k], labels: rr.map(r => `${(r.n / 1000).toFixed(1)}k`), values: rr.map(r => r.chrf) })),
    darkChart({ x: M + 0.2, y: 1.7, w: hw - 0.4, h: 4.15, chartColors: [HEX.gold, "EC8FB5"], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 8,
      showLegend: true, legendPos: "t", valAxisMinVal: 15, valAxisMaxVal: 35, valAxisMajorUnit: 5, showValue: true, dataLabelPosition: "t", dataLabelFontSize: 10,
      dataLabelFormatCode: "0.0", showTitle: true, title: "Baseline test chrF++ vs training pairs", catAxisTitle: "training pairs", objectName: "Learning curve" }));
  s.addText("Each doubling of real data adds about 3–4 chrF++, with no plateau in sight.", { x: M + 0.3, y: 5.95, w: hw - 0.6, h: 0.7, fontSize: 13, color: C.background2, margin: 0, isTextBox: true });

  const x2 = M + hw + 0.3;
  card(s, x2, 1.6, hw, 5.2);
  s.addChart(pres.charts.BAR, [
    { name: "Zero-shot", labels: D.zeroshot.map(r => r.direction), values: D.zeroshot.map(r => r.zeroshot) },
    { name: "Fine-tuned", labels: D.zeroshot.map(r => r.direction), values: D.zeroshot.map(r => r.finetuned) }],
    darkChart({ x: x2 + 0.2, y: 1.7, w: hw - 0.4, h: 4.15, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 45,
      chartColors: ["3E5A7A", HEX.nllb], showLegend: true, legendPos: "t", valAxisMinVal: 0, valAxisMaxVal: 70, valAxisMajorUnit: 10,
      catAxisOrientation: "maxMin", showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFormatCode: "0.0",
      showTitle: true, title: "NLLB dev chrF++: zero-shot vs fine-tuned", objectName: "NLLB zero-shot vs fine-tuned" }));
  s.addText("Fine-tuning helps everywhere. Bodo, never seen in pre-training, jumps by more than 40 points through the Hindi proxy tag.", { x: x2 + 0.3, y: 5.95, w: hw - 0.6, h: 0.7, fontSize: 13, color: C.background2, margin: 0, isTextBox: true });
  s.addNotes("Left: the baseline retrained on 10, 25, 50 and 100% of the Assamese data. Scores keep rising with every doubling, so these pairs are data-limited. That explains why a pretrained model, which is effectively 'more data', wins. Right: NLLB before and after fine-tuning. Bodo goes from about 12–18 to about 55–59 dev chrF++.");
}

/* =============== 9. EXAMPLES =============== */
{
  const s = content("Results", "05 · Results · in practice", "What the outputs look like");
  const pick = ["as-en", "bodo-en"].map(d => D.examples.find(e => e.direction === d)).filter(Boolean);
  const hw = (W - 2 * M - 0.3) / 2;
  pick.forEach((e, i) => {
    const x = M + i * (hw + 0.3), [sl, tl] = e.direction.split("-");
    card(s, x, 1.6, hw, 5.2);
    s.addText(`${LANG[sl]} → ${LANG[tl]}`, { x: x + 0.3, y: 1.72, w: hw - 0.6, h: 0.4, fontSize: 16, bold: true, fontFace: THEME.headFontFace, color: C.background1, margin: 0, isTextBox: true });
    const rows = [["SOURCE", e.src, C.background1, INDIC, null], ["REFERENCE", e.ref, C.accent4, null, null],
      ...MODELS.map(m => [MLABEL[m].replace("Transformer ", "").toUpperCase(), e.hyp[m], C.background1, null, m])];
    rows.forEach(([lab, txt, col, ff, m], j) => {
      const y = 2.25 + j * 0.9;
      if (m) s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.06, w: 0.14, h: 0.14, fill: { color: MCOL[m] }, line: { color: MCOL[m] }, objectName: "model colour" });
      s.addText(lab + (m ? `   chrF++ ${f1(e.chrf[m])}` : ""), { x: x + (m ? 0.52 : 0.3), y, w: hw - 0.8, h: 0.26, fontSize: 10, bold: true, charSpacing: 1, color: C.background2, margin: 0, isTextBox: true });
      s.addText(txt, { x: x + 0.3, y: y + 0.27, w: hw - 0.6, h: 0.6, fontSize: 13, color: col, fontFace: ff || THEME.bodyFontFace, valign: "top", margin: 0, isTextBox: true, fit: "shrink" });
    });
  });
  s.addNotes("Two representative test sentences, chosen automatically (medium length, with a baseline score near its average). The scratch models get the gist but garble names and details. NLLB is fluent and keeps the names.");
}

/* =============== 10. ERROR ANALYSIS =============== */
{
  const s = content("Results", "05 · Results · error analysis", "Where the models fail");
  const CATS = [["reference_noise", "Reference noise"], ["omission", "Omission"], ["mistranslation", "Mistranslation"], ["hallucination", "Hallucination"], ["named_entity", "Named entity / number"], ["script_error", "Script error"]];
  const lw = 7.3;
  card(s, M, 1.6, lw, 5.2);
  s.addChart(pres.charts.BAR, MODELS.map(m => ({ name: MLABEL[m], labels: CATS.map(c => c[1]), values: CATS.map(([c]) => D.errors[m]?.[c] || 0) })),
    darkChart({ x: M + 0.2, y: 1.7, w: lw - 0.4, h: 5.0, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 40,
      chartColors: [HEX.base, HEX.bt, HEX.nllb], showLegend: true, legendPos: "t", catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 40,
      valAxisMajorUnit: 10, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, showTitle: true,
      title: "Error labels among the 64 worst test sentences per model", objectName: "Error categories" }));
  const x2 = M + lw + 0.3, w2 = W - M - x2;
  const ins = [["⅓", "Not model errors", "Many of the worst test pairs are misaligned in the release (for example, a reference that is only a Bible verse number). NLLB often translates them correctly and still scores about 5."],
    ["↓", "Omission dominates", "The scratch models stop early on long or rare-word inputs. NLLB omits less: 31 against 36–37 of 64."],
    ["∞", "Loops and names", "Hallucinations are repetition loops on lists of names. Back-translation adds named-entity errors (5 → 9)."]];
  ins.forEach(([g, h, t], i) => {
    const y = 1.6 + i * 1.77;
    card(s, x2, y, w2, 1.66);
    badge(s, x2 + 0.25, y + 0.22, g);
    s.addText(h, { x: x2 + 0.9, y: y + 0.2, w: w2 - 1.1, h: 0.5, fontSize: 17, bold: true, fontFace: THEME.headFontFace, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: x2 + 0.25, y: y + 0.72, w: w2 - 0.5, h: 0.88, fontSize: 12, color: C.background2, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("We took the 8 lowest-scoring test sentences per direction and model (64 per model) and labelled them with reproducible rules. About a third are misaligned references, which is a data problem, not a model problem. Among real errors, omission dominates for the small models. Script errors and untranslated copying are almost absent.");
}

/* =============== 11. CONCLUSION =============== */
{
  pres.addSection({ title: "Conclusion" });
  const s = content("Conclusion", "06 · Conclusion & future scope", "Transferred knowledge beats generated data");
  s.addText([{ text: "Fine-tuning a pretrained multilingual model lifted average chrF++ from " }, { text: f1(avg("baseline", "chrf++")), options: { color: C.accent4, bold: true } },
    { text: " to " }, { text: f1(avg("nllb_finetune", "chrf++")), options: { color: C.accent4, bold: true } },
    { text: `, and COMET from ${f1(avg("baseline", "comet"))} to ${f1(avg("nllb_finetune", "comet"))}.` }],
    { x: M, y: 1.55, w: W - 2 * M, h: 0.55, fontSize: 18, fontFace: THEME.headFontFace, color: C.background1, margin: 0, isTextBox: true });
  const hw = (W - 2 * M - 0.3) / 2;
  card(s, M, 2.35, hw, 4.45);
  s.addText("Key takeaways", { x: M + 0.3, y: 2.5, w: hw - 0.6, h: 0.45, fontSize: 20, bold: true, fontFace: THEME.headFontFace, color: C.background1, margin: 0, isTextBox: true });
  [["Pretrained transfer dominates.", "NLLB-200 beats the baseline by 4–19 chrF++ in every direction and learns unseen Bodo."],
    ["Back-translation needs a good generator.", "With noisy synthetic data it helps only the smallest pair: an honest negative result."],
    ["Evaluation hygiene matters.", "About a third of the worst test cases are misaligned references, so CIs and error analysis matter."]]
    .forEach(([b, t], i) => {
      const y = 3.1 + i * 1.2;
      badge(s, M + 0.3, y, String(i + 1));
      s.addText([{ text: b + " ", options: { bold: true, color: C.background1 } }, { text: t, options: { color: C.background2 } }],
        { x: M + 1.0, y: y - 0.05, w: hw - 1.3, h: 1.05, fontSize: 14, valign: "top", margin: 0, isTextBox: true });
    });
  const x2 = M + hw + 0.3;
  card(s, x2, 2.35, hw, 4.45);
  s.addText("Future scope", { x: x2 + 0.3, y: 2.5, w: hw - 0.6, h: 0.45, fontSize: 20, bold: true, fontFace: THEME.headFontFace, color: C.background1, margin: 0, isTextBox: true });
  const fut = ["Back-translation with NLLB as the generator, sampling and tuned ratios", "One multilingual model for all 11 pairs; related-language transfer (as → mni)",
    "LoRA / larger NLLB (1.3B, 3.3B) within the same budget", "Decoding fixes: repetition penalty, n-gram blocking", "Evaluation on the official test set with human judgements"];
  s.addText(fut.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < fut.length - 1 } })),
    { x: x2 + 0.3, y: 3.05, w: hw - 0.6, h: 2.55, fontSize: 14, color: C.background1, paraSpaceAfter: 7, valign: "top", margin: 0, isTextBox: true });
  s.addText("Limitations: own held-out split (not leaderboard-comparable), single seed, noisy references, possible overlap between NLLB pre-training data and public Bible/news text.",
    { x: x2 + 0.3, y: 5.75, w: hw - 0.6, h: 0.9, fontSize: 11, color: C.background2, valign: "top", margin: 0, isTextBox: true });
  s.addNotes("Three takeaways, then future work. Be upfront about the limitations: we could not use the official test set, we ran one seed per configuration, and NLLB may have seen some of this public text during pre-training (it reproduces 2.4% of references exactly, against 1.4% for the baseline).");
}

/* =============== 12. THANK YOU =============== */
{
  const s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Conclusion" });
  s.addText("QUESTIONS & DISCUSSION", { x: M, y: 1.1, w: W - 2 * M, h: 0.4, align: "center", fontSize: 13, bold: true, charSpacing: 4, color: C.accent4, isTextBox: true });
  s.addText("Thank you", { placeholder: "title", italic: true });
  s.addText("ধন্যবাদ  ·  থাগৎচরি  ·  Ka lawm e  ·  साबायखुर", { x: M, y: 3.25, w: W - 2 * M, h: 0.55, align: "center", fontSize: 20, color: C.background2, fontFace: INDIC, isTextBox: true });
  s.addText("github.com/AmulyaJain1804/CLNLP-AIML  ·  Experiment-10", { x: M, y: 4.3, w: W - 2 * M, h: 0.4, align: "center", fontSize: 14, color: C.background2, isTextBox: true });
  s.addNotes("Thank the audience in all four languages: Assamese, Manipuri, Mizo and Bodo. Invite questions.");
}

/* pptxgenjs always writes Office's default colour scheme; replace it with THEME's colours so the
 * scheme colours used above resolve correctly, and turn <c:crossesAt val="autoZero"/> (invalid,
 * written by pptxgenjs for value axes) into the equivalent <c:crosses val="autoZero"/>. */
async function setTheme(file, theme) {
  const JSZip = require(require.resolve("jszip", { paths: [path.dirname(require.resolve("pptxgenjs"))] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${theme.name}">` + slots.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
  const part = "ppt/theme/theme1.xml";
  const xml = await zip.file(part).async("string");
  zip.file(part, xml.replace(/<a:clrScheme[\s\S]*?<\/a:clrScheme>/, () => scheme)
    .replace(/(<a:(?:theme|fontScheme)[^>]*?name=")[^"]*"/g, (_, head) => `${head}${theme.name}"`));
  for (const name of Object.keys(zip.files)) {
    if (zip.files[name].dir) continue;
    if (!name.startsWith("ppt/charts/") || !name.endsWith(".xml")) continue;
    const c = await zip.file(name).async("string");
    zip.file(name, c.split('<c:crossesAt val="autoZero"/>').join('<c:crosses val="autoZero"/>'));
  }
  for (const name of Object.keys(zip.files)) if (zip.files[name].dir) delete zip.files[name];  // folder entries are not parts
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await setTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
