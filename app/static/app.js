"use strict";

// Minimal build-free UI over the public JSON API: search, clients and a document inspector.

const view = document.getElementById("view");
const apiKeyInput = document.getElementById("api-key");
let pollTimer = null;

// ---------------------------------------------------------------------------------------------
// Helpers

function storageGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage may be unavailable */ }
}

function sessionGet(key) {
  try { return sessionStorage.getItem(key); } catch { return null; }
}
function sessionSet(key, value) {
  try { sessionStorage.setItem(key, value); } catch { /* storage may be unavailable */ }
}

apiKeyInput.value = storageGet("advisor-api-key") || "";
apiKeyInput.addEventListener("change", () => storageSet("advisor-api-key", apiKeyInput.value.trim()));

class ApiError extends Error {
  constructor(status, body) {
    super(describeError(status, body));
    this.status = status;
  }
}

function describeError(status, body) {
  const detail = body && body.detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => `${(d.loc || []).slice(1).join(".") || "request"}: ${d.msg}`).join("; ");
  }
  return `${status}: ${detail || "Request failed"}`;
}

async function api(method, path, body) {
  const headers = { "Content-Type": "application/json" };
  const key = apiKeyInput.value.trim();
  if (key) headers["X-API-Key"] = key;
  const response = await fetch(path, {
    method, headers, body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, data);
  return data;
}

// Tiny DOM builder: h("a", {href: "#"}, "text", child). Strings become text nodes (escaped).
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (name.startsWith("on")) el.addEventListener(name.slice(2), value);
    else el.setAttribute(name, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

// Server highlights are HTML-escaped with <mark> tags; keep only bare <mark> elements anyway.
function safeHighlight(html) {
  const template = document.createElement("template");
  template.innerHTML = html || "";
  for (const el of [...template.content.querySelectorAll("*")]) {
    if (el.tagName === "MARK") {
      for (const attr of [...el.attributes]) el.removeAttribute(attr.name);
    } else {
      el.replaceWith(document.createTextNode(el.textContent));
    }
  }
  return template.content;
}

function formatDate(value) {
  return value ? new Date(value).toLocaleString() : "—";
}

function statusBadge(status) {
  return h("span", { class: `badge ${status}` }, status);
}

// Navigation links shown above a page: back to the last search (if any) plus extra links.
function breadcrumbs(...links) {
  const lastSearch = sessionGet("advisor-last-search");
  const q = lastSearch ? new URLSearchParams(lastSearch.split("?")[1] || "").get("q") : null;
  const items = [lastSearch && h("a", { href: lastSearch }, `← Back to search “${q}”`), ...links]
    .filter(Boolean);
  return h("p", { class: "small breadcrumbs" }, items.flatMap((item, i) => (i ? [" · ", item] : [item])));
}

function render(...nodes) {
  view.replaceChildren(...nodes.filter((node) => node !== null && node !== undefined && node !== false));
}

function showError(err) {
  render(h("p", { class: "error" }, err.message || String(err)));
}

// ---------------------------------------------------------------------------------------------
// Router

const routes = [
  [/^#\/search(?:\?(.*))?$/, (m) => renderSearch(new URLSearchParams(m[1] || ""))],
  [/^#\/clients\/?$/, () => renderClients()],
  [/^#\/clients\/([\w-]+)$/, (m) => renderClient(m[1])],
  [/^#\/documents\/([\w-]+)$/, (m) => renderDocument(m[1])],
];

function route() {
  clearTimeout(pollTimer);
  const hash = location.hash || "#/search";
  for (const link of document.querySelectorAll("[data-nav]")) {
    link.classList.toggle("active", hash.startsWith(`#/${link.dataset.nav}`));
  }
  for (const [pattern, handler] of routes) {
    const match = hash.match(pattern);
    if (match) {
      handler(match).catch(showError);
      return;
    }
  }
  location.hash = "#/search";
}

window.addEventListener("hashchange", route);

// ---------------------------------------------------------------------------------------------
// Search

const EXAMPLES = ["NevisWealth", "address proof", "proof of income", "estate planning",
  "Jonathan", "rental properties", "monthly electricity usage"];

async function renderSearch(params) {
  const q = params.get("q") || "";
  const type = params.get("type") || "";
  const input = h("input", { type: "search", name: "q", value: q, placeholder: "Search clients and documents…", autofocus: true });
  const select = h("select", { name: "type" },
    h("option", { value: "" }, "Everything"),
    h("option", { value: "client", selected: type === "client" }, "Clients"),
    h("option", { value: "document", selected: type === "document" }, "Documents"));
  const form = h("form", {
    class: "searchbar",
    onsubmit: (e) => {
      e.preventDefault();
      const next = new URLSearchParams({ q: input.value.trim() });
      if (select.value) next.set("type", select.value);
      location.hash = `#/search?${next}`;
    },
  }, input, select, h("button", { type: "submit" }, "Search"));
  const examples = h("div", { class: "examples small muted" }, "Try: ",
    EXAMPLES.map((ex) => h("a", { href: `#/search?${new URLSearchParams({ q: ex })}` }, ex)));
  const results = h("section");
  render(form, examples, results);
  if (!q) return;
  sessionSet("advisor-last-search", location.hash);

  results.append(h("p", { class: "muted" }, "Searching…"));
  const query = new URLSearchParams({ q });
  if (type) query.set("type", type);
  const started = performance.now();
  const hits = await api("GET", `/search?${query}`);
  const ms = Math.round(performance.now() - started);
  // Document hits carry only `client_id`; load each distinct client once (GET /clients/{id}).
  const clientIds = [...new Set(hits.filter((x) => x.type === "document").map((x) => x.document.client_id))];
  const owners = new Map(await Promise.all(clientIds.map(async (cid) =>
    [cid, await api("GET", `/clients/${cid}`).catch(() => null)])));
  results.replaceChildren(
    h("p", { class: "muted small" }, `${hits.length} result${hits.length === 1 ? "" : "s"} in ${ms} ms`),
    ...hits.map((hit) => renderHit(hit, owners)));
}

function scoreView(score) {
  const bar = h("span", { class: "score-bar" }, h("span", { style: `width:${Math.round(score * 100)}%` }));
  return h("span", { class: "score", title: "relevance score" }, bar, score.toFixed(2));
}

function clientName(client) {
  return client ? `${client.first_name} ${client.last_name}` : "client";
}

function chips(items) {
  return h("div", { class: "chips" }, items.map((item) => h("span", { class: "badge" }, item)));
}

function renderHit(hit, owners) {
  if (hit.type === "client") {
    const c = hit.client;
    return h("article", { class: "card" },
      h("div", { class: "card-head" },
        h("span", { class: "badge type" }, "client"),
        h("a", { class: "card-title", href: `#/clients/${c.id}` }, `${c.first_name} ${c.last_name}`),
        h("span", { class: "muted small" }, c.email),
        scoreView(hit.score)),
      chips(hit.matched_on),
      hit.highlight && h("div", { class: "snippet" }, safeHighlight(hit.highlight)),
      c.description && h("div", { class: "summary small" }, c.description));
  }
  const d = hit.document;
  return h("article", { class: "card" },
    h("div", { class: "card-head" },
      h("span", { class: "badge type" }, "document"),
      h("a", { class: "card-title", href: `#/documents/${d.id}` }, d.title),
      statusBadge(d.status),
      h("a", { class: "small", href: `#/clients/${d.client_id}` }, clientName(owners.get(d.client_id))),
      scoreView(hit.score)),
    chips(hit.matched_on),
    hit.highlight && h("div", { class: "snippet small" }, safeHighlight(hit.highlight)),
    d.summary && h("div", { class: "summary small" }, h("strong", {}, "Summary: "), d.summary));
}

// ---------------------------------------------------------------------------------------------
// Clients

function field(label, input) {
  return h("label", {}, label, input);
}

async function renderClients() {
  const clients = await api("GET", "/clients?limit=200");
  const message = h("p");
  const form = h("form", { class: "stack card" },
    h("h2", {}, "New client"),
    field("First name", h("input", { name: "first_name", required: true })),
    field("Last name", h("input", { name: "last_name", required: true })),
    field("Email", h("input", { name: "email", type: "email", required: true })),
    field("Description", h("textarea", { name: "description", style: "min-height:80px" })),
    field("Social links (one per line)", h("textarea", { name: "social_links", style: "min-height:60px" })),
    h("button", { type: "submit" }, "Create client"),
    message);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = new FormData(form);
    const body = {
      first_name: data.get("first_name"),
      last_name: data.get("last_name"),
      email: data.get("email"),
      description: data.get("description") || null,
      social_links: String(data.get("social_links")).split(/\n+/).map((s) => s.trim()).filter(Boolean),
    };
    try {
      const client = await api("POST", "/clients", body);
      location.hash = `#/clients/${client.id}`;
    } catch (err) {
      message.className = "error";
      message.textContent = err.message;
    }
  });

  const table = h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Name"), h("th", {}, "Email"), h("th", {}, "Created"))),
    h("tbody", {}, clients.map((c) => h("tr", {
      class: "clickable", onclick: () => { location.hash = `#/clients/${c.id}`; },
    }, h("td", {}, `${c.first_name} ${c.last_name}`), h("td", {}, c.email), h("td", {}, formatDate(c.created_at))))));

  render(h("h1", {}, "Clients"),
    h("div", { class: "grid" },
      h("section", { class: "card" }, clients.length ? table : h("p", { class: "muted" }, "No clients yet.")),
      form));
}

async function renderClient(id) {
  const [client, documents] = await Promise.all([
    api("GET", `/clients/${id}`), api("GET", `/clients/${id}/documents`)]);
  const message = h("p");
  const form = h("form", { class: "stack card" },
    h("h2", {}, "Add document"),
    field("Title", h("input", { name: "title", required: true })),
    field("Content", h("textarea", { name: "content", required: true })),
    h("button", { type: "submit" }, "Upload"),
    message);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = new FormData(form);
    try {
      const doc = await api("POST", `/clients/${id}/documents`, {
        title: data.get("title"), content: data.get("content"),
      });
      location.hash = `#/documents/${doc.id}`;
    } catch (err) {
      message.className = "error";
      message.textContent = err.message;
    }
  });

  const docs = documents.length
    ? h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Title"), h("th", {}, "Status"), h("th", {}, "Created"))),
      h("tbody", {}, documents.map((d) => h("tr", {
        class: "clickable", onclick: () => { location.hash = `#/documents/${d.id}`; },
      }, h("td", {}, d.title), h("td", {}, statusBadge(d.status)), h("td", {}, formatDate(d.created_at))))))
    : h("p", { class: "muted" }, "No documents yet.");

  render(
    breadcrumbs(h("a", { href: "#/clients" }, "All clients")),
    h("h1", {}, `${client.first_name} ${client.last_name}`),
    h("dl", { class: "meta card" },
      h("dt", {}, "Email"), h("dd", {}, client.email),
      h("dt", {}, "Description"), h("dd", {}, client.description || "—"),
      h("dt", {}, "Social links"), h("dd", {}, client.social_links.length
        ? client.social_links.map((link) => h("div", {}, link)) : "—"),
      h("dt", {}, "Created"), h("dd", {}, formatDate(client.created_at)),
      h("dt", {}, "ID"), h("dd", { class: "small muted" }, client.id)),
    h("div", { class: "grid" }, h("section", { class: "card" }, h("h2", {}, "Documents"), docs), form));
}

// ---------------------------------------------------------------------------------------------
// Document inspector

function summaryRanges(content, summary) {
  if (!summary) return [];
  const ranges = [];
  for (const raw of summary.split(/(?<=[.!?])\s+/)) {
    const sentence = raw.replace(/…$/, "").trim();
    const start = sentence ? content.indexOf(sentence) : -1;
    if (start >= 0) ranges.push([start, start + sentence.length]);
  }
  return ranges;
}

// Content with summary sentences and the selected chunk highlighted (text nodes only).
function contentView(content, summary, chunk) {
  const sentences = summaryRanges(content, summary);
  const chunkRange = chunk ? [chunk.char_start, chunk.char_end] : null;
  const points = new Set([0, content.length]);
  for (const [s, e] of [...sentences, ...(chunkRange ? [chunkRange] : [])]) { points.add(s); points.add(e); }
  const cuts = [...points].sort((a, b) => a - b);
  const container = h("div", { class: "content-view" });
  for (let i = 0; i < cuts.length - 1; i++) {
    const [s, e] = [cuts[i], cuts[i + 1]];
    if (s === e) continue;
    const classes = [];
    if (sentences.some(([a, b]) => s >= a && e <= b)) classes.push("summary-sentence");
    if (chunkRange && s >= chunkRange[0] && e <= chunkRange[1]) classes.push("chunk-range");
    const text = content.slice(s, e);
    container.append(classes.length ? h("span", { class: classes.join(" ") }, text) : text);
  }
  return container;
}

async function renderDocument(id) {
  const [doc, index] = await Promise.all([
    api("GET", `/documents/${id}`), api("GET", `/documents/${id}/index`)]);
  const client = await api("GET", `/clients/${doc.client_id}`);
  let selectedChunk = null;
  const contentSlot = h("div");
  const drawContent = () => contentSlot.replaceChildren(contentView(doc.content, doc.summary, selectedChunk));
  drawContent();

  const reprocess = h("button", {
    class: "secondary",
    onclick: async () => {
      reprocess.disabled = true;
      await api("POST", `/documents/${id}/reprocess`).catch(showError);
      route();
    },
  }, "Reprocess");

  const chunkRows = index.chunks.map((chunk) => {
    const row = h("tr", {
      class: "clickable",
      onclick: () => {
        selectedChunk = selectedChunk === chunk ? null : chunk;
        for (const r of row.parentNode.children) r.classList.remove("selected");
        if (selectedChunk) row.classList.add("selected");
        drawContent();
        contentSlot.querySelector(".chunk-range")?.scrollIntoView({ block: "nearest" });
      },
    },
    h("td", {}, chunk.ordinal),
    h("td", {}, `${chunk.char_start}–${chunk.char_end}`),
    h("td", {}, chunk.text.split(/\s+/).length),
    h("td", {}, chunk.text.length > 160 ? `${chunk.text.slice(0, 160)}…` : chunk.text));
    return row;
  });

  const isBusy = doc.status === "pending" || doc.status === "processing";
  render(
    breadcrumbs(h("a", { href: `#/clients/${client.id}` }, `All documents of ${client.first_name} ${client.last_name}`)),
    h("section", { class: "card client-card" },
      h("div", { class: "card-head" },
        h("span", { class: "badge type" }, "client"),
        h("a", { class: "card-title", href: `#/clients/${client.id}` }, `${client.first_name} ${client.last_name}`),
        h("span", { class: "muted small" }, client.email)),
      client.description && h("div", { class: "summary small" }, client.description)),
    h("div", { class: "card-head" }, h("h1", {}, doc.title), statusBadge(doc.status), reprocess),
    isBusy && h("p", { class: "muted" }, "Indexing in progress — this page refreshes automatically."),
    doc.error && h("p", { class: "error" }, `Last error: ${doc.error}`),
    h("dl", { class: "meta card" },
      h("dt", {}, "Created"), h("dd", {}, formatDate(doc.created_at)),
      h("dt", {}, "Indexed"), h("dd", {}, formatDate(doc.processed_at)),
      h("dt", {}, "Indexing time"), h("dd", {}, index.processing_ms === null ? "—" : `${index.processing_ms} ms`),
      h("dt", {}, "Attempts"), h("dd", {}, index.attempts),
      h("dt", {}, "Embedding model"), h("dd", {}, index.embedding_model),
      h("dt", {}, "ID"), h("dd", { class: "small muted" }, doc.id)),
    h("h2", {}, "Summary"),
    h("p", {}, doc.summary || h("span", { class: "muted" }, "Not available yet.")),
    h("h2", {}, "Content"),
    h("div", { class: "legend" },
      h("span", {}, h("i", { style: "background:var(--summary)" }), "summary sentence"),
      h("span", {}, h("i", { style: "background:var(--chunk)" }), "selected chunk")),
    contentSlot,
    h("h2", {}, `Chunks (${index.chunks.length})`),
    index.chunks.length
      ? h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "Chars"), h("th", {}, "Words"), h("th", {}, "Text"))),
        h("tbody", {}, chunkRows))
      : h("p", { class: "muted" }, "No chunks yet: semantic search starts once indexing finishes."),
    h("h2", {}, "Full-text index (top lexemes)"),
    h("p", { class: "muted small" }, "Stemmed terms in the Postgres inverted index. Weight A = title, B = content."),
    h("div", { class: "lexemes" }, index.lexemes.map((lx) =>
      h("span", { class: "badge lexeme" }, `${lx.lexeme} ×${lx.count} (${[...new Set(lx.weights)].join("")})`))));

  if (isBusy) pollTimer = setTimeout(route, 1000);
}

route();
