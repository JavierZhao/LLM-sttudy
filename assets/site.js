/* LLM Interview Prep: shared page behavior.
   Builds sidebar, table of contents, pager and progress tracking from
   window.LLM_MANIFEST, then renders math (KaTeX) and code (highlight.js).
   Every storage access is wrapped: pages must work with storage disabled. */
(function () {
  "use strict";

  var M = window.LLM_MANIFEST || { parts: [] };
  var body = document.body;
  var ROOT = (body.getAttribute("data-root") || ".").replace(/\/$/, "");
  var CURRENT = body.getAttribute("data-page") || "";

  // ---------- storage helpers ----------
  function load(key, fallback) {
    try { var v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); }
    catch (e) { return fallback; }
  }
  function save(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* ignore */ }
  }
  var done = load("llm-done", {}) || {};

  function allPages() {
    var out = [];
    M.parts.forEach(function (part) {
      part.pages.forEach(function (p) { out.push(Object.assign({ part: part }, p)); });
    });
    return out;
  }
  function href(p) { return ROOT + "/pages/" + p.slug + ".html"; }
  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (html != null) e.innerHTML = html;
    return e;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  // ---------- theme ----------
  var themeOrder = ["auto", "light", "dark"];
  var theme = load("llm-theme", "auto");
  var mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function effectiveTheme() {
    if (theme === "light" || theme === "dark") return theme;
    return mq && mq.matches ? "dark" : "light";
  }
  function applyTheme() {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    var dark = effectiveTheme() === "dark";
    var l = document.getElementById("hljs-light"), d = document.getElementById("hljs-dark");
    if (l) l.disabled = dark;
    if (d) d.disabled = !dark;
    var b = document.getElementById("theme-btn");
    if (b) b.textContent = "Theme: " + theme;
  }
  if (mq && mq.addEventListener) mq.addEventListener("change", applyTheme);

  // ---------- sidebar ----------
  function buildSidebar() {
    var side = document.getElementById("sidebar");
    if (!side) return;
    var pages = allPages();
    var written = pages.filter(function (p) { return p.status !== "planned"; });
    var nDone = pages.filter(function (p) { return done[p.n]; }).length;
    var html = '<div class="side-brand"><a href="' + ROOT + '/index.html">' + esc(M.title || "Home") + '</a>' +
      '<div class="side-tools"><button class="icon-btn" id="theme-btn" type="button">Theme</button></div></div>' +
      '<div class="side-progress">' + nDone + ' / ' + pages.length + ' pages done · ' + written.length + ' written' +
      '<div class="bar"><span style="width:' + (100 * nDone / Math.max(1, pages.length)).toFixed(1) + '%"></span></div></div>';
    M.parts.forEach(function (part) {
      html += '<div class="side-part">Part ' + esc(part.id) + ' · ' + esc(part.title) + '</div><ul class="side-list">';
      part.pages.forEach(function (p) {
        var inner = '<span class="num">' + p.n + '</span><span>' + esc(p.title) + '</span>' +
          '<span class="done-mark">' + (done[p.n] ? "✓" : "") + '</span>';
        if (p.status === "planned") {
          html += '<li><span class="planned" title="Not written yet">' + inner + '</span></li>';
        } else {
          html += '<li><a href="' + href(p) + '"' + (p.n === CURRENT ? ' class="current" aria-current="page"' : '') + '>' + inner + '</a></li>';
        }
      });
      html += '</ul>';
    });
    side.innerHTML = html;
    var cur = side.querySelector("a.current");
    // scroll only the sidebar (scrollIntoView would also scroll the window)
    if (cur) side.scrollTop = Math.max(0, cur.offsetTop - side.clientHeight / 2);
    document.getElementById("theme-btn").addEventListener("click", function () {
      theme = themeOrder[(themeOrder.indexOf(theme) + 1) % themeOrder.length];
      save("llm-theme", theme);
      applyTheme();
    });
  }

  function setupNavToggle() {
    var side = document.getElementById("sidebar");
    if (!side) return;
    var tog = el("button", { "class": "nav-toggle", type: "button", "aria-label": "Open navigation" }, "☰ Contents");
    tog.addEventListener("click", function (ev) { ev.stopPropagation(); body.classList.toggle("nav-open"); });
    body.appendChild(tog);
    document.addEventListener("click", function (ev) {
      if (body.classList.contains("nav-open") && !side.contains(ev.target)) body.classList.remove("nav-open");
    });
  }

  // ---------- headings, anchors, toc ----------
  function slugify(s) {
    return s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9\s-]/g, "").trim().replace(/\s+/g, "-").slice(0, 60);
  }
  function buildToc() {
    var content = document.getElementById("content");
    var toc = document.getElementById("toc");
    if (!content) return;
    var hs = content.querySelectorAll("h2, h3");
    var used = {};
    var items = [];
    hs.forEach(function (h) {
      if (h.closest(".page-header") || h.closest("details")) return;
      if (!h.id) {
        var base = slugify(h.textContent) || "section", id = base, i = 2;
        while (used[id] || document.getElementById(id)) id = base + "-" + i++;
        h.id = id;
      }
      used[h.id] = true;
      var a = el("a", { "class": "anchor", href: "#" + h.id, "aria-hidden": "true" }, "#");
      h.appendChild(a);
      items.push(h);
    });
    if (!toc || !items.length) return;
    var html = '<div class="toc-title">On this page</div><ul>';
    items.forEach(function (h) {
      var text = h.cloneNode(true);
      var an = text.querySelector(".anchor"); if (an) an.remove();
      html += '<li class="' + h.tagName.toLowerCase() + '"><a href="#' + h.id + '">' + esc(text.textContent.trim()) + '</a></li>';
    });
    toc.innerHTML = html + "</ul>";
    if ("IntersectionObserver" in window) {
      var links = {};
      toc.querySelectorAll("a").forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
      var obs = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            toc.querySelectorAll("a.active").forEach(function (a) { a.classList.remove("active"); });
            var a = links[e.target.id]; if (a) a.classList.add("active");
          }
        });
      }, { rootMargin: "0px 0px -70% 0px" });
      items.forEach(function (h) { obs.observe(h); });
    }
  }

  // ---------- header: done toggle ----------
  function buildDoneToggle() {
    if (!CURRENT || CURRENT === "index") return;
    var meta = document.querySelector(".page-header .meta");
    if (!meta) return;
    var b = el("button", { "class": "icon-btn done-toggle", type: "button" });
    function paint() {
      b.textContent = done[CURRENT] ? "✓ Done" : "Mark as done";
      b.classList.toggle("is-done", !!done[CURRENT]);
    }
    b.addEventListener("click", function () {
      if (done[CURRENT]) delete done[CURRENT]; else done[CURRENT] = true;
      save("llm-done", done); paint(); buildSidebar(); applyTheme();
    });
    paint();
    meta.appendChild(b);
  }

  // ---------- pager ----------
  function buildPager() {
    var pager = document.querySelector(".pager");
    if (!pager || !CURRENT) return;
    var pages = allPages().filter(function (p) { return p.status !== "planned" || p.n === CURRENT; });
    var i = pages.findIndex(function (p) { return p.n === CURRENT; });
    if (i < 0) return;
    var html = "";
    if (i > 0) { var p = pages[i - 1]; html += '<a class="prev" href="' + href(p) + '"><span class="dir">← Previous</span>' + p.n + " · " + esc(p.title) + "</a>"; }
    if (i < pages.length - 1) { var q = pages[i + 1]; html += '<a class="next" href="' + href(q) + '"><span class="dir">Next →</span>' + q.n + " · " + esc(q.title) + "</a>"; }
    pager.innerHTML = html;
  }

  // ---------- Q&A controls ----------
  function buildQaControls() {
    var qs = document.querySelectorAll("details.qa");
    if (!qs.length) return;
    var first = qs[0];
    var bar = el("div", { "class": "qa-controls" });
    function btn(label, fn) { var b = el("button", { "class": "icon-btn", type: "button" }, label); b.addEventListener("click", fn); bar.appendChild(b); }
    btn("Expand all answers", function () { qs.forEach(function (d) { if (d.style.display !== "none") d.open = true; }); });
    btn("Collapse all", function () { qs.forEach(function (d) { d.open = false; }); });
    ["all", "warmup", "core", "deep"].forEach(function (t) {
      btn(t === "all" ? "All tiers" : t === "warmup" ? "Warm-up only" : t === "core" ? "Core only" : "Deep-dive only", function () {
        qs.forEach(function (d) { d.style.display = (t === "all" || d.getAttribute("data-tier") === t) ? "" : "none"; });
      });
    });
    first.parentNode.insertBefore(bar, first);
  }

  // ---------- code: copy buttons ----------
  function buildCopyButtons() {
    document.querySelectorAll("pre > code").forEach(function (code) {
      var b = el("button", { "class": "icon-btn copy-btn", type: "button" }, "Copy");
      b.addEventListener("click", function () {
        var text = code.innerText;
        var ok = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1200); };
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(ok, function () {});
        } catch (e) { /* ignore */ }
      });
      code.parentNode.appendChild(b);
    });
  }

  // ---------- index page ----------
  function buildIndex() {
    var root = document.getElementById("index-root");
    if (!root) return;
    var filter = load("llm-index-filter", "all");
    function render() {
      var html = "";
      M.parts.forEach(function (part) {
        var rows = part.pages.filter(function (p) { return filter === "all" || (filter === "p0" && p.tier === "P0") || (filter === "todo" && !done[p.n]); });
        if (!rows.length) return;
        var mins = part.pages.reduce(function (s, p) { return s + (p.minutes || 0); }, 0);
        html += '<section class="part-block"><h2 id="part-' + part.id + '">Part ' + esc(part.id) + ' · ' + esc(part.title) +
          ' <span class="muted small">(~' + Math.round(mins / 60 * 10) / 10 + ' h)</span></h2>';
        rows.forEach(function (p) {
          var planned = p.status === "planned";
          var title = planned ? '<span class="title">' + esc(p.title) + ' <span class="muted small">(planned)</span></span>'
                              : '<a class="title" href="' + href(p) + '">' + esc(p.title) + '</a>';
          html += '<div class="page-row' + (planned ? " planned" : "") + (done[p.n] ? " is-done" : "") + '">' +
            '<span class="num">' + p.n + '</span>' + title +
            '<span class="badge ' + p.tier.toLowerCase() + '">' + p.tier + '</span>' +
            '<span class="mins">' + p.minutes + ' min</span>' +
            (p.desc ? '<div class="desc">' + esc(p.desc) + '</div>' : '') + '</div>';
        });
        html += "</section>";
      });
      root.innerHTML = html || '<p class="muted">Nothing matches this filter.</p>';
      document.querySelectorAll(".index-filters button").forEach(function (b) {
        b.setAttribute("aria-pressed", String(b.getAttribute("data-filter") === filter));
      });
    }
    document.querySelectorAll(".index-filters button").forEach(function (b) {
      b.addEventListener("click", function () { filter = b.getAttribute("data-filter"); save("llm-index-filter", filter); render(); });
    });
    render();
  }

  // ---------- math & code rendering ----------
  function renderMath() {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(document.getElementById("content") || document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false,
      strict: "ignore",
      trust: false,
      macros: {
        "\\R": "\\mathbb{R}",
        "\\E": "\\mathbb{E}",
        "\\softmax": "\\operatorname{softmax}",
        "\\KL": "\\mathrm{KL}",
        "\\dmodel": "d_{\\text{model}}",
        "\\dff": "d_{\\text{ff}}",
        "\\concat": "\\operatorname{concat}",
        "\\argmax": "\\operatorname*{arg\\,max}",
        "\\argmin": "\\operatorname*{arg\\,min}",
        "\\RMSNorm": "\\operatorname{RMSNorm}",
        "\\LayerNorm": "\\operatorname{LayerNorm}",
        "\\RoPE": "\\operatorname{RoPE}",
        "\\clip": "\\operatorname{clip}",
        "\\sg": "\\operatorname{sg}"
      }
    });
  }
  function renderCode() {
    if (!window.hljs) return;
    document.querySelectorAll("pre code").forEach(function (b) {
      if (!/\blanguage-/.test(b.className)) b.classList.add("language-python");
      try { window.hljs.highlightElement(b); } catch (e) { /* ignore */ }
    });
  }

  buildSidebar();
  setupNavToggle();
  applyTheme();
  buildDoneToggle();
  buildQaControls();
  renderMath();
  buildToc();
  buildPager();
  renderCode();
  buildCopyButtons();
  buildIndex();
})();
