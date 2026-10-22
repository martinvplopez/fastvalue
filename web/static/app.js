/* ============================================================
   FastValue: frontend
   All server data is escaped before it touches innerHTML.
   ============================================================ */

(() => {
    'use strict';

    // ------------------------------------------------------------ utilities

    const $ = (sel, root = document) => root.querySelector(sel);
    const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

    const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ESC[c]);
    const safeUrl = (u) => (typeof u === 'string' && /^https?:\/\//i.test(u) ? u : null);
    const TICKER_RE = /^[A-Z0-9^][A-Z0-9.\-=^]{0,11}$/;  // same rule as the server (web/app.py)

    const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
    const fmt = {
        num: (v, d = 2) => (isNum(v) ? v.toFixed(d) : 'n/a'),
        signed: (v, d = 0) => (isNum(v) ? `${v > 0 ? '+' : ''}${v.toFixed(d)}` : 'n/a'),
        pct: (v, d = 1) => (isNum(v) ? `${(v * 100).toFixed(d)}%` : 'n/a'),
        // A change that rounds to zero prints as 0.0% with no sign, so it never reads as a gain or a loss.
        spct: (v, d = 1) => {
            if (!isNum(v)) return 'n/a';
            const text = (v * 100).toFixed(d);
            return Number(text) === 0 ? `${(0).toFixed(d)}%` : `${v > 0 ? '+' : ''}${text}%`;
        },
        // Currency-aware: the Portfolio tab is in its own currency (EUR after a Trade Republic import), and a
        // listing's price is shown in the listing's currency (GBp for London, CAD for Toronto, ...).
        money: (v, d = 2, ccy = 'USD') => {
            if (!isNum(v)) return 'n/a';
            const code = String(ccy || 'USD');
            const n = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
            const sign = v < 0 ? '-' : '';
            if (code === 'GBp' || code === 'GBX') return `${sign}${n}p`;
            const sym = { USD: '$', EUR: '€', GBP: '£', JPY: '¥' }[code.toUpperCase()];
            return sym ? `${sign}${sym}${n}` : `${sign}${n} ${code}`;
        },
        ratio: (v) => (v === null || v === undefined ? 'n/a' : !isNum(v) ? 'inf' : v.toFixed(2)),
    };
    // Icons: one 16px set at a 1.6px stroke in currentColor, like the brand mark. No text glyph stands in for an icon.
    const ICON_PATHS = { x: 'M4 4l8 8M12 4l-8 8', up: 'M4 10l4-4 4 4', down: 'M4 6l4 4 4-4', check: 'M3 8.5l3.2 3.2L13 5' };
    const icon = (name) => `<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="${ICON_PATHS[name]}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    // The count with the right noun ("1 note", "2 notes"), so a label never reads "1 wins".
    // One day format for every visible date: 21 Oct 2026. Anything that is not an ISO date (a status word, a month) comes back as it was.
    const MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const fmtDay = (v) => {
        const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(v ?? ''));
        return m ? `${Number(m[3])} ${MONTH_ABBR[Number(m[2]) - 1]} ${m[1]}` : String(v ?? '');
    };
    const plural = (n, one, many) => `${n} ${Number(n) === 1 ? one : many}`;
    const capFirst = (s) => { const v = String(s ?? ''); return v.charAt(0).toUpperCase() + v.slice(1); };
    // A machine enum (UNAVAILABLE, NOT_RATED) as one sentence-case phrase ("Unavailable", "Not rated").
    const enumLabel = (v) => capFirst(String(v || '').toLowerCase().replace(/_/g, ' '));
    // The data-quality flag of an agent: the word stays visible, the tone is a warning for anything but live data.
    const qualityBadge = (q) => { const k = String(q || 'LIVE').toLowerCase(); return `<span class="badge ${k === 'live' ? 'live' : 'warn'}">${esc(enumLabel(k))}</span>`; };
    const titleCase = (s) => String(s).split('_').map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
    const decisionLabel = (d) => String(d || '').replace(/_/g, ' ');
    // Class names come from server values: keep only [a-z-] so a stored value can never leave the attribute.
    const cls = (v, fallback) => String(v || fallback).toLowerCase().replace(/_/g, '-').replace(/[^a-z-]/g, '');
    const actionClass = (a) => `action-${cls(a, 'hold')}`;
    // Actions are what to do, not ratings: sentence case words ("HOLD" is the action key for "Nothing to do").
    const ACTION_LABEL = { EXIT: 'Exit', TRIM: 'Trim', REVIEW: 'Look at', ADD: 'Add', HOLD: 'Nothing to do', ENTER: 'Enter', ENTER_STAGED: 'Enter in stages', WATCH: 'Watch', AVOID: 'Avoid', NOT_RATED: 'Not rated' };
    const actionLabel = (a) => ACTION_LABEL[a] || String(a || '').replace(/_/g, ' ');
    // Where a score sits against the owner's entry and exit levels, in words and numbers (the bands themselves are never shown).
    const levelZone = (score, lv) => levelRelation(score, lv).text;
    const TIP_TIMING = 'Entry timing: a second score, from price and the market (trend, pullbacks, volume, options activity, the market regime and the VIX), that says when to buy and how much. It never changes the owner score.';
    const SETUP_LABELS = { pullback_in_uptrend: 'pullback in uptrend', extended: 'extended', trend_broken: 'trend broken', neutral: 'no setup' };
    const setupLabel = (s) => SETUP_LABELS[s] || (s ? String(s).replace(/_/g, ' ') : '');
    const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    // An (i) that explains what sits before it; the text is read by screen readers and opens on hover, focus and tap.
    const infoTip = (tip) => `<span class="info" tabindex="0" role="note" aria-label="${esc(tip)}" data-tip="${esc(tip)}">i</span>`;
    const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
    // Writes that skip a value already in place: an unchanged write still queues DOM mutation records (which the
    // edge cues re-measure on), and an unchanged innerHTML rebuilds the whole block.
    const setIf = (el, prop, value) => { if (el && el[prop] !== value) el[prop] = value; };
    const lastHtml = new WeakMap();
    // Only for blocks written by one function and holding no form state, where equal markup means an equal DOM.
    function setHtml(el, html) {
        if (!el || lastHtml.get(el) === html) return;
        el.innerHTML = html;
        lastHtml.set(el, html);
    }
    const clone = (o) => JSON.parse(JSON.stringify(o));
    const shortName = (n) => String(n || '').replace(/ Agent$/, '');

    // Errors stay until dismissed (WCAG 2.2.1) and are announced as alerts; other toasts hide on a timer that
    // pauses while the pointer or focus is on them. At most two stay on screen: a third removes the oldest.
    function toast(message, kind = 'info', ms = 5000, action) {
        if (kind === 'error' && String(message).includes(SERVER_DOWN)) return;  // the banner says it once, with the way out
        const host = $('#toast-host');
        const el = document.createElement('div');
        const isError = kind === 'error';
        el.className = `toast toast-${kind}`;  // prefixed: a bare "info" class is the (i) tip icon
        el.setAttribute('role', isError ? 'alert' : 'status');
        const text = document.createElement('span');
        text.textContent = message;
        el.appendChild(text);
        if (action && action.label) {
            const act = document.createElement('button');
            act.type = 'button';
            act.className = 'btn btn-sm toast-action';
            act.textContent = action.label;
            act.addEventListener('click', () => { el.remove(); if (action.onClick) action.onClick(); });
            el.appendChild(act);
        }
        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'icon-btn toast-close';
        close.setAttribute('aria-label', 'Dismiss');
        close.innerHTML = icon('x');
        close.addEventListener('click', () => el.remove());
        el.appendChild(close);
        while (host.children.length >= 2) host.firstElementChild.remove();
        host.appendChild(el);
        if (isError) return;
        let timer = null;
        let left = ms;
        let startedAt = 0;
        const start = () => { startedAt = Date.now(); timer = setTimeout(() => el.remove(), left); };
        const pause = () => { if (timer === null) return; clearTimeout(timer); timer = null; left = Math.max(1500, left - (Date.now() - startedAt)); };
        const resume = () => { if (timer === null && el.isConnected) start(); };
        el.addEventListener('mouseenter', pause);
        el.addEventListener('focusin', pause);
        el.addEventListener('mouseleave', resume);
        el.addEventListener('focusout', resume);
        start();
    }

    // The local server stopped (its Terminal window was closed): say so once, in one banner that reconnects by itself,
    // instead of a "Failed to fetch" toast per action. Every request goes through fetchLocal to notice it.
    const SERVER_DOWN = 'FastValue is not answering';
    let serverTimer = null;
    let appReady = false;
    function showServerDown() {
        const banner = $('#server-down');
        if (!banner || !banner.hidden) return;
        banner.hidden = false;
        const probe = async () => {
            try {
                const res = await fetch('/api/agents', { cache: 'no-store' });
                if (!res.ok) return;
            } catch (_) { return; }
            clearInterval(serverTimer);
            serverTimer = null;
            banner.hidden = true;
            if (!appReady) { location.reload(); return; }  // it stopped before the page finished starting
            toast('FastValue is back.', 'success');
        };
        serverTimer = setInterval(probe, 5000);
        $('#server-retry').onclick = probe;
    }

    async function fetchLocal(path, options) {
        try {
            return await fetch(path, options);
        } catch (err) {
            if (!(err instanceof TypeError)) throw err;
            showServerDown();
            throw new Error(SERVER_DOWN);
        }
    }

    async function api(path, { method = 'GET', body } = {}) {
        const res = await fetchLocal(path, {
            method,
            headers: body ? { 'Content-Type': 'application/json' } : undefined,
            body: body ? JSON.stringify(body) : undefined,
        });
        let data = {};
        try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
        if (!res.ok || data.error) throw new Error(data.error || `${res.status} ${res.statusText}`);
        return data;
    }

    function setBusy(btn, busy) {
        if (!btn) return;
        btn.disabled = busy;
        btn.classList.toggle('loading', busy);
        if (busy) btn.setAttribute('aria-busy', 'true'); else btn.removeAttribute('aria-busy');
    }

    // Switched-on agents that make reviews and scans slower (slow_note in their metadata).
    const agentLabel = (a) => a.name.replace(/ Agent$/, '');
    const slowAgents = () => (state.agents || []).filter((a) => a.slow_note && ((state.config || {}).enabled || {})[a.key] !== false);

    // While a review or scan runs long, and after it ran long, name the agent that slows it and how to switch it off.
    // Returns the function to call when the request ends.
    function slowHint(sel, noun) {
        const el = $(sel);
        if (!el) return () => {};
        el.hidden = true;
        const started = performance.now();
        const show = (lead) => {
            const slow = slowAgents();
            if (!slow.length) return;
            el.innerHTML = `${esc(lead)} ${slow.map((a) => `${esc(agentLabel(a))} ${esc(a.slow_note)}`).join(' ')} <a href="#weights" data-goto="weights">Open Weights</a>`;
            el.hidden = false;
        };
        const timer = setTimeout(() => show(`The ${noun} is taking a while.`), 8000);
        return () => {
            clearTimeout(timer);
            const secs = (performance.now() - started) / 1000;
            if (secs >= 20) show(`That ${noun} took ${secs.toFixed(0)} seconds.`);
            else el.hidden = true;
        };
    }

    // Settings saved before the rename to FastValue used the "quant-" prefix.
    function storageGet(key) { try { return localStorage.getItem(key) ?? localStorage.getItem(key.replace(/^fastvalue-/, 'quant-')); } catch (_) { return null; } }
    function storageSet(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* private mode */ } }

    // Score color for heat cells: green for positive, red for negative.
    function heatStyle(score) {
        if (!isNum(score)) return '';
        const strength = Math.min(60, Math.abs(score) * 0.6) + 8;
        const color = score >= 0 ? 'var(--pos)' : 'var(--neg)';
        return `background: color-mix(in srgb, ${color} ${strength.toFixed(0)}%, transparent);`;
    }

    // ------------------------------------------------------------ charts

    const charts = {};
    const legends = {};

    function destroyChart(id) {
        if (charts[id]) { charts[id].remove(); delete charts[id]; }
        delete legends[id];
    }

    // A chart is a picture of numbers: its text alternative names the series and the range, and the values it draws
    // are also in the page's tables and legends. Callers pass the sentence; without one the card's heading names it.
    const chartDay = (t) => (typeof t === 'number' ? new Date(t * 1000).toISOString().slice(0, 10) : (t && t.year ? `${t.year}-${String(t.month).padStart(2, '0')}-${String(t.day).padStart(2, '0')}` : String(t || '')));
    function chartSummary(title, lines, f = (v) => fmt.num(v, 0)) {
        const have = lines.filter((l) => l.data && l.data.length);
        if (!have.length) return title;
        const first = have.map((l) => chartDay(l.data[0].time)).sort()[0];
        const last = have.map((l) => chartDay(l.data[l.data.length - 1].time)).sort().pop();
        return `${title}, ${first} to ${last}: ${have.map((l) => `${l.label} ends at ${f(l.data[l.data.length - 1].value)}`).join('; ')}.`;
    }
    function makeChart(id, extra = {}, label = '') {
        destroyChart(id);
        const el = document.getElementById(id);
        el.innerHTML = '';
        let alt = label;
        if (!alt) {
            const head = el.closest('.card')?.querySelector('h3, h4');
            const clone = head ? head.cloneNode(true) : null;
            if (clone) clone.querySelectorAll('.info').forEach((n) => n.remove());
            alt = clone ? `${clone.textContent.trim()}, chart` : 'Chart';
        }
        el.setAttribute('role', 'img');
        el.setAttribute('aria-label', alt);
        // Recessive furniture: no vertical grid, hairline horizontal grid, no axis borders, a solid hairline crosshair.
        const hair = { color: cssVar('--border-strong'), width: 1, style: 0, labelBackgroundColor: cssVar('--text-3') };
        const chart = LightweightCharts.createChart(el, {
            autoSize: true,
            layout: { background: { type: 'solid', color: 'transparent' }, textColor: cssVar('--text-3'), fontFamily: "Inter, -apple-system, 'Segoe UI', sans-serif", fontSize: 12 },
            grid: { vertLines: { visible: false }, horzLines: { color: cssVar('--grid') } },
            rightPriceScale: { borderVisible: false },
            timeScale: { borderVisible: false },
            crosshair: { mode: 0, vertLine: hair, horzLine: hair },
            ...extra,
            // The browser's locale would write month names in its own language inside an English page.
            localization: { locale: 'en-GB', ...(extra.localization || {}) },
        });
        charts[id] = chart;
        el.querySelectorAll('table').forEach((t) => t.setAttribute('aria-hidden', 'true'));
        return chart;
    }

    // A legend entry is [label, color]: a static swatch (bars, heatmaps). Or an object for a line on a chart:
    //   { key, label, color, series, lines, dash, hidden, block }
    //   series: one series or an array; lines: price lines; dash: dashed swatch; hidden: hidden until the user shows it;
    //   block: a square swatch.
    // With { chart } (the id given to makeChart), each object with series or lines is a button that hides or shows them.
    // The price scale refits, and the choice is kept per chart in this browser. One series always stays visible, unless
    // `base` says the chart keeps a series of its own outside the legend (the candles of a price chart).
    function legend(containerId, items, { chart: chartId = null, base = false } = {}) {
        const box = $(`#${containerId}`);
        const swatch = (it) => `<span class="legend-swatch${it.dash ? ' dash' : ''}${it.block ? ' block' : ''}" style="--swatch:${esc(it.color)}"></span>`;
        const asList = (v) => (v ? (Array.isArray(v) ? v : [v]) : []);
        const norm = items.map((it) => (Array.isArray(it) ? it : { ...it, series: asList(it.series), lines: asList(it.lines) }));
        const objs = norm.filter((it) => !Array.isArray(it));
        const buttonSeries = chartId && (base || objs.filter((it) => it.series.length).length >= 2);  // a lone line that cannot be hidden is drawn static
        const isButton = (it) => chartId && (it.series.length ? buttonSeries : it.lines.length > 0);
        if (chartId) { box.setAttribute('role', 'group'); box.setAttribute('aria-label', 'Lines in the chart'); }
        box.innerHTML = norm.map((it) => {
            if (Array.isArray(it)) return `<span class="legend-item"><span class="legend-swatch" style="--swatch:${esc(it[1])}"></span>${esc(it[0])}</span>`;
            if (!isButton(it)) return `<span class="legend-item">${swatch(it)}${esc(it.label)}</span>`;
            return `<button type="button" class="legend-item" data-legend-chart="${esc(chartId)}" data-legend-key="${esc(it.key)}" aria-pressed="true">${swatch(it)}<span class="legend-label">${esc(it.label)}</span></button>`;
        }).join('');
        if (!chartId) return;
        const stored = legendStored(chartId);
        const reg = { base, entries: new Map() };
        legends[chartId] = reg;
        objs.filter(isButton).forEach((o) => {
            reg.entries.set(String(o.key), {
                // A copy: a price line's options() is the chart's live object, so hiding the line would overwrite it.
                series: o.series, lines: o.lines.map((line) => ({ line, options: { ...line.options() } })),
                visible: true, wantsHidden: typeof stored[o.key] === 'boolean' ? !stored[o.key] : !!o.hidden,
            });
        });
        const seriesButtons = [...reg.entries.values()].filter((e) => e.series.length);
        if (!base && seriesButtons.length && seriesButtons.every((e) => e.wantsHidden)) seriesButtons[0].wantsHidden = false;
        reg.entries.forEach((entry, key) => { if (entry.wantsHidden) setEntryVisible(chartId, key, false); delete entry.wantsHidden; });
        syncKeepOne(chartId);
    }

    const legendStored = (chartId) => {
        try {
            const v = JSON.parse(storageGet(`fastvalue-legend-${chartId}`));
            return v && typeof v === 'object' && !Array.isArray(v) ? v : {};
        } catch (_) { return {}; }
    };
    const legendSave = (chartId, obj) => storageSet(`fastvalue-legend-${chartId}`, JSON.stringify(obj));
    // A table checkbox that puts a line back on the chart also clears any earlier "hide" of it.
    function legendForget(chartId, key) {
        const stored = legendStored(chartId);
        if (key in stored) { delete stored[key]; legendSave(chartId, stored); }
    }

    function legendButton(chartId, key) {
        return [...document.querySelectorAll(`[data-legend-chart="${chartId}"]`)].find((b) => b.dataset.legendKey === String(key));
    }

    function setEntryVisible(chartId, key, on) {
        const entry = (legends[chartId] || { entries: new Map() }).entries.get(String(key));
        if (!entry) return;
        entry.visible = on;
        entry.series.forEach((s) => { s.applyOptions({ visible: on }); s.priceScale().applyOptions({ autoScale: true }); });
        entry.lines.forEach(({ line, options }) => line.applyOptions(on ? options : { lineVisible: false, axisLabelVisible: false, title: '' }));
        const btn = legendButton(chartId, key);
        if (btn) { btn.setAttribute('aria-pressed', String(on)); btn.classList.toggle('is-off', !on); }
    }

    // On a chart without a base series, the only line still showing cannot be hidden.
    function syncKeepOne(chartId) {
        const reg = legends[chartId];
        if (!reg) return;
        const visible = [...reg.entries].filter(([, e]) => e.series.length && e.visible);
        reg.entries.forEach((entry, key) => {
            const btn = legendButton(chartId, key);
            if (!btn) return;
            const last = !reg.base && entry.series.length > 0 && entry.visible && visible.length === 1;
            if (last) { btn.setAttribute('aria-disabled', 'true'); btn.title = 'One line always stays on the chart'; }
            else { btn.removeAttribute('aria-disabled'); btn.removeAttribute('title'); }
        });
    }

    function toggleLegend(chartId, key) {
        const reg = legends[chartId];
        const entry = reg && reg.entries.get(String(key));
        if (!entry) return;
        const btn = legendButton(chartId, key);
        if (btn && btn.getAttribute('aria-disabled') === 'true') return;
        const on = !entry.visible;
        setEntryVisible(chartId, key, on);
        const stored = legendStored(chartId);
        stored[key] = on;
        legendSave(chartId, stored);
        syncKeepOne(chartId);
    }

    function setupLegends() {
        document.addEventListener('click', (e) => {
            const b = e.target.closest('.legend [data-legend-key]');
            if (b) toggleLegend(b.dataset.legendChart, b.dataset.legendKey);
        });
    }

    const MONEY_FORMAT = { type: 'price', precision: 0, minMove: 1 };
    // Account values on the axis: whole euros with thousands separators.
    const MONEY_AXIS = { localization: { priceFormatter: (v) => Math.round(v).toLocaleString('en-US') } };
    const seriesColor = (name) => (name && name.startsWith('series-') ? cssVar(`--${name}`) : name || cssVar('--series-1'));
    // The indices have the same colour in every chart: MSCI World orange, S&P 500 green, Nasdaq 100 amber.
    const BENCH_COLORS = { msci_world: '--cat-2', sp500: '--cat-3', nasdaq100: '--cat-4' };
    const benchColor = (id, i = 0) => cssVar(BENCH_COLORS[id] || ['--cat-6', '--cat-other'][i % 2]);

    // ------------------------------------------------------------ state

    const state = {
        agents: [],
        agentByKey: {},
        config: null,
        draft: null,
        dirty: false,
        regimes: [],
        backtestable: [],
        frequencies: [],
        activeRegime: 'BULL',
        analysis: null,
        analyzing: false,
        flowSelected: 'decision',
        watchlist: [],
        watchData: null,
        watchSort: { key: 'rank', dir: 1 },
        scanning: false,
        scoreBoard: null,
        sbt: null,
        bt: null,
        bk: null,
        bkSearch: null,
        bkShown: new Set(),
        bkGrid: null,
        bkMode: 'compare',
        bkJobMode: null,  // the Compare or search run in flight, so progress shows in its own mode only
        jobs: {},         // job id by tag, so a Stop button can cancel it
        bkUniverse: 'portfolio',
        labOptions: null,
        strategies: {},
        timers: {},
        portfolio: null,
        portfolioLimits: {},
        pfReview: null,
        pfCandidate: null,
        pfImport: null,
        pfSection: null,
        pfAllocDim: 'position',
        pfPerfRange: '1Y',
        pfExposureDim: 'sector',
        pfMoverPeriod: '1Y',
        pfActionFilter: 'ALL',
        pfAccount: null,
        pfDetailTicker: null,
        pfAccountLoading: false,
        pfMonitor: null,
        rec: null,
        recAllLearnings: false,
        recAllHoldings: false,
    };

    // ------------------------------------------------------------ navigation

    // The address bar follows the view. A new view is a history step (Back and Forward move between tabs); a deep link
    // such as #backtest=record/cash or #analyze=KO is kept whole. `routed` is the hash the page last acted on, so a
    // popstate and a hashchange for one move run the route once.
    let routed = location.hash;
    function setHash(hash, { push = false } = {}) {
        if (location.hash !== hash) history[push ? 'pushState' : 'replaceState'](null, '', hash);
        routed = location.hash;
    }

    function showView(name, { push = true, load = true } = {}) {
        const changed = !$(`#view-${name}`)?.classList.contains('active');
        $$('.tab').forEach((t) => {
            const on = t.dataset.view === name;
            t.classList.toggle('active', on);
            t.setAttribute('aria-selected', String(on));
            if (on && t.parentElement.scrollWidth > t.parentElement.clientWidth) t.scrollIntoView({ inline: 'center', block: 'nearest' });
        });
        $$('.view').forEach((v) => {
            const on = v.id === `view-${name}`;
            setIf(v, 'hidden', !on);
            v.classList.toggle('active', on);
        });
        const label = ($(`#tab-${name}`) || {}).textContent || 'FastValue';
        setIf($('#view-title'), 'textContent', label);
        document.title = `${label} · FastValue`;
        if (location.hash.replace('#', '').split('=')[0] !== name) {
            const shown = name === 'analyze' && state.analysis && state.analysis.ticker ? `#analyze=${encodeURIComponent(state.analysis.ticker)}` : `#${name}`;
            setHash(shown, { push: push && changed });
        }
        routed = location.hash;
        if (changed) window.scrollTo({ top: 0, behavior: 'instant' });  // a new tab opens at its top
        if (!load) return;
        if (name === 'weights') renderWeights();
        if (name === 'portfolio' && !state.portfolio) loadPortfolio();
        if (name === 'backtest') { refreshBkOptions(); setBkMode(state.bkMode, { hash: false }); }
    }

    // Acts on a location hash: opens the view and, for a deep link, runs what it names. Called at load, on Back and
    // Forward, and when an in-app <a href="#..."> link is followed.
    function route(hash) {
        const [view, arg] = hash.replace('#', '').split('=');
        const legacy = view === 'score-backtest' || view === 'lab';  // links from before the Backtest tab merged Score Backtest and Strategy Lab
        if (legacy) { showView('backtest', { push: false }); setBkMode(view === 'lab' ? 'classic' : 'timing'); }
        else if (document.getElementById(`view-${view || 'analyze'}`)) showView(view || 'analyze', { push: false });  // no hash is the first tab
        if (view === 'analyze' && arg) {
            const symbol = decodeURIComponent(arg).toUpperCase();
            if (!state.analysis || state.analysis.ticker !== symbol) runAnalyze(symbol);
        }
        if (view === 'watchlist' && arg === 'scan') scanWatchlist();
        if (view === 'backtest' && arg) {
            const [mode, section] = arg.split('/');
            if (mode === 'record' && REC_SECTIONS.includes(section)) state.recSection = section;
            setBkMode(mode, { hash: false });
            if (mode === 'record' && state.rec) setRecSection(state.recSection);
        }
        if (view === 'score-backtest' && arg) { $('#sbt-ticker').value = decodeURIComponent(arg).toUpperCase(); $('#sbt-form').requestSubmit(); }
        // #portfolio=review runs the review; #portfolio=review/risk also opens a section.
        if (view === 'portfolio' && arg && arg.startsWith('review')) {
            const asked = arg.split('/')[1];
            const section = asked === 'holdings' ? 'actions' : asked;  // Holdings is the old Actions section
            if (PF_SECTIONS.includes(section)) state.pfSection = section;
            const reveal = () => { if (asked && !$('#pf-dash').hidden) $('#pf-dash').scrollIntoView({ block: 'start' }); };
            if (state.pfReview) { renderReview(state.pfReview); reveal(); } else reviewPortfolio().then(reveal);
        }
        if (view === 'portfolio' && arg && !arg.startsWith('review')) { $('#pf-candidate-ticker').value = decodeURIComponent(arg).toUpperCase(); assessCandidate(); }
    }

    function setupNav() {
        $$('.tab').forEach((t) => t.addEventListener('click', () => showView(t.dataset.view)));
        document.addEventListener('click', (e) => {
            const go = e.target.closest('[data-goto]');
            if (!go) return;
            e.preventDefault();
            showView(go.dataset.goto);
            window.scrollTo({ top: 0 });
        });
        setupPlaceLinks();
        setupMonitor();
        setupAbout();
        $('.skip-link').addEventListener('click', (e) => {
            e.preventDefault();  // the hash is the route, not an anchor
            $('#main').focus();
        });
        const onNav = () => {
            if (location.hash === routed) return;
            routed = location.hash;
            route(location.hash);
        };
        window.addEventListener('popstate', onNav);
        window.addEventListener('hashchange', onNav);
    }

    // ------------------------------------------------------------ keyboard and screen-reader semantics

    // Tells a screen-reader user what just finished (polite status), without a visual toast.
    function announce(message) {
        const el = $('#sr-status');
        if (!el) return;
        el.textContent = '';
        setTimeout(() => { el.textContent = message; }, 50);  // a changed text is what gets read, so clear first
    }

    // Moves focus to the heading of a finished result (import, review, analysis) and announces it.
    // The heading gets tabindex -1 so it can take focus without becoming a Tab stop.
    function focusResult(target, message) {
        const el = typeof target === 'string' ? $(target) : target;
        if (message) announce(message);
        if (!el) return;
        if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '-1');
        el.focus({ preventScroll: false });
    }

    const setAttr = (el, name, value) => { if (el.getAttribute(name) !== value) el.setAttribute(name, value); };

    // One place gives every segmented control and the main tab bar real semantics, read from the .active class:
    // tablist (tabs, roving tabindex), radiogroup (radio, roving tabindex) or group of toggles (aria-pressed).
    // The role on the container picks the mode; data-seg="toggle" overrides it.
    function syncSeg(group) {
        const buttons = [...group.querySelectorAll(':scope > .seg, :scope > .tab')];
        if (!buttons.length) return;
        const declared = group.dataset.seg || ({ tablist: 'tabs', radiogroup: 'radio' })[group.getAttribute('role')] || 'toggle';
        setAttr(group, 'role', declared === 'tabs' ? 'tablist' : declared === 'radio' ? 'radiogroup' : 'group');
        const roving = declared !== 'toggle';
        const active = buttons.find((b) => b.classList.contains('active')) || (roving ? buttons[0] : null);
        buttons.forEach((b) => {
            const on = b === active && (b.classList.contains('active') || roving);
            const pressed = b.classList.contains('active');
            if (declared === 'tabs') {
                setAttr(b, 'role', 'tab');
                setAttr(b, 'aria-selected', String(pressed));
                b.removeAttribute('aria-pressed'); b.removeAttribute('aria-checked');
            } else if (declared === 'radio') {
                setAttr(b, 'role', 'radio');
                setAttr(b, 'aria-checked', String(pressed));
                b.removeAttribute('aria-pressed'); b.removeAttribute('aria-selected');
            } else {
                if (['tab', 'radio'].includes(b.getAttribute('role'))) b.removeAttribute('role');
                setAttr(b, 'aria-pressed', String(pressed));
                b.removeAttribute('aria-selected'); b.removeAttribute('aria-checked');
            }
            if (roving) setAttr(b, 'tabindex', on ? '0' : '-1'); else b.removeAttribute('tabindex');
        });
    }

    function setupSegs() {
        const SEL = '.segmented, .tabs';
        const syncAll = (root) => {
            if (root.matches && root.matches(SEL)) syncSeg(root);
            if (root.querySelectorAll) root.querySelectorAll(SEL).forEach(syncSeg);
        };
        syncAll(document);
        const observer = new MutationObserver((records) => {
            const groups = new Set();
            for (const m of records) {
                if (m.type === 'attributes') {
                    const g = m.target.closest && m.target.closest(SEL);
                    if (g && m.target.matches('.seg, .tab')) groups.add(g);
                } else {
                    const host = m.target.closest && m.target.closest(SEL);
                    if (host) groups.add(host);
                    m.addedNodes.forEach((n) => { if (n.nodeType === 1 && (n.matches(SEL) || n.querySelector(SEL))) syncAll(n); });
                }
            }
            groups.forEach(syncSeg);
        });
        observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
        // Arrow keys, Home and End move between the tabs or radios of a list and select the one reached.
        document.addEventListener('keydown', (e) => {
            if (e.altKey || e.ctrlKey || e.metaKey || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) return;
            const current = e.target instanceof Element ? e.target.closest('[role=tab], [role=radio]') : null;
            const list = current && current.parentElement;
            if (!list || !['tablist', 'radiogroup'].includes(list.getAttribute('role'))) return;
            const items = [...list.children].filter((b) => b.getAttribute('role') === current.getAttribute('role') && !b.disabled && !b.hidden);
            const vertical = e.key === 'ArrowUp' || e.key === 'ArrowDown';
            if (vertical && list.getAttribute('role') === 'tablist') return;
            const i = items.indexOf(current);
            let next = i;
            if (e.key === 'Home') next = 0;
            else if (e.key === 'End') next = items.length - 1;
            else next = (i + (e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 1) + items.length) % items.length;
            e.preventDefault();
            const target = items[next];
            if (!target || target === current) return;
            const signature = Object.entries(target.dataset).map(([k, v]) => `[data-${k.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}="${CSS.escape(v)}"]`).join('');
            target.click();
            // A click may rebuild the list (Weights roles, regimes): focus the same control in what replaced it.
            const focusTarget = target.isConnected ? target : (signature ? document.querySelector(`[role=${current.getAttribute('role')}]${signature}`) : null);
            if (focusTarget) focusTarget.focus();
        });
    }

    // Tips (.term, .info, .peek): Escape closes the open one (WCAG 1.4.13), and every .term carries its text as a
    // description. The descriptions sit once each in a hidden store, so the term's own text stays what it was.
    function setupTips() {
        const store = document.createElement('div');
        store.id = 'tip-store';
        store.hidden = true;
        document.body.appendChild(store);
        const ids = new Map();
        const describe = (root) => {
            const terms = root.matches && root.matches('.term[data-tip]') ? [root] : [];
            if (root.querySelectorAll) terms.push(...root.querySelectorAll('.term[data-tip]:not([aria-describedby])'));
            terms.forEach((t) => {
                if (t.hasAttribute('aria-describedby') || !t.dataset.tip) return;
                let id = ids.get(t.dataset.tip);
                if (!id) {
                    id = `tip-${ids.size + 1}`;
                    ids.set(t.dataset.tip, id);
                    const span = document.createElement('span');
                    span.id = id;
                    span.textContent = t.dataset.tip;
                    store.appendChild(span);
                }
                t.setAttribute('aria-describedby', id);
            });
        };
        describe(document);
        new MutationObserver((records) => {
            records.forEach((m) => m.addedNodes.forEach((n) => { if (n.nodeType === 1 && !store.contains(n)) describe(n); }));
        }).observe(document.body, { childList: true, subtree: true });
        const OPEN = '.term, .info, .peek';
        document.addEventListener('keydown', (e) => {
            if (e.key !== 'Escape') return;
            const open = new Set($$(':is(.term, .info, .peek):hover'));
            const focused = document.activeElement instanceof Element ? document.activeElement.closest(OPEN) : null;
            if (focused) open.add(focused);
            open.forEach((el) => el.classList.add('tip-closed'));
        });
        // A closed tip opens again the next time the pointer or focus arrives.
        const reopen = (e) => { const el = e.target instanceof Element ? e.target.closest(OPEN) : null; if (el) el.classList.remove('tip-closed'); };
        document.addEventListener('focusin', reopen);
        document.addEventListener('pointerover', (e) => {
            if (!(e.target instanceof Element)) return;
            const el = e.target.closest(OPEN);
            if (el && !(e.relatedTarget instanceof Element && el.contains(e.relatedTarget))) el.classList.remove('tip-closed');
        });
    }

    // ------------------------------------------------------------ market clock + theme

    function updateMarketPill() {
        const parts = Object.fromEntries(
            new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', weekday: 'short', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' })
                .formatToParts(new Date()).map((p) => [p.type, p.value]),
        );
        const minutes = Number(parts.hour) * 60 + Number(parts.minute);
        const open = !['Sat', 'Sun'].includes(parts.weekday) && minutes >= 570 && minutes < 960;
        const pill = $('#market-pill');
        setIf(pill, 'textContent', `${open ? 'Market open' : 'Market closed'} ${parts.hour}:${parts.minute} ET`);
        pill.classList.toggle('open', open);
    }

    const THEME_ICONS = {
        sun: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        moon: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a9 9 0 1 0 9 9 7 7 0 0 1-9-9z" fill="currentColor"/></svg>',
    };
    const currentTheme = () => document.documentElement.dataset.theme
        || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');

    // The toggle names the theme it switches to, and shows that theme's icon: a sun in dark, a moon in light.
    function syncThemeToggle() {
        const btn = $('#theme-toggle');
        if (!btn) return;
        const dark = currentTheme() === 'dark';
        setAttr(btn, 'aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
        setAttr(btn, 'title', dark ? 'Switch to light theme' : 'Switch to dark theme');
        setIf(btn, 'innerHTML', dark ? THEME_ICONS.sun : THEME_ICONS.moon);
    }

    function applyTheme(theme) {
        if (theme) document.documentElement.dataset.theme = theme;
        else delete document.documentElement.dataset.theme;
        syncThemeToggle();
    }

    function setupTheme() {
        applyTheme(storageGet('fastvalue-theme'));
        window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', syncThemeToggle);
        $('#theme-toggle').addEventListener('click', () => {
            const current = currentTheme();
            const next = current === 'light' ? 'dark' : 'light';
            applyTheme(next);
            storageSet('fastvalue-theme', next);
            rerenderCharts();
        });
    }

    function rerenderCharts() {
        if (state.analysis) { renderPriceChart(state.analysis); renderHistory(state.analysis); }
        if (state.sbt) { renderSbtCharts(state.sbt); }
        if (state.bt) { renderBtCharts(state.bt); }
        if (state.bk) { renderBkChart(state.bk); }
        if (state.rec && !$('#rec-results').hidden) { renderRecord(state.rec); }
        if (state.pfReview) { renderPerf(); renderRiskPane(); renderCycle(); }
        if (state.regimeOpen && state.regime && state.regime.data) renderRegimeChart();
    }

    // ============================================================ ANALYZE

    async function runAnalyze(ticker, { silent = false, record = true } = {}) {
        if (state.analyzing) return;
        const symbol = (ticker || $('#analyze-ticker').value).trim().toUpperCase();
        const input = $('#analyze-ticker');
        if (!symbol) { input.focus(); return; }
        input.value = symbol;
        input.removeAttribute('aria-invalid');
        const notice = $('#analyze-notice');
        const results = $('#analyze-results');
        if (!silent && !TICKER_RE.test(symbol)) {
            notice.innerHTML = `<div class="warning"><strong>Not a ticker symbol</strong>Use the exchange symbol, like BRK-B, SHEL.L or SAP.DE: letters, digits, '.', '-', '^' or '=', up to 12 characters.</div>`;
            flagTicker();
            return;
        }
        state.analyzing = true;
        if (!silent) notice.innerHTML = '';
        const btn = $('#analyze-button');
        setBusy(btn, true);
        results.setAttribute('aria-busy', 'true');
        const progress = (sym, extra = '') => {
            notice.innerHTML = `<div class="warning note" id="analyze-progress" role="status"><strong>Analyzing ${esc(sym)}</strong>Reading prices, filings and news. This usually takes 5 to 30 seconds.</div>${extra}`;
        };
        if (!silent) {
            progress(symbol);
            if (!results.hidden) results.classList.add('is-stale');
        }
        // A silent refresh rebuilds the page: keep the open agent drawers and the focused step.
        const open = silent ? $$('#agent-groups details[open]').map((el) => (el.closest('.agent-card') || {}).id).filter(Boolean) : [];
        const active = silent ? document.activeElement : null;
        const focusKey = active && active !== document.body ? (active.id ? `#${active.id}` : active.dataset && active.dataset.node ? `[data-node="${active.dataset.node}"]` : '') : '';
        try {
            let target = symbol;
            if (ISIN_LIKE.test(symbol) && !silent) {
                // An ISIN is resolved to its listing first, so the result is titled by ticker and keyed once.
                let res = { suggestions: [] };
                try { res = await api(`/api/suggest/${encodeURIComponent(symbol)}`); } catch (_) { /* analyze the ISIN itself */ }
                const found = res.suggestions || [];
                if (res.bond || found.length > 1) { suggestTickers(symbol, res); return; }
                if (found.length === 1) {
                    target = found[0].ticker;
                    input.value = target;
                    progress(target, `<div class="notes"><div class="warning note"><strong>Listing</strong>${esc(symbol)} is ${esc((found[0].name || target).replace(/\.$/, ''))}: showing ${esc(target)}.</div></div>`);
                }
            }
            const data = await api('/api/analyze', { method: 'POST', body: { ticker: target, record } });
            state.analysis = data;
            state.analysisStale = false;
            renderAnalysis(data);
            syncWeightsNotes();
            if (silent) {
                open.forEach((id) => { const det = document.querySelector(`#${id} details`); if (det) det.open = true; });
                const again = focusKey && document.querySelector(focusKey);
                if (again && again !== document.activeElement) again.focus({ preventScroll: true });
            }
            if (!silent) {
                if ($('#view-analyze').classList.contains('active')) setHash(`#analyze=${encodeURIComponent(target)}`);  // a reload or a shared link re-runs it
                $('#analyze-results').scrollIntoView({ block: 'start' });
                const rated = (data.owner || {}).applicable !== false && data.decision !== 'NOT_RATED';
                focusResult('#hero .hero-ticker', `${data.ticker} analysed: ${rated ? `owner score ${fmt.signed(data.composite_score, 0)}, ${levelRelation(data.composite_score, data.thresholds).text}` : 'not rated, outside the value framework'}.`);
            }
            if (state.draft) refreshPreview();
        } catch (err) {
            if (/no market data/i.test(err.message)) { suggestTickers(input.value || symbol); if (!silent) flagTicker(); }
            else {
                toast(`Analysis failed: ${err.message}`, 'error');
                if (/invalid ticker/i.test(err.message) && !silent) flagTicker();
            }
        } finally {
            state.analyzing = false;
            results.removeAttribute('aria-busy');
            results.classList.remove('is-stale');
            const p = $('#analyze-progress');
            if (p) p.remove();
            setBusy(btn, false);
        }
    }

    // Put the cursor back on a ticker that did not work, marked invalid until the next keystroke.
    function flagTicker() {
        const input = $('#analyze-ticker');
        input.setAttribute('aria-invalid', 'true');
        input.focus();
        input.select();
    }

    // A ticker Yahoo does not know: say so under the ticker box (visible with or without earlier results), and offer
    // the listings its search finds.
    async function suggestTickers(symbol, found = null) {
        const host = $('#analyze-notice');
        let res = found;
        if (!res) {
            host.innerHTML = `<div class="warning"><strong>No data for ${esc(symbol)}</strong>Looking for listings with that name or ISIN...</div>`;
            res = { suggestions: [] };
            try {
                res = await api(`/api/suggest/${encodeURIComponent(symbol)}`);
            } catch (_) { /* the warning below covers it */ }
        }
        const chips = (res.suggestions || []).map((x) => `<button class="chip" type="button" data-analyze="${esc(x.ticker)}"><span class="mono">${esc(x.ticker)}</span>${x.name || x.exchange ? ` <span class="muted small">${esc([x.name, x.exchange].filter(Boolean).join(' · '))}</span>` : ''}</button>`).join(' ');
        const note = res.note ? `<p class="hint">${esc(res.note)}</p>` : '';
        if (res.bond) {
            host.innerHTML = `<div class="warning note"><strong>No data for ${esc(symbol)}</strong>${esc(res.note)}</div>`;
            return;
        }
        host.innerHTML = found && chips
            ? `<div class="warning note"><strong>${esc(symbol)} has several listings</strong>Choose one to analyze: <span class="chip-row">${chips}</span>${note}</div>`
            : `<div class="warning"><strong>No prices for ${esc(symbol)}</strong>${chips
                ? `Yahoo has no prices under that symbol. Did you mean: <span class="chip-row">${chips}</span>`
                : res.isin ? 'Its search found no listing for that ISIN either.' : 'It may be misspelled, delisted (bought out or bankrupt) or listed under another symbol: add the exchange suffix (SAP.DE, SHEL.L, SHOP.TO) or paste its ISIN.'}${note}</div>`;
    }

    function renderAnalysis(d) {
        $('#analyze-empty').hidden = true;
        $('#analyze-results').hidden = false;
        $('#analyze-refresh-field').hidden = false;
        $('#analyze-button').classList.add('btn-primary');
        renderHero(d);
        if ((d.owner || {}).applicable !== false && d.decision !== 'NOT_RATED') renderHeroStrip(d);
        renderWarnings(d);
        renderContributions(d);
        renderPlan(d);
        renderPriceChart(d);
        renderHistory(d);
        loadCompany(d.ticker);
        renderAgents(d);
        renderFlow(d);
    }

    const TIP_OWNER = 'Owner score: how good the business is to own for 12 months, from -100 to +100. It is the verdict. The entry and exit levels set in the Weights tab turn it into actions in the review and the backtest.';
    const levelsOf = (th) => ({ entry: Number((th || {}).buy), exit: Number((th || {}).sell) });
    // Where one score sits against the two levels, in words and numbers. No verdict wording.
    function levelRelation(score, th) {
        const { entry, exit } = levelsOf(th);
        if (!isNum(score) || !isNum(entry) || !isNum(exit)) return { key: 'none', text: '' };
        if (score >= entry) return { key: 'entry', text: `at or above the entry level (${fmt.signed(entry, 0)})` };
        if (score <= exit) return { key: 'exit', text: `at or below the exit level (${fmt.signed(exit, 0)})` };
        return { key: 'between', text: `between the levels (${fmt.signed(exit, 0)} and ${fmt.signed(entry, 0)})` };
    }
    // ---- score strip: owner scores of several companies on one -100..+100 axis, with the two levels.
    // The company in view gets a fill from 0 and a marker; same-industry peers are labelled dots; every other company
    // analysed is a plain dot. Hover or focus a dot to read it, click it to open that company.
    const STRIP_TIP = 'Every dot is a company you analysed in the last 90 days, at its latest owner score recomposed on your current weights, so the scores compare. Labelled dots are in the same industry (the Competitive Position agent\'s peers). Select a dot to open that company.';
    const stripPos = (v) => ((Math.max(-100, Math.min(100, v)) + 100) / 2).toFixed(2);
    function scoreStripHtml({ self = null, items = [], thresholds, caption = '', label = 'Owner scores' }) {
        const { entry, exit } = levelsOf(thresholds);
        const fill = self && isNum(self.score)
            ? `<span class="strip-fill ${self.score >= 0 ? 'pos' : 'neg'}" style="left:${stripPos(Math.min(0, self.score))}%;width:${(Math.min(100, Math.abs(self.score)) / 2).toFixed(2)}%"></span>` : '';
        const dots = items.filter((o) => isNum(o.score)).map((o) => `<button type="button" class="strip-dot ${o.kind || 'other'}" style="left:${stripPos(o.score)}%"
            data-analyze="${esc(o.ticker)}" data-strip-note="${esc(o.note || '')}" aria-label="${esc(o.note || o.ticker)}"></button>`).join('');
        const labelled = [...(self ? [{ ...self, kind: 'self' }] : []), ...items.filter((o) => o.label && isNum(o.score))];
        const labels = labelled.map((o) => `<span class="strip-label ${o.kind || 'other'}" style="left:${stripPos(o.score)}%">${esc(o.ticker)}</span>`).join('');
        const marker = self && isNum(self.score) ? `<span class="strip-marker" style="left:${stripPos(self.score)}%" aria-hidden="true"></span>` : '';
        return `<div class="strip" role="group" aria-label="${esc(label)} from -100 to +100, exit level ${esc(fmt.signed(exit, 0))}, entry level ${esc(fmt.signed(entry, 0))}">
                <div class="strip-labels" aria-hidden="true">${labels}</div>
                <div class="strip-axis">
                    <div class="strip-track"></div><span class="strip-zero"></span>
                    ${isNum(exit) ? `<span class="strip-level" style="left:${stripPos(exit)}%"></span>` : ''}${isNum(entry) ? `<span class="strip-level" style="left:${stripPos(entry)}%"></span>` : ''}
                    ${fill}${dots}${marker}
                </div>
                <div class="scale-labels pinned"><span class="at-start">-100</span>${isNum(exit) ? `<span class="before" style="left:${stripPos(exit)}%">Exit ${esc(fmt.signed(exit, 0))}</span>` : ''}${isNum(entry) ? `<span class="after" style="left:${stripPos(entry)}%">Entry ${esc(fmt.signed(entry, 0))}</span>` : ''}<span class="at-end">+100</span></div>
                <div class="strip-readout small" aria-live="polite" data-default="${esc(caption)}">${esc(caption)}</div>
            </div>`;
    }
    // Laid out once the strip is in the page, to measure it. Dots that would overlap stack above and below the track
    // (a beeswarm), so a crowded part of the scale reads as a pile instead of one blot. Labels sit in rows above the
    // axis: a label that would overlap another moves up a row (three rows at most; beyond that the dot's text says it).
    const STRIP_STEP = 24, STRIP_DOT = 24, STRIP_MAX_LEVEL = 4;   // a dot is a 24px tap target (WCAG 2.5.8)
    function layoutStrips(root = document) {
        root.querySelectorAll('.strip').forEach((strip) => {
            const axis = strip.querySelector('.strip-axis');
            const width = axis.clientWidth;
            if (!width) return;
            const placed = new Map();   // level -> x positions
            let top = 0, bottom = 0;
            [...axis.querySelectorAll('.strip-dot')].map((el) => ({ el, x: parseFloat(el.style.left) / 100 * width }))
                .sort((a, b) => a.x - b.x).forEach((d) => {
                    const order = [0];
                    for (let k = 1; k <= STRIP_MAX_LEVEL; k += 1) order.push(-k, k);
                    const level = order.find((lv) => (placed.get(lv) || []).every((x) => Math.abs(x - d.x) >= STRIP_DOT)) ?? 0;
                    placed.set(level, [...(placed.get(level) || []), d.x]);
                    d.el.style.setProperty('--level', level);
                    top = Math.min(top, level); bottom = Math.max(bottom, level);
                });
            const mid = 12 + (-top) * STRIP_STEP;
            axis.style.setProperty('--mid', `${mid}px`);
            axis.style.height = `${mid + 12 + bottom * STRIP_STEP}px`;

            const host = strip.querySelector('.strip-labels');
            const labels = [...host.children].map((el) => ({ el, x: parseFloat(el.dataset.x ?? el.style.left) / 100 * width, w: el.offsetWidth }))
                .sort((a, b) => (a.el.classList.contains('self') ? -1 : b.el.classList.contains('self') ? 1 : a.x - b.x));
            const rows = [];
            labels.forEach((l) => {
                l.el.dataset.x = l.el.dataset.x ?? parseFloat(l.el.style.left);
                const left = Math.max(0, Math.min(width - l.w, l.x - l.w / 2));
                let row = rows.findIndex((spans) => spans.every(([a, b]) => left + l.w + 6 <= a || left >= b + 6));
                if (row < 0 && rows.length < 3) { rows.push([]); row = rows.length - 1; }
                l.el.hidden = row < 0;
                if (row < 0) return;
                rows[row].push([left, left + l.w]);
                l.el.style.transform = 'none';
                l.el.style.left = `${left}px`;
                l.el.style.bottom = `${row * 16}px`;
            });
            host.style.height = `${Math.max(1, rows.length) * 16}px`;
            // Dots whose label did not fit are named in the readout, so no company is only a dot.
            const readout = strip.querySelector('.strip-readout');
            if (readout) {
                const base = readout.dataset.base ?? (readout.dataset.base = readout.dataset.default);
                const hidden = labels.filter((l) => l.el.hidden).map((l) => l.el.textContent);
                const text = hidden.length ? `${base} Not labelled: ${hidden.join(', ')}.` : base;
                if (readout.dataset.default !== text) {
                    const atRest = readout.textContent === readout.dataset.default;
                    readout.dataset.default = text;
                    if (atRest) readout.textContent = text;
                }
            }
        });
    }
    function setupStrips() {
        const readout = (dot, on) => {
            const box = dot.closest('.strip')?.querySelector('.strip-readout');
            if (box) box.textContent = on ? dot.dataset.stripNote : box.dataset.default;
        };
        const touch = window.matchMedia('(hover: none)');
        ['pointerover', 'focusin'].forEach((ev) => document.addEventListener(ev, (e) => { const d = e.target instanceof Element && e.target.closest('.strip-dot'); if (d) readout(d, true); }));
        ['pointerout', 'focusout'].forEach((ev) => document.addEventListener(ev, (e) => { const d = e.target instanceof Element && e.target.closest('.strip-dot'); if (d) readout(d, false); }));
        document.addEventListener('click', (e) => {
            const d = e.target instanceof Element && e.target.closest('.strip-dot, .strip-readout .chip, .strip-peers .chip');
            if (!d) return;
            // On touch the first tap on a dot reads it (with a button to open it); a second tap, or the button, opens the company.
            if (d.matches('.strip-dot') && touch.matches && d.getAttribute('aria-pressed') !== 'true') {
                const strip = d.closest('.strip');
                strip.querySelectorAll('.strip-dot[aria-pressed]').forEach((x) => x.removeAttribute('aria-pressed'));
                d.setAttribute('aria-pressed', 'true');
                const box = strip.querySelector('.strip-readout');
                if (box) box.innerHTML = `${esc(d.dataset.stripNote)} <button type="button" class="chip" data-analyze="${esc(d.dataset.analyze)}">Open ${esc(d.dataset.analyze)}</button>`;
                return;
            }
            showView('analyze');
            runAnalyze(d.dataset.analyze);
        });
        let t = null;
        window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(() => layoutStrips(), 120); });
    }
    // The latest score of every company analysed (GET /api/scores), fetched again after each new analysis.
    async function loadScoreBoard(force = false) {
        if (!force && state.scoreBoard && Date.now() - state.scoreBoard.at < 60000) return state.scoreBoard.data;
        const data = await api('/api/scores');
        state.scoreBoard = { at: Date.now(), data };
        return data;
    }
    const ordinal = (n) => `${n}${(n % 100 >= 11 && n % 100 <= 13) ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' }[n % 10] || 'th')}`;
    function boardNote(r, extra) {
        const flags = [extra, r.held ? 'in your portfolio' : '', r.watched ? 'on your watchlist' : ''].filter(Boolean).join(', ');
        return `${r.ticker} ${fmt.signed(r.score, 1)}${flags ? `, ${flags}` : ''} (analysed ${fmtDay(r.analysed)})`;
    }
    // The hero's phone layout (points, score details drawer) is chosen when it renders; redraw it when the width crosses 640px.
    window.matchMedia('(max-width: 640px)').addEventListener('change', () => {
        const d = state.analysis;
        if (!d || !$('#hero').children.length) return;
        renderHero(d);
        if ((d.owner || {}).applicable !== false && d.decision !== 'NOT_RATED') renderHeroStrip(d);
        renderWarnings(d);
    });
    // The hero's strip: this company against the others analysed, its rank, and its industry peers.
    async function renderHeroStrip(d) {
        const host = $('#hero-strip');
        if (!host) return;
        const self = { ticker: d.ticker, score: d.composite_score };
        const draw = (items, caption, after = '') => {
            host.innerHTML = scoreStripHtml({ self, items, thresholds: d.thresholds, caption, label: `Owner score of ${d.ticker} against the other companies analysed` }) + after;
            layoutStrips(host);
        };
        draw([], '');
        let board;
        try { board = await loadScoreBoard(true); } catch (_) { return; }
        if (state.analysis !== d) return;   // another analysis started meanwhile
        const comp = d.agent_assessments.find((a) => a.agent_key === 'competitive');
        const peerList = ((comp && comp.raw_data && comp.raw_data.peers) || []).map((p) => String(p.ticker).toUpperCase());
        const peers = new Set(peerList);
        const others = (board.rows || []).filter((r) => r.ticker !== d.ticker);
        const items = others.map((r) => ({ ticker: r.ticker, score: r.score, kind: peers.has(r.ticker) ? 'peer' : 'other', label: peers.has(r.ticker), note: boardNote(r, peers.has(r.ticker) ? 'same industry' : '') }));
        const rank = 1 + others.filter((r) => r.score > d.composite_score).length;
        const caption = others.length
            ? `${ordinal(rank)} of ${plural(others.length + 1, 'company', 'companies')} you analysed in the last ${board.max_age_days} days, on your current weights.`
            : 'No other companies analysed in the last 90 days yet: analyse a few to compare them here.';
        const scored = others.filter((r) => peers.has(r.ticker));   // highest first, as the board is sorted
        const missing = peerList.filter((t) => t !== d.ticker && !scored.some((r) => r.ticker === t));
        const link = (t, text) => `<button type="button" class="chip" data-analyze="${esc(t)}">${text}</button>`;
        const peerLine = peerList.length ? `<p class="strip-peers small"><span class="muted">Same industry:</span> ${scored.length
            ? scored.map((r) => `${link(r.ticker, esc(r.ticker))} <span class="${r.score >= 0 ? 'pos' : 'neg'}">${esc(fmt.signed(r.score, 1))}</span>`).join(' ')
            : '<span class="muted">none analysed yet</span>'}${missing.length ? `<br><span class="muted">Not analysed yet:</span> ${missing.map((t) => link(t, esc(t))).join(' ')}` : ''}</p>` : '';
        draw(items, caption, peerLine);
    }

    // The scored rules behind one layer's score: rule, points and agent, largest first. Points are on each agent's own scale.
    function layerRules(d, layer) {
        const { contribByKey } = agentLayers(d);
        const out = [];
        d.agent_assessments.forEach((a) => {
            const c = contribByKey[a.agent_key] || {};
            if (c.layer !== layer || !c.included || !Array.isArray((a.raw_data || {}).score_breakdown)) return;
            a.raw_data.score_breakdown.forEach((b) => { if (isNum(b.points) && b.points !== 0) out.push({ factor: b.factor, points: b.points, agent: shortName(a.agent_name) }); });
        });
        return out.sort((x, y) => Math.abs(y.points) - Math.abs(x.points));
    }
    const pointsRows = (list) => `<table class="breakdown"><tbody>${list.map((r) => `<tr><td>${esc(r.factor)} <span class="muted small">${esc(r.agent)}</span></td><td class="pts ${r.points > 0 ? 'pos' : 'neg'}">${esc(fmt.signed(r.points, 1))}</td></tr>`).join('')}</tbody></table>`;
    // The entry line and the "Entry" price in the plan only make sense when the score is at or above the entry level and the margin of safety allows new money.
    const atEntry = (d) => (d.owner || {}).applicable !== false && d.decision !== 'NOT_RATED' && isNum(d.composite_score)
        && d.composite_score >= Number((d.thresholds || {}).buy) && !(d.owner || {}).capped;
    function capText(d) {
        const fund = d.agent_assessments.find((a) => a.agent_key === 'fundamentals');
        const evf = fund && fund.data_quality !== 'UNAVAILABLE' && fund.raw_data ? fund.raw_data.ev_to_fcf : null;
        const cap = (d.owner || {}).max_ev_to_fcf;
        return isNum(evf) && isNum(cap) ? `EV/FCF ${fmt.num(evf, 0)}x is at or above the ${fmt.num(cap, 0)}x cap: no new money (margin of safety).` : 'Margin of safety: no new money at this price.';
    }

    function freshnessBadge(f) {
        if (!f) return '';
        const cls = f.status === 'LIVE' ? 'live' : f.status === 'STALE' ? 'unavailable' : 'accent';
        const label = f.status === 'LIVE' ? 'Live intraday' : f.status === 'STALE' ? `Stale: ${plural(f.sessions_behind, 'session', 'sessions')} behind` : 'Last close';
        return `<span class="badge term quiet ${cls}" tabindex="0" data-tip="${esc(f.note)}">${esc(label)} ${esc(f.as_of)}</span>`;
    }

    function renderHero(d) {
        const phone = window.matchMedia('(max-width: 640px)').matches;
        const TOP_HERO_POINTS = phone ? 3 : 5;
        const candles = d.chart?.candles || [];
        const prev = candles.length > 1 ? candles[candles.length - 2].close : null;
        const change = prev ? d.current_price / prev - 1 : null;
        const o = d.owner || {};
        const t = d.timing || {};
        const rated = o.applicable !== false && d.decision !== 'NOT_RATED';
        const ownerUsed = d.contributions.filter((c) => c.included && c.layer === 'owner').length;
        const timingUsed = d.contributions.filter((c) => c.included && c.layer !== 'owner').length;
        const ownerPts = rated ? layerRules(d, 'owner') : [];
        const timingPts = layerRules(d, 'timing');
        const statsHtml = `<div class="stat-row hero-stats">
                ${rated ? `<div class="stat"><div class="stat-label">Owner confidence</div><div class="stat-value">${esc(fmt.num(d.overall_confidence, 0))}%</div></div>` : ''}
                <div class="stat"><div class="stat-label">Setup risk</div><div class="stat-value">${esc(titleCase(d.risk_rating || ''))}, ${esc(fmt.num(d.risk_score, 1))}/10</div></div>
                ${rated ? `<div class="stat"><div class="stat-label">Owner agents</div><div class="stat-value">${ownerUsed} with data</div></div>` : ''}
                <div class="stat"><div class="stat-label">Timing agents</div><div class="stat-value">${timingUsed} with data</div></div>
            </div>`;
        const compareHtml = rated ? `<div class="hero-compare">
                <div class="layer-label"><span class="term" tabindex="0" data-tip="${esc(STRIP_TIP)}">Against the companies you analysed</span> <span class="muted">owner scores on your current weights</span></div>
                <div id="hero-strip"></div>
            </div>` : '';
        $('#hero').innerHTML = `
            <div>
                <div class="hero-ticker">${esc(d.ticker)}</div>
                <div class="hero-price">${esc(fmt.money(d.current_price))} <span class="${change >= 0 ? 'pos' : 'neg'}">${esc(fmt.spct(change, 2))}</span></div>
                <div class="hero-meta">
                    ${freshnessBadge(d.freshness)}
                    <span class="muted small" id="analyze-updated">Updated ${esc(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false }))}</span>
                    <button type="button" class="badge badge-button" data-regime-open aria-expanded="${state.regimeOpen ? 'true' : 'false'}" aria-controls="regime-card">Market regime: ${esc(rgName(d.market_regime))}<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg></button>
                </div>
            </div>
            <div class="hero-score">
                <div class="layer-label"><span class="term" tabindex="0" data-tip="${esc(TIP_OWNER)}">Owner score</span> <span class="muted">the verdict, horizon ${esc(d.horizon || '12 months')}</span></div>
                <div class="hero-score-top">
                    ${rated
                        ? `<span class="hero-score-value ${d.composite_score >= 0 ? 'pos' : 'neg'}">${esc(fmt.signed(d.composite_score, 1))}</span><span class="muted small">/ 100</span>`
                        : '<span class="decision decision-not-rated">Not rated</span><span class="muted small">outside the value framework: no company fundamentals</span>'}
                </div>
                ${rated ? `<div class="small">${esc(capFirst(levelRelation(d.composite_score, d.thresholds).text))}${phone && isNum(d.overall_confidence) ? ` · ${esc(fmt.num(d.overall_confidence, 0))}% confidence` : ''}</div>` : ''}
                <div class="warnings" id="analyze-warnings"></div>
                ${ownerPts.length ? `<div class="layer-label">Points behind the score <span class="muted">largest first</span></div>${pointsRows(ownerPts.slice(0, TOP_HERO_POINTS))}
                <div class="hint">Points are on each agent's own scale; the score blends them by weight and confidence.${ownerPts.length > TOP_HERO_POINTS ? ` ${ownerPts.length - TOP_HERO_POINTS} more in the flow.` : ''} <button type="button" class="btn btn-ghost btn-sm" data-flow-node="owner">All points and how they combine</button></div>` : ''}
                <div class="layer-label"><span class="term" tabindex="0" data-tip="${esc(TIP_TIMING)}">Entry timing</span> <span class="muted">no vote: when and how much</span></div>
                <div class="hero-score-top">
                    <span class="mono ${(t.score || 0) >= 0 ? 'pos' : 'neg'}">${esc(fmt.signed(t.score || 0, 0))}</span>
                    <span class="muted small">trend ${esc(String(t.trend || 'n/a').toLowerCase())}${t.setup ? `, ${esc(setupLabel(t.setup))}` : ''}${isNum(t.momentum_12_1) ? `, 12-1m momentum ${esc(fmt.spct(t.momentum_12_1, 0))}` : ''}</span>
                </div>
                ${timingPts.length ? pointsRows(timingPts.slice(0, phone ? 2 : 3)) : ''}
            </div>
            ${phone ? `<details class="drawer hero-drawer"><summary>Score details</summary><div class="drawer-body">${statsHtml}${compareHtml}</div></details>` : `${statsHtml}${compareHtml}`}`;
        const drawer = $('#hero .hero-drawer');
        if (drawer) drawer.addEventListener('toggle', () => { if (drawer.open) layoutStrips($('#hero-strip')); });
    }

    function renderWarnings(d) {
        const notes = ((d.timing || {}).notes || []).slice(1); // the first note only restates that the panel has no vote
        $('#analyze-warnings').innerHTML = (d.warnings || [])
            .map((w) => `<div class="warning"><strong>${/^Timing agents disagree/.test(w) ? 'Timing' : 'Check'}</strong>${esc(w)}</div>`)
            .join('') + notes.map((n) => `<div class="warning note"><strong>Timing</strong>${esc(n)}</div>`).join('');
    }

    function contribRow(c, maxAbs) {
        if (!c.included) {
            return `<div class="contrib-row excluded">
                <span class="contrib-name">${esc(shortName(c.agent_name))}</span>
                <span class="contrib-bar"><span class="contrib-reason">${esc(c.exclusion_reason)}</span></span>
                <span class="contrib-value">n/a</span>
            </div>`;
        }
        const width = (Math.abs(c.contribution) / maxAbs) * 50;
        const left = c.contribution >= 0 ? 50 : 50 - width;
        return `<div class="contrib-row term quiet" tabindex="0" data-tip="Raw ${esc(c.raw_score)}, normalized ${esc(c.normalized_score)}, confidence ${esc(c.confidence)}%">
            <span class="contrib-name">${esc(shortName(c.agent_name))}<small>${esc(fmt.pct(c.weight_share, 0))}</small></span>
            <span class="contrib-bar"><span class="contrib-fill ${c.contribution >= 0 ? 'pos' : 'neg'}" style="left:${left}%;width:${width}%"></span></span>
            <span class="contrib-value ${c.contribution >= 0 ? 'pos' : 'neg'}">${esc(fmt.signed(c.contribution, 1))}</span>
        </div>`;
    }

    function renderContributions(d) {
        const sortRows = (rows) => [...rows].sort(byAgentOrder((c) => c.agent_key));
        const owner = sortRows(d.contributions.filter((c) => c.layer === 'owner'));
        const timing = sortRows(d.contributions.filter((c) => c.layer === 'timing' || !c.layer));
        const maxAbs = Math.max(10, ...d.contributions.filter((c) => c.included).map((c) => Math.abs(c.contribution)));
        $('#contrib-caption').textContent = 'Points each agent added to its layer (share of weight after confidence)';
        const group = (title, sub, rows, total) => `<div class="layer-head"><strong>${esc(title)}</strong><span class="muted small">${esc(sub)}</span></div>`
            + rows.map((c) => contribRow(c, maxAbs)).join('')
            + `<div class="contrib-row contrib-total"><span>${esc(title)}</span><span></span><span class="contrib-value ${total >= 0 ? 'pos' : 'neg'}">${esc(fmt.signed(total, 1))}</span></div>`;
        const groups = {
            owner: () => group('Owner score', 'the verdict', owner, d.composite_score),
            timing: () => group('Entry timing', 'no vote', timing, (d.timing || {}).score || 0),
        };
        $('#contrib-chart').innerHTML = layerOrder().filter((l) => groups[l]).map((l) => groups[l]()).join('');
    }

    function renderPlan(d) {
        $('#plan-method').textContent = d.stop_loss_method;
        const isBuy = atEntry(d);
        $('#plan-grid').innerHTML = `
            <div class="stat"><div class="stat-label">${isBuy ? 'Entry' : 'Price'}</div><div class="stat-value">${esc(fmt.money(d.suggested_entry))}</div></div>
            <div class="stat stop"><div class="stat-label">Stop reference</div><div class="stat-value">${esc(fmt.money(d.suggested_stop_loss))}</div></div>
            <div class="stat target"><div class="stat-label">Target</div><div class="stat-value">${esc(fmt.money(d.suggested_target_price))}</div></div>
            <div class="stat"><div class="stat-label">Reward / risk</div><div class="stat-value">${esc(fmt.num(d.reward_risk_ratio, 2))}x</div></div>
            <div class="stat"><div class="stat-label">ATR 14</div><div class="stat-value">${esc(fmt.money(d.atr_14))}</div></div>
            <div class="stat"><div class="stat-label">Stop distance</div><div class="stat-value">${esc(fmt.pct(Math.abs(d.suggested_entry - d.suggested_stop_loss) / d.suggested_entry, 1))}</div></div>`;

        const price = d.current_price;
        const levels = [...(d.key_levels || []), { label: 'Current price', price, kind: 'current', source: '', distance_pct: 0 }]
            .sort((a, b) => b.price - a.price);
        $('#levels-list').innerHTML = levels.length > 1
            ? levels.map((l) => `<div class="level-row ${l.kind === 'current' ? 'current' : ''}">
                    <span>${l.kind !== 'current' ? `<span class="level-kind ${esc(l.kind)}"></span>` : ''}${esc(l.label)} <span class="level-source">${esc((l.source || '').replace(' Agent', ''))}</span></span>
                    <span class="mono">${esc(fmt.money(l.price))}</span>
                    <span class="mono ${l.distance_pct >= 0 ? 'pos' : 'neg'}">${l.kind === 'current' ? '' : esc(fmt.signed(l.distance_pct, 1)) + '%'}</span>
                </div>`).join('')
            : '<p class="muted small">No agent reported price levels.</p>';
    }

    function renderPriceChart(d) {
        const cs = d.chart.candles || [];
        const chart = makeChart('analyze-price-chart', {}, cs.length ? `${d.ticker} daily price, last ${cs.length} sessions, from ${fmt.money(cs[0].close)} to ${fmt.money(cs[cs.length - 1].close)}; lines: SMA 50, SMA 200, with volume bars.` : `${d.ticker} daily price`);
        const up = cssVar('--pos'), down = cssVar('--neg');
        const candles = chart.addCandlestickSeries({ upColor: up, downColor: down, borderUpColor: up, borderDownColor: down, wickUpColor: up, wickDownColor: down });
        candles.setData(d.chart.candles);

        const vol = chart.addHistogramSeries({ priceScaleId: 'vol', priceFormat: { type: 'volume' }, color: cssVar('--border-strong') });
        chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
        vol.setData(d.chart.volume);

        const s50 = chart.addLineSeries({ color: cssVar('--series-1'), lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
        s50.setData(d.chart.sma50);
        const s200 = chart.addLineSeries({ color: cssVar('--series-2'), lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
        s200.setData(d.chart.sma200);
        // PB EMA channel (D-48): 200-period EMAs of the closes and (dashed) of the highs. Off until shown in the legend.
        const pbColor = cssVar('--series-7');
        const pbLines = [['pb_ema_close', 0], ['pb_ema_high', 2]].filter(([k]) => (d.chart[k] || []).length).map(([k, style]) => {
            const s = chart.addLineSeries({ color: pbColor, lineWidth: 1, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
            s.setData(d.chart[k]);
            return s;
        });

        const line = (price, color, title, style = 0) => candles.createPriceLine({ price, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title });
        if (atEntry(d)) line(d.suggested_entry, cssVar('--accent'), 'Entry');
        const stopLine = line(d.suggested_stop_loss, down, 'Stop', 2);
        const targetLine = line(d.suggested_target_price, up, 'Target', 2);

        // Nearest agent levels only, to keep the chart readable.
        const agentLines = (d.key_levels || [])
            .filter((l) => Math.abs(l.distance_pct) <= 15)
            .sort((a, b) => Math.abs(a.distance_pct) - Math.abs(b.distance_pct))
            .slice(0, 4)
            .map((l) => candles.createPriceLine({
                price: l.price, color: l.kind === 'support' ? up : l.kind === 'resistance' ? down : cssVar('--series-3'),
                lineWidth: 1, lineStyle: 3, axisLabelVisible: false, title: l.label,
            }));

        chart.timeScale().fitContent();
        $('#price-chart-title').textContent = `${d.ticker} daily, last ${d.chart.candles.length} sessions`;
        legend('price-legend', [
            { key: 'sma50', label: 'SMA 50', color: cssVar('--series-1'), series: s50 },
            { key: 'sma200', label: 'SMA 200', color: cssVar('--series-2'), series: s200 },
            ...(pbLines.length ? [{ key: 'pb_ema', label: 'PB EMA 200', color: pbColor, series: pbLines, hidden: true }] : []),
            { key: 'stop', label: 'Stop', color: down, lines: [stopLine] },
            { key: 'target', label: 'Target', color: up, lines: [targetLine] },
            ...(agentLines.length ? [{ key: 'levels', label: 'Agent levels', color: cssVar('--series-3'), lines: agentLines }] : []),
        ], { chart: 'analyze-price-chart', base: true });
    }

    function renderHistory(d) {
        const hist = (d.history || []).filter((h) => h.decision !== 'NOT_RATED').map((h) => ({ time: Math.floor(Date.parse(h.generated_at) / 1000), value: h.composite_score }))
            .filter((p) => Number.isFinite(p.time));
        const points = [];
        hist.forEach((p) => { if (!points.length || p.time > points[points.length - 1].time) points.push(p); });
        const card = $('#history-card');
        if (points.length < 2) { card.hidden = true; destroyChart('history-chart'); return; }
        card.hidden = false;
        const chart = makeChart('history-chart', { timeScale: { borderVisible: false, timeVisible: true } },
            `${d.ticker} owner score over ${points.length} analyses, from ${fmt.signed(points[0].value, 1)} to ${fmt.signed(points[points.length - 1].value, 1)}; entry level ${fmt.signed(d.thresholds.buy, 0)}, exit level ${fmt.signed(d.thresholds.sell, 0)}.`);
        const series = chart.addBaselineSeries({
            baseValue: { type: 'price', price: 0 },
            priceFormat: { type: 'price', precision: 0, minMove: 1 },
            topLineColor: cssVar('--pos'), bottomLineColor: cssVar('--neg'),
            topFillColor1: 'transparent', topFillColor2: 'transparent', bottomFillColor1: 'transparent', bottomFillColor2: 'transparent',
            lineWidth: 2,
        });
        series.applyOptions({
            autoscaleInfoProvider: (original) => {
                const r = original();
                if (!r || !r.priceRange) return r;
                return { ...r, priceRange: { minValue: Math.min(r.priceRange.minValue - 5, d.thresholds.sell - 5), maxValue: Math.max(r.priceRange.maxValue + 5, d.thresholds.buy + 5) } };
            },
        });
        series.setData(points);
        const th = d.thresholds;
        [['buy', 'Entry level'], ['sell', 'Exit level']].forEach(([k, title]) =>
            series.createPriceLine({ price: th[k], color: cssVar('--text-3'), lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title }));
        chart.timeScale().fitContent();
    }

    const STATUS_CLASS = { Positive: 'live', Negative: 'unavailable', Neutral: '', 'Check manually': 'accent' };

    // A scrolling table is a named region the keyboard can reach. Without a label here, setupScrollRegions names it
    // from the heading above it.
    const regionAttrs = (label) => (label ? ` tabindex="0" role="region" aria-label="${esc(label)}"` : '');
    function tableHtml(columns, rows, { compact = true, wrapCols = [], label = '' } = {}) {
        const head = columns.map((c) => `<th>${esc(c)}</th>`).join('');
        const body = rows.map((r) => `<tr>${r.map((cell, i) => {
            const url = safeUrl(cell);
            const content = url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">Open</a>` : esc(cell ?? '');
            const status = typeof cell === 'string' && Object.hasOwn(STATUS_CLASS, cell) ? STATUS_CLASS[cell] : null;
            return `<td class="${wrapCols.includes(i) ? 'wrap' : ''}">${status ? `<span class="badge ${status}">${content}</span>` : content}</td>`;
        }).join('')}</tr>`).join('');
        return `<div class="table-wrap table-scroll"${regionAttrs(label)}><table class="table ${compact ? 'compact' : ''}"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    }

    function agentLinks(raw) {
        const links = Array.isArray(raw.links) ? [...raw.links] : [];
        const seen = new Set(links.map((l) => l.url));
        [['source_url', raw.data_source_label || 'Source'], ['alt_source_url', 'Secondary source'], ['sec_url', 'SEC EDGAR']]
            .forEach(([k, label]) => { if (raw[k] && !seen.has(raw[k])) { links.push({ label, url: raw[k] }); seen.add(raw[k]); } });
        return links
            .map((l) => ({ ...l, url: safeUrl(l.url) }))
            .filter((l) => l.url)
            .map((l) => `<a href="${esc(l.url)}" target="_blank" rel="noopener noreferrer">${esc(l.label)}</a>`)
            .join('');
    }

    // Layers and agents in the user's order (Weights tab: roles and order). The
    // layer of each agent comes from the analysis itself (its contribution), so
    // an old analysis keeps the roles it was scored with.
    const LAYERS = {
        owner: { title: 'Owner agents', short: 'Decides', sub: 'Make up the owner score, the verdict on a 12-month view.' },
        timing: { title: 'Timing agents', short: 'Timing', sub: 'No vote. They inform when to enter, how to stage the entry and how much to buy.' },
        info: { title: 'Informative agents', short: 'Informative', sub: 'Shown with their evidence, never scored.' },
    };
    const LAYER_KEYS = ['owner', 'timing', 'info'];
    const TRADING_WARNING = 'In the 2011-2026 S&P 500 test, letting trading signals vote made 12.2% a year against 15.6% for the owner score alone.';
    const layerOrder = (cfg = state.config) => ((cfg && cfg.layer_order) || LAYER_KEYS);
    const roleOf = (key, cfg = state.config) => ((cfg && cfg.roles) || {})[key] || (state.agentByKey[key] || {}).role || 'timing';
    const agentRank = (key, cfg = state.config) => {
        const i = ((cfg && cfg.agent_order) || []).indexOf(key);
        return i < 0 ? Number.MAX_SAFE_INTEGER : i;
    };
    const byAgentOrder = (keyOf, cfg = state.config) => (x, y) => agentRank(keyOf(x), cfg) - agentRank(keyOf(y), cfg);

    function agentLayers(d) {
        const contribByKey = Object.fromEntries(d.contributions.map((c) => [c.agent_key, c]));
        const groups = { owner: [], timing: [], info: [] };
        d.agent_assessments.forEach((a) => {
            const layer = contribByKey[a.agent_key] ? contribByKey[a.agent_key].layer : roleOf(a.agent_key);
            (groups[layer] || groups.timing).push(a);
        });
        LAYER_KEYS.forEach((l) => groups[l].sort(byAgentOrder((a) => a.agent_key)));
        return { ...groups, contribByKey };
    }

    function renderAgents(d) {
        const layers = agentLayers(d);
        $('#agent-groups').innerHTML = layerOrder().filter((l) => layers[l].length).map((l) => `
            <div class="agent-group">
                <div class="section-head"><h3>${esc(LAYERS[l].title)}</h3><span class="muted small">${esc(LAYERS[l].sub)}</span></div>
                <div class="agent-grid">${layers[l].map((a) => agentCardHtml(a, layers.contribByKey[a.agent_key] || {})).join('')}</div>
            </div>`).join('');
    }

    const TOP_RULES = 3;
    const FILING_NOTES_TIP = 'Quoted from the latest 10-K annual report. Evidence only: these notes carry no points, because they could not be tested on past filings.';

    // Compact card: what the agent added, its top rules, and everything else behind one toggle.
    function agentCardHtml(a, c) {
        const raw = a.raw_data || {};
        const meta = state.agentByKey[a.agent_key] || {};
        const quality = String(a.data_quality || 'LIVE').toLowerCase();
        const available = quality !== 'unavailable';
        const layerName = { owner: 'owner score', timing: 'entry timing' }[c.layer];

        const rules = Array.isArray(raw.score_breakdown) ? [...raw.score_breakdown].sort((x, y) => Math.abs(y.points) - Math.abs(x.points)) : [];
        const top = rules.filter((r) => r.points !== 0).slice(0, TOP_RULES);
        const rest = rules.filter((r) => !top.includes(r));
        const ruleTable = (list) => `<table class="breakdown"><tbody>${list.map((b) => `<tr class="${b.points === 0 ? 'zero' : ''}"><td>${esc(b.factor)}</td><td class="pts ${b.points > 0 ? 'pos' : b.points < 0 ? 'neg' : ''}">${esc(fmt.signed(b.points, 1))}</td></tr>`).join('')}</tbody></table>`;
        const ruleTexts = new Set(rules.map((b) => b.factor));
        // Key factors often restate a scored rule with its points appended, e.g. "ROIC 20% (+8)".
        const extraFactors = (a.key_factors || []).filter((f) => !ruleTexts.has(String(f).replace(/ \([+-]?\d+(\.\d+)?\)$/, '')));

        const drawer = [];
        drawer.push(`<p class="muted small">${esc(meta.category || '')}${a.data_as_of ? ` · data as of ${esc(fmtDay(a.data_as_of))}` : ''}</p>`);
        if (top.length && a.summary) drawer.push(`<p class="agent-summary">${esc(a.summary)}</p>`);
        // Optional blocks any agent can fill: a company-against-peers table and quoted notes from the annual report.
        const lead = raw.lead_table;
        if (lead && Array.isArray(lead.columns) && Array.isArray(lead.rows) && lead.rows.length) {
            const tip = lead.tip ? ` <span class="info" tabindex="0" role="note" aria-label="${esc(lead.tip)}" data-tip="${esc(lead.tip)}">i</span>` : '';
            const tips = Array.isArray(lead.tips) ? lead.tips : [];
            drawer.push(`<h4>${esc(lead.title || '')}${tip}</h4>${detailTable(lead.columns, lead.rows.map((r, i) => r.map((cell, j) => (j === 0 && tips[i]
                ? `<span class="term" tabindex="0" data-tip="${esc(tips[i])}">${esc(cell)}</span>` : esc(cell)))), { num: lead.columns.map((_, j) => j).slice(1) })}`);
        }
        if (Array.isArray(raw.filing_notes) && raw.filing_notes.length) {
            drawer.push(`<h4>From the annual report <span class="info" tabindex="0" role="note" aria-label="${esc(FILING_NOTES_TIP)}" data-tip="${esc(FILING_NOTES_TIP)}">i</span></h4><ul class="factor-list">${raw.filing_notes.map((n) => {
                const url = safeUrl(n.url);
                return `<li><strong>${esc(n.topic)}.</strong> ${esc(n.text)}${n.quote ? ` <q class="muted">${esc(n.quote)}</q>` : ''}${url ? ` <a href="${esc(url)}" target="_blank" rel="noopener noreferrer">10-K</a>` : ''}</li>`;
            }).join('')}</ul>`);
        }
        if (rest.length) drawer.push(`<h4>Other rules</h4>${ruleTable(rest)}`);
        if (extraFactors.length) drawer.push(`<ul class="factor-list">${extraFactors.map((f) => `<li>${esc(f)}</li>`).join('')}</ul>`);
        if (Array.isArray(raw.metrics_table) && raw.metrics_table.length) {
            const longValues = raw.metrics_table.some((m) => String(m.value).length > 40);
            drawer.push(tableHtml(['Metric', 'Value'], raw.metrics_table.map((m) => [m.metric, m.value]), { wrapCols: longValues ? [1] : [] }));
        }
        if (Array.isArray(raw.terms) && raw.terms.length) {
            drawer.push(`<h4>Terms</h4><ul class="factor-list">${raw.terms.map((t) => `<li><strong>${esc(t.term)}.</strong> ${esc(t.text)}</li>`).join('')}</ul>`);
        }
        if (raw.table && Array.isArray(raw.table.columns) && Array.isArray(raw.table.rows) && raw.table.rows.length) {
            drawer.push(`${raw.table.title ? `<h4>${esc(raw.table.title)}</h4>` : ''}${tableHtml(raw.table.columns, raw.table.rows, { wrapCols: raw.table.columns.map((col, i) => (/headline|insider|text|evidence|item/i.test(col) ? i : -1)) })}`);
        } else if (Array.isArray(raw.transactions_table) && raw.transactions_table.length) {
            drawer.push(tableHtml(['Insider', 'Type', 'Shares', 'Value', 'Date'],
                raw.transactions_table.map((t) => [t.insider, t.type, isNum(t.shares) ? t.shares.toLocaleString('en-US') : t.shares, isNum(t.value) ? fmt.money(t.value, 0) : '', t.date])));
        }
        (Array.isArray(raw.extra_tables) ? raw.extra_tables : []).forEach((t) => {
            if (t && Array.isArray(t.columns) && Array.isArray(t.rows) && t.rows.length) {
                drawer.push(`${t.title ? `<h4>${esc(t.title)}</h4>` : ''}${tableHtml(t.columns, t.rows)}`);
            }
        });
        const links = agentLinks(raw);
        if (links) drawer.push(`<div class="link-row">${links}</div>`);
        if (a.error) drawer.push(`<p class="hint">Error: ${esc(a.error)}</p>`);

        const points = c.included
            ? `<span class="agent-pts ${c.contribution >= 0 ? 'pos' : 'neg'}">${esc(fmt.signed(c.contribution, 1))}</span><span class="muted small">${esc(fmt.pct(c.weight_share, 0))} of the ${esc(layerName)}</span>`
            : `<span class="agent-pts muted">n/a</span><span class="muted small">${esc(EXCLUSION_SHORT[c.exclusion_reason] || c.exclusion_reason || 'not counted')}</span>`;
        const moreCount = rest.length + extraFactors.length;
        return `<article class="card agent-card ${available ? '' : 'is-unavailable'}" id="agent-card-${esc(a.agent_key)}">
            <div class="agent-head">
                <div class="agent-title">
                    <h3>${esc(shortName(a.agent_name))}</h3>
                    <span class="muted small">score ${esc(fmt.signed(a.score, 0))} · ${esc(fmt.num(a.confidence, 0))}% confidence${quality !== 'live' ? ` ${qualityBadge(a.data_quality)}` : ''}</span>
                </div>
                <div class="agent-points">${points}</div>
            </div>
            ${top.length ? ruleTable(top) : `<p class="agent-summary">${esc(a.summary)}</p>`}
            ${raw.caption ? `<p class="hint">${esc(raw.caption)}</p>` : ''}
            <details class="drawer"><summary>All rules and evidence${moreCount ? ` (${moreCount} more)` : ''}</summary><div class="drawer-body">${drawer.join('')}</div></details>
        </article>`;
    }

    // ------------------------------------------------ decision flow diagram
    // Nodes are HTML buttons placed on a grid: left and width in % of the
    // diagram, top and height in px. Edges are one SVG stretched to the same box
    // (x in %, y in px) with non-scaling strokes, so no resize handler is needed.

    const FLOW_COLS = { agents: [0, 30], layer: [37, 57], checks: [63, 81], result: [86, 100] };
    const FLOW_NODE_H = 46;
    const FLOW_GAP = 8;
    const FLOW_HUB_H = 84;
    const EXCLUSION_SHORT = {
        'Weight is 0 (informational only)': 'weight 0, evidence only',
        'No usable data': 'no usable data',
        'Disabled in weights config': 'disabled',
        'Informative: shown as evidence, not scored': 'informative, not scored',
    };

    const ptsHtml = (v, digits = 1) => `<span class="${v > 0 ? 'pos' : v < 0 ? 'neg' : ''}">${esc(fmt.signed(v, digits))}</span>`;

    function ownerChecks(d) {
        const o = d.owner || {};
        const th = o.thresholds || d.thresholds;
        const rated = o.applicable !== false && d.decision !== 'NOT_RATED';
        const relation = levelRelation(d.composite_score, th);
        const fund = d.agent_assessments.find((a) => a.agent_key === 'fundamentals');
        const raw = (fund && fund.data_quality !== 'UNAVAILABLE' && fund.raw_data) || {};
        const evf = isNum(raw.ev_to_fcf) ? `${fmt.num(raw.ev_to_fcf, 1)}x` : null;
        const cap = o.max_ev_to_fcf;
        const framework = rated
            ? { status: 'ok', label: 'Framework applies', detail: 'Company statements are available and the sector is inside the value framework.' }
            : { status: 'bad', label: 'Not rated', detail: o.not_rated_reason || 'No owner-layer data: the value framework does not apply.' };
        const coverage = { status: (o.coverage || 0) >= 0.7 ? 'ok' : 'warn', label: `Owner data ${fmt.pct(o.coverage || 0, 0)}` };
        coverage.detail = `${fmt.pct(o.coverage || 0, 0)} of the owner weight had usable data${coverage.status === 'warn' ? '; confidence reduced' : ''}.`;
        let safety;
        if (!rated) safety = { status: 'na', label: 'Safety cap: not checked', detail: 'Only rated companies are checked.' };
        else if (o.capped) safety = { status: 'warn', label: 'Safety cap: no new money', detail: (d.warnings || []).find((w) => w.startsWith('Margin of safety')) || capText(d) };
        else if (cap === null || cap === undefined) safety = { status: 'na', label: 'Safety cap: off', detail: 'The EV/FCF cap on new money is switched off in the Weights tab.' };
        else if (raw.is_financial) safety = { status: 'na', label: 'Safety cap: exempt', detail: 'Financial companies are exempt: EV/FCF does not apply to them.' };
        else if (relation.key !== 'entry') safety = { status: 'na', label: 'Safety cap: not needed', detail: `The score is not at the entry level, so the cap does not apply${evf ? ` (EV/FCF ${evf}, cap ${cap}x)` : ''}.` };
        else safety = { status: 'ok', label: 'Safety cap: passed', detail: `EV/FCF ${evf || 'n/a'} is below the ${cap}x cap.` };
        return { rated, relation, th, framework, coverage, safety };
    }

    // Cells are trusted HTML: callers escape their values.
    function detailTable(columns, rows, { num = [], wrap = [], label = '' } = {}) {
        const cls = (i) => [num.includes(i) ? 'num' : '', wrap.includes(i) ? 'wrap' : ''].filter(Boolean).join(' ');
        // On phones the table stacks: the first cell (the rule or name) spans, the others sit under their labels.
        const cell = (v, i, tag) => (tag === 'td' && columns.length > 1
            ? `<td class="${[cls(i), i === 0 ? 'span' : ''].filter(Boolean).join(' ')}"${i > 0 && columns[i] ? ` data-label="${esc(columns[i])}"` : ''}>${v}</td>`
            : `<${tag} class="${cls(i)}">${v}</${tag}>`);
        const head = columns.map((c, i) => cell(esc(c), i, 'th')).join('');
        const body = rows.map((r) => {
            const cells = Array.isArray(r) ? r : r.cells;
            return `<tr class="${Array.isArray(r) ? '' : esc(r.cls || '')}">${cells.map((v, i) => cell(v, i, 'td')).join('')}</tr>`;
        }).join('');
        return `<div class="table-wrap"${regionAttrs(label)}><table class="table compact stack"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    }

    function renderFlow(d) {
        const { owner, timing, info, contribByKey } = agentLayers(d);
        const o = d.owner || {};
        const t = d.timing || {};
        const checks = ownerChecks(d);
        const maxAbs = Math.max(10, ...d.contributions.filter((c) => c.included).map((c) => Math.abs(c.contribution)));
        const nodes = [];
        const edges = [];
        const labels = [];
        const box = (col, top, height) => `left:${FLOW_COLS[col][0]}%;width:${FLOW_COLS[col][1] - FLOW_COLS[col][0]}%;top:${top}px;height:${height}px`;
        const node = (id, col, top, height, cls, html, aria) => nodes.push(
            `<button type="button" class="flow-node ${cls}" data-node="${esc(id)}" style="${box(col, top, height)}" aria-pressed="false" aria-label="${esc(aria)}">${html}</button>`);
        const edge = (path, cls, width) => edges.push(`<path class="flow-edge ${cls}" d="${path}" style="stroke-width:${width.toFixed(1)}px"/>`);
        const curve = (x1, y1, x2, y2) => { const mx = (x1 + x2) / 2; return `M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}`; };
        const label = (cls, style, text) => labels.push(`<div class="${cls}" style="${style}">${esc(text)}</div>`);

        [['agents', '1. Agents'], ['layer', '2. Layer score'], ['checks', '3. Checks'], ['result', '4. Result']]
            .forEach(([col, text]) => label('flow-col-label', `left:${FLOW_COLS[col][0]}%;top:0`, text));

        const band = (list, top, layer, hub = true) => {
            const stackH = list.length * (FLOW_NODE_H + FLOW_GAP) - FLOW_GAP;
            const height = Math.max(hub ? FLOW_HUB_H : FLOW_NODE_H, stackH);
            const hubTop = top + (height - FLOW_HUB_H) / 2;
            const agentsTop = top + (height - stackH) / 2;
            list.forEach((a, i) => {
                const c = contribByKey[a.agent_key] || {};
                const on = !!c.included;
                const sign = c.contribution >= 0 ? 'pos' : 'neg';
                const y = agentsTop + i * (FLOW_NODE_H + FLOW_GAP);
                const target = list.length > 1 ? hubTop + 16 + (i * (FLOW_HUB_H - 32)) / (list.length - 1) : hubTop + FLOW_HUB_H / 2;
                if (hub) {
                    edge(curve(FLOW_COLS.agents[1], y + FLOW_NODE_H / 2, FLOW_COLS.layer[0], target),
                        on ? sign : 'off', on ? 1.5 + (6 * Math.abs(c.contribution)) / maxAbs : 1);
                }
                const sub = on
                    ? `<span>score ${esc(fmt.signed(a.score, 0))} · ${esc(fmt.num(a.confidence, 0))}% conf</span><span>${esc(fmt.pct(c.weight_share, 0))} share</span>`
                    : `<span>${esc(EXCLUSION_SHORT[c.exclusion_reason] || c.exclusion_reason || 'not counted')}</span>`;
                node(`agent:${a.agent_key}`, 'agents', y, FLOW_NODE_H, `${layer}${on ? '' : ' is-off'}`,
                    `<span class="fn-row"><span class="fn-name">${esc(shortName(a.agent_name))}</span><span class="fn-pts ${on ? sign : ''}">${on ? esc(fmt.signed(c.contribution, 1)) : 'n/a'}</span></span><span class="fn-row fn-sub">${sub}</span>`,
                    `${a.agent_name}: ${on ? `${fmt.signed(c.contribution, 1)} points` : 'not counted'}`);
            });
            return { hubTop, cy: hub ? hubTop + FLOW_HUB_H / 2 : top + height / 2, bottom: top + height };
        };

        // One band per layer, in the user's layer order. Owner: agents -> owner score ->
        // checks -> decision. Timing: agents -> entry timing -> entry read, with no path to
        // the decision. Informative: agents only, scored nowhere.
        const drawOwner = (top) => {
            const b = band(owner, top, 'owner');
            const used = owner.filter((a) => (contribByKey[a.agent_key] || {}).included).length;
            node('owner', 'layer', b.hubTop, FLOW_HUB_H, 'owner', checks.rated
                ? `<span class="fn-title">Weighted score</span><span class="fn-big ${d.composite_score >= 0 ? 'pos' : 'neg'}">${esc(fmt.signed(d.composite_score, 1))}</span><span class="fn-sub">${esc(fmt.num(o.confidence, 0))}% confidence · ${used} of ${plural(owner.length, 'agent', 'agents')}</span>`
                : '<span class="fn-title">Weighted score</span><span class="fn-big muted">n/a</span><span class="fn-sub">no owner rating</span>', 'Weighted score');
            node('checks', 'checks', b.hubTop, FLOW_HUB_H, 'owner',
                `<span class="fn-title">Checks</span><span class="fn-check ${checks.framework.status}">${esc(checks.framework.label)}</span><span class="fn-check ${checks.safety.status}">${esc(checks.safety.label)}</span>`, 'Checks');
            node('decision', 'result', b.hubTop, FLOW_HUB_H, 'owner', checks.rated
                ? `<span class="fn-title">Owner score</span><span class="fn-big ${d.composite_score >= 0 ? 'pos' : 'neg'}">${esc(fmt.signed(d.composite_score, 1))}</span><span class="fn-sub">${esc(checks.relation.key === 'entry' ? 'at or above entry level' : checks.relation.key === 'exit' ? 'at or below exit level' : 'between the levels')}</span>`
                : '<span class="fn-title">Owner score</span><span class="fn-big muted">Not rated</span><span class="fn-sub">outside the framework</span>',
                `Owner score ${checks.rated ? `${fmt.signed(d.composite_score, 1)}, ${checks.relation.text}` : 'not rated'}`);
            edge(`M${FLOW_COLS.layer[1]} ${b.cy} L${FLOW_COLS.checks[0]} ${b.cy}`, 'main', 2);
            edge(`M${FLOW_COLS.checks[1]} ${b.cy} L${FLOW_COLS.result[0]} ${b.cy}`, 'main', 2);
            return b.bottom;
        };
        const drawTiming = (top) => {
            const b = band(timing, top, 'timing');
            const used = timing.filter((a) => (contribByKey[a.agent_key] || {}).included).length;
            node('timing', 'layer', b.hubTop, FLOW_HUB_H, 'timing',
                `<span class="fn-title">Entry timing</span><span class="fn-big ${(t.score || 0) >= 0 ? 'pos' : 'neg'}">${esc(fmt.signed(t.score || 0, 0))}</span><span class="fn-sub">${esc(rgName(t.regime || d.market_regime))} profile · ${used} of ${plural(timing.length, 'agent', 'agents')}</span>`, 'Entry timing');
            node('entry', 'result', b.hubTop, FLOW_HUB_H, 'timing',
                `<span class="fn-title">Entry read</span><span class="fn-sub">trend ${esc(String(t.trend || 'n/a').toLowerCase())}${t.setup ? `, ${esc(setupLabel(t.setup))}` : ''}</span><span class="fn-sub">${plural((t.notes || []).slice(1).length, 'note', 'notes')} on when and how much</span>`,
                'Entry read');
            edge(`M${FLOW_COLS.layer[1]} ${b.cy} L${FLOW_COLS.result[0]} ${b.cy}`, 'info', 1.5);
            labels.push(`<div class="flow-edge-label" style="left:${FLOW_COLS.checks[0]}%;width:${FLOW_COLS.checks[1] - FLOW_COLS.checks[0]}%;top:${b.cy - 9}px"><span>informs, no vote</span></div>`);
            return b.bottom;
        };
        const drawInfo = (top) => {
            const b = band(info, top, 'info', false);
            labels.push(`<div class="flow-edge-label left" style="left:${FLOW_COLS.layer[0]}%;width:${FLOW_COLS.result[1] - FLOW_COLS.layer[0]}%;top:${b.cy - 9}px"><span>shown as evidence, not scored</span></div>`);
            return b.bottom;
        };
        const BANDS = {
            owner: ['Owner agents: decide', drawOwner],
            timing: ['Timing agents: inform, no vote', drawTiming],
            info: ['Informative agents: evidence only', drawInfo],
        };
        let y = 24;
        layerOrder().filter((l) => BANDS[l] && (l !== 'info' || info.length)).forEach((l, i) => {
            if (i > 0) {
                y += 18;
                labels.push(`<div class="flow-sep" style="top:${y}px"></div>`);
                y += 8;
            }
            // In the node list, so a phone's stacked order reads band label, agents, score, checks, result.
            nodes.push(`<div class="flow-band-label" style="left:0;top:${y}px">${esc(BANDS[l][0])}</div>`);
            y = BANDS[l][1](y + 20);
        });

        const height = Math.ceil(y + 4);
        const el = $('#flow-diagram');
        el.style.height = `${height}px`;
        el.innerHTML = `<svg class="flow-edges" viewBox="0 0 100 ${height}" preserveAspectRatio="none" aria-hidden="true">${edges.join('')}</svg>${labels.join('')}${nodes.join('')}`;
        selectFlowNode(state.flowSelected || 'decision');
    }

    function selectFlowNode(id) {
        const d = state.analysis;
        if (!d) return;
        if (!$$('#flow-diagram .flow-node').some((n) => n.dataset.node === id)) id = 'decision';
        state.flowSelected = id;
        $$('#flow-diagram .flow-node').forEach((n) => {
            const on = n.dataset.node === id;
            n.classList.toggle('is-selected', on);
            n.setAttribute('aria-pressed', String(on));
        });
        $('#flow-detail').innerHTML = id.startsWith('agent:') ? agentFlowDetail(d, id.slice(6))
            : id === 'owner' || id === 'timing' ? layerFlowDetail(d, id)
            : id === 'checks' ? checksFlowDetail(d)
            : id === 'entry' ? entryFlowDetail(d)
            : decisionFlowDetail(d);
    }

    function agentFlowDetail(d, key) {
        const a = d.agent_assessments.find((x) => x.agent_key === key);
        if (!a) return '';
        const c = d.contributions.find((x) => x.agent_key === key) || {};
        const isOwner = c.layer === 'owner';
        const isInfo = c.layer === 'info';
        const layerName = isOwner ? 'owner score' : 'entry timing';
        const scale = ((state.config || {}).score_scales || {})[key] ?? (state.agentByKey[key] || {}).score_scale;
        const breakdown = Array.isArray((a.raw_data || {}).score_breakdown)
            ? [...a.raw_data.score_breakdown].sort((x, y) => Math.abs(y.points) - Math.abs(x.points)) : [];
        const rules = breakdown.length
            ? detailTable(['Rule', 'Points'], [...breakdown.map((b) => [esc(b.factor), ptsHtml(b.points)]), { cls: 'total', cells: ['Raw score', ptsHtml(a.score)] }], { num: [1], wrap: [0] })
            : `<p class="muted small">This agent reports no rule-by-rule breakdown. Raw score ${esc(fmt.signed(a.score, 0))}.</p>`;
        const steps = c.included
            ? detailTable(['Step', 'Value'], [
                ['Raw score', ptsHtml(c.raw_score)],
                [`Normalized: tanh(raw / ${esc(fmt.num(scale, 0))}) x 100`, ptsHtml(c.normalized_score)],
                ['Weight set in the Weights tab', esc(fmt.num(c.base_weight, 2))],
                [`Effective weight at ${esc(fmt.num(c.confidence, 0))}% confidence`, esc(fmt.num(c.effective_weight, 3))],
                [`Share of the ${layerName}`, esc(fmt.pct(c.weight_share, 1))],
                { cls: 'total', cells: ['Points added (normalized x share)', ptsHtml(c.contribution)] },
            ], { num: [1] })
            : isInfo ? '<p class="muted small">Informative: shown with its evidence and scored in neither layer. Change its role in the Weights tab.</p>'
            : `<p class="muted small">Not counted in the ${layerName}: ${esc(c.exclusion_reason || 'no data')}.${isOwner ? ' Its evidence is still shown.' : ''}</p>`;
        return `<div class="flow-detail-head">
                <h3>${esc(a.agent_name)}</h3>
                <span class="badge ${isOwner ? 'owner' : 'timing'}">${isOwner ? 'Owner' : isInfo ? 'Informative' : 'Timing'}</span>
                ${qualityBadge(a.data_quality)}
                <span class="spacer"></span>
                <button class="btn btn-ghost" type="button" data-goto-agent="${esc(key)}">Open the agent card</button>
            </div>
            <p class="muted small">${esc(a.summary)}</p>
            <div class="grid-2"><div><h4>Rules that scored</h4>${rules}</div><div><h4>${isInfo ? 'Role' : `How it enters the ${esc(layerName)}`}</h4>${steps}</div></div>`;
    }

    function layerFlowDetail(d, layer) {
        const isOwner = layer === 'owner';
        const L = (isOwner ? d.owner : d.timing) || {};
        const { owner, timing, contribByKey } = agentLayers(d);
        const rows = (isOwner ? owner : timing).map((a) => {
            const c = contribByKey[a.agent_key] || {};
            const name = esc(shortName(a.agent_name));
            return c.included
                ? [name, ptsHtml(c.raw_score, 0), ptsHtml(c.normalized_score), esc(fmt.num(c.base_weight, 2)), `${esc(fmt.num(c.confidence, 0))}%`, esc(fmt.pct(c.weight_share, 0)), ptsHtml(c.contribution)]
                : { cls: 'is-off', cells: [name, '', '', esc(fmt.num(c.base_weight || 0, 2)), '', '', esc(EXCLUSION_SHORT[c.exclusion_reason] || c.exclusion_reason || 'not counted')] };
        });
        rows.push({ cls: 'total', cells: [isOwner ? 'Weighted score' : 'Entry timing', '', '', '', '', '', ptsHtml(isOwner ? d.composite_score : L.score || 0)] });
        const th = L.thresholds || {};
        const regime = rgName(L.regime || d.market_regime);
        const kvs = [
            ['Confidence', `${fmt.num(L.confidence, 0)}%`],
            ['Coverage', fmt.pct(L.coverage, 0)],
            ['Dispersion', fmt.num(L.dispersion, 1)],
            isOwner ? ['Horizon', d.horizon || '12 months'] : ['Weight profile', regime],
        ];
        const note = isOwner
            ? `Owner weights are the same in every market regime. The entry level is ${fmt.signed(th.buy, 0)} and the exit level ${fmt.signed(th.sell, 0)}.`
            : `Weights come from the ${regime} profile, picked by the Macro agent's regime. The timing score never changes the owner score.`;
        const pts = layerRules(d, layer);
        const ptsBlock = pts.length
            ? `<h4>Points behind the score <span class="muted small">largest first, on each agent's own scale</span></h4>${detailTable(['Rule', 'Agent', 'Points'], pts.map((r) => [esc(r.factor), esc(r.agent), ptsHtml(r.points)]), { num: [2], wrap: [0] })}` : '';
        return `<div class="flow-detail-head"><h3>${isOwner ? 'Weighted score' : 'Entry timing'}</h3><span class="badge ${isOwner ? 'owner' : 'timing'}">${isOwner ? 'The verdict' : 'No vote'}</span></div>
            ${detailTable(['Agent', 'Raw', 'Normalized', 'Weight', 'Confidence', 'Share', 'Points'], rows, { num: [1, 2, 3, 4, 5, 6] })}
            ${ptsBlock}
            <div class="stat-row">${kvs.map(([k, v]) => `<div class="stat"><div class="stat-label">${esc(k)}</div><div class="stat-value">${esc(v)}</div></div>`).join('')}</div>
            <p class="hint">Each raw score is normalized with tanh and weighted by its weight and confidence (effective weight = weight x (floor + (1 - floor) x confidence)); the layer score is the weighted mean. Confidence falls when agents disagree (dispersion) or data is missing (coverage). ${esc(note)}</p>`;
    }

    function checksFlowDetail(d) {
        const k = ownerChecks(d);
        const result = (s) => `<span class="fn-check ${s.status}">${s.status === 'ok' ? 'Pass' : s.status === 'warn' ? 'Applied' : s.status === 'bad' ? 'Fail' : 'n/a'}</span>`;
        const rows = [['Value framework applies', result(k.framework), esc(k.framework.detail)]];
        if (k.rated) rows.push(['Owner data coverage', result(k.coverage), esc(k.coverage.detail)]);
        rows.push(['Margin of safety: no new money at or above the EV/FCF cap', result(k.safety), esc(k.safety.detail)]);
        return `<div class="flow-detail-head"><h3>Checks</h3><span class="badge owner">Owner layer</span></div>
            ${detailTable(['Check', 'Result', 'Detail'], rows, { wrap: [2] })}
            <p class="hint">${k.rated ? `Owner score ${esc(fmt.signed(d.composite_score, 1))}, ${esc(k.relation.text)}. The checks never change the score.` : 'Not rated: the entry timing is shown for information only.'} The cap and the levels are set in the Weights tab.</p>`;
    }

    function decisionFlowDetail(d) {
        const k = ownerChecks(d);
        const th = k.th;
        const zones = [
            ['entry', 'At or above the entry level', `${fmt.signed(th.buy, 0)} or more`],
            ['between', 'Between the levels', `${fmt.signed(th.sell, 0)} to ${fmt.signed(th.buy, 0)}`],
            ['exit', 'At or below the exit level', `${fmt.signed(th.sell, 0)} or less`],
        ];
        const rows = zones.map(([key, name, range]) => {
            const here = k.rated && key === k.relation.key;
            return { cls: here ? 'is-current' : '', cells: [esc(name), esc(range), here ? `owner score ${esc(fmt.signed(d.composite_score, 1))}` : ''] };
        });
        const why = !k.rated ? k.framework.detail
            : (d.owner || {}).capped ? `The owner score ${fmt.signed(d.composite_score, 1)} is ${k.relation.text}. ${capText(d)}`
            : `The owner score ${fmt.signed(d.composite_score, 1)} is ${k.relation.text}.`;
        const pts = k.rated ? layerRules(d, 'owner') : [];
        const checkRows = [['Value framework applies', k.framework.status === 'ok' ? 'Pass' : 'Not rated'], [k.safety.label, k.safety.status === 'warn' ? 'No new money' : k.safety.status === 'ok' ? 'Pass' : 'n/a']];
        return `<div class="flow-detail-head"><h3>Owner score</h3>${k.rated ? ptsHtml(d.composite_score) : '<span class="decision sm decision-not-rated">Not rated</span>'}<span class="muted small">horizon ${esc(d.horizon || '12 months')}</span></div>
            <p class="small">${esc(why)}</p>
            ${k.rated ? detailTable(['Level', 'Owner score', ''], rows) : ''}
            ${pts.length ? `<h4>Points behind the score <span class="muted small">largest first, on each agent's own scale</span></h4>${detailTable(['Rule', 'Agent', 'Points'], pts.map((r) => [esc(r.factor), esc(r.agent), ptsHtml(r.points)]), { num: [2], wrap: [0] })}` : ''}
            <h4>Checks</h4>${detailTable(['Check', 'Result'], checkRows.map(([a, b]) => [esc(a), esc(b)]))}
            <p class="hint">The entry level (${esc(fmt.signed(th.buy, 0))}) and the exit level (${esc(fmt.signed(th.sell, 0))}) turn the score into actions in the portfolio review and the backtest. The strong levels (${esc(fmt.signed(th.strong_buy, 0))} and ${esc(fmt.signed(th.strong_sell, 0))}) only group scores in the backtest tables and raise watchlist alerts. Only the owner layer reaches this box: the entry timing never changes the score.</p>`;
    }

    function entryFlowDetail(d) {
        const t = d.timing || {};
        const ev = t.event_risk || {};
        const vix = t.vix || {};
        const rows = [
            ['Entry timing score', ptsHtml(t.score || 0, 0)],
            ['Trend (50 / 200-day)', esc(`${String(t.trend || 'n/a').toLowerCase()}${isNum(t.sma_50) && isNum(t.sma_200) ? `, SMA 50 ${fmt.money(t.sma_50)} vs SMA 200 ${fmt.money(t.sma_200)}` : ''}`)],
            ['Setup', esc(t.setup ? setupLabel(t.setup) : 'n/a')],
            ['12-1 month momentum', esc(isNum(t.momentum_12_1) ? `${fmt.spct(t.momentum_12_1, 0)}${t.value_trap_check ? ', value-trap check' : ''}` : 'n/a')],
            ['Event risk', esc(ev.elevated ? (isNum(ev.days_to_earnings) ? `elevated, earnings in ${plural(ev.days_to_earnings, 'day', 'days')}` : 'elevated, options price a larger move') : 'normal')],
            ['VIX', esc(isNum(vix.level) ? `${fmt.num(vix.level, 1)}${vix.extreme ? ', extreme fear (40+)' : ''}` : 'n/a')],
            ['Market regime', esc(rgName(t.regime || d.market_regime) || 'n/a')],
        ];
        const notes = (t.notes || []).slice(1); // the first note only restates that the panel has no vote
        const pts = layerRules(d, 'timing');
        return `<div class="flow-detail-head"><h3>Entry read</h3><span class="badge timing">No vote</span></div>
            <div class="grid-2">
                <div>${detailTable(['Reading', 'Value'], rows, { wrap: [1] })}${pts.length ? `<h4>Points behind the timing score</h4>${detailTable(['Rule', 'Agent', 'Points'], pts.slice(0, 8).map((r) => [esc(r.factor), esc(r.agent), ptsHtml(r.points)]), { num: [2], wrap: [0] })}` : ''}</div>
                <div><h4>What it means for the entry</h4>${notes.length ? `<ul class="factor-list">${notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : '<p class="muted small">No timing notes: no setup, event or regime to act on.</p>'}</div>
            </div>`;
    }

    function setupAnalyze() {
        $('#analyze-notice').addEventListener('click', (e) => {
            const chip = e.target.closest('[data-analyze]');
            if (chip) runAnalyze(chip.dataset.analyze);
        });
        $('#analyze-form').addEventListener('submit', (e) => { e.preventDefault(); runAnalyze(); });
        const tickerInput = $('#analyze-ticker');
        tickerInput.addEventListener('focus', (e) => e.target.select());
        tickerInput.addEventListener('input', () => {
            tickerInput.removeAttribute('aria-invalid');
            $('#analyze-button').classList.add('btn-primary');  // the one filled button once the form is in use
        });
        api('/api/portfolio').then((p) => { if ((p.positions || []).length) $('#empty-cta').textContent = 'Review your portfolio'; }).catch(() => {});
        $('#flow-diagram').addEventListener('click', (e) => {
            const n = e.target.closest('[data-node]');
            if (n) selectFlowNode(n.dataset.node);
        });
        $('#hero').addEventListener('click', (e) => {
            const b = e.target.closest('[data-flow-node]');
            if (!b) return;
            selectFlowNode(b.dataset.flowNode);
            $('#flow-detail').scrollIntoView({ block: 'start' });
        });
        $('#flow-detail').addEventListener('click', (e) => {
            const b = e.target.closest('[data-goto-agent]');
            const card = b && document.getElementById(`agent-card-${b.dataset.gotoAgent}`);
            if (!card) return;
            card.scrollIntoView({ block: 'center' });
            card.classList.add('flash');
            setTimeout(() => card.classList.remove('flash'), 1600);
        });
        const refreshSel = $('#analyze-refresh');
        refreshSel.value = storageGet('fastvalue-analyze-refresh') || '0';
        const schedule = () => {
            clearInterval(state.timers.analyze);
            const secs = Number(refreshSel.value);
            storageSet('fastvalue-analyze-refresh', String(secs));
            if (secs > 0) {
                // Only record to history at 5-minute cadence or slower to keep the log readable.
                state.timers.analyze = setInterval(() => {
                    if (state.analysis && !document.hidden) runAnalyze(state.analysis.ticker, { silent: true, record: secs >= 300 });
                }, secs * 1000);
            }
        };
        refreshSel.addEventListener('change', schedule);
        schedule();
    }

    // ============================================================ WATCHLIST

    function renderChips() {
        $('#watch-chips').innerHTML = state.watchlist.length
            ? state.watchlist.map((t) => `<span class="chip">${esc(t)}<button type="button" data-remove="${esc(t)}" aria-label="Remove ${esc(t)}">${icon('x')}</button></span>`).join('')
            : '<p class="hint">Your watchlist is empty. Add the companies you are considering: a scan ranks them by owner score and flags any that cross your entry or exit level.</p>';
        syncWatchResults();
    }

    // The results follow the list: a removed ticker leaves the table, the strip and the notes at once, and an empty
    // list leaves no results and nothing to scan.
    function syncWatchResults() {
        const n = state.watchlist.length;
        if (state.watchData) {
            const keep = (r) => state.watchlist.includes(r.ticker);
            state.watchData = { ...state.watchData, rows: state.watchData.rows.filter(keep), errors: (state.watchData.errors || []).filter(keep) };
            if (!state.watchData.rows.length && !state.watchData.errors.length) state.watchData = null;
        }
        if (!n) state.watchData = null;
        const btn = $('#watch-scan-button');
        if (!state.scanning) btn.disabled = !n;
        const pre = $('#watch-prescan');
        pre.textContent = `Scan now ranks ${n === 1 ? 'this 1 ticker' : `these ${n} tickers`} by owner score. It takes a few seconds, up to a minute the first time.`;
        pre.hidden = !n || !!state.watchData;
        renderWatchTable();
    }

    async function saveWatchlist(previous) {
        try {
            const data = await api('/api/watchlist', { method: 'PUT', body: { tickers: state.watchlist } });
            state.watchlist = data.tickers;
            renderChips();
        } catch (err) {
            if (previous) { state.watchlist = previous; renderChips(); }  // a rejected list must not block later saves
            toast(`Could not save watchlist: ${err.message}`, 'error');
        }
    }

    async function scanWatchlist({ silent = false } = {}) {
        if (state.scanning || !state.watchlist.length) return;
        state.scanning = true;
        const btn = $('#watch-scan-button');
        setBusy(btn, true);
        const slowDone = silent ? () => {} : slowHint('#watch-slow-hint', 'scan');
        if (!silent) {
            const hint = $('#watch-slow-hint');
            hint.textContent = `Scanning ${state.watchlist.length} ticker${state.watchlist.length === 1 ? '' : 's'}.`;
            hint.hidden = false;
        }
        try {
            const data = await api('/api/watchlist/scan', { method: 'POST', body: { tickers: state.watchlist } });
            data.rows.forEach((r, i) => { r.rank = i; });  // the server's order: rated companies by owner score, then funds by timing
            data.levels = clone((state.config || {}).owner_thresholds || {});  // the levels this scan is read with, until the next scan
            state.watchData = data;
            state.watchRestored = false;
            $('#watch-scan-error').innerHTML = '';
            storageSet('fastvalue-watch-last', JSON.stringify(data));
            renderWatchTable();
            const risen = data.rows.filter((r) => r.upgraded);
            if (risen.length) toast(`Score rose past a level since the last scan: ${risen.map((r) => `${r.ticker} ${crossing(r.previous_decision, r.decision).text}`).join(', ')}`, 'success', 9000);
            if ((data.errors || []).length) toast(`${plural(data.errors.length, 'ticker', 'tickers')} could not be analyzed: see the note under the table.`, 'error');
        } catch (err) {
            const old = state.watchData ? ` The results below are from ${watchTime(state.watchData.scanned_at)}.` : '';
            $('#watch-scan-error').innerHTML = `<div class="warning"><strong>Scan failed</strong>${esc(String(err.message).replace(/\.$/, ''))}. This usually clears within a minute.${esc(old)} <button class="btn btn-sm" type="button" data-watch-retry>Try again</button></div>`;
        } finally {
            slowDone();
            state.scanning = false;
            setBusy(btn, false);
            btn.disabled = !state.watchlist.length;
            if (!silent) { const hint = $('#watch-slow-hint'); if (hint.textContent.startsWith('Scanning ')) hint.hidden = true; }
            rearmWatchScan();
        }
    }

    const watchTime = (iso) => { const d = new Date(iso); return isNaN(d) ? 'an earlier scan' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); };
    let rearmWatchScan = () => {};

    // The last scan survives a reload, as a per-viewer convenience (public tickers and scores only).
    function restoreWatchScan() {
        try {
            const data = JSON.parse(storageGet('fastvalue-watch-last') || 'null');
            if (!data || !Array.isArray(data.rows) || !data.rows.length) return;
            state.watchData = data;
            state.watchRestored = true;
            syncWatchResults();
        } catch (_) { /* a damaged copy is ignored */ }
    }

    // The level a score crossed between two scans, from the stored bands: the level that sets the new band.
    const BAND_ORDER = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY'];
    const LEVEL_NAMES = ['strong exit level', 'exit level', 'entry level', 'strong entry level'];
    function crossing(from, to) {
        const o = BAND_ORDER.indexOf(from), n = BAND_ORDER.indexOf(to);
        if (o < 0 || n < 0 || o === n) return { up: n > o, text: 'changed level', tone: '' };
        const i = n > o ? n - 1 : n;
        const entrySide = i >= 2;
        const up = n > o;
        // The badge takes the band the score moved into: only a move to or below the exit level is red.
        const tone = up ? (n >= 3 && o <= 2 ? 'live' : '') : (n <= 1 ? 'unavailable' : n === 2 ? 'warn' : '');
        return { up, tone, text: `${up ? (entrySide ? 'rose to or above' : 'rose above') : (entrySide ? 'fell below' : 'fell to or below')} the ${LEVEL_NAMES[i]}` };
    }

    const WATCH_COLUMNS = [
        { key: 'ticker', label: 'Ticker' },
        { key: 'composite_score', label: 'Owner score', num: true },
        { key: 'flags', label: 'Flags', sort: false },
        { key: 'score_change', label: 'Change', num: true },
        { key: 'top_points', label: 'Main points', sort: false, cls: 'agent-col' },
        { key: 'timing_score', label: 'Entry timing', num: true },
        { key: 'price', label: 'Price', num: true },
        { key: 'change_1d', label: '1D', num: true },
        { key: 'confidence', label: 'Conf.', num: true },
        { key: 'trend', label: 'Trend' },
        { key: 'risk_rating', label: 'Risk' },
    ];
    const WATCH_GROUPS = [['owner', 'Owner score'], ['timing', 'Entry timing'], ['info', 'Informative']];

    function renderWatchTable() {
        const data = state.watchData;
        const results = $('#watch-results'), summary = $('#watch-summary');
        if (!data) {
            results.hidden = true;
            summary.hidden = true;
            ['#watch-summary', '#watch-table', '#watch-strip', '#watch-errors', '#watch-notes'].forEach((s) => { $(s).innerHTML = ''; });
            ['#watch-last', '#watch-regime'].forEach((s) => { $(s).textContent = ''; });
            return;
        }
        const { key, dir } = state.watchSort;
        const rows = [...data.rows].sort((a, b) => {
            const rated = (b.owner_applicable !== false) - (a.owner_applicable !== false);  // funds and indexes stay below the rated companies, either way round
            if (rated) return rated;
            const va = a[key], vb = b[key];
            if (typeof va === 'string' || typeof vb === 'string') return String(va).localeCompare(String(vb)) * dir;
            return ((va ?? -Infinity) - (vb ?? -Infinity)) * dir;
        });

        // Agent columns in layer order. A column that says the same for every company, or has no data in any row, becomes a note.
        const missing = (s) => !s || s.quality === 'UNAVAILABLE';
        const layerIdx = (k) => WATCH_GROUPS.findIndex(([l]) => l === k);
        const agents = [], notes = [];
        [...state.agents].sort(byAgentOrder((a) => a.key)).filter((a) => rows.some((r) => r.agents && r.agents[a.key])).forEach((a) => {
            const cells = rows.map((r) => r.agents?.[a.key]);
            const label = shortName(a.name).split(' & ')[0];
            if (cells.every(missing)) { notes.push(`${label} has no data for any company in this scan, so its column is hidden.`); return; }
            const first = cells.find((c) => c && !missing(c));
            const same = rows.length > 1 && cells.every((c) => !missing(c) && Math.round(c.score) === Math.round(cells[0].score));
            if (same) { notes.push(`${label} gives the same ${a.key === 'macro' ? 'market reading' : 'score'} for every company (${fmt.signed(cells[0].score, 0)}), so it is not shown per row.`); return; }
            agents.push({ ...a, layer: first.layer || 'timing' });
        });
        agents.sort((x, y) => layerIdx(x.layer) - layerIdx(y.layer));

        const th = data.levels || (state.config || {}).owner_thresholds || {};
        const where = (r) => (r.owner_applicable === false || r.decision === 'NOT_RATED' ? 'none'
            : isNum(th.buy) && isNum(th.sell) ? levelRelation(r.composite_score, th).key
            : r.decision === 'BUY' || r.decision === 'STRONG_BUY' ? 'entry' : r.decision === 'SELL' || r.decision === 'STRONG_SELL' ? 'exit' : 'between');
        const counts = rows.reduce((acc, r) => { const k = where(r); acc[k] = (acc[k] || 0) + 1; return acc; }, {});
        const regime = rows.find((r) => r.regime)?.regime;
        const levelSub = (k, name) => (isNum(th[k]) ? `${name} ${fmt.signed(th[k], 0)}` : '');
        summary.innerHTML = [
            ['At or above the entry level', counts.entry || 0, esc(levelSub('buy', 'Entry level'))],
            ['Between the levels', counts.between || 0, ''],
            ['At or below the exit level', counts.exit || 0, esc(levelSub('sell', 'Exit level'))],
            ['Not rated', counts.none || 0, `Outside the value framework ${infoTip('Funds, indexes, utilities, REITs and companies with no statements are not rated by the owner score.')}`],
            ['Level changes', rows.filter((r) => r.decision_changed).length, 'since previous scan'],
        ].map(([l, v, s]) => `<div class="stat"><div class="stat-label">${esc(l)}</div><div class="stat-value">${esc(v)}</div><div class="stat-sub">${s}</div></div>`).join('');
        summary.hidden = false;
        $('#watch-regime').textContent = regime ? `Market regime: ${rgName(regime)}` : '';
        $('#watch-last').textContent = state.watchRestored
            ? `From your scan at ${watchTime(data.scanned_at)}. Scan again for fresh prices.`
            : `Last scan ${watchTime(data.scanned_at)}`;

        const showAgents = state.watchView === 'agents';
        const groups = WATCH_GROUPS.map(([l, name]) => [name, agents.filter((a) => a.layer === l).length]).filter(([, n]) => n);
        const groupRow = agents.length
            ? `<tr class="group-row"><th class="watch-key"><span class="sr-only">Company</span></th><th colspan="${WATCH_COLUMNS.length - 1}"><span class="sr-only">Scan details</span></th>${groups.map(([name, n]) => `<th class="group-head agent-col" colspan="${n}">${esc(name)}</th>`).join('')}</tr>` : '';
        const head = WATCH_COLUMNS.map((c) => {
            const cls = [c.num ? 'num' : '', c.cls || '', c.key === 'ticker' ? 'watch-key' : ''].filter(Boolean).join(' ');
            return c.sort === false ? `<th${cls ? ` class="${cls}"` : ''}>${esc(c.label)}</th>`
                : `<th class="sortable ${cls}" data-sort="${c.key}" aria-sort="${key === c.key ? (dir < 0 ? 'descending' : 'ascending') : 'none'}"><button type="button" class="th-sort">${esc(c.label)}</button></th>`;
        }).join('') + agents.map((a) => `<th class="num agent-col"><span class="term quiet tip-end" tabindex="0" data-tip="${esc(shortName(a.name))}">${esc(shortName(a.name).split(' & ')[0])}</span></th>`).join('');

        const body = rows.map((r) => {
            const flags = [];
            if (r.decision_changed) { const c = crossing(r.previous_decision, r.decision); flags.push(`<span class="badge ${c.tone}">${esc(capFirst(c.text))}</span>`); }
            if (r.conflict) flags.push('<span class="badge warn">Conflict</span>');
            if (r.freshness?.status === 'STALE') flags.push('<span class="badge unavailable">Stale</span>');
            if (r.coverage < 0.7) flags.push('<span class="badge warn">Low coverage</span>');
            const width = Math.abs(r.composite_score) / 2;
            return `<tr class="clickable" data-ticker="${esc(r.ticker)}">
                <td class="mono watch-key"><button type="button" class="co-rowbtn" aria-label="Open ${esc(r.ticker)} in Analyze"><strong>${esc(r.ticker)}</strong></button></td>
                <td class="num">${r.owner_applicable === false ? '<span class="muted term quiet" tabindex="0" data-tip="Outside the value framework">Not rated</span>' : `<span class="score-cell term quiet" tabindex="0" data-tip="${esc(capFirst(levelRelation(r.composite_score, th).text))}"><span class="mini-bar"><span style="left:${r.composite_score >= 0 ? 50 : 50 - width}%;width:${width}%;background:${r.composite_score >= 0 ? 'var(--pos)' : 'var(--neg)'}"></span></span>${esc(fmt.signed(r.composite_score, 1))}</span>`}</td>
                <td>${flags.join(' ')}</td>
                <td class="num ${r.score_change > 0 ? 'pos' : r.score_change < 0 ? 'neg' : ''}">${r.score_change === null ? '<span class="muted">new</span>' : esc(fmt.signed(r.score_change, 1))}</td>
                <td class="small agent-col">${(r.top_points || []).length ? (r.top_points || []).map((p) => `<div class="term quiet" tabindex="0" data-tip="${esc(p.factor)}, ${esc(p.agent)}">${esc(String(p.factor).split(' (')[0])} ${ptsHtml(p.points, 0)} pts</div>`).join('') : '<span class="muted">n/a</span>'}</td>
                <td class="num ${(r.timing_score || 0) > 0 ? 'pos' : (r.timing_score || 0) < 0 ? 'neg' : ''}">${r.setup ? `<span class="term quiet" tabindex="0" data-tip="${esc(setupLabel(r.setup))}">${esc(fmt.signed(r.timing_score || 0, 0))}</span>` : esc(fmt.signed(r.timing_score || 0, 0))}</td>
                <td class="num">${esc(fmt.money(r.price))}</td>
                <td class="num ${r.change_1d >= 0 ? 'pos' : 'neg'}">${esc(fmt.spct(r.change_1d, 2))}</td>
                <td class="num">${r.owner_applicable === false ? '<span class="muted">n/a</span>' : esc(fmt.num(r.confidence, 0)) + '%'}</td>
                <td>${esc(String(r.trend || 'n/a').toLowerCase())}</td>
                <td>${esc(titleCase(r.risk_rating || ''))}</td>
                ${agents.map((a) => {
                    const s = r.agents?.[a.key];
                    if (missing(s)) return '<td class="num agent-col"><span class="heat na">n/a</span></td>';
                    if (!s.included) return `<td class="num agent-col"><span class="heat na term quiet" tabindex="0" data-tip="${esc(a.name)} (not weighted)">${esc(fmt.signed(s.score, 0))}</span></td>`;
                    return `<td class="num agent-col"><span class="heat" style="${heatStyle(s.score)}">${esc(fmt.signed(s.score, 0))}</span></td>`;
                }).join('')}
            </tr>`;
        }).join('');

        const table = $('#watch-table');
        table.classList.toggle('show-agents', showAgents);
        table.innerHTML = `<thead>${groupRow}<tr>${head}</tr></thead><tbody>${body}</tbody>`;
        $$('#watch-view .seg').forEach((b) => b.classList.toggle('active', (b.dataset.watchView === 'agents') === showAgents));
        $('#watch-view').hidden = !agents.length && !rows.some((r) => (r.top_points || []).length);
        $('#watch-notes').innerHTML = notes.map((n) => `<div class="warning">${esc(n)}</div>`).join('');
        results.hidden = false;
        $('#watch-prescan').hidden = true;
        const rated = rows.filter((r) => where(r) !== 'none' && isNum(r.composite_score));
        $('#watch-strip').innerHTML = rated.length > 1 ? scoreStripHtml({
            items: rated.map((r) => ({ ticker: r.ticker, score: r.composite_score, label: true, note: `${r.ticker} ${fmt.signed(r.composite_score, 1)}, ${levelRelation(r.composite_score, th).text}` })),
            thresholds: th, label: 'Owner scores of the scanned companies',
            caption: window.matchMedia('(hover: none)').matches ? 'Owner scores of this scan on one scale. Tap a dot to read it.' : 'Owner scores of this scan on one scale. Point at a dot to read it, click it to open the company.',
        }) : '';
        layoutStrips($('#watch-strip'));
        const errs = data.errors || [];
        const reason = (e) => (e.error === 'No market data' ? 'Yahoo has no prices under that symbol.' : `${e.error}.`);
        $('#watch-errors').innerHTML = errs.length ? `<div class="warning"><strong>${errs.length === 1 ? `No data for ${esc(errs[0].ticker)}` : `No analysis for ${plural(errs.length, 'ticker', 'tickers')}`}</strong>${errs.length === 1 ? esc(reason(errs[0])) : errs.map((e) => `${esc(e.ticker)}: ${esc(reason(e))}`).join(' ')}
            <div class="small">What to do: check the symbol, or add the exchange suffix (SAP.DE, SHEL.L); open it in Analyze to see the listings Yahoo finds, then replace it here. <span class="chip-row">${errs.map((e) => `<button class="chip" type="button" data-watch-analyze="${esc(e.ticker)}">Find <span class="mono">${esc(e.ticker)}</span> in Analyze</button><button class="chip" type="button" data-watch-remove="${esc(e.ticker)}">Remove <span class="mono">${esc(e.ticker)}</span></button>`).join('')}</span></div></div>` : '';
        syncWeightsNotes();
        $('#watch-footnote').textContent = 'Open a row for the full analysis. Agent columns show each agent\'s score on a -100 to +100 scale.';
    }

    function removeWatchTicker(t) {
        state.watchlist = state.watchlist.filter((x) => x !== t);
        renderChips();
        saveWatchlist();
        if (!state.watchlist.length) $('#watch-add-input').focus();
    }

    function setupWatchlist() {
        const input = $('#watch-add-input'), msg = $('#watch-add-msg');
        const say = (text, invalid) => {
            msg.textContent = text;
            msg.hidden = !text;
            if (invalid) input.setAttribute('aria-invalid', 'true'); else input.removeAttribute('aria-invalid');
        };
        input.addEventListener('input', () => say('', false));
        $('#watch-add-form').addEventListener('submit', (e) => {
            e.preventDefault();
            const added = input.value.split(/[\s,;]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
            const bad = added.filter((t) => !TICKER_RE.test(t));
            const dupes = [...new Set(added.filter((t) => TICKER_RE.test(t) && state.watchlist.includes(t)))];
            const previous = [...state.watchlist];
            added.filter((t) => TICKER_RE.test(t)).forEach((t) => { if (!state.watchlist.includes(t)) state.watchlist.push(t); });
            input.value = bad.join(', ');  // what was not a ticker stays, to be corrected
            const lines = [];
            if (bad.length) lines.push(`${bad.join(', ')} ${bad.length === 1 ? 'is not a ticker symbol' : 'are not ticker symbols'}: use letters, digits and . - ^ =, up to 12 characters.`);
            if (dupes.length) lines.push(`${dupes.join(', ')} ${dupes.length === 1 ? 'is' : 'are'} already on your watchlist.`);
            say(lines.join(' '), bad.length > 0);
            if (state.watchlist.length !== previous.length) saveWatchlist(previous);
        });
        $('#watch-chips').addEventListener('click', (e) => {
            const t = e.target.closest('[data-remove]')?.dataset.remove;
            if (!t) return;
            removeWatchTicker(t);
        });
        $('#watch-scan-button').addEventListener('click', () => scanWatchlist());
        $('#watch-errors').addEventListener('click', (e) => {
            if (e.target.closest('[data-watch-rescan]')) { scanWatchlist(); return; }
            const t = e.target.closest('[data-watch-analyze]')?.dataset.watchAnalyze;
            if (t) { showView('analyze'); runAnalyze(t); return; }
            const r = e.target.closest('[data-watch-remove]')?.dataset.watchRemove;
            if (r) removeWatchTicker(r);
        });
        $('#watch-scan-error').addEventListener('click', (e) => { if (e.target.closest('[data-watch-retry]')) scanWatchlist(); });
        $('#watch-view').addEventListener('click', (e) => {
            const v = e.target.closest('[data-watch-view]')?.dataset.watchView;
            if (!v) return;
            state.watchView = v;
            renderWatchTable();
        });
        $('#watch-table').addEventListener('click', (e) => {
            const th = e.target.closest('th[data-sort]');
            if (th) {
                const key = th.dataset.sort;
                state.watchSort = { key, dir: state.watchSort.key === key ? -state.watchSort.dir : -1 };
                renderWatchTable();
                $(`#watch-table th[data-sort="${key}"] .th-sort`)?.focus();  // the table was rebuilt: keep the keyboard on the header
                return;
            }
            const row = e.target.closest('tr[data-ticker]');
            if (row) { showView('analyze'); runAnalyze(row.dataset.ticker); }
        });
        const sel = $('#watch-refresh');
        sel.value = storageGet('fastvalue-watch-refresh') || '0';
        // One timer, re-armed after every scan, so "Next scan" is always one interval after the last one.
        const next = $('#watch-next');
        const schedule = () => {
            clearTimeout(state.timers.watch);
            const secs = Number(sel.value);
            storageSet('fastvalue-watch-refresh', String(secs));
            next.textContent = '';
            if (!(secs > 0)) return;
            state.timers.watch = setTimeout(async () => {
                if (!document.hidden) await scanWatchlist({ silent: true });
                schedule();
            }, secs * 1000);
            next.textContent = `Next scan ${watchTime(Date.now() + secs * 1000)}`;
        };
        rearmWatchScan = schedule;
        sel.addEventListener('change', schedule);
        schedule();
    }

    // ============================================================ WEIGHTS

    // The browser-side twin of config/weights.py validate_config (plus the "someone must decide" rule of
    // save_scoring_config). Returns the first message and the fields to mark; state.badInputs holds fields
    // the user cleared, whose draft value stays at the previous number.
    const LEVEL_MSG = 'Levels must run from high to low: strong entry, entry, exit, strong exit, each between -100 and +100.';
    const NO_DECIDER_MSG = 'At least one deciding agent must stay on with a weight above 0, or every company comes back not rated.';
    const hasDecider = (d) => (state.agents || []).some((a) => roleOf(a.key, d) === 'owner' && d.enabled[a.key] !== false && ((d.owner_weights || {})[a.key] || 0) > 0);
    function validateDraft(d) {
        const errors = [];
        (state.badInputs || new Map()).forEach((msg, sel) => errors.push({ sel, msg }));
        const levels = (th, prefix) => {
            const v = THRESHOLD_FIELDS.map(([k]) => (th || {})[k]);
            const inRange = v.map((x) => Number.isFinite(x) && x >= -100 && x <= 100);
            const ordered = v[0] >= v[1] && v[1] > v[2] && v[2] >= v[3];
            if (inRange.every(Boolean) && ordered) return;
            THRESHOLD_FIELDS.forEach(([k], i) => { if (!inRange[i] || !ordered) errors.push({ sel: `#${prefix}-${k}`, msg: LEVEL_MSG }); });
        };
        levels(d.owner_thresholds || d.thresholds, 'oth');
        levels(d.thresholds, 'th');
        const bad = (w) => !Number.isFinite(w) || w < 0 || w > 1;
        Object.entries(d.owner_weights || {}).forEach(([k, w]) => { if (bad(w)) errors.push({ sel: `[data-owner-weight-num="${k}"]`, msg: 'Weights run from 0 to 1.' }); });
        Object.values(d.regime_weights || {}).forEach((p) => Object.entries(p).forEach(([k, w]) => { if (bad(w)) errors.push({ sel: `[data-weight-num="${k}"]`, msg: 'Weights run from 0 to 1.' }); }));
        const cap = d.owner_max_ev_to_fcf;
        if (cap != null && !(Number.isFinite(cap) && cap > 0)) errors.push({ sel: '#owner-cap', msg: 'The safety cap is a number above 0, or blank for no cap.' });
        const floor = d.confidence_floor;
        if (!(Number.isFinite(floor) && floor >= 0 && floor <= 1)) errors.push({ sel: '#confidence-floor', msg: 'Weights run from 0 to 1.' });
        if (!hasDecider(d)) errors.push({ sel: '', msg: NO_DECIDER_MSG });
        return { ok: !errors.length, errors, message: errors.length ? errors[0].msg : '' };
    }

    // A note inside a notice area that other code also writes to (suggestions, scan errors).
    function setNote(host, html) {
        if (!host) return;
        let el = host.querySelector('[data-weights-note]');
        if (!html) { if (el) el.remove(); return; }
        if (!el) { el = document.createElement('div'); el.setAttribute('data-weights-note', ''); host.prepend(el); }
        setIf(el, 'innerHTML', html);
    }

    // Which weights the open analysis and scan were made with: unsaved edits are not used, and a save does not rescore them.
    function syncWeightsNotes() {
        const a = state.analysis;
        const w = state.watchData;
        const unsaved = '<div class="warning note"><strong>Unsaved weights</strong>This uses your saved weights. <a href="#weights" data-goto="weights">Review and save your changes</a></div>';
        setNote($('#analyze-notice'), !a ? '' : state.dirty ? unsaved
            : state.analysisStale ? `<div class="warning note"><strong>Scored with your previous weights</strong>Run ${esc(a.ticker)} again to use the saved ones. <button class="btn btn-sm" type="button" data-analyze="${esc(a.ticker)}">Run ${esc(a.ticker)} again</button></div>` : '');
        setNote($('#watch-errors'), !w ? '' : state.dirty ? unsaved
            : w.stale ? '<div class="warning note"><strong>Scanned with your previous weights</strong>Scan again to rank on the saved ones. <button class="btn btn-sm" type="button" data-watch-rescan>Scan again</button></div>' : '');
    }

    // Marks the fields that fail, shows the first reason beside Save, and keeps Save off while the draft is invalid or unchanged.
    function syncValidity() {
        const check = validateDraft(state.draft);
        $$('#view-weights [aria-invalid]').forEach((el) => { el.removeAttribute('aria-invalid'); });
        check.errors.forEach((er) => { const el = er.sel ? $(er.sel) : null; if (el) el.setAttribute('aria-invalid', 'true'); });
        const msg = state.holdMessage || check.message;
        state.holdMessage = '';
        const box = $('#weights-error');
        setIf(box, 'textContent', msg);
        setIf(box, 'hidden', !msg);
        $('#weights-save').disabled = !state.dirty || !check.ok;
        return check;
    }

    function markDirty(dirty = true) {
        state.dirty = dirty;
        const badge = $('#weights-source');
        const [text, className] = dirty ? ['Unsaved changes', 'badge warn']
            : state.hasLocal ? ['Using your saved weights', 'badge accent'] : ['Using defaults', 'badge'];
        setIf(badge, 'textContent', text);
        setIf(badge, 'className', className);
        const tab = $('#tab-weights');
        if (tab) {
            tab.toggleAttribute('data-dirty', dirty);
            tab.setAttribute('aria-label', dirty ? 'Weights, unsaved changes' : 'Weights');
        }
        document.body.classList.toggle('weights-dirty', dirty);
        $('#weights-discard').disabled = !dirty;
        setIf($('#weights-hint'), 'textContent', dirty ? 'Unsaved changes. Nothing uses them until you save.' : 'No changes to save. Saved on this computer.');
        syncValidity();
        syncWeightsNotes();
        renderBkSetup();
    }

    const THRESHOLD_FIELDS = [
        ['strong_buy', 'Strong entry, at or above'], ['buy', 'Entry, at or above'],
        ['sell', 'Exit, at or below'], ['strong_sell', 'Strong exit, at or below'],
    ];
    const LEVEL_NAMES_PLAIN = { strong_buy: 'strong entry level', buy: 'entry level', sell: 'exit level', strong_sell: 'strong exit level' };

    const WEIGHT_HEAD = '<div class="weight-row weight-head" aria-hidden="true"><span>On</span><span>Agent</span><span></span><span>Weight</span><span>Share</span></div>';
    const tagTip = (label, tip) => `<span class="term" tabindex="0" data-tip="${esc(tip)}">${esc(label)}</span>`;

    function weightRow(a, w, on, share, prefix, evidenceOnly = false) {
        const name = shortName(a.name);
        const dis = on ? '' : ' disabled';
        const tags = [esc(a.category)];
        if (evidenceOnly) tags.push(tagTip('evidence only', 'Weight 0: shown with its evidence, adds nothing to the score.'));
        if (a.point_in_time) tags.push(tagTip('replayed in backtests', 'Has point-in-time history, so the backtests can replay it.'));
        if (a.provides_regime) tags.push(tagTip('sets regime', 'Reads the market regime that picks the timing weights.'));
        if (a.slow_note) tags.push(tagTip('slows reviews', `${agentLabel(a)} ${a.slow_note}`));
        return `<div class="weight-row ${on ? '' : 'disabled'}" data-agent="${esc(a.key)}">
            <label class="cb-hit"><input type="checkbox" class="toggle" data-enable="${esc(a.key)}" ${on ? 'checked' : ''} aria-label="Enable ${esc(name)}"></label>
            <div class="weight-name"><strong>${esc(name)}${on ? '' : ' <span class="badge">Off</span>'}</strong><span>${tags.join(' · ')}</span></div>
            <input type="range" min="0" max="1" step="0.01" value="${esc(w)}" data-${prefix}="${esc(a.key)}" aria-label="${esc(name)} weight"${dis}>
            <input type="number" class="input weight-input" min="0" max="1" step="any" value="${esc(w)}" data-${prefix}-num="${esc(a.key)}" aria-label="${esc(name)} weight value"${dis}>
            <span class="weight-share term quiet tip-end" tabindex="0" data-tip="Share of this layer before confidence scaling">${esc(fmt.pct(share, 0))}</span>
        </div>`;
    }

    function updateShares(rows, weights) {
        const total = rows.reduce((sum, r) => sum + (state.draft.enabled[r.dataset.agent] !== false ? (weights[r.dataset.agent] || 0) : 0), 0) || 1;
        rows.forEach((r) => {
            const on = state.draft.enabled[r.dataset.agent] !== false;
            setIf(r.querySelector('.weight-share'), 'textContent', fmt.pct(on ? (weights[r.dataset.agent] || 0) / total : 0, 0));
        });
    }

    // Up/down reorder of one item inside a list (agent order, layer order).
    function moveInList(list, item, dir) {
        const i = list.indexOf(item);
        const j = i + dir;
        if (i < 0 || j < 0 || j >= list.length) return false;
        [list[i], list[j]] = [list[j], list[i]];
        return true;
    }

    const orderedDraftAgents = () => [...state.agents].sort(byAgentOrder((a) => a.key, state.draft));

    function renderRoles(ordered) {
        const draft = state.draft;
        const layers = layerOrder(draft);
        const arrows = (attr, value, i, n, label) => `<span class="order-btns">
                <button type="button" class="icon-btn" ${attr}="${esc(value)}" data-dir="-1" ${i === 0 ? 'disabled' : ''} aria-label="Move ${esc(label)} up" title="Move up">${icon('up')}</button>
                <button type="button" class="icon-btn" ${attr}="${esc(value)}" data-dir="1" ${i === n - 1 ? 'disabled' : ''} aria-label="Move ${esc(label)} down" title="Move down">${icon('down')}</button>
            </span>`;
        setHtml($('#layer-order'), `<span class="inline-label">Order in the analysis ${infoTip('Display order only. It changes nothing in the scores.')}</span>`
            + layers.map((l, i) => `<span class="order-chip">${arrows('data-move-layer', l, i, layers.length, LAYERS[l].short)}${i + 1}. ${esc(LAYERS[l].short)}</span>`).join(''));
        setHtml($('#role-list'), ordered.map((a, i) => {
            const role = roleOf(a.key, draft);
            const weight = role === 'owner' ? ((draft.owner_weights || {})[a.key] || 0)
                : role === 'timing' ? Math.max(0, ...Object.values(draft.regime_weights || {}).map((p) => p[a.key] || 0)) : null;
            const notes = [];
            if (role === 'owner' && a.role !== 'owner') notes.push(`<div class="warning"><strong>Trading signals now vote</strong>${esc(TRADING_WARNING)} <a href="#backtest" data-goto="backtest">Compare your setup in Backtest</a></div>`);
            if (weight === 0 && a.role !== role) notes.push(`<span class="hint">Weight 0 in this layer, so it is not counted yet. Set its weight below.</span>`);
            return `<div class="role-row">
                ${arrows('data-move-agent', a.key, i, ordered.length, a.name)}
                <div class="weight-name"><strong>${esc(shortName(a.name))}</strong><span>${esc(a.category)}${a.role !== role ? ` · default: ${esc(LAYERS[a.role] ? LAYERS[a.role].short.toLowerCase() : a.role)}` : ''}</span></div>
                <div class="segmented role-seg" role="radiogroup" aria-label="Role of ${esc(a.name)}">${LAYER_KEYS.map((r) => `<button type="button" class="seg ${r === role ? 'active' : ''}" data-role-key="${esc(a.key)}" data-role="${r}" role="radio" aria-checked="${r === role}">${esc(LAYERS[r].short)}</button>`).join('')}</div>
                ${notes.length ? `<div class="role-note">${notes.join('')}</div>` : ''}
            </div>`;
        }).join(''));
    }

    function renderWeights() {
        if (!state.draft) return;
        const draft = state.draft;
        draft.roles = draft.roles || {};
        draft.agent_order = draft.agent_order || state.agents.map((a) => a.key);
        draft.layer_order = draft.layer_order || [...LAYER_KEYS];
        state.badInputs = new Map();  // the inputs are rebuilt from the draft, so a cleared field is back to its number
        const ordered = orderedDraftAgents();
        renderRoles(ordered);
        // The roles drawer opens by itself when a role differs from the shipped one.
        const drawer = $('#roles-drawer');
        if (drawer && state.agents.some((a) => roleOf(a.key, draft) !== a.role)) drawer.open = true;
        const ownerAgents = ordered.filter((a) => roleOf(a.key, draft) === 'owner');
        const timingAgents = ordered.filter((a) => roleOf(a.key, draft) === 'timing');
        const ownerWeights = state.draft.owner_weights || (state.draft.owner_weights = {});
        const ownerTotal = ownerAgents.reduce((sum, a) => sum + (state.draft.enabled[a.key] !== false ? (ownerWeights[a.key] || 0) : 0), 0) || 1;
        $('#owner-weights-list').innerHTML = WEIGHT_HEAD + ownerAgents.map((a) => {
            const w = ownerWeights[a.key] ?? 0;
            const on = state.draft.enabled[a.key] !== false;
            return weightRow(a, w, on, on ? w / ownerTotal : 0, 'owner-weight', w === 0);
        }).join('');
        const ownerTh = state.draft.owner_thresholds || (state.draft.owner_thresholds = { ...state.draft.thresholds });
        $('#owner-threshold-grid').innerHTML = THRESHOLD_FIELDS
            .map(([k, label]) => `<div class="field"><label for="oth-${k}">${esc(label)}</label><input id="oth-${k}" class="input" type="number" min="-100" max="100" step="any" value="${esc(ownerTh[k])}" data-owner-threshold="${k}"></div>`).join('')
            + `<div class="field"><label for="owner-cap">Safety cap (EV/FCF)</label><input id="owner-cap" class="input" type="number" min="0" step="any" placeholder="No cap" value="${state.draft.owner_max_ev_to_fcf == null ? '' : esc(state.draft.owner_max_ev_to_fcf)}" data-owner-cap="1"><span class="hint">No new money at or above this multiple. ${infoTip('Margin of safety: at or above this EV/FCF a business gets no new money, whatever its score. The score itself does not change. Leave blank for no cap.')}</span></div>`;

        const regimes = state.regimes;
        // Open on the regime the market is in now (until the user picks one), and say when every profile is the same.
        const now = state.analysis?.market_regime || state.watchData?.rows?.[0]?.regime;
        if (!state.regimePicked && regimes.includes(now)) state.activeRegime = now;
        const same = state.regimeSame ?? regimesAreSame(state.draft.regime_weights, regimes);
        setIf($('#regime-same'), 'checked', same);
        setIf($('#regime-tabs'), 'hidden', same);
        $('#regime-tabs').innerHTML = regimes.map((r) => `<button type="button" class="seg ${r === state.activeRegime ? 'active' : ''}" data-regime="${esc(r)}">${esc(rgName(r))}${r === now ? ' (now)' : ''}</button>`).join('');
        const profile = state.draft.regime_weights[state.activeRegime] || {};
        const enabledTotal = timingAgents.reduce((sum, a) => sum + (state.draft.enabled[a.key] !== false ? (profile[a.key] || 0) : 0), 0) || 1;
        $('#weights-list').innerHTML = WEIGHT_HEAD + timingAgents.map((a) => {
            const w = profile[a.key] ?? 0;
            const on = state.draft.enabled[a.key] !== false;
            return weightRow(a, w, on, on ? w / enabledTotal : 0, 'weight');
        }).join('') || '<p class="hint">No timing agents. Set an agent\'s role to Timing above.</p>';

        $('#threshold-grid').innerHTML = THRESHOLD_FIELDS
            .map(([k, label]) => `<div class="field"><label for="th-${k}">${esc(label)}</label><input id="th-${k}" class="input" type="number" min="-100" max="100" step="any" value="${esc(state.draft.thresholds[k])}" data-threshold="${k}"></div>`).join('');

        const floor = $('#confidence-floor');
        floor.value = state.draft.confidence_floor;
        $('#confidence-floor-value').textContent = fmt.pct(Number(state.draft.confidence_floor), 0);
        syncValidity();
    }

    const regimesAreSame = (profiles, regimes) => {
        const sig = (r) => JSON.stringify(Object.entries(profiles[r] || {}).map(([k, v]) => [k, Math.round(v * 1e4)]).sort());
        return regimes.every((r) => sig(r) === sig(regimes[0]));
    };

    const refreshPreview = debounce(async () => {
        const body = $('#weights-preview-body');
        const a = state.analysis;
        if (!a || !state.draft || !validateDraft(state.draft).ok) return;  // an invalid draft is explained beside Save
        try {
            const res = await api('/api/rescore', {
                method: 'POST',
                body: {
                    assessments: a.agent_assessments.map((x) => ({ agent_key: x.agent_key, agent_name: x.agent_name, score: x.score, confidence: x.confidence, signal: x.signal, data_quality: x.data_quality, raw_data: { regime: x.raw_data?.regime, ev_to_fcf: x.raw_data?.ev_to_fcf, fcf: x.raw_data?.fcf, is_financial: x.raw_data?.is_financial, excluded_sector: x.raw_data?.excluded_sector, sector: x.raw_data?.sector } })),
                    config: state.draft,
                    regime: a.market_regime,
                },
            });
            const ownerNow = res.owner?.composite_score ?? res.composite_score;
            const timingNow = res.timing?.composite_score ?? 0;
            const delta = ownerNow - a.composite_score;
            const draftUnrated = res.decision === 'NOT_RATED';
            const savedUnrated = a.decision === 'NOT_RATED';
            body.classList.remove('muted', 'small');
            body.innerHTML = `
                <p class="muted small">${esc(a.ticker)}, market regime ${esc(rgName(a.market_regime))}: same agent data, draft weights.</p>
                <div class="stat-row" style="margin-top:10px">
                    <div class="stat"><div class="stat-label">Owner, saved</div><div class="stat-value">${esc(fmt.signed(a.composite_score, 1))}</div><div class="stat-sub">${a.decision === 'NOT_RATED' ? 'Not rated' : esc(levelRelation(a.composite_score, a.thresholds).text)}</div></div>
                    <div class="stat"><div class="stat-label">Owner, draft</div><div class="stat-value">${draftUnrated ? 'Not rated' : esc(fmt.signed(ownerNow, 1))}</div><div class="stat-sub">${draftUnrated ? '' : esc(levelRelation(ownerNow, state.draft.owner_thresholds || state.draft.thresholds).text)}</div></div>
                    <div class="stat"><div class="stat-label">Timing, saved</div><div class="stat-value">${esc(fmt.signed((a.timing || {}).score || 0, 0))}</div></div>
                    <div class="stat"><div class="stat-label">Timing, draft</div><div class="stat-value">${esc(fmt.signed(timingNow, 0))}</div></div>
                </div>
                ${draftUnrated ? `<p class="small" style="margin-top:8px">${savedUnrated ? `${esc(a.ticker)} is not rated with the saved or the draft weights.` : `With these settings ${esc(a.ticker)} is not rated: no deciding agent counts.`}</p>`
                    : `<p class="small" style="margin-top:8px">Owner score change <strong class="${delta >= 0 ? 'pos' : 'neg'}">${esc(fmt.signed(delta, 1))}</strong>, confidence ${esc(fmt.num(res.owner?.confidence ?? res.confidence, 0))}%.</p>`}`;
        } catch (err) {
            body.innerHTML = `<span class="neg">${esc(err.message)}</span>`;
        }
    }, 250);

    // A saved or reset config changes the display order now; roles and weights apply to the next analysis.
    function applySavedConfig(res) {
        state.config = res.config;
        state.draft = clone(res.config);
        state.regimeSame = undefined;
        // What is open was made with the weights before this save: say so, and offer to redo it.
        if (state.analysis) state.analysisStale = true;
        if (state.watchData) state.watchData.stale = true;
        const sbtEntry = $('#sbt-entry');
        if (sbtEntry && isNum(res.config.thresholds?.buy)) sbtEntry.value = res.config.thresholds.buy;
        const sbtExit = $('#sbt-exit');
        if (sbtExit && isNum(res.config.thresholds?.sell)) sbtExit.value = res.config.thresholds.sell;
        if (Array.isArray(res.backtestable)) { state.backtestable = res.backtestable; renderSbtAgents(); }
        if (state.analysis) renderAnalysis(state.analysis);
        renderBkSetup();
    }

    function setupWeights() {
        const view = $('#view-weights');
        view.addEventListener('click', (e) => {
            const r = e.target.closest('[data-regime]')?.dataset.regime;
            if (r) { state.activeRegime = r; state.regimePicked = true; renderWeights(); return; }
            if (!state.draft) return;
            const agentBtn = e.target.closest('[data-move-agent]');
            const layerBtn = e.target.closest('[data-move-layer]');
            const roleBtn = e.target.closest('[data-role-key]');
            if (agentBtn || layerBtn) {
                const moved = agentBtn
                    ? moveInList(state.draft.agent_order, agentBtn.dataset.moveAgent, Number(agentBtn.dataset.dir))
                    : moveInList(state.draft.layer_order, layerBtn.dataset.moveLayer, Number(layerBtn.dataset.dir));
                if (moved) { markDirty(true); renderWeights(); } // display order only: no rescore
            } else if (roleBtn) {
                const key = roleBtn.dataset.roleKey;
                const role = roleBtn.dataset.role;
                if (roleOf(key, state.draft) === role) return;
                const deciders = state.agents.filter((a) => a.key !== key && roleOf(a.key, state.draft) === 'owner');
                if (role !== 'owner' && roleOf(key, state.draft) === 'owner' && !deciders.length) {
                    toast('At least one agent must decide, or every ticker is not rated.', 'error');
                    return;
                }
                state.draft.roles = { ...state.draft.roles, [key]: role };
                markDirty(true);
                renderWeights();
                refreshPreview();
            }
        });
        view.addEventListener('input', (e) => {
            const t = e.target;
            if (!state.draft) return;
            const ownerWeights = state.draft.owner_weights || (state.draft.owner_weights = {});
            const bad = state.badInputs || (state.badInputs = new Map());
            // A weight snaps into 0..1 and the slider and the number show the same value; a cleared field keeps the old draft value.
            const weightEdit = (key, isOwner) => {
                const numSel = `[data-${isOwner ? 'owner-weight' : 'weight'}-num="${key}"]`;
                const raw = t.value === '' ? NaN : Number(t.value);
                if (!Number.isFinite(raw)) { bad.set(numSel, 'Weights run from 0 to 1.'); return; }
                bad.delete(numSel);
                const v = Math.round(Math.min(1, Math.max(0, raw)) * 1e4) / 1e4;
                const targets = isOwner ? [ownerWeights] : (state.regimeSame ? Object.values(state.draft.regime_weights) : [state.draft.regime_weights[state.activeRegime]]);
                targets.forEach((p) => { p[key] = v; });
                const other = t.dataset.weight || t.dataset.ownerWeight ? $(numSel) : $(`[data-${isOwner ? 'owner-weight' : 'weight'}="${key}"]`);
                if (other) other.value = v;
                if (raw !== v && t.type === 'number') t.value = v;
                if (isOwner) updateShares($$('#owner-weights-list .weight-row:not(.weight-head)'), ownerWeights);
                else updateShares($$('#weights-list .weight-row:not(.weight-head)'), state.draft.regime_weights[state.activeRegime]); // no re-render: keeps slider focus
                renderRoles(orderedDraftAgents()); // the weight-0 notes live in the roles card
            };
            if (t.dataset.weight || t.dataset.weightNum) {
                weightEdit(t.dataset.weight || t.dataset.weightNum, false);
            } else if (t.dataset.ownerWeight || t.dataset.ownerWeightNum) {
                weightEdit(t.dataset.ownerWeight || t.dataset.ownerWeightNum, true);
            } else if (t.dataset.threshold || t.dataset.ownerThreshold) {
                const owner = !!t.dataset.ownerThreshold;
                const k = t.dataset.threshold || t.dataset.ownerThreshold;
                const target = owner ? (state.draft.owner_thresholds || (state.draft.owner_thresholds = {})) : state.draft.thresholds;
                const v = t.value === '' ? NaN : Number(t.value);
                if (Number.isFinite(v)) { target[k] = v; bad.delete(`#${t.id}`); } else bad.set(`#${t.id}`, `Enter a number for the ${LEVEL_NAMES_PLAIN[k]}.`);
            } else if (t.dataset.ownerCap) {
                state.draft.owner_max_ev_to_fcf = t.value === '' ? null : Number(t.value);
            } else if (t.id === 'confidence-floor') {
                state.draft.confidence_floor = Number(t.value);
                $('#confidence-floor-value').textContent = fmt.pct(Number(t.value), 0);
            } else {
                return;
            }
            markDirty(true);
            refreshPreview();
        });
        view.addEventListener('change', (e) => {
            if (!state.draft) return;
            if (e.target.id === 'regime-same') {
                // Ticking it copies the open profile to all four, so one edit then changes every regime.
                state.regimeSame = e.target.checked;
                if (state.regimeSame) {
                    const from = state.draft.regime_weights[state.activeRegime];
                    state.regimes.forEach((r) => { state.draft.regime_weights[r] = { ...from }; });
                }
                markDirty(true);
                renderWeights();
                refreshPreview();
                return;
            }
            const num = e.target.matches('.weight-input') && e.target.value === '';
            if (num) { renderWeights(); return; }  // a weight left empty goes back to its number
            const key = e.target.dataset.enable;
            if (!key) return;
            const hadDecider = hasDecider(state.draft);
            state.draft.enabled[key] = e.target.checked;
            if (hadDecider && !hasDecider(state.draft)) {
                state.draft.enabled[key] = true;  // the last deciding agent stays on
                state.holdMessage = NO_DECIDER_MSG;
                renderWeights();
                return;
            }
            markDirty(true);
            renderWeights();
            refreshPreview();
        });
        // Puts a config back (Undo after a save): the saved file, or no file at all when there was none.
        const restoreConfig = async (prev, hadLocal) => {
            try {
                const res = hadLocal ? await api('/api/config', { method: 'PUT', body: prev }) : await api('/api/config/reset', { method: 'POST' });
                applySavedConfig(res);
                state.hasLocal = hadLocal;
                markDirty(false);
                renderWeights();
                refreshPreview();
                toast('Previous weights restored.', 'success');
            } catch (err) {
                toast(`Undo failed: ${err.message}`, 'error');
            }
        };
        $('#weights-save').addEventListener('click', async () => {
            const btn = $('#weights-save');
            if (!validateDraft(state.draft).ok) { syncValidity(); return; }
            const prev = clone(state.config);
            const hadLocal = state.hasLocal;
            setBusy(btn, true);
            try {
                const res = await api('/api/config', { method: 'PUT', body: state.draft });
                applySavedConfig(res);
                state.hasLocal = true;
                setBusy(btn, false);
                markDirty(false);
                renderWeights();
                toast('Weights saved. Analyses and scans from now on use them.', 'success', 9000, { label: 'Undo', onClick: () => restoreConfig(prev, hadLocal) });
            } catch (err) {
                setBusy(btn, false);
                markDirty(state.dirty);
                toast(`Not saved: ${err.message}`, 'error');
            }
        });
        $('#weights-discard').addEventListener('click', () => {
            state.draft = clone(state.config);
            state.regimeSame = undefined;
            markDirty(false);
            renderWeights();
            refreshPreview();
        });
        const resetBox = $('#weights-reset-confirm');
        $('#weights-reset').addEventListener('click', () => {
            resetBox.hidden = false;
            $('#weights-reset-no').focus();
        });
        $('#weights-reset-no').addEventListener('click', () => { resetBox.hidden = true; $('#weights-reset').focus(); });
        $('#weights-reset-yes').addEventListener('click', async () => {
            try {
                const res = await api('/api/config/reset', { method: 'POST' });
                applySavedConfig(res);
                state.hasLocal = false;
                resetBox.hidden = true;
                markDirty(false);
                renderWeights();
                refreshPreview();
                toast('Default weights restored.', 'success');
                $('#weights-reset').focus();
            } catch (err) {
                toast(`Reset failed: ${err.message}`, 'error');
            }
        });
        window.addEventListener('beforeunload', (e) => { if (state.dirty) { e.preventDefault(); e.returnValue = ''; } });
    }

    // ============================================================ SCORE BACKTEST

    // Backtest forms: one message under the run row with the field marked, in English and with its fix; no toast and no
    // native bubble (the forms are novalidate). Always returns false, so a check can `return formError(...)`.
    function formError(form, message, field) {
        const box = form.querySelector('[data-form-error]');
        form.querySelectorAll('[aria-invalid]').forEach((f) => { f.removeAttribute('aria-invalid'); f.removeAttribute('aria-describedby'); });
        if (!box) return false;
        box.textContent = message || '';
        box.hidden = !message;
        if (message && field) {
            field.setAttribute('aria-invalid', 'true');
            field.setAttribute('aria-describedby', box.id);
            field.focus();
        }
        return false;
    }
    const clearFormError = (form) => formError(form, '');
    const SYMBOL_FIX = 'Check the symbol; London listings end in .L, for example IWDA.L.';
    const notNumberIn = (el, min, max) => {
        const text = el.value.trim();
        const n = Number(text);
        return text === '' || !Number.isFinite(n) || n < min || n > max;
    };
    // The server's wording for a bad symbol, with what to do about it; other messages pass through.
    function runErrorMessage(err, ticker) {
        const text = String((err && err.message) || err || '');
        if (/^No (price history|data available) for/.test(text)) return `${text.replace(/\.$/, '')}. ${SYMBOL_FIX}`;
        if (/^Invalid ticker symbol/.test(text)) return `${ticker || 'That'} is not a ticker symbol. ${SYMBOL_FIX}`;
        return text;
    }

    function renderSbtAgents() {
        const agents = [...state.agents].sort(byAgentOrder((a) => a.key));
        $('#sbt-agents').innerHTML = agents.map((a) => {
            const ok = state.backtestable.includes(a.key);
            const why = a.point_in_time ? 'Not a timing agent in your weights (role or disabled)' : 'No point-in-time history available';
            return `<label class="check ${ok ? '' : 'disabled'}"><input type="checkbox" value="${esc(a.key)}" ${ok ? 'checked' : 'disabled'}>${esc(a.name)}</label>${infoTip(ok ? a.description : why)}`;
        }).join('');
        const noHistory = agents.filter((a) => !a.point_in_time).map((a) => a.name);
        const otherRole = agents.filter((a) => a.point_in_time && !state.backtestable.includes(a.key)).map((a) => a.name);
        $('#sbt-agents-hint').textContent = [
            noHistory.length ? `Not replayable without look-ahead: ${noHistory.join(', ')}. Their data is only available as of today.` : '',
            otherRole.length ? `Not in your entry timing (role or disabled in the Weights tab): ${otherRole.join(', ')}.` : '',
        ].filter(Boolean).join(' ');
    }

    function setupScoreBacktest() {
        $('#sbt-frequency').innerHTML = state.frequencies.map((f) => `<option value="${esc(f)}" ${f === 'weekly' ? 'selected' : ''}>${esc(titleCase(f))}</option>`).join('');
        $('#sbt-entry').value = state.config?.thresholds?.buy ?? 15;
        $('#sbt-exit').value = state.config?.thresholds?.sell ?? 0;
        renderSbtAgents();

        $('#sbt-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const form = e.currentTarget;
            const ticker = $('#sbt-ticker').value.trim().toUpperCase();
            if (!TICKER_RE.test(ticker)) return formError(form, runErrorMessage({ message: 'Invalid ticker symbol' }, ticker), $('#sbt-ticker'));
            if (notNumberIn($('#sbt-entry'), -100, 100)) return formError(form, 'The entry score goes from -100 to 100.', $('#sbt-entry'));
            if (notNumberIn($('#sbt-exit'), -100, 100)) return formError(form, 'The exit score goes from -100 to 100.', $('#sbt-exit'));
            if (notNumberIn($('#sbt-horizon'), 1, 252)) return formError(form, 'The calibration horizon goes from 1 to 252 bars.', $('#sbt-horizon'));
            if ($('#sbt-years').value.trim() !== '' && notNumberIn($('#sbt-years'), 1, 20)) return formError(form, 'Years go from 1 to 20, or leave it empty for all.', $('#sbt-years'));
            if (!$$('#sbt-agents input:checked').length) return formError(form, 'Tick at least one agent to replay.', $('#sbt-agents input'));
            clearFormError(form);
            const btn = $('#sbt-run');
            setBusy(btn, true);
            try {
                const years = $('#sbt-years').value;
                const data = await api('/api/score-backtest', {
                    method: 'POST',
                    body: {
                        ticker: $('#sbt-ticker').value.trim().toUpperCase(),
                        frequency: $('#sbt-frequency').value,
                        entry_threshold: Number($('#sbt-entry').value),
                        exit_threshold: Number($('#sbt-exit').value),
                        forward_days: Number($('#sbt-horizon').value) || 20,
                        years: years ? Number(years) : null,
                        agents: $$('#sbt-agents input:checked').map((i) => i.value),
                    },
                });
                state.sbt = data;
                renderSbt(data);
            } catch (err) {
                formError(form, runErrorMessage(err, ticker), /ticker|price history|data available/i.test(err.message) ? $('#sbt-ticker') : null);
            } finally {
                setBusy(btn, false);
            }
        });
    }

    // [label, value, sub, valueCls, subCls]: a label of [text, tip] gets a tip; the sub carries the comparison with its tone.
    const statLabel = (l) => (Array.isArray(l) ? tagTip(l[0], l[1]) : esc(l));
    function statCards(items) {
        return items.map(([label, value, sub, cls, subCls]) => `<div class="stat"><div class="stat-label">${statLabel(label)}</div><div class="stat-value ${cls || ''}">${esc(value)}</div>${sub ? `<div class="stat-sub ${subCls || ''}">${esc(sub)}</div>` : ''}</div>`).join('');
    }

    const STAT_TIPS = {
        cagr: 'Compound annual growth: the yearly return that turns the start value into the end value.',
        sharpeCash: 'Yearly return above 3-month T-bills for each unit of volatility. Higher is better.',
        sharpe: 'Yearly return for each unit of volatility, with no cash rate deducted. Higher is better.',
        sortino: 'Like Sharpe, but only falls count as risk.',
        drawdown: 'The deepest fall from a previous high.',
        profitFactor: 'Gains of winning trades divided by losses of losing trades.',
        weaker: 'The worse of the two halves against buy and hold. A setup has to win in both to count.',
        same: 'How many other setups in the grid traded exactly the same.',
        evFcf: 'Enterprise value over free cash flow: what you pay for each dollar of cash the business makes in a year.',
    };
    // The comparison with buy and hold as words with their tone, so the value itself stays neutral: [sub, valueCls, subCls].
    function againstBuyHold(mine, theirs, show, [better, worse] = ['ahead of', 'behind']) {
        if (!isNum(mine) || !isNum(theirs)) return [`buy and hold ${show(theirs)}`, '', ''];
        const up = mine >= theirs;
        return [`${up ? better : worse} buy and hold (${show(theirs)})`, '', up ? 'pos' : 'neg'];
    }

    function renderSbt(d) {
        $('#sbt-results').hidden = false;
        const s = d.strategy_stats, b = d.benchmark_stats;
        const notes = [...(d.notes || [])];
        $('#sbt-notes').innerHTML = notes.map((n) => `<div class="warning note"><strong>Note</strong>${esc(n)}</div>`).join('')
            + `<div class="muted small">${esc(d.ticker)}, ${esc(d.evaluations)} ${esc(d.frequency)} evaluations from ${esc(d.date_range.start)} to ${esc(d.date_range.end)} using ${esc(d.agents_used.map((a) => a.name).join(', '))}.</div>`;
        $('#sbt-stats').innerHTML = statCards([
            ['Total return', fmt.pct(s.total_return), ...againstBuyHold(s.total_return, b.total_return, fmt.pct)],
            [['CAGR', STAT_TIPS.cagr], fmt.pct(s.cagr), `buy and hold ${fmt.pct(b.cagr)}`],
            [['Sharpe', STAT_TIPS.sharpe], fmt.num(s.sharpe), ...againstBuyHold(s.sharpe, b.sharpe, fmt.num)],
            [['Max drawdown', STAT_TIPS.drawdown], fmt.pct(s.max_drawdown), ...againstBuyHold(s.max_drawdown, b.max_drawdown, fmt.pct, ['shallower than', 'deeper than'])],
            ['Win rate', fmt.pct(s.win_rate, 0), `${plural(s.num_trades, 'trade', 'trades')}`],
            [['Profit factor', STAT_TIPS.profitFactor], fmt.ratio(s.profit_factor), ''],
            ['Time invested', fmt.pct(s.exposure, 0), ''],
        ]);

        const cal = d.calibration;
        $('#sbt-calibration-caption').textContent = `${cal.horizon_bars}-bar forward return after each evaluation`;
        const cth = cal.thresholds || {};
        const calRange = (dec) => {
            if (!['strong_buy', 'buy', 'sell', 'strong_sell'].every((k) => isNum(cth[k]))) return 'n/a';
            const f = (k) => fmt.signed(cth[k], 0);
            return { STRONG_BUY: `at or above ${f('strong_buy')}`, BUY: `${f('buy')} to below ${f('strong_buy')}`, HOLD: `above ${f('sell')} to below ${f('buy')}`,
                SELL: `above ${f('strong_sell')} to ${f('sell')}`, STRONG_SELL: `at or below ${f('strong_sell')}` }[dec] || 'n/a';
        };
        const rows = cal.buckets.map((bk) => `<tr>
            <td>${esc(calRange(bk.decision))}</td>
            <td class="num">${esc(bk.count)}</td>
            <td class="num ${bk.mean_forward_return >= 0 ? 'pos' : 'neg'}">${esc(fmt.spct(bk.mean_forward_return, 2))}</td>
            <td class="num">${esc(fmt.spct(bk.median_forward_return, 2))}</td>
            <td class="num">${esc(fmt.pct(bk.hit_rate, 0))}</td>
        </tr>`).join('');
        $('#sbt-calibration').innerHTML = `<thead><tr><th>Score</th><th class="num">Count</th><th class="num">Mean fwd</th><th class="num">Median fwd</th><th class="num">Hit rate</th></tr></thead>
            <tbody>${rows}<tr><td class="muted">All evaluations</td><td class="num">${esc(cal.observations)}</td><td class="num">${esc(fmt.spct(cal.baseline_mean_forward_return, 2))}</td><td></td><td class="num">${esc(fmt.pct(cal.baseline_hit_rate, 0))}</td></tr></tbody>`;
        const ic = cal.information_coefficient;
        $('#sbt-ic').innerHTML = isNum(ic)
            ? `Rank correlation between score and forward return (information coefficient): <strong class="${ic > 0 ? 'pos' : 'neg'}">${esc(fmt.num(ic, 3))}</strong>. ${ic > 0.05 ? 'Higher scores were followed by better returns.' : ic < -0.05 ? 'Higher scores were followed by worse returns: the current weights or thresholds are not predictive at this horizon.' : 'No meaningful relationship at this horizon.'}`
            : 'Not enough observations for a rank correlation.';

        $('#sbt-trades-caption').textContent = `${plural(d.trades.length, 'trade', 'trades')}`;
        $('#sbt-trades').innerHTML = `<thead><tr><th>Entry</th><th class="num">Price</th><th>Exit</th><th class="num">Price</th><th class="num">Return</th><th class="num">Bars</th></tr></thead><tbody>${
            [...d.trades].reverse().map((t) => `<tr>
                <td>${esc(t.entry_date)}</td><td class="num">${esc(fmt.money(t.entry_price))}</td>
                <td>${t.open ? '<span class="badge accent">Open</span>' : esc(t.exit_date)}</td><td class="num">${esc(fmt.money(t.exit_price))}</td>
                <td class="num ${t.return >= 0 ? 'pos' : 'neg'}">${esc(fmt.spct(t.return, 1))}</td><td class="num">${esc(t.bars_held)}</td></tr>`).join('')
        }</tbody>`;

        renderSbtCharts(d);
        $('#sbt-results').scrollIntoView({ block: 'start' });
    }

    function renderSbtCharts(d) {
        const eq = makeChart('sbt-equity-chart');
        const strategy = eq.addLineSeries({ color: cssVar('--series-1'), lineWidth: 2, priceFormat: MONEY_FORMAT });
        strategy.setData(d.equity);
        const buyHold = eq.addLineSeries({ color: cssVar('--series-8'), lineWidth: 1.5, lineStyle: 2, priceFormat: MONEY_FORMAT });
        buyHold.setData(d.benchmark_equity);
        eq.timeScale().fitContent();
        legend('sbt-equity-legend', [
            { key: 'strategy', label: 'Score strategy', color: cssVar('--series-1'), series: strategy },
            { key: 'buy_hold', label: 'Buy and hold', color: cssVar('--series-8'), series: buyHold },
        ], { chart: 'sbt-equity-chart' });

        const sc = makeChart('sbt-score-chart');
        const series = sc.addBaselineSeries({
            baseValue: { type: 'price', price: 0 },
            topLineColor: cssVar('--pos'), bottomLineColor: cssVar('--neg'),
            topFillColor1: 'transparent', topFillColor2: 'transparent', bottomFillColor1: 'transparent', bottomFillColor2: 'transparent',
            lineWidth: 1.5,
            autoscaleInfoProvider: () => ({ priceRange: { minValue: -100, maxValue: 100 } }),
        });
        series.setData(d.score_series);
        const entryLine = series.createPriceLine({ price: d.entry_threshold, color: cssVar('--pos'), lineWidth: 1, lineStyle: 2, title: 'Entry' });
        const exitLine = series.createPriceLine({ price: d.exit_threshold, color: cssVar('--neg'), lineWidth: 1, lineStyle: 2, title: 'Exit' });
        sc.timeScale().fitContent();
        legend('sbt-score-legend', [
            { key: 'composite', label: 'Composite', color: cssVar('--pos'), series },
            { key: 'thresholds', label: 'Entry / exit thresholds', color: cssVar('--text-3'), lines: [entryLine, exitLine] },
        ], { chart: 'sbt-score-chart' });
    }

    // ============================================================ STRATEGY LAB

    function setupLab() {
        const select = $('#bt-strategy');
        const label = (k) => state.strategies[k].label || titleCase(k);
        select.innerHTML = Object.keys(state.strategies).sort((a, b) => label(a).localeCompare(label(b)))
            .map((k) => `<option value="${esc(k)}">${esc(label(k))}</option>`).join('');
        if (state.strategies.sma_crossover) select.value = 'sma_crossover';
        const onChange = () => {
            const info = state.strategies[select.value];
            if (!info) return;
            $('#bt-description').textContent = info.description;
            // A percent field shows whole percents (5 for 0.05) and is divided by 100 on submit.
            const shown = (spec, v) => (spec.percent && v !== undefined ? Number((v * 100).toFixed(6)) : v);
            $('#bt-params').innerHTML = Object.entries(info.params || {}).map(([key, spec]) => `
                <div class="field"><label for="param-${esc(key)}">${esc(spec.label || key)}</label>
                <input class="input" type="number" id="param-${esc(key)}" data-param="${esc(key)}" data-type="${esc(spec.type)}"${spec.percent ? ' data-percent' : ''} value="${esc(shown(spec, spec.default))}"
                    ${spec.min !== undefined ? `min="${esc(shown(spec, spec.min))}"` : ''} ${spec.max !== undefined ? `max="${esc(shown(spec, spec.max))}"` : ''} step="${spec.percent ? '0.1' : spec.type === 'float' ? '0.01' : '1'}"></div>`).join('');
        };
        select.addEventListener('change', onChange);
        onChange();

        $('#bt-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const form = e.currentTarget;
            const ticker = $('#bt-ticker').value.trim().toUpperCase();
            if (!TICKER_RE.test(ticker)) return formError(form, runErrorMessage({ message: 'Invalid ticker symbol' }, ticker), $('#bt-ticker'));
            for (const input of $$('[data-param]')) {
                const min = input.min === '' ? -Infinity : Number(input.min);
                const max = input.max === '' ? Infinity : Number(input.max);
                if (notNumberIn(input, min, max)) {
                    const name = $(`label[for="${input.id}"]`).textContent;
                    return formError(form, Number.isFinite(min) && Number.isFinite(max) ? `${name} goes from ${min} to ${max}.` : `Enter a number for ${name.toLowerCase()}.`, input);
                }
            }
            clearFormError(form);
            const btn = $('#bt-run');
            setBusy(btn, true);
            try {
                const params = {};
                $$('[data-param]').forEach((i) => {
                    const v = i.dataset.type === 'float' ? parseFloat(i.value) : parseInt(i.value, 10);
                    params[i.dataset.param] = i.hasAttribute('data-percent') ? v / 100 : v;
                });
                const data = await api('/api/backtest', { method: 'POST', body: { ticker, strategy: select.value, params } });
                state.bt = data;
                renderBt(data);
            } catch (err) {
                formError(form, runErrorMessage(err, ticker), /ticker|price history|data available/i.test(err.message) ? $('#bt-ticker') : null);
            } finally {
                setBusy(btn, false);
            }
        });

    }

    function renderBt(d) {
        $('#bt-results').hidden = false;
        const s = d.strategy_stats, b = d.benchmark_stats;
        // Fewer than 20 trades is too thin a sample for a win rate or a profit factor (N-50); no calculation changes.
        const thin = s.num_trades < 20;
        $('#bt-stats').innerHTML = statCards([
            ['Total return', fmt.pct(s.total_return), ...againstBuyHold(s.total_return, b.total_return, fmt.pct)],
            [['CAGR', STAT_TIPS.cagr], fmt.pct(s.cagr), `buy and hold ${fmt.pct(b.cagr)}`],
            [['Sharpe', STAT_TIPS.sharpe], fmt.num(s.sharpe), ...againstBuyHold(s.sharpe, b.sharpe, fmt.num)],
            [['Sortino', STAT_TIPS.sortino], fmt.num(s.sortino), `buy and hold ${fmt.num(b.sortino)}`],
            [['Max drawdown', STAT_TIPS.drawdown], fmt.pct(s.max_drawdown), ...againstBuyHold(s.max_drawdown, b.max_drawdown, fmt.pct, ['shallower than', 'deeper than'])],
            ['Win rate', fmt.pct(s.win_rate, 0), thin ? `${plural(s.num_trades, 'trade', 'trades')}: too few to judge` : `${plural(s.num_trades, 'trade', 'trades')}`],
            [['Profit factor', STAT_TIPS.profitFactor], fmt.ratio(s.profit_factor), ''],
        ]);
        $('#bt-thin-hint').hidden = !thin;
        $('#bt-chart-title').textContent = `${d.ticker}: ${(state.strategies[d.strategy] || {}).label || titleCase(d.strategy)}`;
        renderBtCharts(d);
    }

    function renderBtCharts(d) {
        const chart = makeChart('bt-price-chart');
        const up = cssVar('--pos'), down = cssVar('--neg');
        const candles = chart.addCandlestickSeries({ upColor: up, downColor: down, borderUpColor: up, borderDownColor: down, wickUpColor: up, wickDownColor: down });
        candles.setData(d.chart.candles);
        const markers = [
            ...d.chart.buy_signals.map((m) => ({ ...m, color: up })),
            ...d.chart.sell_signals.map((m) => ({ ...m, color: down })),
        ].sort((a, b) => a.time.localeCompare(b.time));
        if (markers.length) candles.setMarkers(markers);
        const items = [];
        (d.chart.overlays.lines || []).forEach((l) => {
            const color = seriesColor(l.color);
            const line = chart.addLineSeries({ color, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
            line.setData(l.data);
            items.push({ key: l.name, label: l.name, color, series: line });
        });
        legend('bt-legend', items, { chart: 'bt-price-chart', base: true });
        chart.timeScale().fitContent();

        const sub = (d.chart.overlays.subplots || [])[0];
        $('#bt-subplot-card').hidden = !sub;
        if (sub) {
            $('#bt-subplot-title').textContent = sub.name;
            const sc = makeChart('bt-subplot-chart');
            const main = sc.addLineSeries({ color: cssVar('--series-3'), lineWidth: 1.5 });
            main.setData(sub.data);
            (sub.levels || []).forEach((lv) => main.createPriceLine({ price: lv, color: cssVar('--text-3'), lineWidth: 1, lineStyle: 2 }));
            if (sub.signal_data) sc.addLineSeries({ color: cssVar('--series-2'), lineWidth: 1.5 }).setData(sub.signal_data);
            if (sub.histogram) {
                sc.addHistogramSeries({ priceScaleId: 'right' }).setData(sub.histogram.map((p) => ({ ...p, color: p.value >= 0 ? up : down })));
            }
            sc.timeScale().fitContent();
        } else {
            destroyChart('bt-subplot-chart');
        }

        const eq = makeChart('bt-equity-chart');
        const strategy = eq.addLineSeries({ color: cssVar('--series-1'), lineWidth: 2, priceFormat: MONEY_FORMAT });
        strategy.setData(d.equity);
        const buyHold = eq.addLineSeries({ color: cssVar('--series-8'), lineWidth: 1.5, lineStyle: 2, priceFormat: MONEY_FORMAT });
        buyHold.setData(d.benchmark_equity);
        eq.timeScale().fitContent();
        legend('bt-equity-legend', [
            { key: 'strategy', label: 'Strategy', color: cssVar('--series-1'), series: strategy },
            { key: 'buy_hold', label: 'Buy and hold', color: cssVar('--series-8'), dash: true, series: buyHold },
        ], { chart: 'bt-equity-chart' });
    }

    // ============================================================ PORTFOLIO

    const pfCcy = () => (state.portfolio && state.portfolio.currency) || 'EUR';

    const IMPORT_JOB_KEY = 'fastvalue-import-job';
    const clearImportJob = () => { try { localStorage.removeItem(IMPORT_JOB_KEY); } catch (_) { /* private mode */ } };

    // A reload during an import: show its progress again and wait for it, so the holdings and the import card load together.
    async function resumeImport(job) {
        showImportProgress(true, { stage: 'Finishing your import' });
        try {
            await pollJob(job, (p) => showImportProgress(true, { ...p, stage: p.stage || 'Finishing your import' }));
            state.pfJustImported = true;
            return true;
        } catch (_) { /* the job is gone or failed: the stored import below is the truth */ return false; } finally {
            clearImportJob();
            showImportProgress(false);
        }
    }

    async function loadPortfolio() {
        const pending = storageGet(IMPORT_JOB_KEY);
        const resumed = pending ? await resumeImport(pending) : false;
        try {
            const [data, imp] = await Promise.all([api('/api/portfolio'), api('/api/portfolio/import')]);
            state.portfolio = data;
            state.portfolioLimits = data.limits || {};
            state.pfImport = imp;
            renderPositions();
            renderImport();
            if (resumed && imp.imported) $('#pf-import').scrollIntoView({ block: 'start' });
        } catch (err) {
            toast(`Could not load the portfolio: ${err.message}`, 'error');
        }
        loadLastReview();
        api('/api/monitor').then(renderMonitor).catch(() => { $('#pf-monitor').hidden = true; });
    }

    // The last review stays on screen after a reload while the holdings and cash are unchanged (kept on this machine by the server).
    async function loadLastReview() {
        if (state.pfReview || state.pfJustImported) return;  // right after an import the next step is the Review button
        try {
            const saved = await api('/api/portfolio/review/last');
            if (!saved || !saved.reviewed_at || state.pfReview) return;
            state.pfReview = saved;
            renderReview(saved);
            const at = new Date(saved.reviewed_at);
            $('#pf-todo-caption').textContent = `From your last review, ${at.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })} ${at.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}. Review again for fresh prices.`;
        } catch (_) { /* no saved review: the form is enough */ }
    }

    // ------------------------------------------------ daily check (core/monitor.py)

    const MONITOR_TIP = 'The daily check reviews every holding and the watchlist each weekday after the US close, stores the ratings and news tone so the 30/90-day changes and the news test have history, and raises an alert when a holding\'s action or a watchlist owner decision changes. Entry timing never raises an alert: it is not a rule.';

    function monitorAlert(kind, title, text, fixHtml) {
        return `<div class="alert-item ${kind === 'risk' ? 'risk' : ''}">
                <span class="alert-kind">${kind === 'risk' ? 'Risk' : 'Note'}</span><span class="alert-title">${esc(title)}</span>${esc(text)}
                ${fixHtml ? `<div class="alert-fix"><strong>What to do:</strong> ${fixHtml}</div>` : ''}</div>`;
    }

    function renderMonitor(m) {
        const card = $('#pf-monitor');
        state.pfMonitor = m || null;
        renderTodoChanges();
        if (!m || !(state.portfolio && (state.portfolio.positions || []).length)) { card.hidden = true; return; }
        const sched = m.schedule || {};
        const alerts = m.alerts || [];
        const latest = alerts.filter((a) => a.at === m.checked_at);
        const earlier = alerts.filter((a) => a.at !== m.checked_at);
        const failed = (m.errors || []).length > 0;
        card.hidden = false;
        const when = m.checked_at ? `Last check ${String(m.checked_at).replace('T', ' ').slice(0, 16)}.` : 'No check has run yet.';
        const plan = sched.installed ? `Scheduled weekdays at ${sched.time || '?'}.`
            : sched.unsupported ? 'Scheduling works on macOS for now; you can run a check whenever you like.'
                : 'Schedule it to review your holdings and watchlist every weekday after the US close and tell you when an action changes.';
        const covered = m.checked_at && sched.installed ? ` It covered ${plural(m.holdings || 0, 'holding', 'holdings')} and ${plural(m.watchlist || 0, 'watchlist name', 'watchlist names')}.` : '';
        $('#pf-monitor-caption').innerHTML = `<span>${esc(`${when} ${plan}${covered}`)}</span> ${infoTip(MONITOR_TIP)}`;
        setHtml($('#pf-monitor-actions'), `${sched.installed ? '<button class="btn btn-ghost btn-sm" type="button" data-monitor-uninstall>Stop the daily check</button>'
            : sched.unsupported ? '' : '<button class="btn btn-sm" type="button" data-monitor-install>Schedule the daily check</button>'}
            <button class="btn btn-sm btn-ghost" type="button" data-monitor-run>Run a check now</button>`);
        const fragment = (link) => {
            const hash = String(link || '').split('#')[1] || '';
            return /^(portfolio|analyze)=[A-Za-z0-9.\-=^]{1,12}$/.test(hash) ? `#${hash}` : null;
        };
        const item = (a) => {
            const href = fragment(a.link);
            const day = String(a.at || '').slice(0, 10);
            return monitorAlert(a.kind, `${day} ${a.title}`, a.text, `${esc(a.fix)}${href ? ` <a href="${esc(href)}">Open</a>` : ''}`);
        };
        const parts = [];
        if (failed) {
            parts.push(monitorAlert('risk', 'The last check could not analyze everything', m.errors.slice(0, 4).join('; '),
                'Usually a ticker without prices: fix its ticker in Your holdings (or the import card), then review again.'));
        }
        latest.slice(0, 12).forEach((a) => parts.push(item(a)));
        $('#pf-monitor-alerts').innerHTML = parts.join('');
        const drawer = $('#pf-monitor-earlier');
        drawer.hidden = !earlier.length;
        $('#pf-monitor-earlier-summary').textContent = `Earlier alerts (${earlier.length})`;
        $('#pf-monitor-earlier-alerts').innerHTML = earlier.slice(0, 12).map(item).join('');
    }

    // The footer's "Your data and costs": what is free, what leaves the computer (the list comes from
    // config/data_sources.py) and the one control that deletes everything FastValue keeps.
    const DELETE_ALL_CONFIRM = 'Delete everything FastValue keeps about you on this computer: holdings and import, watchlist, saved weights, analysis history, daily check and alerts, SEC contact and shareable pages? Cached market data stays. This cannot be undone.';
    function setupAbout() {
        const toggle = $('#about-toggle');
        const panel = $('#about-data');
        toggle.addEventListener('click', () => {
            const open = panel.hidden;
            panel.hidden = !open;
            toggle.setAttribute('aria-expanded', String(open));
            if (open) panel.scrollIntoView({ block: 'nearest' });
        });
        $('#about-sources').innerHTML = (state.dataSources || []).map((s) => `<li><strong>${esc(s.name)}</strong> (${esc(s.host)})${s.optional ? ', optional' : ''}: ${esc(s.why)}. Sends ${esc(s.sends)}.</li>`).join('');
        $('#about-version').textContent = state.version ? `FastValue ${state.version}` : '';
        $('#data-delete').addEventListener('click', async (e) => {
            if (!window.confirm(DELETE_ALL_CONFIRM)) return;
            const btn = e.currentTarget;
            setBusy(btn, true);
            try {
                await api('/api/data', { method: 'DELETE' });
                location.reload();
            } catch (err) {
                setBusy(btn, false);
                toast(`Nothing was deleted: ${err.message}`, 'error');
            }
        });
    }

    // Schedule, stop or run the daily check from the card (the server does what `python -m core.monitor` does).
    function setupMonitor() {
        $('#pf-monitor').addEventListener('click', async (e) => {
            const btn = e.target.closest('[data-monitor-install], [data-monitor-uninstall], [data-monitor-run]');
            if (!btn || btn.disabled) return;
            const run = 'monitorRun' in btn.dataset;
            const path = run ? '/api/monitor/run' : `/api/monitor/${'monitorInstall' in btn.dataset ? 'install' : 'uninstall'}`;
            setBusy(btn, true);
            try {
                const res = await api(path, { method: 'POST' });
                const status = run ? await pollJob(res.job, () => {}) : res;
                renderMonitor(status);
                const sched = status.schedule || {};
                announce(run ? 'Check finished.' : sched.installed ? `Daily check scheduled weekdays at ${sched.time}.` : 'Daily check stopped.');
                if (run) toast('Check finished.', 'success');
            } catch (err) {
                toast(`${run ? 'The check failed' : 'The daily check could not be changed'}: ${err.message}`, 'error');
            } finally {
                setBusy(btn, false);
            }
        });
    }

    // ------------------------------------------------ broker import

    function showImportProgress(on, p = {}) {
        setIf($('#pf-import-progress'), 'hidden', !on);
        setIf($('#pf-import-button'), 'hidden', on);
        if (!on) return;
        const share = p.total ? Math.min(1, p.done / p.total) : 0;
        $('#pf-import-progress-fill').style.width = `${(share * 100).toFixed(0)}%`;
        setIf($('#pf-import-progress-text'), 'textContent', p.stage || 'Working');
    }

    function brokerLabel(key) {
        const b = (state.brokers || []).find((x) => x.key === key);
        return b ? b.label : 'broker';
    }

    // One privacy promise for every broker, said once next to the import button.
    const IMPORT_PRIVACY = 'Read on this computer; the file itself is not kept. Names, IBANs, transfer texts and card merchants are dropped as it is read. To find prices, only instrument identifiers (ISIN, ticker or fund name) are looked up online.';

    function renderBrokerChoice() {
        const sel = $('#pf-import-broker');
        const list = state.brokers || [];
        if (!list.length) { sel.hidden = true; return; }
        const current = state.brokerChosen || storageGet('fastvalue-broker') || (state.pfImport && state.pfImport.source) || state.brokerDefault || list[0].key;
        sel.innerHTML = list.map((b) => `<option value="${esc(b.key)}">${esc(b.label)}</option>`).join('');
        sel.value = list.some((b) => b.key === current) ? current : list[0].key;
        const chosen = list.find((b) => b.key === sel.value) || list[0];
        const imported = state.pfImport && state.pfImport.imported;
        $('#pf-import-button').textContent = imported ? 'Import a newer export' : (chosen.button || `Import ${chosen.label} export`);
        const hint = $('#pf-import-hint');
        hint.hidden = false;
        hint.innerHTML = `${esc(chosen.hint || '')}${chosen.tip ? ` ${infoTip(chosen.tip)}` : ''}${chosen.template ? ' <a href="/api/brokers/template" download>Download the template</a>' : ''}<br>${esc(IMPORT_PRIVACY)}`;
        setIf($('#pf-template-help'), 'hidden', !chosen.template);
    }

    // The failure stays under the import button until the next try, with the way forward; a file from the other broker is one click away.
    function showImportError(message, other) {
        const act = other ? `<div class="head-actions"><button class="btn btn-sm" type="button" data-import-as="${esc(other.key)}">Import as ${esc(other.label)}</button></div>` : '';
        const host = $('#pf-import-error');
        host.innerHTML = `<div class="warning"><strong>Not imported</strong>${esc(message)}${act}</div>`;
        host.scrollIntoView({ block: 'nearest' });
    }

    const clearImportError = () => { $('#pf-import-error').innerHTML = ''; };

    // One verdict for the import, read by the toast and by the card's summary.
    function importHealth(d) {
        const failed = (d.checks || []).filter((c) => !c.ok).length;
        const attention = (d.instruments || []).filter((r) => !r.usable || (r.status === 'check' && r.source !== 'manual'));
        const notes = [...(d.notes || []), ...(d.warnings || [])].length;
        const issues = [];
        if (failed) issues.push(`${failed} check${failed === 1 ? '' : 's'} failed`);
        if (attention.length) issues.push(`${attention.length} holding${attention.length === 1 ? ' needs' : 's need'} a look`);
        if (notes) issues.push(`${notes} note${notes === 1 ? '' : 's'} to read`);
        return { healthy: !issues.length, failed, attention, issues };
    }

    async function importExport(file) {
        if (!file) return;
        state.lastImportFile = file;
        clearImportError();
        const form = new FormData();
        form.append('file', file);
        form.append('broker', $('#pf-import-broker').value || state.brokerDefault || 'trade_republic');
        setBusy($('#pf-import-button'), true);
        showImportProgress(true, { stage: 'Reading the export' });
        try {
            const res = await fetchLocal('/api/portfolio/import', { method: 'POST', body: form });
            let data = {};
            try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
            if (!res.ok || data.error) {
                const failure = new Error(data.error || `${res.status} ${res.statusText}`);
                if (data.other_broker) failure.other = { key: data.other_broker, label: data.other_broker_label || brokerLabel(data.other_broker) };
                throw failure;
            }
            storageSet(IMPORT_JOB_KEY, data.job);
            const result = await pollJob(data.job, (p) => showImportProgress(true, p));
            clearImportJob();
            state.pfImport = result;
            state.portfolio = null;
            state.pfReview = null;
            state.pfAccount = null;
            state.pfJustImported = true;
            $('#pf-dash').hidden = true;
            $('#pf-todo').hidden = true;
            await loadPortfolio();
            // The stored status carries no notes: keep this import's own (a typed holding it replaced, costs now read in EUR).
            if (state.pfImport && state.pfImport.imported) {
                state.pfImport.notes = result.notes || [];
                renderImport();
            }
            const health = importHealth(state.pfImport || result);
            const lead = `Imported ${plural(result.rows, 'booking', 'bookings')}.`;
            toast(health.healthy ? `${lead} Every check passed. Next: review what to do now.`
                : `${lead} ${health.issues.join(', ')}${health.attention.length ? `: confirm ${health.attention.length === 1 ? 'its ticker' : 'their tickers'} below` : ''}.`,
            health.healthy ? 'success' : (health.failed ? 'error' : 'info'), 8000);
            $('#pf-import').scrollIntoView({ block: 'start' });
            focusResult('#pf-import-title', `Imported ${plural(result.rows, 'booking', 'bookings')}. ${health.healthy ? 'Every check passed.' : health.issues.join(', ')}.`);
        } catch (err) {
            showImportError(err.message, err.other);
        } finally {
            clearImportJob();
            setBusy($('#pf-import-button'), false);
            showImportProgress(false);
            $('#pf-import-file').value = '';
        }
    }

    const SOURCE_STATUS = {
        ok: ['fits', 'pos'], check: ['check', 'warn-text'], mismatch: ['wrong listing', 'neg'], no_prices: ['no prices', 'neg'],
        unmapped: ['not found', 'neg'], not_priced: ['not priced', 'muted'], no_trades: ['no trades to check', 'muted'],
    };

    function sourceStatus(r) {
        const [label, cls] = SOURCE_STATUS[r.status] || [r.status, ''];
        return `<span class="${cls}">${esc(label)}</span>${r.source === 'manual' ? ' <span class="badge">set by you</span>' : ''}`;
    }

    function candidateChips(r) {
        if (r.bond) return `<div class="muted small">${esc(r.note || '')}</div>`;
        const list = (r.candidates || []).map((c) => (typeof c === 'string' ? { ticker: c, median_gap: null } : c)).filter((c) => c && c.ticker);
        if (!list.length) return '';
        return `<div class="chip-row candidate-row"><span class="muted small">Listings found:</span>${infoTip('Fit is the median gap between that listing\'s prices and your trade prices. Above 3% it is probably another share class or currency line. A listing without a fit has no prices to compare with your trades.')}${list.map((c) => `<button class="chip" type="button" data-candidate="${esc(c.ticker)}" data-isin="${esc(r.isin)}"><span class="mono">${esc(c.ticker)}</span>${c.median_gap != null ? ` <span class="small ${c.median_gap <= 0.03 ? 'pos' : c.median_gap <= 0.08 ? 'warn-text' : 'neg'}">fit ${esc(fmt.pct(c.median_gap, 1))}</span>` : ''}</button>`).join('')}</div>`;
    }

    // "Keep KO" while the box holds the ticker already matched to a listing that needs a look; "Use" once it is changed.
    const editLabel = (r, value) => (r.status === 'check' && r.ticker && value === r.ticker ? `Keep ${r.ticker}` : 'Use');

    function tickerEditor(r) {
        return `<form class="inline-edit" data-isin="${esc(r.isin)}" data-matched="${esc(r.status === 'check' ? r.ticker || '' : '')}"><input class="input input-compact input-ticker" value="${esc(r.ticker || '')}" placeholder="Ticker" maxlength="12" spellcheck="false" aria-label="Ticker for ${esc(r.name || r.isin)}"><button class="btn btn-sm" type="submit">${esc(editLabel(r, r.ticker))}</button></form>`;
    }

    // What a listing that "needs a look" means, in the cell that asks about it.
    function checkProblem(r) {
        if (r.status !== 'check' || r.median_gap == null) return `${sourceStatus(r)}${r.median_gap != null ? ` <span class="muted small">fit ${esc(fmt.pct(r.median_gap, 1))}</span>` : ''}`;
        return `<span class="warn-text">Needs a look.</span> Your trade prices sit ${esc(fmt.pct(r.median_gap, 1))} from this listing's daily range. Above 3% it can be another share class or currency line.${r.ticker ? ` Keep ${esc(r.ticker)} if that is the share you bought.` : ''}`;
    }

    // After an import the next step sits first in the card: review, or read the track record. It goes after the first review.
    function renderImportNext(d) {
        const on = !!state.pfJustImported && !state.pfReview;
        setIf($('#pf-import-next'), 'hidden', !on);
        if (on) setIf($('#pf-import-next-text'), 'textContent', `Imported ${plural(d.rows, 'booking', 'bookings')}. Next: see what to do with each holding.`);
    }

    // One primary action in the Portfolio card at a time: import while empty, then review, then none ("Review again" carries it).
    function syncPortfolioState() {
        const p = state.portfolio;
        if (!p) return;
        const holdings = (p.positions || []).length > 0;
        const imported = !!(state.pfImport && state.pfImport.imported);
        const empty = !holdings && !imported;
        const reviewed = !!state.pfReview;
        const bar = !!state.pfJustImported && !reviewed && imported;
        const typing = empty && !!state.pfTypeIn;
        setIf($('#pf-empty'), 'hidden', !empty || typing);
        setIf($('#pf-card-title'), 'hidden', empty && !typing);
        setIf($('#pf-positions-drawer'), 'hidden', empty);
        setIf($('#pf-add-form'), 'hidden', empty && !typing);
        setIf($('#pf-cash-form'), 'hidden', empty && !typing);
        setIf($('#pf-review-button'), 'hidden', empty);
        setIf($('#pf-import-remove'), 'hidden', !imported);  // next to the import button: leaving is as easy to find as arriving
        setIf($('#pf-type-in'), 'hidden', !empty || typing);
        const primary = (el, on) => { if (el) el.classList.toggle('btn-primary', on); };
        primary($('#pf-import-button'), empty);
        primary($('#pf-review-button'), holdings && !reviewed && !bar);
    }

    function renderImport() {
        const d = state.pfImport;
        const card = $('#pf-import');
        renderBrokerChoice();
        if (!d || !d.imported) { card.hidden = true; syncPortfolioState(); return; }
        card.hidden = false;
        $('#pf-import-title').textContent = `${d.source_label || brokerLabel(d.source)} import`;
        const s = d.summary || {};
        const eur = (v, digits = 0) => fmt.money(v, digits, d.currency || 'EUR');
        const years = Object.entries(s.realised_by_year || {});
        $('#pf-import-caption').textContent = `${plural(d.rows, 'booking', 'bookings')}, ${fmtDay(d.period?.first)} to ${fmtDay(d.period?.last)}, imported ${fmtDay(d.imported_at)}`;
        $('#pf-import-stats').innerHTML = statCards([
            ['Cash', eur(s.cash, 2), 'reconciles with every booking'],
            ['Net deposited', eur(s.net_external_flows), `in ${eur(s.deposits)}, out ${eur(-(s.withdrawals || 0) - (s.card_spending || 0))} incl. card`],
            ['Realised P&L, FIFO', eur((s.realised_total || {}).fifo), years.map(([y, v]) => `${y} ${eur(v.fifo)}`).join(' · ')],
            ['Dividends and interest', eur((s.dividends_gross || 0) + (s.interest_gross || 0)), `gross; ${eur(Math.abs(s.taxes))} tax withheld in all`],
            ['Fees', eur(Math.abs(s.fees)), `${plural(s.round_trips_closed || 0, 'closed round trip', 'closed round trips')}`],
            ['Open holdings', String(s.open_positions ?? 'n/a'), `${eur(s.broker_cost_of_open_positions ?? s.cost_of_open_positions)} at cost incl. buy fees`],
        ]);
        $('#pf-import-checks').innerHTML = (d.checks || []).map((c) => `<span class="check-chip term quiet ${c.ok ? 'ok' : 'bad'}" tabindex="0" data-tip="${esc(c.detail)}">${icon(c.ok ? 'check' : 'x')} ${esc(c.name)}${c.ok ? '' : `: ${esc(c.detail)}`}</span>`).join('');
        $('#pf-import-notes').innerHTML = [...(d.notes || []), ...(d.warnings || [])].map((n) => `<div class="warning"><strong>Note</strong>${esc(n)}</div>`).join('');
        const rows = d.instruments || [];
        const health = importHealth(d);
        const attention = health.attention;
        $('#pf-import-attention').innerHTML = attention.length ? `
            <div class="card-head sub"><h4>Holdings that need a ticker</h4><span class="muted small">Counted at cost in the review until they have a price source; private funds and rights stay out</span></div>
            <div class="table-wrap"><table class="table compact stack"><thead><tr><th>Holding</th><th>ISIN</th><th class="num">Cost</th><th>Problem</th><th>Ticker</th></tr></thead><tbody>${
                attention.map((r) => `<tr><td class="span">${esc(r.name || '')}</td><td class="mono small" data-label="ISIN">${esc(r.isin)}</td><td class="num" data-label="Cost">${esc(eur(r.cost))}</td>
                    <td class="span wrap" data-label="Problem">${checkProblem(r)}${r.reason ? `<div class="muted small">${esc(r.reason)}</div>` : ''}${candidateChips(r)}</td>
                    <td class="span" data-label="Ticker">${tickerEditor(r)}</td></tr>`).join('')}</tbody></table></div>` : '';
        const healthy = health.healthy;
        $('#pf-import-caption').textContent += healthy ? ' · every check passed' : '';
        $('#pf-import-body').open = !healthy;
        $('#pf-import-body-summary').textContent = healthy ? 'Totals and checks' : `Totals and checks: ${health.issues.join(', ')}`;
        renderImportNext(d);
        syncPortfolioState();
        $('#pf-import-sources').innerHTML = `<thead><tr><th>Holding</th><th>ISIN</th><th>Ticker</th><th>Currency</th><th class="num">Fit</th><th>Status</th><th>Change</th></tr></thead><tbody>${
            rows.map((r) => `<tr><td class="span">${esc(r.name || '')}</td><td class="mono small" data-label="ISIN">${esc(r.isin)}</td><td class="mono" data-label="Ticker">${esc(r.ticker || '')}</td><td data-label="Currency">${esc(r.currency || '')}</td>
                <td class="num" data-label="Fit">${r.median_gap != null ? esc(fmt.pct(r.median_gap, 1)) : ''}</td><td class="span" data-label="Status">${sourceStatus(r)}</td><td class="span" data-label="Change">${tickerEditor(r)}</td></tr>`).join('')}</tbody>`;
    }

    async function setInstrumentTicker(isin, ticker) {
        try {
            const data = await api(`/api/portfolio/import/instrument/${encodeURIComponent(isin)}`, { method: 'PUT', body: { ticker } });
            state.pfImport = data;
            const row = (data.instruments || []).find((r) => r.isin === isin);
            toast(row && row.usable ? `${isin}: ${row.ticker} saved${row.median_gap != null ? `, fits your trade prices within ${fmt.pct(row.median_gap, 1)}` : ''}.` : `${isin}: saved, but it still has no usable prices.`, row && row.usable ? 'success' : 'error');
            const pf = await api('/api/portfolio');
            state.portfolio = pf;
            renderPositions();
            renderImport();
            markReviewStale();
        } catch (err) {
            toast(`Not saved: ${err.message}`, 'error');
        }
    }

    async function removeImport() {
        if (!window.confirm('Forget the imported export? This removes its positions, the cash it set, the ticker map and the track-record report. Positions you added by hand stay.')) return;
        try {
            state.pfImport = await api('/api/portfolio/import', { method: 'DELETE' });
            state.portfolio = null;
            await loadPortfolio();
            toast('Import removed.', 'success');
        } catch (err) {
            toast(`Could not remove the import: ${err.message}`, 'error');
        }
    }

    // Positions from any broker import carry that broker's key as their source; hand-typed ones say 'manual'.
    const IMPORT_BADGE = { trade_republic: 'TR', generic: 'CSV' };
    const isImported = (r) => !!r.source && r.source !== 'manual';

    function renderPositions() {
        const p = state.portfolio;
        if (!p) return;
        $('#pf-cash').value = p.cash || 0;
        const ccy = pfCcy();
        $('#pf-add-cost').labels[0].textContent = `Average cost (${ccy})`;
        $('#pf-cash').previousElementSibling.textContent = `Cash (${ccy})`;
        const rows = p.positions || [];
        renderCurrency(ccy, rows.length > 0);
        $('#pf-positions').innerHTML = rows.length
            ? `<thead><tr><th>Ticker</th><th class="num">Shares</th><th class="num">Avg cost (${esc(ccy)})</th><th class="num">Cost basis</th><th>Opened</th><th>Owner at entry</th><th></th></tr></thead><tbody>${
                rows.map((r) => `<tr data-ticker="${esc(r.ticker)}">
                    <td class="mono span"><strong>${esc(r.ticker)}</strong>${isImported(r) ? ` <span class="badge term quiet" tabindex="0" data-tip="Imported from your ${esc(brokerLabel(r.source))} export">${esc(IMPORT_BADGE[r.source] || 'import')}</span>` : ''}${r.name ? `<div class="muted small sans">${esc(r.name)}</div>` : ''}</td>
                    <td class="num" data-label="Shares">${esc(r.shares.toLocaleString('en-US', { maximumFractionDigits: 6 }))}</td>
                    <td class="num" data-label="Average cost">${esc(fmt.money(r.avg_cost, 2, ccy))}</td>
                    <td class="num" data-label="Cost basis">${esc(fmt.money(r.shares * r.avg_cost, 0, ccy))}</td>
                    <td data-label="Opened">${esc(r.opened || '')}</td>
                    <td class="span" data-label="Owner at entry">${isNum(r.owner_score_at_entry) ? `<span class="mono">${esc(fmt.signed(r.owner_score_at_entry, 0))}</span>${String(r.rated_at || '').startsWith('replay') ? ' <span class="muted small term quiet" tabindex="0" data-tip="Replayed point in time by My track record">replayed</span>' : ''}` : '<span class="muted small term quiet" tabindex="0" data-tip="No owner score from when you bought. Funds and metals are never rated; a stock gets one when you analyse it, or from My track record (US companies with SEC filings, replayed for the month you bought).">no rating</span>'}</td>
                    <td class="span">${isImported(r) ? '<span class="muted small term quiet" tabindex="0" data-tip="Change it in your broker and import again">from import</span>' : `<button type="button" class="btn btn-sm btn-danger" data-remove-position="${esc(r.ticker)}" aria-label="Remove ${esc(r.ticker)}">Remove</button>`}</td>
                </tr>`).join('')}</tbody>`
            : `<tbody><tr><td class="muted">No holdings yet. Import your broker's export above, or type a holding in.</td></tr></tbody>`;
        const cost = rows.reduce((t, r) => t + r.shares * r.avg_cost, 0);
        $('#pf-positions-summary').textContent = rows.length
            ? `Your holdings: ${rows.length} position${rows.length === 1 ? '' : 's'}, ${fmt.money(cost, 0, ccy)} at cost, ${fmt.money(p.cash || 0, 0, ccy)} cash`
            : 'Your holdings';
        if (!rows.length) $('#pf-positions-drawer').open = true;
        syncPortfolioState();
    }

    // Resolves to true when the holding was saved, so the form keeps what was typed after a failure.
    async function savePosition(position) {
        try {
            const data = await api('/api/portfolio/position', { method: 'POST', body: position });
            state.portfolio = data;
            renderPositions();
            markReviewStale();
            const r = (data.positions || []).find((x) => x.ticker === position.ticker) || position;
            toast(`${r.ticker} saved: ${r.shares.toLocaleString('en-US', { maximumFractionDigits: 6 })} share${r.shares === 1 ? '' : 's'} at ${fmt.money(r.avg_cost, 2, pfCcy())} average cost.`, 'success');
            return true;
        } catch (err) {
            if (/invalid ticker/i.test(err.message)) showTickerError(position.ticker);
            else toast(`Not saved: ${err.message}`, 'error');
            return false;
        }
    }

    // Resolves to true when Yahoo's search lists the typed symbol (or cannot be asked); otherwise shows the warning with
    // the listings it did find and resolves to false, so nothing is saved until the person picks one or saves anyway.
    async function checkListing(ticker) {
        const host = $('#pf-add-warning');
        const btn = $('#pf-add-form button[type=submit]');
        setBusy(btn, true);
        let res = null;
        try {
            res = await api(`/api/suggest/${encodeURIComponent(ticker)}`);
        } catch (_) { return true; } finally { setBusy(btn, false); }
        const list = (res && res.suggestions) || [];
        if (res && res.bond) return true;
        if (list.some((x) => String(x.ticker || '').toUpperCase() === ticker)) return true;
        const chips = list.slice(0, 6).map((x) => `<button class="chip" type="button" data-fill-ticker="${esc(x.ticker)}"><span class="mono">${esc(x.ticker)}</span>${x.name ? ` <span class="muted small">${esc(x.name)}</span>` : ''}${x.exchange ? ` <span class="muted small">${esc(x.exchange)}</span>` : ''}</button>`).join(' ');
        host.innerHTML = `<div class="warning"><strong>Not saved yet</strong>We found no listing for ${esc(ticker)}. Check the symbol; listings outside the US often need a suffix, such as SAP.DE or SHEL.L.${chips ? ` Did you mean: <span class="chip-row">${chips}</span>` : ''}<div class="head-actions"><button class="btn btn-ghost btn-sm" type="button" data-save-anyway>Save anyway</button></div></div>`;
        return false;
    }

    function showTickerError(ticker) {
        const el = $('#pf-add-ticker-error');
        el.textContent = ticker ? `${ticker} is not a ticker: use letters, digits, dots or dashes, like KO or SAP.DE.` : 'Enter a ticker, like KO or SAP.DE.';
        el.hidden = false;
        $('#pf-add-ticker').setAttribute('aria-invalid', 'true');
        $('#pf-add-ticker').focus();
    }

    function clearTickerError() {
        $('#pf-add-ticker-error').hidden = true;
        $('#pf-add-ticker').removeAttribute('aria-invalid');
    }

    // The portfolio currency is chosen while the portfolio is empty; costs typed later are read in it.
    function renderCurrency(ccy, hasHoldings) {
        const sel = $('#pf-currency');
        if (![...sel.options].some((o) => o.value === ccy)) sel.add(new Option(ccy, ccy), 0);
        sel.value = ccy;
        const imported = !!(state.pfImport && state.pfImport.imported);
        sel.disabled = imported || hasHoldings;
        const hint = $('#pf-currency-hint');
        hint.hidden = !sel.disabled;
        hint.textContent = imported ? 'Set by your import.' : 'Remove your holdings to change the currency.';
    }

    async function saveCurrency() {
        if (!state.portfolio) return;
        const currency = $('#pf-currency').value;
        try {
            const data = await api('/api/portfolio', { method: 'PUT', body: { ...state.portfolio, currency } });
            state.portfolio = data;
            renderPositions();
            toast(`Portfolio currency set to ${data.currency}.`, 'success');
        } catch (err) {
            $('#pf-currency').value = pfCcy();
            toast(`Not saved: ${err.message}`, 'error');
        }
    }

    // Same words as the server's refusal (data/portfolio.py imported_message), shown before any request is sent.
    function showAddWarning(r) {
        $('#pf-add-warning').innerHTML = `<div class="warning"><strong>Not saved</strong>${esc(r.ticker)} comes from your ${esc(brokerLabel(r.source))} import. To change it, import a newer export: a holding typed here would replace the imported one.<div class="head-actions"><button class="btn btn-sm" type="button" data-add-import>Import a newer export</button><button class="btn btn-sm btn-ghost" type="button" data-add-cancel>Cancel</button></div></div>`;
    }

    async function removePosition(ticker) {
        try {
            const data = await api(`/api/portfolio/position/${encodeURIComponent(ticker)}`, { method: 'DELETE' });
            state.portfolio = data;
            renderPositions();
            markReviewStale();
        } catch (err) {
            toast(`Could not remove ${ticker}: ${err.message}`, 'error');
        }
    }

    async function saveCash() {
        if (!state.portfolio) return;
        const btn = $('#pf-cash-save');
        const err = $('#pf-cash-error');
        if (btn.disabled) return;
        err.hidden = true;
        setBusy(btn, true);
        try {
            const data = await api('/api/portfolio', { method: 'PUT', body: { ...state.portfolio, cash: Number($('#pf-cash').value) || 0 } });
            state.portfolio = data;
            renderPositions();
            markReviewStale();
            toast('Cash saved.', 'success');
        } catch (e) {
            // Put the saved amount back and say why, under the field.
            $('#pf-cash').value = state.portfolio.cash || 0;
            err.textContent = /negative/i.test(e.message)
                ? `Cash cannot be negative; it is back to ${fmt.money(state.portfolio.cash || 0, 0, pfCcy())}.`
                : `Not saved: ${e.message}`;
            err.hidden = false;
        } finally {
            setBusy(btn, false);
        }
    }

    // The review describes the holdings as they were: say so, and make the way to refresh it the main action.
    function markReviewStale() {
        const card = $('#pf-todo');
        if (!card || card.hidden) return;
        if (!card.querySelector('.pf-stale')) {
            const w = document.createElement('div');
            w.className = 'warning pf-stale';
            w.innerHTML = '<strong>Out of date</strong>Your holdings changed after this review. Review again to update what to do.';
            const head = card.querySelector('.card-head');
            head.insertAdjacentElement('afterend', w);
        }
        const re = $('#pf-rereview');
        re.classList.add('btn-primary');
    }

    async function reviewPortfolio() {
        const btn = $('#pf-review-button');
        if (btn.disabled) return;
        if (!state.portfolio) await loadPortfolio();
        if (!((state.portfolio && state.portfolio.positions) || []).length) {
            $('#pf-todo').hidden = false;
            $('#pf-todo-caption').textContent = '';
            $('#pf-todo-settings').innerHTML = '';
            $('#pf-todo-changes').innerHTML = '';
            $('#pf-todo-room').innerHTML = '';
            $('#pf-todo-rest').hidden = true;
            $('#pf-todo-groups').innerHTML = '<div class="warning note"><strong>Nothing to review yet</strong>Import your broker export above, or type a holding in, then review.</div>';
            $('#pf-plan').hidden = true;
            $('#pf-rereview').hidden = true;
            $('#pf-dash').hidden = true;
            return;
        }
        setBusy(btn, true);
        setBusy($('#pf-rereview'), true);
        setBusy($('#pf-next-review'), true);
        const slowDone = slowHint('#pf-slow-hint', 'review');
        $('#pf-todo').setAttribute('aria-busy', 'true');
        try {
            const data = await api('/api/portfolio/review', { method: 'POST', body: {} });
            state.pfReview = data;
            renderReview(data);
            const todo = data.todo || [];
            const doNow = todo.filter((g) => MAIN_GROUPS.includes(g.key)).reduce((n, g) => n + g.count, 0);
            const nothing = (todo.find((g) => g.key === 'rest') || {}).count || 0;
            focusResult('#pf-todo h3', `Review ready: ${doNow} to do, ${nothing} with nothing to do.`);
            $('#pf-todo').scrollIntoView({ block: 'start' });
            (data.errors || []).forEach((e) => toast(`${e.ticker}: ${e.error}. Counted at cost.`, 'error'));
            if (state.pfImport && state.pfImport.imported) loadAccountHistory();
        } catch (err) {
            toast(`Review failed: ${err.message}`, 'error');
        } finally {
            slowDone();
            $('#pf-todo').removeAttribute('aria-busy');
            setBusy(btn, false);
            setBusy($('#pf-rereview'), false);
            setBusy($('#pf-next-review'), false);
        }
    }

    // ------------------------------------------------ portfolio dashboard
    // Charts follow the dataviz rules: validated categorical slots (--cat-*), at most six named slices
    // plus Other and Cash, a legend or table twin for every chart, and hover text that is never the only
    // way to read a value.

    const PF_SECTIONS = ['overview', 'actions', 'risk', 'movers', 'cycle'];
    const ALLOC_DIMS = [['position', 'Holdings'], ['sector', 'Sector'], ['region', 'Region'], ['asset_type', 'Type'], ['currency', 'Currency']];
    const EXPOSURE_DIMS = [['sector', 'Sector'], ['cycle', 'Cycle'], ['industry', 'Industry'], ['country', 'Country'], ['region', 'Region'], ['asset_type', 'Asset type'], ['currency', 'Currency']];
    const PERF_RANGES = [['1Y', 12], ['3Y', 36], ['5Y', 60], ['ALL', null]];
    const MOVER_PERIODS = [['1M', '1 month'], ['3M', '3 months'], ['YTD', 'This year'], ['1Y', '1 year']];
    const MAX_SLICES = 6;
    const DIM_HINTS = {
        position: 'Each holding at market value; holdings without prices are counted at cost.',
        sector: 'Funds are split by their own sector mix (look-through, from Yahoo); commodities, bonds and cash have no sector.',
        cycle: 'Sectors grouped by how much their profits depend on the economy (cyclical, sensitive, defensive). Funds are looked through.',
        industry: 'Companies by industry; funds count as many industries.',
        country: 'Where a company has its headquarters (Yahoo); a fund names its countries only when its name does (S&P 500, Japan).',
        region: 'A company by its headquarters; a fund by its name (MSCI World and All-World count as global).',
        asset_type: 'Stocks, ETFs and funds, commodity trackers (physical gold or silver), bonds and crypto.',
        currency: 'The currency each holding is priced in, which is the currency risk you carry in your own currency. Cash counts in your currency.',
    };
    const PF_TIPS = {
        mix: "Today's mix: the holdings you have now, at today's weights, replayed over past prices (cash earns nothing). It shows the risk you hold now, not what you earned: that is your account's line, from your actual trades.",
        volatility: 'How much the portfolio swings in a typical year (annualized standard deviation of weekly returns over two years, in your currency). Cash lowers it.',
        beta: 'How strongly the portfolio follows the benchmark: 1.2 means it moved 1.2% for every 1% of the benchmark in a week, on average over two years.',
        diversification: 'Diversification ratio: the average swing of your holdings divided by the swing of the whole portfolio. 1 means nothing cancels out; 1.5 means a third of the swings cancel.',
        badmonth: 'Bad month: one month in twenty was worse than this for today\'s mix over the last three years (5th percentile of 21-trading-day returns).',
        tracking: 'Tracking error: how far the portfolio strays from the benchmark in a typical year. Near 0 means you hold the index in practice.',
        effective: 'Effective positions: how many equal-sized holdings would give the same concentration. Ten holdings with one at 50% behave like about three. Core funds are left out: a broad fund is not one bet.',
    };

    const money = (v, d = 0) => fmt.money(v, d, (state.pfReview && state.pfReview.currency) || pfCcy());
    const smoney = (v, d = 0) => (isNum(v) ? `${v > 0 ? '+' : ''}${money(v, d)}` : 'n/a');
    const sgnCls = (v) => (isNum(v) && v !== 0 ? (v > 0 ? 'pos' : 'neg') : '');

    function pfStat(label, value, sub = '', cls = '', tip = '') {
        const head = tip ? `<span class="term" tabindex="0" data-tip="${esc(tip)}">${esc(label)}</span>` : esc(label);
        return `<div class="stat"><div class="stat-label">${head}</div><div class="stat-value ${cls}">${esc(value)}</div>${sub ? `<div class="stat-sub">${esc(sub)}</div>` : ''}</div>`;
    }

    function segButtons(host, items, active, attr) {
        $(host).innerHTML = items.map(([k, label]) => `<button class="seg ${k === active ? 'active' : ''}" type="button" data-${attr}="${esc(k)}">${esc(label)}</button>`).join('');
    }

    function setPfSection(name) {
        if (!PF_SECTIONS.includes(name)) name = 'overview';
        state.pfSection = name;
        storageSet('fastvalue-pf-section', name);
        $$('#pf-sections .seg').forEach((s) => { const on = s.dataset.pfSection === name; s.classList.toggle('active', on); s.setAttribute('aria-selected', String(on)); });
        $$('#pf-dash .pf-pane').forEach((p) => { p.hidden = p.dataset.pane !== name; });
        if (name === 'overview') renderPerf();
    }

    function renderReview(d) {
        const a = d.analytics || {};
        $('#pf-dash').hidden = false;
        $('#pf-plan').hidden = false;
        $('#pf-rereview').hidden = false;
        $('#pf-review-caption').textContent = `${a.benchmark ? `Benchmark ${a.benchmark.label} · ` : ''}values in ${d.currency}`;
        $('#pf-positions-drawer').open = false;
        renderTodo(d);
        renderPfSummary(d);
        renderAllocation();
        renderActionsTable();
        $('#pf-detail').hidden = true;
        renderRiskPane();
        renderMovers();
        renderCycle();
        setPfSection(state.pfSection || storageGet('fastvalue-pf-section') || 'overview');
        state.pfJustImported = false;
        if (state.pfImport && state.pfImport.imported) renderImportNext(state.pfImport);
        syncPortfolioState();
    }

    function renderPfSummary(d) {
        const s = d.summary || {};
        const a = d.analytics || {};
        const mv = (a.movers || {}).totals || {};
        const p1y = ((a.performance || {}).periods || []).find((p) => p.key === '1Y');
        const bench = (a.benchmark || {}).label || 'benchmark';
        const total = isNum(mv.total_pnl) ? mv.total_pnl : s.unrealized_pnl;
        const pnlSub = [isNum(s.unrealized_pnl_pct) ? `${fmt.spct(s.unrealized_pnl_pct, 1)} on cost` : '', mv.dividends ? `${money(mv.dividends)} dividends` : '',
            isNum(mv.fx_effect) ? `currency ${smoney(mv.fx_effect)}` : ''].filter(Boolean).join(' · ');
        const cap = state.portfolioLimits.max_position_weight || 1;
        const hold = p1y && isNum(p1y.holdings);
        const mix1y = p1y ? (hold ? p1y.holdings : p1y.mix) : null;
        $('#pf-summary').innerHTML = [
            pfStat('Total value', money(s.total_value), `${money(s.invested_value)} invested · ${fmt.pct(s.cash_weight, 0)} cash`),
            pfStat('P&L since bought', smoney(total), pnlSub, sgnCls(total)),
            pfStat(hold ? "Today's holdings, 1 year" : "Today's mix, 1 year", p1y ? fmt.spct(mix1y, 1) : 'n/a', p1y && isNum(p1y.benchmark) ? `${bench} ${fmt.spct(p1y.benchmark, 1)}` : '', p1y ? sgnCls(mix1y) : '', hold ? PERF_HOLDINGS_TIP : PF_TIPS.mix),
            pfStat('Volatility', fmt.pct(s.volatility, 0), s.bad_month ? `bad month ${money(s.bad_month.amount)}` : 'a year, weekly returns', '', PF_TIPS.volatility),
            pfStat('Largest single holding', s.largest_ticker ? `${s.largest_ticker} ${fmt.pct(s.largest_weight, 0)}` : 'n/a', `cap ${fmt.pct(cap, 0)}, core funds have none`, s.largest_weight > cap ? 'neg' : ''),
        ].join('');
    }

    // ---- what to do now (the review's to-do list, core/portfolio.py)

    const CLASS_NAME = { core: 'core fund', fund: 'other fund', stock: 'stock', commodity: 'metal or commodity', derivative: 'derivative', unpriced: 'no price' };
    const CLASS_LABEL_ONE = { core: 'Core fund', fund: 'Other fund', stock: 'Stock', commodity: 'Metal or commodity', derivative: 'Derivative', unpriced: 'No price yet' };
    const CLASS_ORDER = ['core', 'fund', 'stock', 'commodity', 'derivative', 'unpriced'];
    const GROUP_TIPS = {
        price: 'Holdings without a price source are counted at cost, so the weights of the others are not overstated.',
        room: 'Owner score at or above the entry level, below the 10% starting size and cash available, but no pullback now, so the Add rule stays silent. Not an action: Check a ticker before you buy sizes a top-up if you want one.',
    };
    const NOTHING_TO_DO = '<p class="muted small">Nothing needs doing. Every holding keeps its action; the checks of your plan pass.</p>';
    const MAIN_GROUPS = ['sell', 'trim', 'look', 'add', 'price'];
    const ISIN_LIKE = /^[A-Z]{2}[A-Z0-9]{9}[0-9]$/;
    const PLAN_TITLES = { drift: 'Core share', budget: 'Derivatives budget' };
    const pct0 = (v) => (isNum(v) && Math.abs(v) < 0.005 ? '0%' : fmt.pct(v, 0));
    const dayLabel = (iso) => (iso ? new Date(`${String(iso).slice(0, 10)}T00:00:00`).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) : '');

    function freshnessText(d) {
        const f = d.freshness || {};
        const when = new Date(d.reviewed_at);
        const prices = [];
        if (f.live) prices.push(`${f.live} live`);
        const closes = Object.entries(f.close_days || {});   // close days of the closed-session rows only; live rows are dated later
        if (closes.length) prices.push(closes.map(([day, n]) => `${n} at the ${dayLabel(day)} close`).join(', '));
        else if (f.close) prices.push(`${f.close} at the last close`);
        const text = `Reviewed ${dayLabel(d.reviewed_at)}, ${when.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}.${prices.length ? ` Prices: ${prices.join(', ')}.` : ''}`;
        const stale = (f.stale || []).length
            ? ` <span class="warn-text term quiet" tabindex="0" data-tip="Price data is missing a completed session. Review again after the next close, or when Yahoo catches up.">${f.stale.length} stale: ${esc(f.stale.join(', '))}</span>` : '';
        return `${esc(text)}${stale}`;
    }

    function reviewSettingsText(d) {
        const s = d.settings || {};
        const parts = Object.entries(s.owner_weights || {}).map(([k, w]) => `${(s.labels || {})[k] || k} ${fmt.pct(w, 0)}`);
        const weights = parts.length > 1 ? `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}` : (parts[0] || 'no owner agent');
        const cap = isNum(s.cap) ? `; no new money at or above ${s.cap}x EV/FCF (margin of safety)` : '';
        return `Decided by ${s.saved ? 'your saved weights' : "FastValue's default weights (you have not saved your own yet)"}: entry level ${esc(fmt.signed(s.buy, 0))}, exit level ${esc(fmt.signed(s.sell, 0))}. `
            + `<a href="#weights" data-goto="weights">Edit in Weights</a>. Decision support only, not financial advice. ${infoTip(`Owner weights: ${weights}${cap}.`)}`;
    }

    // Holding alerts of the latest daily check, each with its fix: what changed since the review you last looked at.
    function renderTodoChanges() {
        const host = $('#pf-todo-changes');
        const m = state.pfMonitor;
        const latest = m ? (m.alerts || []).filter((a) => a.at === m.checked_at && a.scope === 'holding') : [];
        host.innerHTML = latest.length
            ? `<div class="card-head sub"><h4>Changed at the last daily check (${esc(dayLabel(m.checked_at))})</h4></div>${latest.slice(0, 8).map((a) => monitorAlert(a.kind, a.title, a.text, esc(a.fix))).join('')}`
            : '';
    }

    // The owner score next to the instruction: stocks only; funds and metals are never rated.
    function todoScoreHtml(row, th) {
        if (!row || row.class !== 'stock') return '';
        if (!isNum(row.owner_score)) return '<div class="muted small">Owner score: not rated</div>';
        const rel = levelRelation(row.owner_score, th).text;
        return `<div class="muted small">Owner score ${esc(fmt.signed(row.owner_score, 0))}${rel ? `, ${esc(rel)}` : ''}</div>`;
    }

    function todoRowHtml(i, rows, th) {
        const can = !!rows[i.ticker];
        const byIsin = ISIN_LIKE.test(i.ticker);   // an imported holding without a ticker is listed by its name
        const label = [byIsin ? i.ticker : i.name, CLASS_NAME[i.class]].filter(Boolean).join(' · ');
        const name = esc(byIsin && i.name ? i.name : i.ticker);
        return `<tr class="${can ? 'clickable' : ''}"${can ? ` data-todo-ticker="${esc(i.ticker)}"` : ''}>
            <td class="wrap"><span class="mono">${can ? `<button type="button" class="co-rowbtn" aria-expanded="false" aria-controls="todo-detail-${esc(i.ticker)}"><strong>${name}</strong></button>` : `<strong>${name}</strong>`}</span><div class="muted small sans">${esc(label)}</div>${todoScoreHtml(rows[i.ticker], th)}</td>
            <td class="wrap">${placeLinks(i.why)}${i.fix && i.fix !== 'Nothing to do.' ? `<div class="muted small">What to do: ${placeLinks(i.fix)}</div>` : ''}${i.risk_note ? `<div class="muted small">${placeLinks(i.risk_note)}</div>` : ''}</td>
            <td class="num">${esc(fmt.pct(i.weight, 1))}</td></tr>`;
    }

    function todoTableHtml(items, rows, th) {
        return `<div class="table-wrap"><table class="table compact"><thead><tr><th>Holding</th><th>Why</th><th class="num">Weight</th></tr></thead><tbody>${items.map((i) => todoRowHtml(i, rows, th)).join('')}</tbody></table></div>`;
    }

    const levelsFor = (d) => ({ buy: (d.settings || {}).buy, sell: (d.settings || {}).sell });

    function todoGroupHtml(g, d) {
        const rows = Object.fromEntries((d.positions || []).map((r) => [r.ticker, r]));
        const head = g.action ? `<span class="action ${actionClass(g.action)}">${esc(actionLabel(g.action))}</span>` : esc(g.label);
        const tip = GROUP_TIPS[g.key] ? ` <span class="info" tabindex="0" role="note" aria-label="${esc(GROUP_TIPS[g.key])}" data-tip="${esc(GROUP_TIPS[g.key])}">i</span>` : '';
        return `<div class="card-head sub"><h4>${head}${tip}</h4><span class="muted small">${g.count} holding${g.count === 1 ? '' : 's'}, ${esc(fmt.pct(g.weight, 1))} of the portfolio</span></div>${todoTableHtml(g.items, rows, levelsFor(d))}`;
    }

    function restHtml(g, d) {
        const rows = Object.fromEntries((d.positions || []).map((r) => [r.ticker, r]));
        return CLASS_ORDER.map((k) => {
            const items = g.items.filter((i) => i.class === k);
            if (!items.length) return '';
            const w = items.reduce((t, i) => t + i.weight, 0);
            const label = ((d.classes || []).find((c) => c.key === k) || {}).label || k;
            return `<div class="card-head sub"><h4>${esc(label)}</h4><span class="muted small">${items.length} holding${items.length === 1 ? '' : 's'}, ${esc(fmt.pct(w, 1))} of the portfolio</span></div>${todoTableHtml(items, rows, levelsFor(d))}`;
        }).join('');
    }

    // A fix or hint that names another place (written by the server as plain words) links to it. Takes the text,
    // returns escaped HTML. Section names in "Backtest > My track record > Cash" become part of the link.
    const PLACE_LINKS = [
        [/Should I enter\?|Check a ticker before you buy/g, (m) => `<a href="#portfolio" data-pf-go="candidate">${m}</a>`],
        [/Risk &gt; Exposure/g, (m) => `<a href="#portfolio" data-pf-go="exposure">${m}</a>`],
        [/Winners &amp; losers|Movers/g, (m) => `<a href="#portfolio" data-pf-go="movers">${m}</a>`],
        [/Backtest &gt; My track record(?: &gt; (Cash|Rules lab|Rules|Entries &amp; exits|Entries|Your style|Holdings))?/g,
            (m, section) => `<a href="#backtest=record${section ? `/${recSectionKey(section)}` : ''}">${m}</a>`],
        [/under Your plan/g, () => 'under <a href="#portfolio" data-pf-go="plan">Your plan</a>'],
        [/import card \(Price source of every holding\)/g, (m) => `<a href="#portfolio" data-pf-go="sources">${m}</a>`],
    ];
    const recSectionKey = (label) => ({ cash: 'cash', rules: 'rules', 'rules lab': 'rules', entries: 'entries', 'entries &amp; exits': 'entries', 'your style': 'style', holdings: 'holdings' })[label.toLowerCase()] || 'overview';
    function placeLinks(text) {
        return PLACE_LINKS.reduce((html, [pattern, make]) => html.replace(pattern, make), esc(text));
    }

    function alertItemHtml(kind, title, text, fix) {
        return `<div class="alert-item ${kind === 'risk' ? 'risk' : ''}">
                <span class="alert-kind">${kind === 'risk' ? 'Risk' : 'Note'}</span><span class="alert-title">${esc(title)}</span>${placeLinks(text)}
                ${fix ? `<div class="alert-fix"><strong>What to do:</strong> ${placeLinks(fix)}</div>` : ''}</div>`;
    }

    // One handler for the in-page places (the places in another tab are plain #hash links, which the router follows).
    function setupPlaceLinks() {
        const scrollTo = (el) => { if (el) el.scrollIntoView({ block: 'center' }); };
        $('#view-portfolio').addEventListener('click', (e) => {
            const link = e.target.closest('[data-pf-go]');
            if (!link) return;
            e.preventDefault();
            const go = link.dataset.pfGo;
            if (go === 'candidate') { const input = $('#pf-candidate-ticker'); scrollTo(input); input.focus({ preventScroll: true }); }
            else if (go === 'exposure') { setPfSection('risk'); scrollTo($('#pf-exposure-table')); }
            else if (go === 'movers') { setPfSection('movers'); scrollTo($('#pf-dash .pf-pane[data-pane="movers"]')); }
            else if (go === 'plan') { const edit = $('#pf-plan-edit'); if (!edit.hidden) edit.click(); scrollTo($('#pf-plan-form').hidden ? edit : $('#pf-plan-form')); }
            else if (go === 'sources') { const drawer = $('#pf-import-sources').closest('details'); drawer.open = true; scrollTo(drawer); }
        });
    }

    // Alerts that ask for something now go in the to-do card with the plan's failed checks; information goes to Risk.
    // Returns how many items the to-do card shows.
    function renderAlerts(d) {
        const alerts = d.alerts || [];
        const now = alerts.filter((a) => (a.where === 'todo' || a.where === 'plan') && !a.attached);
        const hasCore = isNum((d.plan || {}).core_target);
        // No core share set: the plan line already says so once, with the way to set it.
        const checks = ((d.plan || {}).checks || []).filter((c) => c.status !== 'ok' && (c.key !== 'drift' || hasCore));
        const items = [...now.map((a) => alertItemHtml(a.kind, a.title, a.text, a.fix)),
            ...checks.map((c) => alertItemHtml(c.status === 'look' ? 'risk' : 'note', PLAN_TITLES[c.key] || 'Your plan', c.text, c.fix))];
        $('#pf-alerts').innerHTML = items.join('');
        const info = alerts.filter((a) => a.where === 'risk');
        $('#pf-risk-alerts').innerHTML = info.map((a) => alertItemHtml(a.kind, a.title, a.text, a.fix)).join('');
        $('#pf-risk-alerts-card').hidden = !info.length;
        return items.length;
    }

    function renderPfPlan(d) {
        const p = d.plan || {};
        const target = isNum(p.core_target) ? `core funds ${fmt.pct(p.core_target, 0)} of what you have invested` : '';
        const budget = isNum(p.derivatives_budget) ? `derivatives budget ${money(p.derivatives_budget)} a year` : '';
        const line = $('#pf-plan-line');
        if (target || budget) line.textContent = `Your plan: ${[target, budget].filter(Boolean).join(', ')}.`;
        else line.innerHTML = 'No plan yet. Set the share you want in core funds and the review checks how far you are from it. <button class="link-btn" type="button" data-plan-set>Set your plan</button>';
        const sp = p.split || {};
        const tips = { Core: 'Broad index, bond and money-market funds.', Satellite: 'Stocks and sector or theme funds.', Metals: 'Physical metal and other single-commodity trackers.', Speculative: 'Derivatives and crypto trackers.' };
        const stats = ['Core', 'Satellite', 'Metals', 'Speculative'].filter((b) => b === 'Core' || b === 'Satellite' || sp[b] > 0)
            .map((b) => pfStat(b, fmt.pct(sp[b], 0), b === 'Core' && isNum(p.core_target) ? `of invested · plan ${fmt.pct(p.core_target, 0)}` : 'of invested', '', tips[b]));
        stats.push(pfStat('Cash', fmt.pct(p.cash_weight, 0), 'of the whole portfolio'));
        $('#pf-plan-split').innerHTML = stats.join('');
    }

    function renderTodo(d) {
        $('#pf-todo').hidden = false;
        $('#pf-todo').querySelectorAll('.pf-stale').forEach((el) => el.remove());
        $('#pf-rereview').classList.remove('btn-primary');
        $('#pf-todo-caption').innerHTML = freshnessText(d);
        $('#pf-todo-settings').innerHTML = reviewSettingsText(d);
        renderTodoChanges();
        const planItems = renderAlerts(d);
        renderPfPlan(d);
        const groups = d.todo || [];
        const main = groups.filter((g) => MAIN_GROUPS.includes(g.key));
        $('#pf-todo-groups').innerHTML = main.length ? main.map((g) => todoGroupHtml(g, d)).join('') : (planItems ? '' : NOTHING_TO_DO);
        const room = groups.find((g) => g.key === 'room');
        $('#pf-todo-room').innerHTML = room ? todoGroupHtml(room, d) : '';
        const rest = groups.find((g) => g.key === 'rest');
        const drawer = $('#pf-todo-rest');
        drawer.hidden = !rest;
        if (rest) {
            const tally = {};
            rest.items.forEach((i) => {
                const k = i.class === 'stock' ? ({ hold_band: 'hold', thesis_intact: 'buy', outside_framework: 'outside' }[i.rule] || 'stock') : i.class;
                tally[k] = (tally[k] || 0) + 1;
            });
            const names = { hold: ['between the levels', 'between the levels'], buy: ['at or above the entry level', 'at or above the entry level'], outside: ['outside the framework', 'outside the framework'],
                core: ['core fund', 'core funds'], fund: ['other fund', 'other funds'], commodity: ['metal or commodity', 'metals and commodities'] };
            const bits = Object.keys(names).filter((k) => tally[k]).map((k) => `${tally[k]} ${names[k][tally[k] === 1 ? 0 : 1]}`);
            $('#pf-todo-rest-summary').textContent = `Nothing to do: ${rest.count} holding${rest.count === 1 ? '' : 's'}${bits.length ? ` (${bits.join(', ')})` : ''}`;
            $('#pf-todo-rest-body').innerHTML = restHtml(rest, d);
            drawer.open = false;
        }
        if (!isNum((d.plan || {}).core_target) && !isNum((d.plan || {}).derivatives_budget)) $('#pf-plan-form').hidden = true;
    }

    function toggleTodoDetail(tr) {
        const ticker = tr.dataset.todoTicker;
        const next = tr.nextElementSibling;
        const btn = tr.querySelector('.co-rowbtn');
        if (next && next.classList.contains('detail') && next.dataset.detailFor === ticker) {
            next.remove();
            if (btn) btn.setAttribute('aria-expanded', 'false');
            return;
        }
        const row = ((state.pfReview || {}).positions || []).find((r) => r.ticker === ticker);
        if (!row) return;
        const detail = document.createElement('tr');
        detail.className = 'detail';
        detail.dataset.detailFor = ticker;
        detail.id = `todo-detail-${ticker}`;
        detail.innerHTML = `<td colspan="3" class="wrap">${positionDetailHtml(row, state.pfReview.currency || pfCcy())}</td>`;
        tr.after(detail);
        if (btn) btn.setAttribute('aria-expanded', 'true');
    }

    async function savePlan() {
        const core = $('#pf-plan-core').value.trim();
        const budget = $('#pf-plan-budget').value.trim();
        const body = { core_target: core === '' ? null : Number(core) / 100, derivatives_budget: budget === '' ? null : Number(budget) };
        const last = (state.pfReview || {}).plan || {};
        if (isNum(last.invested_value)) Object.assign(body, { core_share: isNum(last.core_share) ? last.core_share : null, invested_value: last.invested_value });
        try {
            const data = await api('/api/portfolio/plan', { method: 'PUT', body });
            state.portfolio = data;
            if (state.pfReview) {
                // The server re-checks the plan against the last review's core share, so the checks match the plan line at once.
                const fresh = data.review_plan || {};
                state.pfReview.plan = { ...last, ...data.plan, ...fresh };
                const drift = (fresh.checks || []).find((c) => c.key === 'drift');
                (state.pfReview.positions || []).filter((r) => r.class === 'core' && fresh.checks).forEach((r) => {
                    r.checks = (r.checks || []).filter((c) => c.key !== 'plan');
                    if (drift) r.checks.push({ key: 'plan', label: 'Your plan', status: drift.status, text: drift.text, fix: drift.fix });
                });
                const planItems = renderAlerts(state.pfReview);
                renderPfPlan(state.pfReview);
                if (!(state.pfReview.todo || []).some((g) => MAIN_GROUPS.includes(g.key))) $('#pf-todo-groups').innerHTML = planItems ? '' : NOTHING_TO_DO;
            }
            $('#pf-plan-form').hidden = true;
            toast('Plan saved.', 'success');
        } catch (err) {
            toast(`Plan not saved: ${err.message}`, 'error');
        }
    }

    async function saveNotes(ticker, notes) {
        try {
            const data = await api(`/api/portfolio/position/${encodeURIComponent(ticker)}/notes`, { method: 'PUT', body: { notes } });
            state.portfolio = data;
            const row = ((state.pfReview || {}).positions || []).find((r) => r.ticker === ticker);
            if (row) row.notes = notes;
            toast(`Notes on ${ticker} saved.`, 'success');
        } catch (err) {
            toast(`Notes not saved: ${err.message}`, 'error');
        }
    }

    // ---- allocation donut

    function foldSlices(groups) {
        const named = groups.filter((g) => !g.cash && g.weight > 0);
        const cash = groups.filter((g) => g.cash && g.weight > 0);
        const head = named.slice(0, MAX_SLICES).map((g, i) => ({ ...g, color: `var(--cat-${i + 1})` }));
        const tail = named.slice(MAX_SLICES);
        if (tail.length) {
            const add = (k) => tail.reduce((s, g) => s + (g[k] || 0), 0);
            head.push({ label: `Other (${tail.length})`, value: add('value'), weight: add('weight'), risk_share: tail.some((g) => isNum(g.risk_share)) ? add('risk_share') : null,
                tickers: tail.flatMap((g) => g.tickers || []), members: tail.map((g) => g.label), color: 'var(--cat-other)' });
        }
        cash.forEach((g) => head.push({ ...g, label: g.label === 'Cash' ? 'Cash' : `${g.label} (incl. cash)`, color: 'var(--cat-cash)' }));
        return head;
    }

    function arcPath(r, ri, a0, a1) {
        const c = 100;
        const p = (rad, ang) => `${(c + rad * Math.cos(ang)).toFixed(2)} ${(c + rad * Math.sin(ang)).toFixed(2)}`;
        const large = a1 - a0 > Math.PI ? 1 : 0;
        return `M ${p(r, a0)} A ${r} ${r} 0 ${large} 1 ${p(r, a1)} L ${p(ri, a1)} A ${ri} ${ri} 0 ${large} 0 ${p(ri, a0)} Z`;
    }

    function donutSvg(slices, centerValue, centerLabel) {
        const total = slices.reduce((s, x) => s + Math.max(0, x.weight || 0), 0) || 1;
        let a0 = -Math.PI / 2;
        const paths = slices.map((s, i) => {
            const frac = Math.max(0, s.weight || 0) / total;
            if (frac <= 0) return '';
            const a1 = a0 + frac * 2 * Math.PI;
            const d = frac > 0.9999 ? `${arcPath(92, 60, a0, a0 + Math.PI)} ${arcPath(92, 60, a0 + Math.PI, a0 + 2 * Math.PI)}` : arcPath(92, 60, a0, a1);
            a0 = a1;
            const tip = `${s.label}: ${fmt.pct(s.weight, 1)} · ${money(s.value)}${isNum(s.risk_share) ? ` · ${fmt.pct(s.risk_share, 0)} of risk` : ''}`;
            return `<path class="slice" d="${d}" style="fill:${s.color}" stroke="var(--surface)" stroke-width="2" data-slice="${i}" data-ctip="${esc(tip)}"></path>`;
        }).join('');
        return `<svg viewBox="0 0 200 200" role="img" aria-label="Allocation donut; the table beside it lists every slice">${paths}
            <text class="donut-center-value" x="100" y="99" text-anchor="middle" font-size="18">${esc(centerValue)}</text>
            <text class="donut-center-label" x="100" y="118" text-anchor="middle" font-size="10">${esc(centerLabel)}</text></svg>`;
    }

    function renderAllocation() {
        const a = (state.pfReview || {}).analytics;
        const dim = state.pfAllocDim || 'position';
        segButtons('#pf-alloc-dims', ALLOC_DIMS, dim, 'alloc-dim');
        if (!a) { $('#pf-donut').innerHTML = ''; $('#pf-alloc-table').innerHTML = '<tbody><tr><td class="muted">No figures yet: add a holding with a price, then review.</td></tr></tbody>'; return; }
        const slices = foldSlices(a.exposures[dim] || []);
        const names = Object.fromEntries((state.pfReview.positions || []).map((r) => [r.ticker, r.name]));
        const top = slices.find((s) => !s.cash) || {};
        $('#pf-donut').innerHTML = donutSvg(slices, top.weight ? fmt.pct(top.weight, 0) : '', top.label ? String(top.label).slice(0, 18) : '');
        const hasRisk = slices.some((s) => isNum(s.risk_share) && !s.cash);
        $('#pf-alloc-table').innerHTML = `<thead><tr><th>${dim === 'position' ? 'Holding' : 'Group'}</th><th class="num">Weight</th><th class="num">Value</th>${hasRisk ? `<th class="num"><span class="term" tabindex="0" data-tip="Share of the portfolio's swings that come from this group (weekly returns, two years). Larger than its weight: it swings the portfolio more than its size suggests.">Risk</span></th>` : ''}</tr></thead><tbody>${
            slices.map((s, i) => `<tr data-slice="${i}"><td><span class="swatch" style="background:${s.color}"></span>${dim === 'position' && !s.members && !s.cash ? `<span class="mono">${esc(s.label)}</span> <span class="muted small">${esc((names[s.label] || '').slice(0, 26))}</span>` : esc(s.label)}${s.members ? `<div class="muted small">${esc(s.members.slice(0, 8).join(', '))}${s.members.length > 8 ? ', ...' : ''}</div>` : ''}</td>
                <td class="num">${esc(fmt.pct(s.weight, 1))}</td><td class="num">${esc(money(s.value))}</td>${hasRisk ? `<td class="num">${s.cash ? '' : esc(pct0(s.risk_share))}</td>` : ''}</tr>`).join('')}</tbody>`;
        $('#pf-alloc-hint').textContent = DIM_HINTS[dim] || '';
    }

    function focusSlice(i) {
        const donut = $('#pf-donut');
        donut.classList.toggle('has-focus', i !== null);
        $$('#pf-donut .slice').forEach((p) => p.classList.toggle('is-focus', p.dataset.slice === String(i)));
        $$('#pf-alloc-table tr[data-slice]').forEach((r) => r.classList.toggle('is-focus', r.dataset.slice === String(i)));
    }

    // ---- performance

    const PERF_HOLDINGS_TIP = "Today's holdings: what you hold now, at today's weights within the invested part, replayed over past prices. Cash is left out, as in My track record, where it has its own section. It shows the risk you hold now, not what you earned: that is your invested part's line, from your actual trades.";

    function perfLines() {
        const a = (state.pfReview || {}).analytics || {};
        const s = (a.performance || {}).series || {};
        const g = (state.pfAccount && state.pfAccount.growth) || {};
        const lines = [];
        const hold = s.holdings && s.holdings.length ? s.holdings : s.mix;   // older analytics: the mix with cash
        if (hold && hold.length) lines.push({ key: 'holdings', label: s.holdings ? "Today's holdings" : "Today's mix", data: hold, color: cssVar('--cat-1'), width: 2 });
        if (g.invested_part && g.invested_part.length) lines.push({ key: 'invested', label: 'Your invested part', data: g.invested_part, color: cssVar('--cat-5'), width: 2 });
        if (g.account && g.account.length) lines.push({ key: 'account', label: 'Whole account, cash included', data: g.account, color: cssVar('--series-8'), width: 1.5, dash: true, hidden: true });
        const benches = (a.benchmarks && a.benchmarks.length) ? a.benchmarks : (s.benchmark ? [{ id: 'benchmark', label: (a.benchmark || {}).label || 'Benchmark' }] : []);
        benches.forEach((b, i) => {
            const data = b.id === 'benchmark' ? s.benchmark : (s.benchmarks || {})[b.id];
            if (data && data.length) lines.push({ key: b.id, label: b.label, data, color: benchColor(b.id, i), width: 1.5, hidden: b.id === 'sp500' || b.id === 'nasdaq100' });
        });
        return lines;
    }

    function rebase(data, from) {
        const sel = data.filter((p) => p.time >= from);
        if (!sel.length) return [];
        const base = sel[0].value;
        return base ? sel.map((p) => ({ time: p.time, value: +(p.value / base * 100).toFixed(2) })) : [];
    }

    function renderPerf() {
        if (!state.pfReview || $('#pf-dash').hidden || state.pfSection !== 'overview') return;
        const range = state.pfPerfRange || '1Y';
        const ranges = PERF_RANGES.filter(([k]) => k !== 'ALL' || state.pfAccount);
        segButtons('#pf-perf-range', ranges.map(([k]) => [k, k === 'ALL' ? 'Since first trade' : k]), range, 'perf-range');
        const lines = perfLines();
        if (!lines.length) { destroyChart('pf-perf-chart'); $('#pf-perf-chart').innerHTML = '<p class="muted small">Not enough price history yet.</p>'; return; }
        const end = lines.reduce((m, l) => (l.data[l.data.length - 1].time > m ? l.data[l.data.length - 1].time : m), '');
        const months = (PERF_RANGES.find(([k]) => k === range) || [])[1];
        let from;
        if (months) { const d = new Date(`${end}T00:00:00Z`); d.setUTCMonth(d.getUTCMonth() - months); from = d.toISOString().slice(0, 10); } else {
            const acct = lines.find((l) => l.key === 'invested') || lines.find((l) => l.key === 'account');
            from = acct ? acct.data[0].time : lines.reduce((m, l) => (l.data[0].time < m ? l.data[0].time : m), end);
        }
        const chart = makeChart('pf-perf-chart', {}, chartSummary(`Your portfolio rebased to 100 from ${from}`, lines.map((l) => ({ label: l.label, data: rebase(l.data, from) }))));
        const byKey = {};
        lines.forEach((l) => {
            byKey[l.key] = chart.addLineSeries({ color: l.color, lineWidth: l.width, lineStyle: l.dash ? 2 : 0, priceLineVisible: false, lastValueVisible: false, priceFormat: { type: 'price', precision: 1, minMove: 0.1 } });
            byKey[l.key].setData(rebase(l.data, from));
        });
        legend('pf-perf-legend', lines.map((l) => ({ key: l.key, label: l.label, color: l.color, dash: l.dash, hidden: l.hidden, series: byKey[l.key] })), { chart: 'pf-perf-chart' });
        chart.timeScale().fitContent();

        const a = state.pfReview.analytics || {};
        const periods = (a.performance || {}).periods || [];
        const perfCell = (total, annual, low) => {
            const v = isNum(annual) ? annual : total;
            if (!isNum(v)) return '<td class="num muted">n/a</td>';
            return `<td class="num ${sgnCls(v)}${low ? ' muted' : ''}">${esc(fmt.spct(v, 1))}${low ? '*' : ''}</td>`;
        };
        const keys = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y'].filter((k) => periods.some((p) => p.key === k));
        const by = Object.fromEntries(periods.map((p) => [p.key, p]));
        const hasHold = periods.some((p) => isNum(p.holdings));
        const holdRow = `<tr><td><span class="term" tabindex="0" data-tip="${esc(hasHold ? PERF_HOLDINGS_TIP : PF_TIPS.mix)}">${hasHold ? "Today's holdings" : "Today's mix"}</span></td>${keys.map((k) => {
            const p = by[k];
            if (!p) return '<td class="num muted">n/a</td>';
            const low = isNum(p.coverage) && p.coverage < 0.9;
            return hasHold ? perfCell(p.holdings, p.holdings_annual, low) : perfCell(p.mix, p.mix_annual, low);
        }).join('')}</tr>`;
        const benches = (a.benchmarks && a.benchmarks.length) ? a.benchmarks : [{ id: null, label: (a.benchmark || {}).label || 'Benchmark' }];
        const benchRows = benches.map((b) => `<tr><td>${esc(b.label)}</td>${keys.map((k) => {
            const p = by[k];
            if (!p) return '<td class="num muted">n/a</td>';
            const hasMap = b.id && p.benchmarks && b.id in p.benchmarks;
            return hasMap ? perfCell(p.benchmarks[b.id], (p.benchmarks_annual || {})[b.id]) : perfCell(p.benchmark, p.benchmark_annual);
        }).join('')}</tr>`).join('');
        $('#pf-perf-table').innerHTML = keys.length ? `<thead><tr><th></th>${keys.map((k) => `<th class="num">${k}${by[k] && isNum(hasHold ? by[k].holdings_annual : by[k].mix_annual) ? ' <span class="muted small">/yr</span>' : ''}</th>`).join('')}</tr></thead><tbody>${holdRow}${benchRows}</tbody>` : '';
        // On a phone each line is a block: the period labels move into the cells, and the periods the chart does not show drop out.
        $$('#pf-perf-table tbody tr').forEach((tr) => [...tr.cells].forEach((td, i) => {
            if (!i) { td.classList.add('span'); return; }
            td.dataset.label = keys[i - 1];
            if (['3M', '6M', '5Y'].includes(keys[i - 1])) td.classList.add('hide-sm');
        }));
        const acct = state.pfAccount && state.pfAccount.summary;
        const lowCov = periods.some((p) => isNum(p.coverage) && p.coverage < 0.9);
        const acctBenches = acct && acct.benchmarks && acct.benchmarks.length
            ? acct.benchmarks.map((b) => `${b.label} ${fmt.spct(b.annual, 1)}`) : (acct ? [`MSCI World ${fmt.spct(acct.benchmark_annual, 1)}`] : []);
        const acctYield = acct && isNum(acct.twr_invested_annual) ? acct.twr_invested_annual : null;
        // One line under the chart; the full explanation sits behind its (i), the low-coverage flag stays on its own line.
        const perfMore = [
            `Each line starts at 100 at the start of the range (or its first day). ${hasHold ? "Today's holdings replays what you hold now at today's weights, cash left out" : "Today's mix replays the holdings you have now"}; it is the risk you hold, not what you earned.`,
            acct && acctYield !== null ? `Your invested part, from your actual trades: ${fmt.spct(acctYield, 1)} a year time-weighted since ${fmtDay(state.pfAccount.period.first_trade)}, against ${acctBenches.join(', ')} over the same days. Cash is left out of every line except the whole account's (click it in the legend); its effect is in Backtest > My track record > Cash.`
                : (acct ? `Your account, from your actual trades: ${fmt.spct(acct.twr_annual, 1)} a year time-weighted since ${fmtDay(state.pfAccount.period.first_trade)}, against ${fmt.spct(acct.benchmark_annual, 1)} for MSCI World. The full breakdown is in Backtest > My track record.` : ''),
            'Select a name to hide or show its line.',
        ].filter(Boolean).join(' ');
        $('#pf-perf-hint').innerHTML = `${esc(`Each line starts at 100. ${hasHold ? "Today's holdings" : "Today's mix"} replays what you own now${acct && (acctYield !== null || isNum(acct.twr_annual)) ? '; your invested part is what you earned' : ''}.`)} ${infoTip(perfMore)}`;
        // The account against its benchmarks a year, in view: the same numbers as the (i) above.
        const acctLine = acct && acctYield !== null ? `Your invested part: ${fmt.spct(acctYield, 1)} a year; ${acctBenches.join(', ')}.`
            : (acct && isNum(acct.twr_annual) ? `Your account: ${fmt.spct(acct.twr_annual, 1)} a year; ${acctBenches.join(', ')}.` : '');
        setIf($('#pf-perf-acct'), 'hidden', !acctLine);
        $('#pf-perf-acct').textContent = acctLine;
        setIf($('#pf-perf-note'), 'hidden', !lowCov);
        $('#pf-perf-note').textContent = lowCov ? '* Some holdings have no prices that far back; they count as cash in those periods.' : '';
    }

    function showPerfProgress(on, p = {}) {
        setIf($('#pf-perf-progress'), 'hidden', !on);
        if (!on) return;
        const share = p.total ? Math.min(1, p.done / p.total) : 0;
        $('#pf-perf-progress-fill').style.width = `${(share * 100).toFixed(0)}%`;
        setIf($('#pf-perf-progress-text'), 'textContent', `Your account over time: ${p.stage || 'loading'}`);
    }

    async function loadAccountHistory() {
        if (state.pfAccountLoading || state.pfAccount) return;
        state.pfAccountLoading = true;
        showPerfProgress(true, { stage: 'starting' });
        try {
            const { job } = await api('/api/portfolio/performance', { method: 'POST', body: {} });
            state.pfAccount = await pollJob(job, (p) => showPerfProgress(true, p));
            renderPerf();
        } catch (err) {
            toast(`Account history not loaded: ${err.message}`, 'error');
        } finally {
            state.pfAccountLoading = false;
            showPerfProgress(false);
        }
    }

    // ---- actions table and detail

    const VS_ENTRY_TIP = 'Owner score now minus the owner score when you bought: stored when you added the position, or replayed for the month you bought by My track record. A rating taken more than a month after the purchase says nothing about the thesis at entry, so it is not compared.';
    const FROM_HIGH_TIP = 'Fall from the high since you bought. A stock between the entry and exit levels gets a look when it is down 25%, or by its own usual yearly swing if that is larger (a volatile stock falls that far in an ordinary year).';
    const CHECK_WORD = { ok: ['ok', 'pos'], look: ['look', 'warn-text'], na: ['n/a', 'muted'] };

    function ownerCell(r) {
        if (r.rated === false) {
            const note = r.rating_note || 'Not rated by the owner framework.';
            return `<span class="decision sm decision-not-rated term" tabindex="0" data-tip="${esc(note)}">not rated</span>`;
        }
        const zone = levelZone(r.owner_score, r.owner_levels);
        return `<span class="mono">${esc(fmt.signed(r.owner_score, 0))}</span>${zone ? `<div class="muted small">${esc(zone)}</div>` : ''}`;
    }

    function renderActionsTable() {
        const d = state.pfReview;
        if (!d) return;
        const ccy = d.currency || pfCcy();
        const filter = state.pfActionFilter || 'ALL';
        const all = d.positions || [];
        const counts = (d.summary || {}).actions || {};
        segButtons('#pf-action-filter', [['ALL', `All ${all.length}`], ...['EXIT', 'TRIM', 'REVIEW', 'ADD', 'HOLD'].filter((k) => counts[k]).map((k) => [k, `${actionLabel(k)} ${counts[k]}`])], filter, 'action-filter');
        const cap = state.portfolioLimits.max_position_weight || 1;
        const exempt = state.portfolioLimits.cap_exempt || [];
        const total = (d.summary || {}).total_value || 1;
        const section = (label, count, weight, table) => `<div class="card-head sub"><h4>${esc(label)}</h4><span class="muted small">${count} holding${count === 1 ? '' : 's'}, ${esc(fmt.pct(weight, 1))} of the portfolio</span></div>${table}`;
        const sections = CLASS_ORDER.map((k) => {
            const rows = all.filter((r) => (r.class || 'stock') === k && (filter === 'ALL' || r.action === filter));
            const off = filter === 'ALL' ? (d.unpriced || []).filter((u) => u.class === k) : [];
            if (!rows.length && !off.length) return '';
            const label = ((d.classes || []).find((c) => c.key === k) || {}).label || k;
            const weight = rows.reduce((t, r) => t + r.weight, 0) + off.reduce((t, u) => t + u.value / total, 0);
            const body = rows.map((r) => `<tr class="clickable" data-pf-ticker="${esc(r.ticker)}">
                <td class="mono"><button type="button" class="co-rowbtn" aria-expanded="${state.pfDetailTicker === r.ticker}" aria-controls="pf-detail" aria-label="Evidence for ${esc(r.ticker)}"><strong>${esc(r.ticker)}</strong></button>${r.name ? `<div class="muted small sans">${esc(r.name)}</div>` : ''}</td>
                <td><span class="action ${actionClass(r.action)}">${esc(actionLabel(r.action))}</span></td>
                <td class="wrap">${esc(r.why || (r.reasons || [])[0] || '')}</td>
                <td data-label="Weight" class="num ${r.weight > cap && !exempt.includes(r.class) ? 'neg' : ''}">${esc(fmt.pct(r.weight, 1))}${exempt.includes(r.class) ? '<div class="muted small">no cap</div>' : ''}</td>
                <td data-label="P&amp;L" class="num ${sgnCls(r.total_pnl ?? r.pnl)}">${esc(fmt.spct(r.pnl_pct, 1))}<div class="muted small">${esc(fmt.money(r.total_pnl ?? r.pnl, 0, ccy))}</div></td>
                <td data-label="Owner">${ownerCell(r)}</td>
                <td data-label="From high" class="num ${(r.class || 'stock') === 'stock' && (r.drawdown?.from_high || 0) <= -(r.drawdown_trigger || state.portfolioLimits.drawdown_review || 1) ? 'neg' : ''}">${esc(fmt.pct(r.drawdown?.from_high, 0))}</td>
            </tr>`).join('') + off.map((u) => `<tr class="is-off"><td class="mono"><strong>${esc(ISIN_LIKE.test(u.ticker) && u.name ? u.name : u.ticker)}</strong>${u.name && !ISIN_LIKE.test(u.ticker) ? `<div class="muted small sans">${esc(u.name)}</div>` : ''}</td><td><span class="muted small">no price</span></td><td class="wrap muted small">${esc(u.error || 'No market data')}: counted at cost ${esc(fmt.money(u.value, 0, ccy))}.</td><td class="num">${esc(fmt.pct(u.value / total, 1))}</td><td class="num muted">at cost</td><td></td><td></td></tr>`).join('');
            const head = `<thead><tr><th>Holding</th><th>Action</th><th>Why</th><th class="num">Weight</th><th class="num">P&amp;L</th><th>Owner</th><th class="num"><span class="term" tabindex="0" data-tip="${esc(FROM_HIGH_TIP)}">From high</span></th></tr></thead>`;
            return section(label, rows.length + off.length, weight, `<div class="table-wrap"><table class="table compact">${head}<tbody>${body}</tbody></table></div>`);
        }).join('');
        $('#pf-review-table').innerHTML = sections || '<p class="muted small">No holdings with this action.</p>';
    }

    function headlineLink(h) {
        if (!h) return '<span class="muted">none</span>';
        const url = safeUrl(h.url);
        const title = esc(String(h.title || '').slice(0, 140));
        return `${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${title}</a>` : title}<div class="muted small">${esc(h.publisher || '')} ${esc(fmtDay(h.published))}</div>`;
    }

    // What an entry rating is and where it comes from, in plain words (a rating taken long after the purchase is not one).
    function entryStatement(r) {
        const ref = r.entry_ref || {};
        const dec = isNum(r.owner_score_at_entry) ? fmt.signed(r.owner_score_at_entry, 0) : '';
        if (!r.owner_decision_at_entry && !isNum(r.owner_score_at_entry)) return 'No rating from when you bought, and none recorded since.';
        if (ref.valid === false) {
            const months = isNum(ref.days_after_open) ? Math.max(1, Math.round(ref.days_after_open / 30.4)) : null;
            return `No rating from when you bought; the first one FastValue has is from ${esc(fmtDay(ref.at))}${months ? `, ${months} month${months === 1 ? '' : 's'} later` : ''}.`;
        }
        if (ref.kind === 'replay') return `Owner score ${esc(dec)} at the month-end before you bought (replayed from the filings of that time).`;
        return `Owner score ${esc(dec)} when you bought${ref.at ? ` (${esc(fmtDay(ref.at))})` : ''}.`;
    }

    function positionDetailHtml(r, ccy = pfCcy()) {
        const pc = r.price_currency || 'USD';
        const stock = (r.class || 'stock') === 'stock';
        const rated = r.rated !== false;
        const list = (items, cls = '') => items && items.length ? `<ul class="reason-list ${cls}">${items.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : '<p class="muted small">None.</p>';
        const rk = r.risk || {};
        const fx = r.currency_split;
        const news = r.news || {};
        const rev = r.revisions || {};
        const revText = (x) => (x ? `${fmt.signed(x.score, 0)} (${fmt.signed(x.change, 0)})` : 'no analysis recorded');
        const pnl = r.total_pnl ?? r.pnl;
        const pnlSub = [`${fmt.spct(r.pnl_pct, 1)} on price`, r.dividends ? `${fmt.money(r.dividends, 0, ccy)} dividends` : '', fx ? `currency ${smoney(fx.fx_effect)}` : ''].filter(Boolean).join(' · ');
        const ownerSub = !rated ? (r.rating_note || '')
            : (isNum(r.owner_change) ? `${fmt.signed(r.owner_change, 0)} since you bought (${fmt.signed(r.owner_score_at_entry, 0)} then)` : 'no rating from when you bought to compare with');
        const dd = r.drawdown || {};
        const trigger = stock ? ` · looks at a ${fmt.pct(r.drawdown_trigger, 0)} fall${isNum(r.own_volatility) ? ` (one typical year for a stock with ${fmt.pct(r.own_volatility, 0)} volatility)` : ''}` : '';
        const stats = [
            pfStat('Value', fmt.money(r.value, 0, ccy), `${fmt.pct(r.weight, 1)} of the portfolio · price ${fmt.money(r.price, 2, pc)}`),
            pfStat('P&L', fmt.money(pnl, 0, ccy), pnlSub, sgnCls(pnl)),
            pfStat('Held', isNum(r.days_held) ? `${plural(r.days_held, 'day', 'days')}` : 'n/a'),
            pfStat('Owner score now', rated ? fmt.signed(r.owner_score, 0) : 'Not rated', rated ? [levelZone(r.owner_score, r.owner_levels), ownerSub].filter(Boolean).join(' · ') : ownerSub, '', rated ? VS_ENTRY_TIP : ''),
            pfStat('From high', fmt.pct(dd.from_high, 1), `${dd.basis || ''}${trigger}`),
            pfStat('Share of risk', pct0(rk.risk_share), isNum(rk.volatility) ? `volatility ${fmt.pct(rk.volatility, 0)}` : '', '', "Share of the portfolio's swings that come from this holding (weekly returns, two years)."),
        ].join('');

        const ref = r.entry_ref || {};
        const refSource = ref.kind === 'replay' ? `replayed for ${String(r.rated_at_entry || '').replace('replay ', '')}` : ref.kind === 'first_after' ? `first recorded ${fmtDay(r.rated_at_entry)}` : ref.kind === 'stored' ? `stored ${fmtDay(r.rated_at_entry)}` : '';
        const inputs = [rated ? `owner score ${fmt.signed(r.owner_score, 0)}` : '',
            rated && isNum(r.owner_score_at_entry) ? `owner score when you bought ${fmt.signed(r.owner_score_at_entry, 0)}${refSource ? ` (${refSource})` : ''}${ref.valid === false ? ', too late to compare' : ''}` : '',
            `weight ${fmt.pct(r.weight, 1)}`, isNum(dd.from_high) && stock ? `${fmt.pct(dd.from_high, 0)} from the high against a look level of ${fmt.pct(r.drawdown_trigger, 0)}` : ''].filter(Boolean).join(', ');

        const checks = (r.checks || []).length ? `<div class="pf-section"><h4>Checks</h4><ul class="reason-list">${r.checks.map((c) => {
            const [word, tone] = CHECK_WORD[c.status] || [c.status, ''];
            return `<li><span class="${tone}">${esc(word)}</span> ${esc(c.label)}: ${esc(c.text)}${c.fix ? `<div class="muted small">What to do: ${esc(c.fix)}</div>` : ''}</li>`;
        }).join('')}</ul></div>` : '';
        const thesis = stock ? `<div class="pf-section"><h4>Thesis</h4><p class="small">${entryStatement(r)}</p>${list(r.concerns, 'concern')}</div>` : '';
        const risk = [
            stock ? pfStat('Stop reference, for sizing only', fmt.money(r.stop_reference, 2, pc), `${fmt.pct(r.stop_distance_pct, 1)} below the price; not a stop to trade on`) : '',
            isNum(r.days_to_exit) && r.days_to_exit >= 1 ? pfStat('Days to exit', r.days_to_exit.toFixed(1), 'at a tenth of the daily volume') : '',
            pfStat('Beta', isNum(rk.beta) ? rk.beta.toFixed(2) : 'n/a'),
            pfStat('Moves with the rest', isNum(rk.corr_to_rest) ? rk.corr_to_rest.toFixed(2) : 'n/a', 'low or negative: it diversifies', '', 'Weekly correlation with the rest of the portfolio over two years.'),
            stock ? pfStat('12-1 month momentum', fmt.spct(r.momentum_12_1, 0)) : '',
        ].filter(Boolean).join('');
        const timing = stock && (r.timing_notes || []).length > 1 ? `<div class="pf-section"><h4>Timing notes</h4>${list((r.timing_notes || []).slice(1), 'timing')}</div>` : '';
        const changes = stock && rated ? `<p class="small">Owner score now ${esc(fmt.signed(r.owner_score, 0))}. 30 days ago: ${esc(revText(rev.d30))}. 90 days ago: ${esc(revText(rev.d90))}.</p>` : '';
        const newsHtml = news.available ? `<p class="small">News tone <span class="tone ${esc(String(news.signal || '').toLowerCase())}">${esc(String(news.signal || '').toLowerCase())}</span> (${esc(fmt.signed(news.score, 0))}, ${esc(news.headlines)} headlines).</p>
                    <div class="stat-row"><div class="stat"><div class="stat-label">Most negative</div><div class="small">${headlineLink(news.worst)}</div></div><div class="stat"><div class="stat-label">Most positive</div><div class="small">${headlineLink(news.best)}</div></div></div>`
            : (stock ? `<p class="muted small">${esc(news.reason || 'No news sentiment.')}</p>` : '');
        const notices = r.notices || (stock ? r.warnings : []) || [];
        return `
            <div class="pf-head">
                <span class="hero-ticker">${esc(r.ticker)}</span>
                <span class="action ${actionClass(r.action)}">${esc(actionLabel(r.action))}</span>
                ${rated ? `<span class="mono">Owner score ${esc(fmt.signed(r.owner_score, 0))}</span>` : '<span class="decision sm decision-not-rated">not rated</span>'}
                <span class="muted small">${esc([CLASS_LABEL_ONE[r.class] || r.class_label, stock && r.setup ? setupLabel(r.setup) : '', r.sector, r.country].filter(Boolean).join(' · '))}</span>
                <span class="spacer"></span>
                <a href="#analyze=${encodeURIComponent(r.ticker)}" data-open-analysis="${esc(r.ticker)}">Full analysis</a>
            </div>
            ${!rated && r.rating_note ? `<p class="hint">${esc(r.rating_note)}</p>` : ''}
            <div class="stat-row">${stats}</div>
            <div class="pf-section"><h4>Why: ${esc(actionLabel(r.action).toLowerCase())}</h4>${list(r.reasons)}<p class="small muted">Inputs: ${esc(inputs)}.</p>${r.fix ? `<p class="small"><strong>What to do:</strong> ${esc(r.fix)}</p>` : ''}</div>
            ${thesis}${checks}
            ${stock || news.available ? `<div class="pf-section"><h4>News and rating changes</h4>${changes}${newsHtml}</div>` : ''}
            ${notices.length ? `<div class="pf-section"><h4>Warnings</h4>${list(notices, 'concern')}</div>` : ''}
            <details class="drawer"><summary>Risk details</summary><div class="drawer-body"><div class="stat-row">${risk}</div>${timing}</div></details>
            <div class="pf-section"><h4>Your notes</h4>
                <form class="field" data-notes-for="${esc(r.ticker)}"><textarea class="input" rows="2" maxlength="500" placeholder="${r.action === 'EXIT' ? 'Sell it, or write down here why you keep it' : 'Why you hold it, what would change your mind'}" aria-label="Notes on ${esc(r.ticker)}">${esc(r.notes || '')}</textarea>
                <div><button class="btn btn-sm" type="submit">Save notes</button></div></form></div>`;
    }

    function showPositionDetail(ticker) {
        const row = (state.pfReview?.positions || []).find((r) => r.ticker === ticker);
        if (!row) return;
        if (state.pfSection !== 'actions') setPfSection('actions');
        const card = $('#pf-detail');
        card.innerHTML = positionDetailHtml(row, state.pfReview.currency || pfCcy());
        card.hidden = false;
        state.pfDetailTicker = ticker;
        $$('#pf-review-table .co-rowbtn[aria-controls="pf-detail"]').forEach((b) => b.setAttribute('aria-expanded', String(b.closest('tr').dataset.pfTicker === ticker)));
        card.scrollIntoView({ block: 'nearest' });
    }

    // ---- risk

    function hbarRows(items, series) {
        const max = Math.max(0.0001, ...items.flatMap((it) => series.map((s) => it[s.key] || 0)));
        return items.map((it) => `<div class="hbar-row">
                <span class="hbar-label${it.title && it.title !== it.label ? ' term quiet' : ''}"${it.title && it.title !== it.label ? ` tabindex="0" data-tip="${esc(it.title)}"` : ''}>${it.labelHtml || esc(it.label)}</span>
                <span class="hbar-track">${series.map((s) => `<span class="hbar ${s.cls || ''}" style="width:${(((it[s.key] || 0) / max) * 100).toFixed(1)}%;${s.cls ? '' : `background:${s.color}`}" data-ctip="${esc(`${it.label}: ${s.label} ${fmt.pct(it[s.key], 1)}`)}"></span>`).join('')}</span>
                <span class="hbar-values">${series.map((s) => esc(pct0(it[s.key]))).join(' · ')}</span>
            </div>`).join('');
    }

    // A half-width card fits the numbers of 16 holdings; with more (the server sends at most 20) the matrix takes the
    // whole row, so the numbers stay readable without hovering.
    const HEATMAP_HALF_MAX = 16;

    function heatColor(v) {
        if (!isNum(v)) return 'var(--surface-2)';
        const pct = Math.min(100, Math.abs(v) * 100).toFixed(0);
        return `color-mix(in oklab, ${v >= 0 ? 'var(--div-pos)' : 'var(--div-neg)'} ${pct}%, var(--div-mid))`;
    }

    function renderRiskPane() {
        const d = state.pfReview;
        const a = (d || {}).analytics;
        if (!a) { $('#pf-risk-stats').innerHTML = ''; $('#pf-riskbars').innerHTML = '<p class="muted small">No analytics for this review.</p>'; return; }
        const r = a.risk || {};
        const bench = (a.benchmark || {}).label || 'benchmark';
        const dd1 = r.max_drawdown_1y || {};
        const dd3 = r.max_drawdown_3y || {};
        $('#pf-risk-stats').innerHTML = [
            pfStat('Volatility', fmt.pct(r.volatility, 1), isNum(r.benchmark_volatility) ? `${bench} ${fmt.pct(r.benchmark_volatility, 1)}` : 'a year', '', PF_TIPS.volatility),
            pfStat('Beta', isNum(r.beta) ? r.beta.toFixed(2) : 'n/a', isNum(r.correlation_to_benchmark) ? `correlation ${r.correlation_to_benchmark.toFixed(2)}` : '', '', PF_TIPS.beta),
            pfStat('Tracking error', fmt.pct(r.tracking_error, 1), `against ${bench}`, '', PF_TIPS.tracking),
            pfStat('Diversification', isNum(r.diversification_ratio) ? r.diversification_ratio.toFixed(2) : 'n/a', 'holdings swing / portfolio swing', '', PF_TIPS.diversification),
            pfStat('Effective positions', isNum((d.summary || {}).effective_positions) ? String(d.summary.effective_positions) : 'n/a', `of ${(d.summary || {}).single_positions ?? (d.summary || {}).positions} single holdings, core funds not counted`, '', PF_TIPS.effective),
            pfStat('Cleared for new money', fmt.pct((d.summary || {}).owner_buy_weight, 0), (d.summary || {}).owner_sell_weight ? `${fmt.pct(d.summary.owner_sell_weight, 0)} at or below the exit level` : (d.summary || {}).not_rated_weight ? `${fmt.pct(d.summary.not_rated_weight, 0)} stocks not rated` : 'owner score at or above the entry level and EV/FCF under the cap'),
            pfStat('Bad month', r.bad_month ? fmt.pct(r.bad_month.value, 1) : 'n/a', r.bad_month ? money(r.bad_month.amount) : '', '', PF_TIPS.badmonth),
            pfStat('Worst month', r.worst_month ? fmt.pct(r.worst_month.value, 1) : 'n/a', r.worst_month ? `${fmtDay(r.worst_month.start)} to ${fmtDay(r.worst_month.end)}` : '', '', "Worst 21 trading days for today's mix in the last three years."),
            pfStat('Max drawdown 1y', fmt.pct(dd1.value, 1), dd1.peak ? `${dd1.peak} to ${dd1.trough}` : '', '', "Largest fall from a high for today's mix over the last year."),
            pfStat('Max drawdown 3y', fmt.pct(dd3.value, 1), dd3.peak ? `${dd3.peak} to ${dd3.trough}` : '', '', "Largest fall from a high for today's mix over the last three years."),
        ].join('');

        const invested = (d.positions || []).reduce((s, x) => s + x.value, 0) + (d.unpriced || []).reduce((s, x) => s + x.value, 0);
        const items = (d.positions || []).filter((x) => x.risk && isNum(x.risk.risk_share))
            .map((x) => ({ label: x.ticker, value_share: invested > 0 ? x.value / invested : 0, risk_share: x.risk.risk_share }))
            .sort((p, q) => q.risk_share - p.risk_share);
        const series = [{ key: 'value_share', label: 'share of invested value', color: 'var(--cat-1)' }, { key: 'risk_share', label: 'share of risk', color: 'var(--cat-2)' }];
        legend('pf-riskbar-legend', [['Share of invested value', cssVar('--cat-1')], ['Share of risk', cssVar('--cat-2')]]);
        $$('#pf-riskbar-legend .legend-swatch').forEach((s) => s.classList.add('block'));
        $('#pf-riskbars').innerHTML = items.length ? hbarRows(items, series) : '<p class="muted small">Not enough price history for risk figures.</p>';
        $('#pf-risk-hint').textContent = [
            r.basis ? `Based on ${r.basis}.` : '',
            (r.excluded || []).length ? `Left out (under 26 weeks of prices): ${r.excluded.join(', ')}.` : '',
            'A holding whose risk bar is much longer than its value bar swings the portfolio more than its size suggests.',
        ].filter(Boolean).join(' ');

        const c = r.correlation;
        if (c && c.tickers && c.tickers.length >= 2) {
            const n = c.tickers.length;
            const host = $('#pf-heatmap');
            $('#pf-corr-grid').classList.toggle('stacked', n > HEATMAP_HALF_MAX);
            host.classList.add('with-values');
            host.style.gridTemplateColumns = `minmax(44px, max-content) repeat(${n}, minmax(34px, 56px))`;
            const cell = (v) => (!isNum(v) ? '' : `<span class="heat-value ${Math.abs(v) >= 0.55 ? 'on-dark' : ''}">${esc(v.toFixed(2).replace(/^(-?)0\./, '$1.'))}</span>`);
            host.innerHTML = '<span></span>' + c.tickers.map((t) => `<span class="heat-label col">${esc(t)}</span>`).join('')
                + c.tickers.map((t, i) => `<span class="heat-label">${esc(t)}</span>` + c.matrix[i].map((v, j) => (i === j
                    ? '<span class="heat-cell diag"></span>'
                    : `<span class="heat-cell" style="background:${heatColor(v)}" data-ctip="${esc(`${t} / ${c.tickers[j]}: ${isNum(v) ? v.toFixed(2) : 'n/a'}`)}">${cell(v)}</span>`)).join('')).join('');
            $('#pf-heat-scale').innerHTML = `<span>-1 opposite</span><span class="ramp"></span><span>+1 together</span>${c.total > c.shown ? `<span>· largest ${c.shown} of ${plural(c.total, 'holding', 'holdings')}</span>` : ''}`;
        } else {
            $('#pf-heatmap').innerHTML = '<p class="muted small">Needs at least two holdings with half a year of prices.</p>';
            $('#pf-heat-scale').innerHTML = '';
        }
        const pairs = (c && c.high_pairs) || [];
        $('#pf-pairs').innerHTML = pairs.length
            ? `<thead><tr><th>Moving together (${esc(fmt.num(c.high_threshold, 2))}+)</th><th class="num">Correlation</th><th class="num">Together</th></tr></thead><tbody>${pairs.map((p) => `<tr><td class="mono span">${esc(p.pair.join(' / '))}</td><td class="num" data-label="Correlation">${esc(fmt.num(p.value, 2))}</td><td class="num" data-label="Together">${esc(fmt.pct(p.weight, 1))}</td></tr>`).join('')}</tbody>`
            : `<tbody><tr><td class="muted small span">No pair above ${esc(fmt.num(c?.high_threshold ?? 0.7, 2))}: your holdings move fairly independently.${c && c.max ? ` Highest: ${esc(c.max.pair.join(' / '))} at ${esc(fmt.num(c.max.value, 2))}.` : ''}</td></tr></tbody>`;
        const div = (d.positions || []).filter((x) => x.risk && isNum(x.risk.corr_to_rest)).sort((p, q) => p.risk.corr_to_rest - q.risk.corr_to_rest);
        $('#pf-diversifiers').innerHTML = div.length ? `<thead><tr><th>Holding</th><th class="num">With the rest</th><th class="num">Beta</th><th class="num">Volatility</th></tr></thead><tbody>${
            div.map((x) => `<tr><td class="mono">${esc(x.ticker)}</td><td class="num">${esc(x.risk.corr_to_rest.toFixed(2))}</td><td class="num">${isNum(x.risk.beta) ? esc(x.risk.beta.toFixed(2)) : 'n/a'}</td><td class="num">${esc(fmt.pct(x.risk.volatility, 0))}</td></tr>`).join('')}</tbody>` : '';
        renderExposure();
    }

    function renderExposure() {
        const a = (state.pfReview || {}).analytics;
        if (!a) return;
        const dim = state.pfExposureDim || 'sector';
        segButtons('#pf-exposure-dims', EXPOSURE_DIMS, dim, 'exposure-dim');
        const rows = a.exposures[dim] || [];
        const vsBench = (dim === 'sector' || dim === 'cycle') && rows.some((g) => isNum(g.benchmark));
        const maxW = Math.max(0.0001, ...rows.map((g) => g.weight || 0));
        const bench = (a.benchmark || {}).label || 'benchmark';
        $('#pf-exposure-table').innerHTML = `<thead><tr><th>${esc((EXPOSURE_DIMS.find(([k]) => k === dim) || [])[1] || '')}</th><th class="num">Weight</th><th></th><th class="num">Value</th><th class="num">Share of risk</th>${vsBench ? `<th class="num">Of equity</th><th class="num">${esc(bench)}</th><th class="num">Difference</th>` : ''}<th>Holdings</th></tr></thead><tbody>${
            rows.map((g) => `<tr class="${g.cash ? 'is-off' : ''}"><td class="span">${g.tip ? `<span class="term" tabindex="0" data-tip="${esc(g.tip)}">${esc(g.label)}</span>` : esc(g.label)}</td>
                <td class="num" data-label="Weight">${esc(fmt.pct(g.weight, 1))}</td>
                <td class="hide-sm" style="min-width:90px"><span class="hbar" style="display:block;width:${(((g.weight || 0) / maxW) * 100).toFixed(1)}%;background:var(--cat-1)"></span></td>
                <td class="num" data-label="Value">${esc(money(g.value))}</td>
                <td class="num" data-label="Share of risk">${g.cash ? '' : esc(pct0(g.risk_share))}</td>
                ${vsBench ? `<td class="num" data-label="Of equity">${isNum(g.share_of_equity) ? esc(fmt.pct(g.share_of_equity, 1)) : ''}</td><td class="num" data-label="${esc(bench)}">${isNum(g.benchmark) ? esc(fmt.pct(g.benchmark, 1)) : ''}</td><td class="num ${isNum(g.active) && Math.abs(g.active) < 0.05 ? 'muted' : ''}" data-label="Difference">${isNum(g.active) ? esc(`${g.active > 0 ? '+' : ''}${(g.active * 100).toFixed(1)} pts`) : ''}</td>` : ''}
                <td class="wrap span muted small mono" data-label="Holdings">${esc((g.tickers || []).slice(0, 10).join(', '))}${(g.tickers || []).length > 10 ? ` +${g.tickers.length - 10}` : ''}</td></tr>`).join('')}</tbody>`;
        $('#pf-exposure-hint').innerHTML = `${esc(DIM_HINTS[dim] || '')}${vsBench ? ` ${infoTip(`"Of equity" is the share among holdings with a sector, comparable with the ${bench} sector mix (from its fund data).`)}` : ''}`;
    }

    // ---- winners and losers

    function dbarRows(items, valueKey, labelFn, tipFn) {
        const max = Math.max(0.0001, ...items.map((it) => Math.abs(it[valueKey] || 0)));
        // Every value on the same side of zero: one half, the whole width, no midline.
        const oneSide = items.length > 0 && (items.every((it) => (it[valueKey] || 0) >= 0) || items.every((it) => (it[valueKey] || 0) <= 0));
        return items.map((it) => {
            const v = it[valueKey] || 0;
            const w = `${((Math.abs(v) / max) * 100).toFixed(1)}%`;
            const bar = `<span class="dbar ${v >= 0 ? 'pos' : 'neg'}" style="width:${w}" data-ctip="${esc(tipFn(it))}"></span>`;
            const val = `<span class="dbar-value">${esc(labelFn(it))}</span>`;
            const half = oneSide ? '' : '<span class="dbar-half left">' + (v < 0 ? val + bar : '') + '</span>';
            const hold = !oneSide || v >= 0 ? `<span class="dbar-half">${v >= 0 ? bar + val : ''}</span>` : `<span class="dbar-half left">${val + bar}</span>`;
            return `<div class="dbar-row"><span class="hbar-label${it.name ? ' term quiet' : ''}"${it.name ? ` tabindex="0" data-tip="${esc(it.name)}"` : ''}>${esc(it.ticker)}</span>
                <span class="dbar-track${oneSide ? ' one-side' : ''}">${half}${hold}</span></div>`;
        }).join('');
    }

    function renderMovers() {
        const a = (state.pfReview || {}).analytics;
        if (!a) { $('#pf-since-bars').innerHTML = ''; $('#pf-period-bars').innerHTML = ''; return; }
        const m = a.movers || {};
        const t = m.totals || {};
        const since = m.since_entry || [];
        const best = since[0];
        const worst = since[since.length - 1];
        $('#pf-mover-stats').innerHTML = [
            pfStat('P&L since bought', smoney(t.total_pnl), `${smoney(t.pnl)} price${t.dividends ? ` · ${money(t.dividends)} dividends` : ''}`, sgnCls(t.total_pnl)),
            isNum(t.fx_effect) ? pfStat('Currency effect', smoney(t.fx_effect), `on ${t.split_positions} holding${t.split_positions === 1 ? '' : 's'} priced in other currencies`, sgnCls(t.fx_effect), 'Part of your P&L that came from exchange-rate moves since you bought, not from the share prices.') : '',
            best ? pfStat('Best', best.ticker, `${smoney(best.total_pnl)} (${fmt.spct(best.pnl_pct, 0)})`, 'pos') : '',
            worst && worst !== best ? pfStat(worst.total_pnl > 0 ? 'Smallest gain' : 'Worst', worst.ticker, `${smoney(worst.total_pnl)} (${fmt.spct(worst.pnl_pct, 0)})`, sgnCls(worst.total_pnl)) : '',
        ].join('');
        $('#pf-since-bars').innerHTML = since.length ? dbarRows(since, 'total_pnl', (it) => `${smoney(it.total_pnl)} ${fmt.spct(it.pnl_pct, 0)}`,
            (it) => `${it.ticker}: price ${smoney(it.pnl)}${it.dividends ? `, dividends ${money(it.dividends)}` : ''}${isNum(it.days_held) ? `, held ${plural(it.days_held, 'day', 'days')}` : ''}`) : '<p class="muted small">No holdings.</p>';

        const key = state.pfMoverPeriod || '1Y';
        segButtons('#pf-mover-periods', MOVER_PERIODS, key, 'mover-period');
        const p = (m.periods || {})[key];
        $('#pf-period-bars').innerHTML = p && p.items.length ? dbarRows(p.items, 'pnl', (it) => `${smoney(it.pnl)}${isNum(it.return) ? ` ${fmt.spct(it.return, 0)}` : ''}`,
            (it) => `${it.ticker}: return while held ${isNum(it.return) ? fmt.spct(it.return, 1) : 'n/a'}, start ${money(it.start_value)}${it.net_invested ? `, put in ${smoney(it.net_invested)}` : ''}${it.dividends ? `, dividends ${money(it.dividends)}` : ''}`) : '<p class="muted small">No price history for this period.</p>';
        $('#pf-period-hint').innerHTML = p ? `${esc(`Since ${fmtDay(p.start)}: ${smoney(p.total)}. Money you added is not counted as gain.${p.missing.length ? ` No price at the start for ${p.missing.join(', ')}.` : ''}`)} ${infoTip('The percentage is the return while you held it (a holding bought late is measured from its purchase). Imported holdings use your actual trades, savings plans included; holdings added by hand count as bought on the date opened. Positions sold in the period are in Backtest > My track record.')}` : '';

        const fxRows = since.filter((x) => x.currency_split);
        $('#pf-fx-card').hidden = !fxRows.length;
        $('#pf-fx-table').innerHTML = fxRows.length ? `<thead><tr><th>Holding</th><th>Priced in</th><th class="num">Share price move</th><th class="num">Currency move</th><th class="num">Price effect</th><th class="num">Currency effect</th><th class="num">P&amp;L</th></tr></thead><tbody>${
            fxRows.map((x) => { const s = x.currency_split; return `<tr><td class="mono span">${esc(x.ticker)}</td><td data-label="Priced in" class="hide-sm">${esc(s.currency)}</td><td class="num ${sgnCls(s.local_return)}" data-label="Share price move">${esc(fmt.spct(s.local_return, 1))}</td><td class="num ${sgnCls(s.currency_move)}" data-label="Currency move">${esc(fmt.spct(s.currency_move, 1))}</td>
                <td class="num ${sgnCls(s.price_effect)}" data-label="Price effect">${esc(smoney(s.price_effect))}</td><td class="num ${sgnCls(s.fx_effect)}" data-label="Currency effect">${esc(smoney(s.fx_effect))}</td><td class="num ${sgnCls(x.pnl)}" data-label="P&amp;L">${esc(smoney(x.pnl))}</td></tr>`; }).join('')}
            <tr class="total"><td class="span">Total</td><td class="hide-sm"></td><td class="hide-sm"></td><td class="hide-sm"></td><td class="num" data-label="Price effect">${esc(smoney(t.price_effect_foreign))}</td><td class="num" data-label="Currency effect">${esc(smoney(t.fx_effect))}</td><td class="num" data-label="P&amp;L">${esc(smoney(fxRows.reduce((s, x) => s + x.pnl, 0)))}</td></tr></tbody>` : '';
        $('#pf-movers-hint').textContent = 'Values in your currency. "Since you bought" uses your average cost; dividends are those received on imported holdings since the position was opened.';
    }

    // ---- cycle and news

    function renderCycle() {
        const d = state.pfReview;
        const a = (d || {}).analytics;
        if (!a) return;
        const cy = a.cycle || {};
        $('#pf-macro-asof').textContent = cy.as_of ? `as of ${fmtDay(cy.as_of)}` : '';
        $('#pf-macro').innerHTML = (cy.indicators || []).map((i) => pfStat(i.label, i.key === 'regime' ? rgName(i.value) : i.value, i.detail || '')).join('');
        if (!(cy.indicators || []).length) $('#pf-macro').innerHTML = '<p class="muted small">The Macro agent had no data in this review.</p>';
        const groups = cy.groups || [];
        const hasBench = groups.some((g) => isNum(g.benchmark));
        legend('pf-cycle-legend', [['Your equity holdings', cssVar('--cat-1')], ...(hasBench ? [[(a.benchmark || {}).label || 'Benchmark', cssVar('--cat-other')]] : [])]);
        $$('#pf-cycle-legend .legend-swatch').forEach((s) => s.classList.add('block'));
        $('#pf-cycle-bars').innerHTML = groups.length ? hbarRows(groups.map((g) => ({ label: g.label, labelHtml: `<span class="term sans" tabindex="0" data-tip="${esc(g.tip || '')}">${esc(g.label)}</span>`, yours: g.share_of_equity, bench: g.benchmark })),
            [{ key: 'yours', label: 'yours', color: 'var(--cat-1)' }, ...(hasBench ? [{ key: 'bench', label: 'benchmark', color: 'var(--cat-other)' }] : [])])
            : '<p class="muted small">No holdings with a sector.</p>';
        $('#pf-cycle-notes').innerHTML = (cy.notes || []).map((n) => `<div class="warning note"><strong>Note</strong>${esc(n)}</div>`).join('');

        const all = (d.positions || []).slice().sort((p, q) => q.weight - p.weight);
        const changed = (r) => ['d30', 'd90'].some((k) => r.revisions && r.revisions[k] && isNum(r.revisions[k].change) && r.revisions[k].change !== 0);
        const rows = all.filter((r) => (r.news || {}).available || changed(r));
        const rev = (x) => (x && isNum(x.change) ? `<span class="${sgnCls(x.change)}">${esc(fmt.signed(x.change, 0))}</span><div class="muted small">score ${esc(fmt.signed(x.score, 0))}</div>` : '<span class="muted small">no analysis recorded</span>');
        $('#pf-news-wrap').hidden = !rows.length;
        $('#pf-news-table').innerHTML = rows.length ? `<thead><tr><th>Holding</th><th>News tone</th><th>Most negative headline</th><th>Most positive headline</th><th>Owner now</th><th class="num">vs 30 days</th><th class="num">vs 90 days</th><th>Earnings</th></tr></thead><tbody>${
            rows.map((r) => { const n = r.news || {}; const ev = (r.timing_notes || []).find((t) => t.startsWith('Earnings in')); return `<tr>
                <td class="mono"><strong>${esc(r.ticker)}</strong></td>
                <td>${n.available ? `<span class="tone ${esc(String(n.signal || '').toLowerCase())}">${esc(String(n.signal || '').toLowerCase())}</span> <span class="muted small">${esc(fmt.signed(n.score, 0))} · ${esc(n.headlines)}</span>` : '<span class="muted small">no news</span>'}</td>
                <td class="headline small">${n.available ? headlineLink(n.worst) : ''}</td>
                <td class="headline small">${n.available ? headlineLink(n.best) : ''}</td>
                <td>${ownerCell(r)}</td>
                <td class="num">${rev((r.revisions || {}).d30)}</td><td class="num">${rev((r.revisions || {}).d90)}</td>
                <td class="small">${ev ? esc(ev.split(':')[0]) : '<span class="muted">not soon</span>'}</td></tr>`; }).join('')}</tbody>` : '';
        const quiet = all.length - rows.length;
        $('#pf-news-rest').textContent = quiet > 0 && rows.length ? `No headlines or rating changes for the other ${quiet} holding${quiet === 1 ? '' : 's'}.` : '';
        $('#pf-news-fix').hidden = !(all.length && all.every((r) => !(r.news || {}).available));
    }

    function setupChartTips() {
        const tip = document.createElement('div');
        tip.className = 'chart-tip';
        tip.hidden = true;
        tip.setAttribute('role', 'tooltip');
        document.body.appendChild(tip);
        const place = (x, y) => {
            const r = tip.getBoundingClientRect();
            let left = x + 12;
            let top = y + 12;
            if (left + r.width > window.innerWidth - 8) left = x - r.width - 12;
            if (top + r.height > window.innerHeight - 8) top = y - r.height - 12;
            tip.style.left = `${Math.max(8, left)}px`;
            tip.style.top = `${Math.max(8, top)}px`;
        };
        const show = (el, x, y) => {
            setIf(tip, 'textContent', el.dataset.ctip);
            setIf(tip, 'hidden', false);
            place(x, y);
        };
        document.addEventListener('mousemove', (e) => {
            const el = e.target.closest && e.target.closest('[data-ctip]');
            if (!el) { setIf(tip, 'hidden', true); return; }
            show(el, e.clientX, e.clientY);
        });
        // Touch and keyboard read the same values: a tap or focus shows the tip at the element, the next tap elsewhere hides it.
        document.addEventListener('touchstart', (e) => {
            const el = e.target.closest && e.target.closest('[data-ctip]');
            if (!el) { setIf(tip, 'hidden', true); return; }
            const t = e.touches[0];
            show(el, t.clientX, t.clientY);
        }, { passive: true });
        document.addEventListener('focusin', (e) => {
            const el = e.target.closest && e.target.closest('[data-ctip]');
            if (!el) return;
            const r = el.getBoundingClientRect();
            show(el, r.left, r.bottom);
        });
        document.addEventListener('focusout', (e) => { if (e.target.closest && e.target.closest('[data-ctip]')) setIf(tip, 'hidden', true); });
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setIf(tip, 'hidden', true); });
        window.addEventListener('scroll', () => setIf(tip, 'hidden', true), { passive: true });
    }

    async function assessCandidate() {
        const symbol = $('#pf-candidate-ticker').value.trim().toUpperCase();
        if (!symbol) { $('#pf-candidate-ticker').focus(); return; }
        $('#pf-candidate-ticker').value = symbol;
        const btn = $('#pf-candidate-button');
        setBusy(btn, true);
        const notice = $('#pf-candidate-notice');
        notice.innerHTML = '';
        try {
            const data = await api('/api/portfolio/candidate', { method: 'POST', body: { ticker: symbol } });
            state.pfCandidate = data;
            renderCandidate(data);
        } catch (err) {
            // The card below belongs to the previous ticker: hide it, so it does not read as the answer to this one.
            state.pfCandidate = null;
            $('#pf-candidate').hidden = true;
            toast(`Assessment failed: ${err.message}`, 'error');
            if (/no market data/i.test(err.message)) suggestCandidate(symbol);
        } finally {
            setBusy(btn, false);
        }
    }

    // Turns a positive check into a holding without leaving the card: shares and the price paid, in the portfolio currency.
    function candidateAddHtml(d, ccy) {
        const sz = d.sizing || {};
        const shares = isNum(sz.shares) && sz.shares > 0 ? sz.shares : '';
        const cost = shares && isNum(sz.value) ? (sz.value / sz.shares).toFixed(2) : '';
        return `<div class="pf-section">
            <form class="toolbar-inline labelled" id="pf-candidate-add-form" autocomplete="off" novalidate>
                <div class="field"><label for="pf-candidate-shares">Shares</label>
                    <input id="pf-candidate-shares" class="input input-compact pf-num" type="number" inputmode="decimal" step="any" min="0" value="${esc(shares)}"></div>
                <div class="field"><label for="pf-candidate-cost">Average cost (${esc(ccy)})</label>
                    <input id="pf-candidate-cost" class="input input-compact pf-num" type="number" inputmode="decimal" step="any" min="0" value="${esc(cost)}"></div>
                <button class="btn btn-primary" type="submit">Add to my holdings</button>
            </form>
            <p class="hint neg" id="pf-candidate-add-error" role="alert" hidden>Enter the shares and the price you paid.</p>
            <p class="hint">Saves today's owner score as the score when you bought.</p>
        </div>`;
    }

    async function addCandidate() {
        const d = state.pfCandidate;
        if (!d) return;
        const shares = Number($('#pf-candidate-shares').value);
        const cost = Number($('#pf-candidate-cost').value);
        const err = $('#pf-candidate-add-error');
        err.hidden = shares > 0 && cost > 0;
        if (!err.hidden) return;
        // A holding from a broker import changes with a newer import, never by typing over it.
        const held = ((state.portfolio && state.portfolio.positions) || []).find((r) => r.ticker === d.ticker);
        if (held && isImported(held)) { showAddWarning(held); $('#pf-add-warning').scrollIntoView({ block: 'nearest' }); return; }
        const btn = $('#pf-candidate-add-form button[type=submit]');
        setBusy(btn, true);
        const ok = await savePosition({ ticker: d.ticker, shares, avg_cost: cost, opened: new Date().toISOString().slice(0, 10), ...(d.entry_snapshot || {}) });
        if (ok) $('#pf-candidate-add-form').closest('.pf-section').innerHTML = `<p class="hint">${esc(d.ticker)} is in your holdings.</p>`;
        else setBusy(btn, false);
    }

    // The same help as suggestTickers, under the candidate box: the listings Yahoo's search finds for what was typed.
    async function suggestCandidate(symbol) {
        const host = $('#pf-candidate-notice');
        let res = { suggestions: [] };
        try { res = await api(`/api/suggest/${encodeURIComponent(symbol)}`); } catch (_) { /* the warning below covers it */ }
        const chips = (res.suggestions || []).slice(0, 6).map((x) => `<button class="chip" type="button" data-candidate-pick="${esc(x.ticker)}"><span class="mono">${esc(x.ticker)}</span>${x.name ? ` <span class="muted small">${esc(x.name)}</span>` : ''}${x.exchange ? ` <span class="muted small">${esc(x.exchange)}</span>` : ''}</button>`).join(' ');
        host.innerHTML = `<div class="warning"><strong>No data for ${esc(symbol)}</strong>Yahoo has no prices under that symbol. ${chips
            ? `Did you mean: <span class="chip-row">${chips}</span>`
            : 'Check the spelling, or add the exchange suffix (SAP.DE, SHEL.L, SHOP.TO).'}</div>`;
    }

    function renderCandidate(d) {
        const card = $('#pf-candidate');
        const a = d.analysis || {};
        const list = (items, cls = '') => items && items.length ? `<ul class="reason-list ${cls}">${items.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : '<p class="muted small">None.</p>';
        const ccy = d.currency || pfCcy();
        const pc = d.price_currency || 'USD';
        const sizeStats = (sz, st = {}) => `<div class="stat-row">${[
            pfStat('Shares', isNum(sz.shares) ? sz.shares.toLocaleString('en-US') : 'n/a'),
            pfStat('Value', fmt.money(sz.value, 0, ccy)),
            pfStat('Weight', fmt.pct(sz.weight, 1)),
            pfStat('At risk to stop ref.', fmt.money(sz.risk_amount, 0, ccy)),
            pfStat('Binding limit', sz.binding || 'n/a'),
            st.tranches > 1 ? pfStat('Staging', `${st.tranches} tranches`, `about ${plural(st.over_weeks, 'week', 'weeks')}`) : '',
        ].join('')}</div>`;
        if (d.held && d.position) {
            const tu = d.top_up;
            card.innerHTML = `<p class="muted small">You hold ${esc(d.ticker)}: this is the position review.</p>` + positionDetailHtml(d.position, ccy)
                + (tu ? `<div class="pf-section"><h4>Top-up size <span class="info" tabindex="0" role="note" aria-label="How many more shares fit if you want to add: the same limits as a new position, with what you already hold counted against the starting size." data-tip="How many more shares fit if you want to add: the same limits as a new position, with what you already hold counted against the starting size.">i</span></h4>${sizeStats(tu)}
                    <p class="small muted">${esc(tu.note || '')}</p></div>` : '');
        } else {
            const sz = d.sizing;
            const st = d.staging || {};
            const stock = (d.class || 'stock') === 'stock';
            const checks = (d.checks || []).length ? `<div class="pf-section"><h4>Checks</h4><ul class="reason-list">${d.checks.map((c) => {
                const [word, tone] = CHECK_WORD[c.status] || [c.status, ''];
                return `<li><span class="${tone}">${esc(word)}</span> ${esc(c.label)}: ${esc(c.text)}${c.fix ? `<div class="muted small">What to do: ${esc(c.fix)}</div>` : ''}</li>`;
            }).join('')}</ul></div>` : '';
            card.innerHTML = `
                <div class="pf-head">
                    <span class="hero-ticker">${esc(d.ticker)}</span>
                    <span class="action ${actionClass(d.verdict)}">${esc(actionLabel(d.verdict))}</span>
                    ${stock && a.decision !== 'NOT_RATED' ? `<span class="mono">Owner score ${esc(fmt.signed(a.composite_score, 0))}</span>` : ''}
                    ${stock ? `<span class="mono muted"><span class="term" tabindex="0" data-tip="${esc(TIP_TIMING)}">Entry timing</span> ${esc(fmt.signed(a.timing?.score || 0, 0))}</span>` : ''}
                    <span class="muted small">${esc(d.class_label || '')} · ${esc(fmt.money(d.current_price, 2, pc))}${stock ? ` · market regime ${esc(rgName(a.market_regime))}` : ''}</span>
                    <span class="spacer"></span>
                    <a href="#analyze=${encodeURIComponent(d.ticker)}" data-open-analysis="${esc(d.ticker)}">Full analysis</a>
                </div>
                <div class="pf-section"><h4>Verdict</h4>${list(d.reasons)}</div>
                ${sz ? `<div class="pf-section"><h4>Position size</h4>${sizeStats(sz, st)}
                    <p class="small muted">${esc(sz.note || '')}${sz.cash_short ? ` Cash is short by ${esc(fmt.money(sz.cash_short, 0, ccy))}.` : ''}</p>
                    <p class="hint">Entry ${esc(fmt.money(a.suggested_entry, 2, pc))}, stop reference ${esc(fmt.money(a.suggested_stop_loss, 2, pc))}, target ${esc(fmt.money(a.suggested_target_price, 2, pc))}. ${esc(String(a.stop_loss_method || '').replace(/\$(\d[\d,]*(?:\.\d+)?)/g, (m, n) => fmt.money(Number(n.replace(/,/g, '')), 2, pc)))}</p>
                </div>` : ''}
                ${checks}
                ${stock ? `<div class="pf-section"><h4>Owner concerns</h4>${list(d.concerns, 'concern')}</div>
                <div class="pf-section"><h4>Timing notes</h4>${list((d.timing_notes || []).slice(1), 'timing')}</div>` : ''}
                ${fitHtml(d.fit, ccy)}
                ${d.previous && stock ? `<p class="hint">Previous analysis ${esc(fmtDay(d.previous.generated_at))}: owner score ${esc(fmt.signed(d.previous.owner_score ?? d.previous.composite_score, 0))}.</p>` : ''}
                ${stock && a.decision !== 'NOT_RATED' && (d.verdict === 'ENTER' || d.verdict === 'ENTER_STAGED') ? candidateAddHtml(d, ccy) : ''}`;
        }
        card.hidden = false;
        card.scrollIntoView({ block: 'nearest' });
    }

    function fitHtml(f, ccy) {
        if (!f) return '';
        const prof = f.profile || {};
        const o = f.overlap || {};
        const sim = (f.most_similar || []).map((x) => `${x.ticker} ${fmt.num(x.correlation, 2)}`).join(', ');
        if (!isNum(f.volatility_before) && !isNum(f.correlation_to_portfolio)) {
            return f.note ? `<div class="pf-section"><h4>How it fits your portfolio</h4><p class="muted small">${esc(f.note)}</p></div>` : '';
        }
        const corr = f.correlation_to_portfolio;
        const reading = !isNum(corr) ? '' : corr >= 0.8 ? 'It moves very much like what you already hold: it adds little diversification.'
            : corr >= 0.5 ? 'It moves partly with what you hold.' : 'It moves fairly independently of what you hold: it diversifies.';
        return `<div class="pf-section"><h4>How it fits your portfolio</h4>
            <div class="stat-row">${[
                pfStat('Moves with your portfolio', isNum(corr) ? corr.toFixed(2) : 'n/a', '', '', 'Weekly correlation with your current portfolio over two years, in your currency. 1: moves exactly with it; 0: unrelated.'),
                pfStat('Most similar holdings', sim || 'n/a'),
                pfStat('Portfolio volatility', `${fmt.pct(f.volatility_before, 1)} to ${fmt.pct(f.volatility_after, 1)}`),
                pfStat('Its share of risk after', pct0(f.risk_share_after), `for ${fmt.pct(f.add_weight, 0)} of value`),
                pfStat(`Already in ${prof.sector || 'its sector'}`, fmt.pct(o.sector, 0)),
                pfStat('Same price currency', fmt.pct(o.currency, 0)),
            ].join('')}</div>
            <p class="small muted">${esc(reading)} Volatility after assumes ${esc(fmt.money(f.add_value, 0, ccy))} bought from cash (the suggested size, or the new-position cap). ${esc(prof.country ? `Headquarters: ${prof.country}.` : '')}</p></div>`;
    }

    function setupPortfolio() {
        loadBrokers();
        $('#pf-import-broker').addEventListener('change', (e) => { state.brokerChosen = e.target.value; storageSet('fastvalue-broker', e.target.value); clearImportError(); renderBrokerChoice(); });
        $('#pf-import-error').addEventListener('click', (e) => {
            const b = e.target.closest('[data-import-as]');
            if (!b || !state.lastImportFile) return;
            state.brokerChosen = b.dataset.importAs;
            storageSet('fastvalue-broker', state.brokerChosen);
            renderBrokerChoice();
            importExport(state.lastImportFile);
        });
        $('#pf-type-in').addEventListener('click', () => { state.pfTypeIn = true; syncPortfolioState(); $('#pf-add-ticker').focus(); });
        $('#pf-next-review').addEventListener('click', () => reviewPortfolio());
        $('#pf-next-record').addEventListener('click', () => { showView('backtest'); setBkMode('record'); });
        $('#pf-import').addEventListener('input', (e) => {
            const form = e.target.closest && e.target.closest('form.inline-edit');
            if (!form || !form.dataset.matched) return;
            const value = e.target.value.trim().toUpperCase();
            form.querySelector('button').textContent = value === form.dataset.matched ? `Keep ${form.dataset.matched}` : 'Use';
        });
        $('#pf-add-form').addEventListener('submit', (e) => {
            e.preventDefault();
            const ticker = $('#pf-add-ticker').value.trim().toUpperCase();
            const shares = Number($('#pf-add-shares').value);
            const avgCost = Number($('#pf-add-cost').value);
            const err = $('#pf-add-error');
            err.hidden = true;
            if (!ticker) { showTickerError(''); return; }
            if (!(shares > 0)) { err.textContent = 'Shares must be more than 0.'; err.hidden = false; $('#pf-add-shares').focus(); return; }
            if (!(avgCost > 0)) { err.textContent = 'Average cost must be more than 0.'; err.hidden = false; $('#pf-add-cost').focus(); return; }
            if (!TICKER_RE.test(ticker)) { showTickerError(ticker); return; }
            // A holding from a broker import changes with a newer import, never by typing over it.
            const held = ((state.portfolio && state.portfolio.positions) || []).find((r) => r.ticker === ticker);
            if (held && isImported(held)) { showAddWarning(held); return; }
            $('#pf-add-warning').innerHTML = '';
            const save = () => savePosition({ ticker, shares, avg_cost: avgCost, opened: $('#pf-add-opened').value || null }).then((ok) => { if (ok) { $('#pf-add-form').reset(); clearTickerError(); } });
            if (held) { save(); return; }
            checkListing(ticker).then((found) => { if (found) save(); });
        });
        $('#pf-add-ticker').addEventListener('input', () => { $('#pf-add-warning').innerHTML = ''; clearTickerError(); });
        $('#pf-add-form').addEventListener('input', () => { $('#pf-add-error').hidden = true; });
        $('#pf-add-warning').addEventListener('click', (e) => {
            const fill = e.target.closest('[data-fill-ticker]');
            if (fill) {
                $('#pf-add-warning').innerHTML = '';
                $('#pf-add-ticker').value = fill.dataset.fillTicker;
                $('#pf-add-shares').focus();
                return;
            }
            if (e.target.closest('[data-save-anyway]')) {
                $('#pf-add-warning').innerHTML = '';
                const ticker = $('#pf-add-ticker').value.trim().toUpperCase();
                savePosition({ ticker, shares: Number($('#pf-add-shares').value), avg_cost: Number($('#pf-add-cost').value), opened: $('#pf-add-opened').value || null })
                    .then((ok) => { if (ok) { $('#pf-add-form').reset(); clearTickerError(); } });
                return;
            }
            if (e.target.closest('[data-add-import]')) {
                $('#pf-add-warning').innerHTML = '';
                $('#pf-import-file').click();
            } else if (e.target.closest('[data-add-cancel]')) {
                $('#pf-add-warning').innerHTML = '';
                $('#pf-add-form').reset();
                clearTickerError();
                $('#pf-add-ticker').focus();
            }
        });
        $('#pf-positions').addEventListener('click', (e) => {
            const t = e.target.closest('[data-remove-position]')?.dataset.removePosition;
            if (t && window.confirm(`Remove ${t} from the portfolio?`)) removePosition(t);
        });
        $('#pf-cash-form').addEventListener('submit', (e) => { e.preventDefault(); saveCash(); });
        $('#pf-currency').addEventListener('change', saveCurrency);
        $('#pf-import-button').addEventListener('click', () => {
            if (state.picking) return;  // a double click opens one file picker
            state.picking = true;
            setTimeout(() => { state.picking = false; }, 1000);
            $('#pf-import-file').click();
        });
        $('#pf-import-file').addEventListener('change', (e) => importExport(e.target.files[0]));
        $('#pf-import-remove').addEventListener('click', removeImport);
        $('#pf-import').addEventListener('click', (e) => {
            const chip = e.target.closest('[data-candidate]');
            if (!chip) return;
            const input = $(`#pf-import form.inline-edit[data-isin="${chip.dataset.isin}"] input`);
            if (input) { input.value = chip.dataset.candidate; input.focus(); }
        });
        $('#pf-import').addEventListener('submit', (e) => {
            const form = e.target.closest('form.inline-edit');
            if (!form) return;
            e.preventDefault();
            setInstrumentTicker(form.dataset.isin, form.querySelector('input').value.trim().toUpperCase());
        });
        $('#pf-review-button').addEventListener('click', () => reviewPortfolio());
        $('#pf-rereview').addEventListener('click', () => reviewPortfolio());
        $('#pf-review-table').addEventListener('click', (e) => {
            const row = e.target.closest('tr[data-pf-ticker]');
            if (row) showPositionDetail(row.dataset.pfTicker);
        });
        $('#pf-sections').addEventListener('click', (e) => { const b = e.target.closest('[data-pf-section]'); if (b) setPfSection(b.dataset.pfSection); });
        const todoToggle = (e) => {
            const b = e.target.closest('button.co-rowbtn');
            const tr = (b || e.target).closest('tr[data-todo-ticker]');
            if (tr && (b || !e.target.closest('a, button, form, input, textarea, summary'))) toggleTodoDetail(tr);
        };
        $('#pf-todo').addEventListener('click', todoToggle);
        $('#pf-plan-edit').addEventListener('click', () => {
            const form = $('#pf-plan-form');
            form.hidden = !form.hidden;
            if (form.hidden) return;
            const plan = (state.pfReview || {}).plan || (state.portfolio || {}).plan || {};
            $('#pf-plan-core').value = isNum(plan.core_target) ? +(plan.core_target * 100).toFixed(1) : '';
            $('#pf-plan-budget').value = isNum(plan.derivatives_budget) ? plan.derivatives_budget : '';
            $('#pf-plan-core').focus();
        });
        $('#pf-plan-line').addEventListener('click', (e) => { if (e.target.closest('[data-plan-set]')) $('#pf-plan-edit').click(); });
        $('#pf-plan-cancel').addEventListener('click', () => { $('#pf-plan-form').hidden = true; });
        $('#pf-plan-form').addEventListener('submit', (e) => { e.preventDefault(); savePlan(); });
        $('#pf-action-filter').addEventListener('click', (e) => { const b = e.target.closest('[data-action-filter]'); if (b) { state.pfActionFilter = b.dataset.actionFilter; renderActionsTable(); } });
        $('#pf-alloc-dims').addEventListener('click', (e) => { const b = e.target.closest('[data-alloc-dim]'); if (b) { state.pfAllocDim = b.dataset.allocDim; renderAllocation(); } });
        $('#pf-perf-range').addEventListener('click', (e) => { const b = e.target.closest('[data-perf-range]'); if (b) { state.pfPerfRange = b.dataset.perfRange; renderPerf(); } });
        $('#pf-exposure-dims').addEventListener('click', (e) => { const b = e.target.closest('[data-exposure-dim]'); if (b) { state.pfExposureDim = b.dataset.exposureDim; renderExposure(); } });
        $('#pf-mover-periods').addEventListener('click', (e) => { const b = e.target.closest('[data-mover-period]'); if (b) { state.pfMoverPeriod = b.dataset.moverPeriod; renderMovers(); } });
        const sliceOf = (e) => { const el = e.target.closest('[data-slice]'); return el ? Number(el.dataset.slice) : null; };
        ['#pf-donut', '#pf-alloc-table'].forEach((sel) => {
            $(sel).addEventListener('mouseover', (e) => focusSlice(sliceOf(e)));
            $(sel).addEventListener('mouseleave', () => focusSlice(null));
        });
        setupChartTips();
        $('#pf-candidate-form').addEventListener('submit', (e) => { e.preventDefault(); assessCandidate(); });
        $('#view-portfolio').addEventListener('submit', (e) => {
            const form = e.target.closest('form[data-notes-for]');
            if (!form) return;
            e.preventDefault();
            saveNotes(form.dataset.notesFor, form.querySelector('textarea').value);
        });
        $('#view-portfolio').addEventListener('click', (e) => {
            const open = e.target.closest('[data-open-analysis]');
            if (open) { e.preventDefault(); showView('analyze'); runAnalyze(open.dataset.openAnalysis); return; }
            const pick = e.target.closest('[data-candidate-pick]');
            if (pick) { $('#pf-candidate-ticker').value = pick.dataset.candidatePick; assessCandidate(); }
        });
        $('#pf-candidate').addEventListener('submit', (e) => {
            if (e.target.id !== 'pf-candidate-add-form') return;
            e.preventDefault();
            addCandidate();
        });
    }

    // ============================================================ boot

    async function loadBrokers() {
        try {
            const res = await api('/api/brokers');
            state.brokers = res.brokers || [];
            state.brokerDefault = res.default;
        } catch (_) {
            state.brokers = [];
        }
        renderBrokerChoice();
    }

    async function loadMeta() {
        const [meta, strategies, watch] = await Promise.all([
            api('/api/agents'),
            api('/api/strategies'),
            api('/api/watchlist'),
        ]);
        state.agents = meta.agents;
        state.agentByKey = Object.fromEntries(meta.agents.map((a) => [a.key, a]));
        state.config = meta.config;
        state.draft = clone(meta.config);
        state.regimes = meta.regimes;
        state.activeRegime = meta.regimes[0];
        state.backtestable = meta.backtestable;
        state.frequencies = meta.frequencies;
        state.hasLocal = meta.has_local_overrides;
        state.dataSources = meta.data_sources || [];
        state.version = meta.version || '';
        state.strategies = strategies;
        state.watchlist = watch.tickers;
    }

    // ============================================================ BACKTEST
    // One tab, four modes. Compare and search run as background jobs on the
    // server (core/lab.py) and are polled; the classic and timing tools are the
    // single-ticker views that existed before.

    const BK_HELP = {
        compare: 'Your saved setup traded month by month on your portfolio or one ticker, point in time since 2011, against buy and hold of the same allocation, the market and the classic strategies.',
        search: 'Many setups on the same stored scores, ranked by their weaker half of the period. Use it to test ideas, not to tune until something wins.',
        classic: 'One price-only strategy on one ticker, with its indicator chart. These strategies read no fundamentals and no agents.',
        trend: 'Trend, momentum and dip rules that step aside in deep falls, with the checks that say whether to trust a backtest. Research and alerts: nothing here changes a score.',
        timing: 'The point-in-time timing agents replayed daily or weekly on one ticker as an entry and exit rule, with a calibration table. A research view: in FastValue the entry timing has no vote.',
        record: 'Your real account from the broker import: your invested part against MSCI World, the S&P 500 and the Nasdaq 100, your cash on its own, your style in numbers, and rules replayed on your own history.',
    };
    const CONF_LABEL = { exact: 'exact', high: 'high confidence', medium: 'medium confidence', low: 'low confidence' };
    const GRID_SHORT = { shareholder_share: 'Shareholders', buy: 'Entry level', sell: 'Exit level', cap: 'Cap', momentum_vote: 'Momentum vote', entry_gate: 'Gate', rebalance: 'Rebalance' };
    const DECISION_NAMES = { B: 'at or above the entry level', H: 'between the levels', S: 'at or below the exit level', N: 'not rated', '': 'no score' };
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const gridKey = (v) => JSON.stringify(v);

    function bkValueLabel(key, v) {
        if (key === 'shareholder_share') return fmt.pct(v, 0);
        if (key === 'cap') return v === null ? 'off' : `${v}x`;
        if (key === 'momentum_vote') return v === 0 ? 'none' : fmt.num(v, 2);
        if (key === 'entry_gate') return ((state.labOptions || {}).entry_gates || {})[v] || v;
        if (key === 'buy' || key === 'sell') return fmt.signed(v, 0);
        return String(v);
    }

    // A job started with a tag can be stopped: the Stop button posts to the cancel route and the poll ends as "stopped".
    async function runJob(path, body, onProgress, tag) {
        const { job } = await api(path, { method: 'POST', body });
        if (tag) state.jobs[tag] = job;
        try {
            return await pollJob(job, onProgress);
        } finally {
            if (tag && state.jobs[tag] === job) delete state.jobs[tag];
        }
    }

    async function pollJob(job, onProgress) {
        for (;;) {
            await sleep(500);
            const j = await api(`/api/lab/jobs/${job}`);
            if (j.status === 'cancelled') { const stopped = new Error('Stopped'); stopped.stopped = true; throw stopped; }
            onProgress(j.progress || {});
            if (j.status === 'done') return j.result;
            if (j.status === 'error') throw new Error(j.error || 'Backtest failed');
        }
    }

    async function stopJob(tag, btn) {
        const job = state.jobs[tag];
        if (!job) return;
        if (btn) btn.disabled = true;
        try { await api(`/api/lab/jobs/${job}/cancel`, { method: 'POST' }); } catch (_) { /* it had already finished */ }
    }

    // A run ended: say so when the person is on another tab or mode, with a way back to it.
    function bkFinished(mode, message) {
        if (!$('#view-backtest').hidden && state.bkMode === mode) return;
        toast(message, 'success', 12000, { label: 'Open', onClick: () => { showView('backtest'); setBkMode(mode); } });
    }

    // Phase one (prices, scores) fills the first 60% of the bar and phase two the rest, so it never goes back.
    const BK_PHASE_TWO = /^(Step 2 of 2|Classic strategies|Setup )/;
    function showBkProgress(on, p = {}) {
        setIf($('#bk-progress'), 'hidden', !on || state.bkMode !== state.bkJobMode);
        if (!on) {
            state.bkProgressMax = 0;
            const stop = $('#bk-progress [data-job-cancel]');
            if (stop) stop.disabled = false;
            return;
        }
        const share = p.total ? Math.min(1, p.done / p.total) : 0;
        const at = BK_PHASE_TWO.test(p.stage || '') ? 0.6 + 0.4 * share : 0.6 * share;
        state.bkProgressMax = Math.max(state.bkProgressMax || 0, at);
        $('#bk-progress-fill').style.width = `${(state.bkProgressMax * 100).toFixed(0)}%`;
        setIf($('#bk-progress-text'), 'textContent', p.stage || 'Working');
    }

    // The shared run button follows the mode on show: busy in the mode that runs, "is running" in the other, off when the grid is empty or too big.
    function syncBkRun() {
        const mode = state.bkMode;
        if (mode !== 'compare' && mode !== 'search') return;
        const btn = $('#bk-run');
        const job = state.bkJobMode;
        const here = job === mode;
        const other = job && !here;
        setIf(btn, 'textContent', other ? `${job === 'search' ? 'Search' : 'Compare'} is running` : mode === 'search' ? 'Run search' : 'Run backtest');
        btn.disabled = !!other || here || (mode === 'search' && !!state.bkGridBad);
        btn.classList.toggle('loading', here);
        if (here) btn.setAttribute('aria-busy', 'true'); else btn.removeAttribute('aria-busy');
        setIf($('#bk-progress'), 'hidden', !here);
        if (mode !== 'search' && $('#bk-error').textContent === GRID_EMPTY) clearFormError($('#bk-form'));
    }

    function setBkMode(mode, { hash = true } = {}) {
        if (!BK_HELP[mode]) return;
        state.bkMode = mode;
        $$('#bk-modes .seg').forEach((s) => s.classList.toggle('active', s.dataset.mode === mode));
        $('#bk-mode-help').textContent = BK_HELP[mode];
        const own = mode === 'compare' || mode === 'search';
        $('#bk-form').hidden = !own;
        $('#bk-compare-options').hidden = mode !== 'compare';
        $('#bk-grid').hidden = mode !== 'search';
        syncBkRun();
        $('#bk-results').hidden = mode !== 'compare' || !state.bk;
        // The chart is laid out only while its card shows, so draw it again whenever the card comes back (H-17).
        if (mode === 'compare' && state.bk && !$('#bk-results').hidden) requestAnimationFrame(() => renderBkChart(state.bk));
        $('#bk-search-results').hidden = mode !== 'search' || !state.bkSearch;
        $('#bk-classic').hidden = mode !== 'classic';
        $('#bk-trend').hidden = mode !== 'trend';
        if (mode === 'trend') openTrend().then(() => { if (state.trRule) renderTrCharts(state.trRule); });
        $('#bk-timing').hidden = mode !== 'timing';
        $('#bk-record').hidden = mode !== 'record';
        if (mode === 'record') renderRecordForm();
        const target = mode === 'compare' ? '#backtest' : `#backtest=${mode}`;
        if (hash && location.hash.startsWith('#backtest') && location.hash !== target) setHash(target);
    }

    function setBkUniverse(universe) {
        state.bkUniverse = universe;
        $$('#bk-universe .seg').forEach((s) => s.classList.toggle('active', s.dataset.universe === universe));
        $('#bk-ticker-field').hidden = universe !== 'ticker';
        $('#bk-allocation-field').hidden = universe !== 'portfolio';
        renderBkHoldings();
    }

    function renderBkHoldings() {
        const tickers = (state.labOptions || {}).portfolio_tickers || [];
        $('#bk-holdings').textContent = state.bkUniverse !== 'portfolio' ? ''
            : tickers.length ? `Holdings from the Portfolio tab (${tickers.length}): ${tickers.join(', ')}.`
                : 'Your portfolio is empty. Add positions in the Portfolio tab, or test one ticker.';
    }

    async function refreshBkOptions() {
        try {
            state.labOptions = await api('/api/lab/options');
        } catch (err) {
            toast(`Backtest options unavailable: ${err.message}`, 'error');
            return;
        }
        renderBkHoldings();
        renderSecNotice($('#bk-sec'));
        renderBkSetup();
        if (state.bkMode === 'record') renderRecordForm();
    }

    // The saved setup the backtest uses, readable in place (hover) with a way to the tab where it is edited.
    function renderBkSetup() {
        const cfg = state.config;
        const el = $('#bk-setup');
        if (!cfg || !el) return;
        const enabled = (k) => (cfg.enabled || {})[k] !== false;
        const ow = cfg.owner_weights || {};
        const owners = state.agents.filter((a) => roleOf(a.key, cfg) === 'owner' && enabled(a.key));
        const total = owners.reduce((x, a) => x + (ow[a.key] || 0), 0) || 1;
        const deciding = owners.filter((a) => (ow[a.key] || 0) > 0);
        const evidence = owners.filter((a) => !(ow[a.key] > 0));
        const timing = state.agents.filter((a) => roleOf(a.key, cfg) === 'timing' && enabled(a.key));
        const info = state.agents.filter((a) => roleOf(a.key, cfg) === 'info' || !enabled(a.key));
        const th = cfg.owner_thresholds || {};
        const cap = cfg.owner_max_ev_to_fcf;
        const replayed = new Set((state.labOptions || {}).replayed_agents || []);
        const names = (list) => list.map((a) => `${shortName(a.name)}${replayed.size && !replayed.has(a.key) ? '*' : ''}`).join(', ') || 'none';
        const evFcf = tagTip('EV/FCF', STAT_TIPS.evFcf);
        const capText = cap == null ? 'no valuation cap' : `no new money at or above ${esc(cap)}x ${evFcf}`;
        const summary = `${esc(`${deciding.map((a) => `${shortName(a.name)} ${fmt.pct((ow[a.key] || 0) / total, 0)}`).join(' + ') || 'no agent'} decide · entry level ${fmt.signed(th.buy, 0)} · exit level ${fmt.signed(th.sell, 0)}`)} · ${capText}`;
        const rows = [
            ['Decide (owner score)', esc(deciding.map((a) => `${shortName(a.name)} ${fmt.pct((ow[a.key] || 0) / total, 0)}`).join(', ') || 'none')],
            ['Evidence only (weight 0)', esc(names(evidence))],
            ['Entry timing (no vote; used by the entry gate)', esc(names(timing))],
            ['Informative or switched off', esc(names(info))],
            ['Owner levels', esc([['strong_buy', 'Strong entry'], ['buy', 'Entry'], ['sell', 'Exit'], ['strong_sell', 'Strong exit']].map(([k, label]) => `${label} ${fmt.signed(th[k], 0)}`).join(' · '))],
            ['Margin of safety', cap == null ? 'off' : capText],
            ['Confidence floor', esc(fmt.pct(cfg.confidence_floor, 0))],
        ];
        // Shown in place with no hover: a details block that opens on tap, and a real link to the tab where it is edited.
        const open = el.querySelector('details') && el.querySelector('details').open;
        setHtml(el, `<details class="setup-details"${open ? ' open' : ''}><summary><strong>Your saved setup</strong> <span class="muted">${summary}</span></summary>
                <table class="table compact">${rows.map(([k, v]) => `<tr><td class="muted">${esc(k)}</td><td>${v}</td></tr>`).join('')}</table>
                <p class="hint">${state.hasLocal ? 'Saved on this computer.' : "FastValue's default weights (nothing saved yet)."}${replayed.size ? ' * not replayed here: no point-in-time history, so it counts in live analysis only.' : ''}${state.dirty ? ' Unsaved changes in the Weights tab are not used until you save them.' : ''}</p>
            </details>
            <button class="link-btn" type="button" data-goto="weights">Edit in Weights</button>`);
    }

    // An alert always comes with its fix: the SEC contact can be set right here.
    function renderSecNotice(el) {
        if (!el) return;
        const o = state.labOptions || {};
        // A div, not a form: these slots sit inside the backtest forms, and browsers drop nested forms.
        const id = `${el.id || 'sec'}-email`;
        const form = `<div class="sec-form" role="group" aria-label="SEC contact email" hidden>
                <div class="field sec-field">
                    <label for="${id}">Contact email for the SEC</label>
                    <input type="email" id="${id}" class="input input-compact" placeholder="you@example.com" autocomplete="email" data-sec-input>
                    <p class="hint neg" id="${id}-error" role="alert" data-sec-error hidden></p>
                </div>
                <button class="btn btn-sm btn-primary" type="button" data-sec-save>Save</button>
                <button class="btn btn-sm btn-ghost" type="button" data-sec-cancel>Cancel</button>
                <span class="hint">Kept only on this computer and sent only to sec.gov with each request, as the SEC asks of every app. Save an empty field to remove it.</span>
            </div>`;
        if (o.sec_access) {
            el.innerHTML = o.sec_source === 'saved'
                ? `<span class="muted small">SEC filings: contact email saved on this machine. <button class="link-btn" type="button" data-sec-edit>Change</button></span>${form}`
                : '';
            return;
        }
        el.innerHTML = `<div class="warning"><strong>SEC filings</strong>Owner scores in backtests come from each company's 10-K filings at sec.gov, and the SEC asks every app for a contact email. Without one, only filings already saved on this machine are used, so some holdings may show as not rated. <button class="btn btn-sm" type="button" data-sec-edit>Set contact email</button></div>${form}`;
    }

    // A bad address is explained under the field, with the field marked; no toast.
    function secError(form, message) {
        const input = form.querySelector('[data-sec-input]');
        const box = form.querySelector('[data-sec-error]');
        box.textContent = message || '';
        box.hidden = !message;
        if (message) { input.setAttribute('aria-invalid', 'true'); input.setAttribute('aria-describedby', box.id); input.focus(); }
        else { input.removeAttribute('aria-invalid'); input.removeAttribute('aria-describedby'); }
    }

    async function saveSecContact(form) {
        const email = form.querySelector('input').value.trim();
        secError(form, '');
        try {
            const res = await api('/api/settings/sec', { method: 'PUT', body: { email } });
            toast(res.configured ? 'Contact email saved: SEC filings will be downloaded when needed.' : 'Contact email removed.', 'success');
            await refreshBkOptions();
        } catch (err) {
            secError(form, /valid email/i.test(err.message) ? 'That does not look like an email address. Check it and save again.' : err.message);
        }
    }

    // ------------------------------------------------ my track record

    // The server keeps the last result (the share page is written from it): open it again instead of an empty form.
    async function loadLastRecord() {
        if (state.rec || state.recLastTried) return;
        state.recLastTried = true;
        try {
            const last = await api('/api/track-record/last');
            if (state.rec || !last || !last.period) return;
            state.rec = last;
            renderRecord(last);
        } catch (_) { /* no saved result: the form is enough */ }
    }

    function renderRecordForm() {
        const has = (state.labOptions || {}).has_import;
        $('#rec-empty').hidden = !!has;
        $('#rec-run').disabled = !has;
        setIf($('#rec-run'), 'textContent', state.rec ? 'Run again' : 'Replay my account');
        loadLastRecord();
        const sec = $('#rec-sec');
        const needSec = $('#rec-fastvalue').checked && !(state.labOptions || {}).sec_access;
        sec.hidden = !needSec;
        if (needSec) renderSecNotice(sec);
    }

    function showRecProgress(on, p = {}) {
        setIf($('#rec-progress'), 'hidden', !on);
        if (!on) { $('#rec-progress [data-job-cancel]').disabled = false; return; }
        const share = p.total ? Math.min(1, p.done / p.total) : 0;
        $('#rec-progress-fill').style.width = `${(share * 100).toFixed(0)}%`;
        setIf($('#rec-progress-text'), 'textContent', p.stage || 'Working');
    }

    async function runRecord() {
        const btn = $('#rec-run');
        setBusy(btn, true);
        showRecProgress(true, { stage: 'Starting' });
        try {
            const result = await runJob('/api/track-record', { method: $('#rec-method').value, fastvalue: $('#rec-fastvalue').checked },
                (p) => showRecProgress(true, p), 'rec');
            state.rec = result;
            state.recAllLearnings = false;
            state.recAllHoldings = false;
            renderRecord(result);
            if (state.bkMode === 'record' && !$('#view-backtest').hidden) $('#rec-results').scrollIntoView({ block: 'start' });
            bkFinished('record', 'Track record finished: your account is replayed.');
            if (result.fastvalue && result.fastvalue.backfilled) {
                toast(`${plural(result.fastvalue.backfilled, 'imported position', 'imported positions')} now ${result.fastvalue.backfilled === 1 ? 'carries' : 'carry'} the owner score at entry (replayed).`, 'success', 8000);
                state.portfolio = null;
            }
        } catch (err) {
            if (err.stopped) toast('Stopped. Nothing was changed.', 'info');
            else toast(`Track record failed: ${err.message}`, 'error', 10000);
        } finally {
            setBusy(btn, false);
            showRecProgress(false);
        }
    }

    const eur = (v, d = 0) => fmt.money(v, d, 'EUR');
    const signedEur = (v) => (isNum(v) ? `${v > 0 ? '+' : ''}${eur(v)}` : 'n/a');
    const benchList = (d) => ((d.summary || {}).benchmarks || []);  // primary first; [] for a result made before D-36
    const recSection = (x) => x.section || (x.key === 'msci' || x.key === 'cash' ? 'cash' : 'overview');  // old results lack "section"
    const isOldRecord = (d) => !benchList(d).length;
    const OLD_RECORD_WARNING = '<div class="warning"><strong>Older track record</strong>This track record was made before the comparison left cash out. Run it again to see your invested part against MSCI World, the S&amp;P 500 and the Nasdaq 100, and the Cash section. <button class="btn btn-sm" type="button" data-rec-rerun>Run again</button></div>';

    function learningHtml(x, i, { rank = true } = {}) {
        const ev = x.evidence || {};
        const table = ev.rows && ev.rows.length
            ? `<div class="table-wrap"><table class="table compact"><thead><tr>${ev.columns.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${
                ev.rows.map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : '';
        return `<div class="learning conf-${esc(x.confidence)}">
            <button class="learning-head" type="button" aria-expanded="false" data-learning="${i}">
                <span class="learning-rank">${rank ? i + 1 : ''}</span>
                <span class="learning-title">${esc(x.title)}</span>
                <span class="conf-badge conf-${esc(x.confidence)}">${esc(CONF_LABEL[x.confidence] || x.confidence)}</span>
                <span class="learning-eur mono ${signCls(x.eur)}">${isNum(x.eur) ? esc(signedEur(x.eur)) : ''}</span>
            </button>
            <p class="learning-finding">${esc(x.finding)}</p>
            <div class="learning-body" hidden>
                <p><strong>What to do:</strong> ${esc(x.lesson)}</p>
                <p class="muted small">Why this confidence: ${esc(x.why)}${x.n ? ` · ${plural(x.n, 'case', 'cases')}` : ''}</p>
                ${table}
            </div>
        </div>`;
    }

    function renderRecord(d) {
        const s = d.summary || {};
        $('#rec-results').hidden = false;
        $('#rec-caption').textContent = `${d.generated_at ? `Last replayed ${fmtDay(d.generated_at)} · ` : ''}${fmtDay(d.period.first_trade)} to ${fmtDay(d.period.end)} · realised P&L by ${d.method === 'fifo' ? 'FIFO' : 'average cost'}`;
        setIf($('#rec-run'), 'textContent', 'Run again');
        const bl = benchList(d), primary = bl[0];
        const stats = [
            statHtml('Account value', eur(s.end_value), `cash ${eur(s.cash)} · invested ${eur(s.invested_value)}`),
            statHtml('Gain', signedEur(s.gain), `on ${eur(s.net_external_flows)} put in, net`, signCls(s.gain)),
        ];
        if (primary) {
            stats.push(
                statHtml('Invested part a year', fmt.spct(s.twr_invested_annual), `time-weighted, cash left out; money-weighted ${isNum(s.invested_xirr) ? fmt.spct(s.invested_xirr) : 'under a year'}`, signCls(s.twr_invested_annual)),
                statHtml(`${primary.label} a year`, fmt.spct(primary.annual), bl.slice(1).map((b) => `${b.label} ${fmt.spct(b.annual)}`).join(' · ')),
                statHtml(`Same buys in ${primary.label}`, signedEur(primary.buys_difference),
                    isNum(primary.buys_end_value) ? `your holdings ${eur(s.invested_value)} against ${eur(primary.buys_end_value)} in the index` : 'your sales took out more than the index held; see the chart', signCls(primary.buys_difference)),
            );
        } else {
            stats.push(
                statHtml('Invested part a year', fmt.spct(s.twr_invested_annual), `cash was ${fmt.pct(s.avg_cash_share, 0)} of the account on average`),
                statHtml('MSCI World a year', fmt.spct(s.benchmark_annual), `same flows would be ${eur(s.benchmark_end_value)}`),
            );
        }
        $('#rec-stats').innerHTML = stats.join('');
        renderRecChart(d);
        renderRecStyle(d);
        renderRecEntries(d);
        renderRecRules(d.rules_lab);
        renderRecTrips(d);

        const learnings = (d.learnings || []).filter((x) => recSection(x) === 'overview');
        $('#rec-learnings').innerHTML = learnings.slice(0, state.recAllLearnings ? learnings.length : 5).map(learningHtml).join('');
        $('#rec-learnings-more').hidden = state.recAllLearnings || learnings.length <= 5;
        $('#rec-learnings-more').textContent = `Show all ${learnings.length}`;

        const altTable = (rows) => `<thead><tr><th>Alternative</th><th class="num">Account today</th><th class="num">Against what you did</th><th>Note</th></tr></thead><tbody>${
            rows.map((a) => `<tr class="${a.key === 'actual' ? 'is-current' : ''}"><td><strong>${esc(a.label)}</strong>${a.key === 'cash_to_core' ? ' <button class="link-btn" type="button" data-rec-goto="rules">Rules lab</button>' : ''}</td><td data-label="Account today" class="num">${esc(eur(a.end_value))}</td>
                <td data-label="Against what you did" class="num ${a.key === 'actual' ? 'muted' : signCls(a.difference)}">${a.key === 'actual' ? 'reference' : esc(signedEur(a.difference))}</td><td class="wrap muted small">${esc(a.note || '')}</td></tr>`).join('')}</tbody>`;
        const alternatives = d.alternatives || [];
        $('#rec-alternatives').innerHTML = altTable(alternatives.filter((a) => recSection(a) === 'overview'));

        const realisedKey = d.method === 'fifo' ? 'realised_fifo' : 'realised_average';
        const yearHead = bl.length ? bl.map((b) => `<th class="num">${esc(b.label)}</th>`).join('') : '<th class="num">MSCI World</th>';
        const yearIdx = (y) => (bl.length ? bl.map((b) => `<td class="num" data-label="${esc(b.label)}">${esc(fmt.spct((y.benchmarks || {})[b.id]))}</td>`).join('') : `<td class="num" data-label="MSCI World">${esc(fmt.spct(y.benchmark))}</td>`);
        const against = (y) => {
            if (!primary) return '';
            const v = isNum(y.twr_invested) && isNum((y.benchmarks || {})[primary.id]) ? y.twr_invested - y.benchmarks[primary.id] : null;
            return `<td class="num ${signCls(v)}" data-label="Against ${esc(primary.label)}">${esc(isNum(v) ? `${fmt.signed(v * 100, 1)} pts` : 'n/a')}</td>`;
        };
        $('#rec-years').innerHTML = `<thead><tr><th>Year</th><th class="num">Invested part</th>${yearHead}${primary ? `<th class="num">Against ${esc(primary.label)}</th>` : ''}<th class="num">Gain</th><th class="num">Realised</th><th class="num">Dividends</th><th class="num">Fees</th></tr></thead><tbody>${
            (d.years || []).map((y) => `<tr><td class="span"><strong>${esc(y.year)}</strong>${y.partial ? ' <span class="muted small">part</span>' : ''}</td><td class="num" data-label="Invested part">${esc(fmt.spct(y.twr_invested))}</td>${yearIdx(y)}${against(y)}
                <td class="num ${signCls(y.gain)}" data-label="Gain">${esc(signedEur(y.gain))}</td><td class="num" data-label="Realised">${esc(eur(y[realisedKey]))}</td><td class="num" data-label="Dividends">${esc(eur(y.dividends_net))}</td><td class="num" data-label="Fees">${esc(eur(y.fees))}</td></tr>`).join('')}</tbody>`;
        renderRecCash(d, altTable);

        const holdings = d.instruments || [];
        $('#rec-holdings-caption').textContent = `${plural(holdings.length, 'instrument', 'instruments')}; total P&L includes dividends, fees and today's value of what you still hold`;
        const mwTip = 'Money-weighted return a year (XIRR) of the money put into the holding. Not annualised when the money was in for less than a year: a few weeks\' gain scaled up to a year is not a rate you earned.';
        const mwCell = (h) => (isNum(h.xirr) ? esc(fmt.spct(h.xirr)) : `<span class="muted small">${isNum(h.span_days) && h.span_days < 365 ? 'under a year' : 'n/a'}</span>`);
        $('#rec-holdings').innerHTML = `<thead><tr><th>Holding</th><th>Group</th><th>Status</th><th class="num">Bought</th><th class="num">Total P&amp;L</th><th class="num">Return</th><th class="num"><span class="term tip-end" tabindex="0" data-tip="${esc(mwTip)}">Money-weighted a year</span></th></tr></thead><tbody>${
            holdings.slice(0, state.recAllHoldings ? holdings.length : 10).map((h) => `<tr><td>${esc(h.name)} <span class="muted mono small">${esc(h.ticker || '')}</span></td><td>${esc(h.group)}</td><td>${esc(h.status)}</td>
                <td class="num">${esc(eur(h.bought))}</td><td class="num ${signCls(h.total_pnl)}">${esc(signedEur(h.total_pnl))}</td><td class="num">${esc(fmt.spct(h.return_pct))}</td><td class="num">${mwCell(h)}</td></tr>`).join('')}</tbody>`;
        $('#rec-holdings-more').hidden = state.recAllHoldings || holdings.length <= 10;
        $('#rec-holdings-more').textContent = `Show all ${holdings.length}`;
        $('#rec-notes').innerHTML = (isOldRecord(d) ? OLD_RECORD_WARNING : '') + (d.notes || []).map((n) => `<div class="warning note"><strong>Note</strong>${esc(n)}</div>`).join('');
        setRecSection(state.recSection || storageGet('fastvalue-rec-section') || 'overview');
    }

    // ---- track record: sections, your style, entries and exits, rules lab

    const REC_SECTIONS = ['overview', 'cash', 'style', 'entries', 'rules', 'holdings'];
    const REC_RULE_COLORS = ['--cat-1', '--cat-2', '--cat-3', '--cat-4', '--cat-5', '--cat-6'];
    const MAX_RULE_LINES = 6;
    const fmtDays = (v) => (isNum(v) ? `${v.toFixed(0)} days` : 'n/a');
    const confBadge = (c) => `<span class="conf-badge conf-${esc(c)}">${esc(CONF_LABEL[c] || c || '')}</span>`;

    function setRecSection(name) {
        if (!REC_SECTIONS.includes(name)) name = 'overview';
        state.recSection = name;
        storageSet('fastvalue-rec-section', name);
        $$('#rec-sections .seg').forEach((b) => { const on = b.dataset.recSection === name; b.classList.toggle('active', on); b.setAttribute('aria-selected', String(on)); });
        $$('#rec-results .rec-pane').forEach((p) => { p.hidden = p.dataset.pane !== name; });
        if (state.rec && name === 'overview') renderRecChart(state.rec);
        if (state.rec && name === 'cash') renderRecCashCharts(state.rec);
        if (state.rec && name === 'rules') renderRecRulesChart();
        const target = `#backtest=record${name === 'overview' ? '' : `/${name}`}`;
        if (state.bkMode === 'record' && location.hash.startsWith('#backtest') && location.hash !== target) setHash(target);
    }

    function renderRecChart(d) {
        if ($('#rec-chart').closest('.rec-pane').hidden) return;
        const bl = benchList(d);
        const modern = bl.length && (d.growth || {}).benchmarks && d.invested_equity;
        $('#rec-chart-mode').hidden = !modern;
        let lines, axis, format, hint;
        if (modern) {
            if (!state.recChartMode) state.recChartMode = storageGet('fastvalue-rec-chart-mode') === 'eur' ? 'eur' : 'twr';
            const mode = state.recChartMode;
            segButtons('#rec-chart-mode', [['twr', 'Time-weighted'], ['eur', 'In euros']], mode, 'rec-chart-mode');
            const eurMode = mode === 'eur';
            lines = [{ key: 'invested', label: eurMode ? 'Your holdings' : 'Invested part', data: eurMode ? d.invested_equity.invested_part : d.growth.invested_part, color: cssVar('--accent'), width: 2.5 }];
            bl.forEach((b, i) => {
                const data = (eurMode ? d.invested_equity.benchmarks : d.growth.benchmarks)[b.id];
                if (data) lines.push({ key: b.id, label: eurMode ? `Same buys in ${b.label}` : b.label, data, color: benchColor(b.id, i), width: 1.5 });
            });
            axis = eurMode ? MONEY_AXIS : {};
            format = eurMode ? MONEY_FORMAT : { type: 'price', precision: 1, minMove: 0.1 };
            const dipped = bl.filter((b) => b.buys_dipped).map((b) => b.label);
            hint = (eurMode
                ? "Your holdings' value against the same buys and sales put into each index on the same days, every sale and dividend taken out again. Cash is in its own section. Select a name to hide or show its line."
                : 'Growth of 100 from your first trade: your stock and fund choices, with deposits, withdrawals and idle cash left out, against each index. Select a name to hide or show its line.')
                + (eurMode && dipped.length ? ` Your sales at times took out more than the same buys had grown to in ${dipped.join(' and ')}, so its line dips below zero: it is a yardstick, not a holding.` : '');
        } else {
            const eq = d.equity || {};
            lines = [
                { key: 'account', label: 'Your account', data: eq.account, color: cssVar('--accent'), width: 2.5 },
                { key: 'invested', label: 'Put in, net', data: eq.invested, color: cssVar('--series-8'), width: 1.5, dash: true },
                { key: 'benchmark', label: 'Same flows in MSCI World', data: eq.benchmark, color: benchColor('msci_world'), width: 1.5 },
            ].filter((l) => l.data);
            axis = MONEY_AXIS; format = MONEY_FORMAT;
            hint = 'Older track record: your account (cash plus holdings, EUR), the money you put in net, and the same flows in MSCI World. Run it again for the invested part against the indices.';
        }
        const chart = makeChart('rec-chart', axis, chartSummary('Your track record', lines, (v) => fmt.num(v, 0)));
        lines.forEach((l) => {
            l.series = chart.addLineSeries({ color: l.color, lineWidth: l.width, lineStyle: l.dash ? 2 : 0, priceLineVisible: false, lastValueVisible: false, priceFormat: format });
            l.series.setData(l.data);
        });
        legend('rec-legend', lines.map(({ key, label, color, dash, series }) => ({ key, label, color, dash, series })), { chart: 'rec-chart' });
        $('#rec-chart-hint').textContent = hint;
        chart.timeScale().fitContent();
    }

    // ---- track record: cash

    function renderRecCash(d, altTable) {
        const s = d.summary || {}, c = d.cash, bl = benchList(d), primary = bl[0];
        const learnings = (d.learnings || []).filter((x) => recSection(x) === 'cash');
        const old = !c || !primary;
        $$('#rec-results [data-pane="cash"] > .card:not(#rec-cash-learning-card)').forEach((el) => { el.hidden = old; });
        $('#rec-cash-learning-card').hidden = !learnings.length && !old;
        if (old) {
            $('#rec-cash-stats').innerHTML = [
                statHtml('Cash now', eur(s.cash), `${fmt.pct(isNum(s.end_value) && s.end_value > 0 ? s.cash / s.end_value : null, 0)} of the account`),
                statHtml('Average cash share', fmt.pct(s.avg_cash_share, 0), 'of the account since your first trade'),
                statHtml('Whole account a year', fmt.spct(s.twr_annual), `cash included; money-weighted ${fmt.spct(s.xirr)}`, signCls(s.twr_annual)),
                statHtml('Every deposit into MSCI World', signedEur(isNum(s.benchmark_end_value) ? s.benchmark_end_value - s.end_value : null), `${eur(s.benchmark_end_value)} in the index against ${eur(s.end_value)} in your account`),
            ].join('');
            $('#rec-cash-learning').innerHTML = OLD_RECORD_WARNING + learnings.map((x) => learningHtml(x, 0, { rank: false })).join('');
            destroyChart('rec-cash-chart'); destroyChart('rec-cash-share-chart');
            return;
        }
        $('#rec-cash-stats').innerHTML = [
            statHtml('Cash now', eur(c.now), `${fmt.pct(c.share_now, 0)} of the account · ${fmt.pct(c.avg_share, 0)} on average`),
            statHtml('Interest after tax', eur(c.interest_net), isNum(c.interest_rate) ? `about ${fmt.pct(c.interest_rate, 1)} a year on the average balance` : 'n/a'),
            statHtml('Cash effect', isNum(c.effect_annual) ? `${fmt.signed(c.effect_annual * 100, 1)} points a year` : 'n/a',
                `time-weighted: whole account ${fmt.spct(c.twr_annual)} a year, invested part ${fmt.spct(s.twr_invested_annual)}`, signCls(c.effect_annual)),
            statHtml('Whole account a year', fmt.spct(c.twr_annual), `cash included; money-weighted ${fmt.spct(c.xirr)}`, signCls(c.twr_annual)),
            statHtml(`Against every deposit in ${primary.label}`, signedEur(primary.deposits_difference),
                `${eur(primary.deposits_end_value)} in the index · ${signedEur(primary.buys_difference)} your choices, ${signedEur(primary.cash_part)} cash`, signCls(primary.deposits_difference)),
        ].join('');
        const alternatives = (d.alternatives || []);
        $('#rec-cash-alternatives').innerHTML = altTable([...alternatives.filter((a) => a.key === 'actual'), ...alternatives.filter((a) => recSection(a) === 'cash')]);
        $('#rec-cash-years').innerHTML = `<thead><tr><th>Year</th><th class="num">Start</th><th class="num">End</th><th class="num">Put in, net</th><th class="num">Whole account</th><th class="num">Invested part</th><th class="num">Cash effect</th><th class="num">Cash share</th><th class="num">Interest</th><th class="num">Modified Dietz</th></tr></thead><tbody>${
            (d.years || []).map((y) => {
                const effect = isNum(y.twr) && isNum(y.twr_invested) ? y.twr - y.twr_invested : null;
                return `<tr><td>${esc(y.year)}${y.partial ? ' <span class="muted small">part</span>' : ''}</td><td class="num">${esc(eur(y.start_value))}</td><td class="num">${esc(eur(y.end_value))}</td>
                    <td class="num">${esc(eur(y.net_flows))}</td><td class="num ${signCls(y.twr)}">${esc(fmt.spct(y.twr))}</td><td class="num">${esc(fmt.spct(y.twr_invested))}</td>
                    <td class="num ${signCls(effect)}">${esc(isNum(effect) ? `${fmt.signed(effect * 100, 1)} pts` : 'n/a')}</td><td class="num">${esc(fmt.pct(y.avg_cash_share, 0))}</td><td class="num">${esc(eur(y.interest_net))}</td><td class="num">${esc(fmt.spct(y.dietz))}</td></tr>`;
            }).join('')}</tbody>`;
        $('#rec-cash-learning').innerHTML = learnings.map((x) => learningHtml(x, 0, { rank: false })).join('');
        $('#rec-cash-share-caption').textContent = `${fmt.pct(c.avg_share, 0)} on average since ${fmtDay(d.period.first_trade)}`;
        renderRecCashCharts(d);
    }

    function renderRecCashCharts(d) {
        if ($('#rec-cash-chart').closest('.rec-pane').hidden) return;
        const c = d.cash, bl = benchList(d);
        if (!c || !bl.length) return;
        const eq = d.equity || {};
        const lines = [
            { key: 'account', label: 'Your account', data: eq.account, color: cssVar('--accent'), width: 2.5 },
            { key: 'put_in', label: 'Put in, net', data: eq.invested, color: cssVar('--series-8'), width: 1.5, dash: true },
        ];
        bl.forEach((b, i) => {
            const data = (eq.benchmarks || {})[b.id] || (b.primary ? eq.benchmark : null);
            if (data) lines.push({ key: b.id, label: `Every deposit in ${b.label}`, data, color: benchColor(b.id, i), width: 1.5 });
        });
        const chart = makeChart('rec-cash-chart', MONEY_AXIS);
        lines.forEach((l) => {
            l.series = chart.addLineSeries({ color: l.color, lineWidth: l.width, lineStyle: l.dash ? 2 : 0, priceLineVisible: false, lastValueVisible: false, priceFormat: MONEY_FORMAT });
            l.series.setData(l.data || []);
        });
        legend('rec-cash-legend', lines.map(({ key, label, color, dash, series }) => ({ key, label, color, dash, series })), { chart: 'rec-cash-chart' });
        chart.timeScale().fitContent();

        const share = makeChart('rec-cash-share-chart', { localization: { priceFormatter: (v) => `${Math.round(v)}%` } });
        const line = share.addLineSeries({ color: cssVar('--text-3'), lineWidth: 2, priceLineVisible: false, lastValueVisible: false, priceFormat: { type: 'price', precision: 0, minMove: 1 },
            autoscaleInfoProvider: (original) => {   // keep the 10% reserve line and zero in view
                const r = original();
                if (!r || !r.priceRange) return r;
                return { ...r, priceRange: { minValue: Math.min(0, r.priceRange.minValue), maxValue: Math.max(10, r.priceRange.maxValue) } };
            } });
        line.setData((c.share || []).map((p) => ({ time: p.time, value: p.value * 100 })));
        line.createPriceLine({ price: 10, color: cssVar('--text-3'), lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: '10% reserve (rules lab)' });
        share.timeScale().fitContent();
    }

    // One-series horizontal bars: the bar pictures the share, the text beside it carries the numbers.
    function barRows(items, color = 'var(--cat-1)') {
        const top = Math.max(0.0001, ...items.map((it) => Math.abs(it.value || 0)));
        return items.map((it) => `<div class="hbar-row">
            <span class="hbar-label${it.title && it.title !== it.label ? ' term quiet' : ''}"${it.title && it.title !== it.label ? ` tabindex="0" data-tip="${esc(it.title)}"` : ''}>${esc(it.label)}</span>
            <span class="hbar-track"><span class="hbar" style="width:${((Math.abs(it.value || 0) / top) * 100).toFixed(1)}%;background:${color}"${it.tip ? ` data-ctip="${esc(it.tip)}"` : ''}></span></span>
            <span class="hbar-values">${esc(it.text || '')}</span></div>`).join('');
    }

    function cutText(r) {
        return `${r.n} trips · ${fmt.pct(r.hit_rate, 0)} won · ${signedEur(r.pnl)}${isNum(r.pnl_share) ? ` (${fmt.pct(r.pnl_share, 0)} of P&L)` : ''}`;
    }

    function renderRecStyle(d) {
        const sp = d.style;
        const stats = $('#rec-style-stats');
        if (!sp) {
            stats.innerHTML = '';
            ['#rec-pareto', '#rec-holding', '#rec-buckets', '#rec-building'].forEach((id) => { $(id).innerHTML = ''; });
            $('#rec-top5').innerHTML = '';
            $('#rec-lessons').innerHTML = '<p class="muted small">The style profile did not run; see the notes in Overview.</p>';
            return;
        }
        const st = sp.style;
        const rt = st.round_trips, hr = st.hit_rate, pc = st.concentration_of_pnl, b = st.bets, py = st.pyramiding, td = st.today, dv = st.dividends, c = st.costs || {};
        stats.innerHTML = [
            statHtml('Closed round trips', String(rt.closed), `${rt.open} open · median ${fmtDays(rt.median_days_closed)} · ${fmt.pct(rt.share_closed_quickly, 0)} closed within ${rt.quick_days} days`),
            statHtml('Hit rate', fmt.pct(hr.rate, 0), `${plural(hr.wins, 'win', 'wins')}, ${plural(hr.losses, 'loss', 'losses')} · payoff ${fmt.ratio(hr.payoff)} · profit factor ${fmt.ratio(hr.profit_factor)}`),
            statHtml('Average win / loss', `${eur(hr.avg_win)} / ${eur(hr.avg_loss)}`, `expectancy ${signedEur(hr.expectancy)} per closed trip`, signCls(hr.expectancy)),
            statHtml('Positions for 80% of gains', String(pc.positions_for_80pct_of_gains), `of ${pc.positions_with_gains} winners · top 8 trips are ${fmt.pct(pc.top8_share, 0)} of the P&L`),
            statHtml('One-off bets', String(b.discretionary_buys), `median ${eur(b.discretionary_median)} · ${plural(b.discretionary_names, 'name', 'names')} · ${eur(b.discretionary_total)} in all`),
            statHtml('Savings-plan buys', String(b.plan_buys), `${plural(b.plan_names, 'name', 'names')} · ${eur(b.plan_total)} in all`),
            statHtml('Buys per position', fmt.num(py.median_buys_per_trip, 0), `median · ${py.trips_with_5plus_buys} built with 5+ buys carry ${fmt.pct(py.pnl_share_5plus_buys, 0)} of the P&L`),
            statHtml('Top 3 today', fmt.pct(td.top3_weight, 0), `${td.top1_name || 'largest'} ${fmt.pct(td.top1_weight, 0)} · ${plural(td.positions, 'position', 'positions')}, ${fmt.num(td.effective_positions, 1)} effective`),
            statHtml('Cash', fmt.pct(td.cash_share, 0), `now · ${fmt.pct(td.avg_cash_share, 0)} on average`),
            statHtml('Still unrealised', fmt.pct(td.unrealised_share_of_gain, 0), `of the total P&L: ${eur(td.unrealised)} of ${eur(td.total_pnl)}`),
            statHtml('One-off dividends', fmt.pct(dv.one_off_share, 0), `of dividend income · ${plural(dv.names_recurring, 'name', 'names')} ${dv.names_recurring === 1 ? 'pays' : 'pay'} regularly`),
            statHtml('Turnover', fmt.pct(c.turnover_per_year, 0), `of the invested value a year · fees ${eur(Math.abs(c.fees || 0))}`),
        ].join('');

        const trips = (d.round_trips || []).filter((t) => isNum(t.total_pnl)).sort((p, q) => q.total_pnl - p.total_pnl);
        const total = trips.reduce((x, t) => x + t.total_pnl, 0);
        const shown = [...trips.slice(0, 8), ...trips.slice(-4).filter((t) => t.total_pnl < 0 && !trips.slice(0, 8).includes(t))];
        $('#rec-pareto-caption').textContent = `${plural(trips.length, 'round trip', 'round trips')} · top 8 are ${fmt.pct(pc.top8_share, 0)} of the total`;
        $('#rec-pareto').classList.add('wide');
        $('#rec-pareto').innerHTML = shown.length ? dbarRows(shown.map((t) => ({ ticker: t.name, name: t.name, pnl: t.total_pnl, share: total > 0 ? t.total_pnl / total : null })),
            'pnl', (it) => signedEur(it.pnl), (it) => `${it.name}: ${signedEur(it.pnl)}${isNum(it.share) ? `, ${fmt.pct(it.share, 0)} of the total` : ''}`) : '<p class="muted small">No round trips yet.</p>';

        const cuts = sp.cuts || {};
        const hold = cuts['holding period'] || [];
        const nHold = hold.reduce((x, r) => x + r.n, 0) || 1;
        $('#rec-holding').innerHTML = barRows(hold.map((r) => ({ label: r.label, value: r.n / nHold, text: cutText(r), tip: `${r.label}: ${fmt.pct(r.n / nHold, 0)} of round trips, median ${fmtDays(r.median_days)}` })));
        $('#rec-buckets').innerHTML = barRows((st.buckets || []).map((g) => ({ label: g.bucket, value: g.weight_now, text: `${fmt.pct(g.weight_now, 0)} today · ${plural(g.instruments, 'instrument', 'instruments')} · P&L ${signedEur(g.pnl)} (${fmt.spct(g.return_pct, 0)})` })), 'var(--cat-3)');
        const nb = cuts['number of buys'] || [];
        const nNb = nb.reduce((x, r) => x + r.n, 0) || 1;
        $('#rec-building').innerHTML = barRows(nb.map((r) => ({ label: r.label, value: r.n / nNb, text: cutText(r) })), 'var(--cat-2)');
        $('#rec-top5').innerHTML = `<thead><tr><th>Position</th><th class="num">Buys</th><th>Built over</th><th>Held</th><th class="num">P&amp;L</th><th class="num">Return</th><th>Status</th></tr></thead><tbody>${
            (py.top5 || []).map((t) => `<tr><td>${esc(t.name)}</td><td class="num">${esc(t.buys)}</td><td>${esc(fmtDays(t.build_days))}</td><td>${esc(fmtDays(t.days))}</td><td class="num ${signCls(t.pnl)}">${esc(signedEur(t.pnl))}</td><td class="num">${esc(fmt.spct(t.return_pct))}</td><td>${esc(t.outcome)}</td></tr>`).join('')}</tbody>`;

        $('#rec-lessons').innerHTML = (sp.lessons || []).map((l) => `<div class="learning conf-${esc(l.confidence)}"><div class="learning-head static"><span class="learning-title">${esc(l.title)}</span>${confBadge(l.confidence)}</div><p class="lesson-text">${esc(l.text)}</p></div>`).join('')
            || '<p class="muted small">No lessons yet.</p>';
    }

    // Buckets are ranked by the return after the order. After a sale that return is what the name did without you, so the
    // exits table names its columns by that move rather than calling the bucket whose names rose most the best.
    const BUCKET_WORDS = { BUY: 'at or above the entry level', HOLD: 'between the levels', SELL: 'at or below the exit level' };
    const bucketWord = (b) => BUCKET_WORDS[b] || b;
    function tagsTable(tags, horizon, unit = 'buys', heads = ['Best bucket', 'Worst bucket']) {
        if (!tags || !tags.length) return '<tbody><tr><td class="muted small">Not enough tagged orders yet.</td></tr></tbody>';
        return `<thead><tr><th>Condition</th><th>${esc(heads[0])}</th><th>${esc(heads[1])}</th><th class="num">Gap after ${horizon} days</th><th class="num">1st half</th><th class="num">2nd half</th><th class="num">t</th><th>Confidence</th></tr></thead><tbody>${tags.map((t, i) => {
            const b = t.buckets.find((x) => x.bucket === t.best) || {};
            const w = t.buckets.find((x) => x.bucket === t.worst) || {};
            return `<tr class="clickable tag-row" data-tag-row="${i}"><td><button type="button" class="co-rowbtn" aria-expanded="false"><strong>${esc(capFirst(t.tag))}</strong></button></td>
                <td class="wrap">${esc(bucketWord(t.best))} <span class="muted small">${esc(b.n || 0)} ${unit}, ${esc(fmt.spct(b.mean))}</span></td>
                <td class="wrap">${esc(bucketWord(t.worst))} <span class="muted small">${esc(w.n || 0)} ${unit}, ${esc(fmt.spct(w.mean))}</span></td>
                <td class="num ${signCls(t.spread)}">${esc(fmt.spct(t.spread))}</td><td class="num ${signCls(t.first_half)}">${esc(fmt.spct(t.first_half))}</td>
                <td class="num ${signCls(t.second_half)}">${esc(fmt.spct(t.second_half))}</td><td class="num">${t.t == null ? 'n/a' : esc(t.t.toFixed(1))}</td><td>${confBadge(t.confidence)}</td></tr>
            <tr class="detail" data-tag-detail="${i}" hidden><td colspan="8"><div class="table-wrap"><table class="table compact"><thead><tr><th>Bucket</th><th class="num">Orders</th><th class="num">Name-months</th><th class="num">vs MSCI World, ${horizon} days</th><th class="num">180 days</th><th class="num">What the money made</th><th class="num">EUR vs MSCI World</th></tr></thead><tbody>${
                t.buckets.map((x) => `<tr><td>${esc(bucketWord(x.bucket))}</td><td class="num">${esc(x.n)}</td><td class="num">${esc(x.n_eff)}</td><td class="num ${signCls(x.mean)}">${esc(fmt.spct(x.mean))}</td><td class="num">${esc(fmt.spct(x.mean_180))}</td><td class="num">${esc(fmt.spct(x.money_return))}</td><td class="num">${esc(signedEur(x.eur))}</td></tr>`).join('')}</tbody></table></div>
                <p class="muted small">${esc(t.why)}</p></td></tr>`;
        }).join('')}</tbody>`;
    }

    function renderRecCuts() {
        const sp = (state.rec || {}).style;
        const rows = ((sp || {}).cuts || {})[state.recCut] || [];
        $('#rec-cuts').innerHTML = rows.length ? `<thead><tr><th>${esc(capFirst(state.recCut))}</th><th class="num">Round trips</th><th class="num">Closed</th><th class="num">Hit rate</th><th class="num">Median return</th><th class="num">Median held</th><th class="num">Bought</th><th class="num">P&amp;L</th><th class="num">Share of P&amp;L</th></tr></thead><tbody>${
            rows.map((r) => `<tr><td>${esc(r.label)}</td><td class="num">${esc(r.n)}</td><td class="num">${esc(r.closed)}</td><td class="num">${esc(fmt.pct(r.hit_rate, 0))}</td><td class="num">${esc(fmt.spct(r.median_return))}</td><td class="num">${esc(fmtDays(r.median_days))}</td><td class="num">${esc(eur(r.bought))}</td><td class="num ${signCls(r.pnl)}">${esc(signedEur(r.pnl))}</td><td class="num">${esc(fmt.pct(r.pnl_share, 0))}</td></tr>`).join('')}</tbody>`
            : '<tbody><tr><td class="muted small">Nothing to cut yet.</td></tr></tbody>';
    }

    function renderRecEntries(d) {
        const sp = d.style;
        if (!sp) {
            $('#rec-entry-tags').innerHTML = '<tbody><tr><td class="muted small">The style profile did not run; see the notes in Overview.</td></tr></tbody>';
            renderRecWeights(null);
            $('#rec-cuts').innerHTML = ''; $('#rec-exit-stats').innerHTML = ''; $('#rec-exit-tags').innerHTML = ''; $('#rec-cut-dims').innerHTML = '';
            return;
        }
        const e = sp.entries || {};
        $('#rec-entries-caption').textContent = `${plural(e.n || 0, 'buy', 'buys')} tagged with the conditions at the time; the gap is the best bucket against the worst, 90 days later, against MSCI World`;
        $('#rec-entries-protocol').textContent = sp.protocol || '';
        $('#rec-entry-tags').innerHTML = tagsTable(e.tags, 90);
        renderRecWeights(sp.weights);
        const dims = Object.keys(sp.cuts || {});
        if (!dims.includes(state.recCut)) state.recCut = dims[0];
        segButtons('#rec-cut-dims', dims.map((k) => [k, capFirst(k)]), state.recCut, 'rec-cut');
        renderRecCuts();
        const x = sp.exits || {};
        $('#rec-exits-caption').textContent = `${plural(x.n || 0, 'sale', 'sales')}, and what each name did against MSCI World in the 180 days after (positive: it kept beating the index)`;
        const tile = (label, o) => statHtml(label, o && o.n ? fmt.spct(o.mean_excess_180) : 'n/a', `${(o || {}).n || 0} sales · sold at ${fmt.spct((o || {}).gain_at_sale_mean)} on average · ${CONF_LABEL[(o || {}).confidence] || ''}`, o && o.n ? signCls(-(o.mean_excess_180 || 0)) : '');
        $('#rec-exit-stats').innerHTML = [tile('Sold into weakness', x.into_weakness), tile('Steady names sold within 6 months', x.steady_quick), tile('Losers cut within 90 days', x.losers_cut_fast)].join('');
        $('#rec-exit-tags').innerHTML = tagsTable(x.tags, 180, 'sales', ['Rose most after the sale', 'Rose least after the sale']);
    }

    // ---- your history against the weights (Entries & exits): agents, regimes, what it suggests now
    const WR_TONE = { 'in line': 'live', 'your history disagrees': 'warn', mixed: 'warn', 'too few cases': '', 'not testable': '' };
    const WR_EVIDENCE_TONE = { 'Your history and the S&P 500 panel agree': 'live', 'S&P 500 panel only': 'accent', 'Your history only: a hypothesis to test forward': 'warn', 'Keep what you do': '' };
    const wrTip = (t) => `<span class="info" tabindex="0" role="note" aria-label="${esc(t)}" data-tip="${esc(t)}">i</span>`;
    const wrTerm = (label, t) => `<span class="term" tabindex="0" data-tip="${esc(t)}">${esc(label)}</span>`;
    const wrVerdict = (v, strength) => `<span class="badge ${WR_TONE[v] || ''}">${esc(capFirst(v))}</span>${strength === 'leaning' ? ' <span class="muted small">low confidence</span>' : ''}`;
    const wrGapText = (g) => (g && isNum(g.difference)
        ? `${fmt.spct(g.difference)} <span class="muted small">${esc(g.plus_n)} against ${plural(g.minus_n, 'buy', 'buys')} · halves ${esc(fmt.spct(g.first_half))} / ${esc(fmt.spct(g.second_half))} · t ${g.t == null ? 'n/a' : esc(g.t.toFixed(1))}</span>`
        : '<span class="muted small">too few buys on one side</span>');
    const WR_PANEL_WORD = { agrees: 'Agrees with the agent', disagrees: 'Finds it backwards', mixed: 'Mixed', none: 'No comparable number' };

    function wrWeightText(a) {
        if (a.owner_parts) return a.owner_parts.map((p) => `${esc(p.label.replace(/ Agent$/, ''))} ${esc(fmt.pct(p.weight, 0))}`).join('<br>');
        if (!a.enabled) return '<span class="muted small">switched off</span>';
        const by = a.by_regime || {};
        const list = Object.entries(by).map(([r, w]) => `${r.toLowerCase()} ${fmt.pct(w, 0)}`).join(', ');
        const now = isNum(a.weight_today) ? `${esc(fmt.pct(a.weight_today, 0))} <span class="muted small">today</span>` : '';
        return `${now}<div class="muted small">${esc(list)}</div>`;
    }

    function wrAgentRows(agents) {
        return agents.map((a, i) => {
            const tested = a.lines.filter((l) => l.gap);
            const lead = tested.slice().sort((p, q) => (q.gap.n || 0) - (p.gap.n || 0))[0];
            const hist = lead ? `<strong>${esc(capFirst(lead.tag))}</strong> ${wrGapText(lead.gap)}` : `<span class="muted small">${esc(a.note || '')}</span>`;
            const pv = a.panel ? `${esc(WR_PANEL_WORD[a.panel_verdict] || '')}<div class="muted small">${esc(a.panel.text)}</div>` : `<span class="muted small">${esc((WR_PANEL_WORD[a.panel_verdict] || 'No comparable number'))}</span>`;
            const detail = a.lines.length ? `<table class="table compact"><thead><tr><th>Condition</th><th>What the agent reads</th><th>Rewards</th><th>Penalises</th><th>Rewarded minus penalised, ${90} days</th><th>Best and worst bucket in your buys</th><th>S&amp;P 500 panel</th><th>Verdict</th></tr></thead><tbody>${a.lines.map((l) => {
                const h = l.history;
                return `<tr><td class="wrap"><strong>${esc(capFirst(l.tag))}</strong></td><td class="wrap">${esc(l.reads)}</td><td class="wrap">${esc(l.rewarded.join(', ') || 'n/a')}</td><td class="wrap">${esc(l.penalised.join(', ') || 'n/a')}</td>
                    <td class="wrap num">${wrGapText(l.gap)}</td>
                    <td class="wrap">${h ? `${esc(h.best)} ${esc(fmt.spct(((h.buckets || []).find((b) => b.bucket === h.best) || {}).mean))} · ${esc(h.worst)} ${esc(fmt.spct(((h.buckets || []).find((b) => b.bucket === h.worst) || {}).mean))} <span class="muted small">${plural(h.n, 'case', 'cases')} · ${esc(CONF_LABEL[h.confidence] || '')}</span>` : '<span class="muted small">fewer than two buckets with 3 buys</span>'}</td>
                    <td class="wrap">${esc(WR_PANEL_WORD[l.panel.direction] || '')}<div class="muted small">${esc(l.panel.text)}${l.panel.study ? ` (${esc(l.panel.study)})` : ''}</div></td>
                    <td>${wrVerdict(l.verdict, l.strength)}</td></tr>`;
            }).join('')}</tbody></table>${a.lines.some((l) => l.gap && l.gap.why) ? `<p class="muted small">${esc(a.lines.filter((l) => l.gap).map((l) => `${capFirst(l.tag)}: ${l.gap.why}`).join(' · '))}</p>` : ''}` : `<p class="muted small">${esc(a.note || '')}</p>`;
            return `<tr class="clickable tag-row" data-tag-row="${i}"><td><button type="button" class="co-rowbtn" aria-expanded="false"><strong>${esc(a.label.replace(/ Agent$/, ''))}</strong></button><div class="muted small">${a.layer === 'owner' ? 'decides' : 'informs timing, no vote'}</div></td>
                <td class="wrap">${wrWeightText(a)}</td><td class="wrap">${hist}</td><td class="wrap">${pv}</td><td>${wrVerdict(a.verdict, a.strength)}</td></tr>
                <tr class="detail" data-tag-detail="${i}" hidden><td colspan="5"><div class="table-wrap">${detail}</div>${a.by_regime && Object.keys(a.by_regime).length ? `<p class="muted small">Weight by regime: ${esc(Object.entries(a.by_regime).map(([r, w]) => `${r} ${fmt.pct(w, 0)}`).join(', '))}. Change weights in the Weights tab.</p>` : ''}</td></tr>`;
        }).join('');
    }

    function renderRecWeights(w) {
        const hosts = ['#rec-wr-agents', '#rec-wr-unread', '#rec-reg-table'];
        const clear = (msg) => {
            hosts.forEach((h) => { $(h).innerHTML = ''; });
            $('#rec-wr-agents').innerHTML = `<tbody><tr><td class="muted small">${esc(msg)}</td></tr></tbody>`;
            ['#rec-wr-caption', '#rec-wr-summary', '#rec-wr-protocol', '#rec-reg-caption', '#rec-now-caption'].forEach((h) => { $(h).textContent = ''; });
            $('#rec-now-list').innerHTML = '';
        };
        if (!w || w.error) { clear(w && w.error ? 'This part did not run; see the notes in Overview.' : 'Run My track record again to add this part.'); return; }
        const agents = w.agents || [];
        $('#rec-wr-caption').textContent = `weights as saved in the Weights tab · ${w.regime_available ? 'regime at entry added to the tags' : 'regime not available (macro data did not load)'}`;
        $('#rec-wr-agents').innerHTML = `<thead><tr><th>${wrTerm('Agent', 'A scoring module. Open a row to see each condition it reads.')}</th><th>${wrTerm('Weight', 'Its share of its layer: the owner agents split the owner score, each regime has its own timing profile.')}</th><th>What your history says</th><th>${wrTerm('S&P 500 panel', 'The market-wide study of the same condition or of the agent itself (point in time, S&P 500 members, 2011 to 2026). The numbers are in the panel evidence file; the regime study can update them.')}</th><th>${wrTerm('Verdict', 'In line: the buckets this agent rewards did better in your buys. Your history disagrees: they did worse. Too few cases: fewer than 10 usable cases. Not testable: none of your entry tags reads this agent.')}</th></tr></thead><tbody>${wrAgentRows(agents)}</tbody>`;
        const counts = { 'in line': 0, 'your history disagrees': 0, mixed: 0, 'too few cases': 0, 'not testable': 0 };
        agents.forEach((a) => { counts[a.verdict] = (counts[a.verdict] || 0) + 1; });
        $('#rec-wr-summary').textContent = `${plural(counts['in line'], 'agent', 'agents')} in line, ${counts['your history disagrees']} where your history disagrees, ${counts.mixed} mixed, ${counts['too few cases']} with too few cases, ${counts['not testable']} that no entry tag tests. A verdict marked low confidence could be chance and is not acted on.`;
        const un = w.unread || [];
        $('#rec-wr-unread').innerHTML = un.length ? `<thead><tr><th>Condition</th><th>Why no agent</th><th>Best bucket</th><th>Worst bucket</th><th class="num">Gap</th><th class="num">Buys</th><th>Confidence</th></tr></thead><tbody>${un.map((u) => `<tr><td><strong>${esc(capFirst(u.tag))}</strong></td><td class="wrap">${esc(u.note)}</td><td class="wrap">${esc(u.best)}</td><td class="wrap">${esc(u.worst)}</td><td class="num ${signCls(u.spread)}">${esc(fmt.spct(u.spread))}</td><td class="num">${esc(u.n)}</td><td>${confBadge(u.confidence)}</td></tr>`).join('')}</tbody>` : '';
        $('#rec-wr-protocol').textContent = w.protocol || '';

        const regs = w.regimes || [];
        const today = ((w.now || {}).now || {}).regime;
        $('#rec-reg-caption').textContent = regs.length ? `${plural(regs.reduce((x, r) => x + r.n, 0), 'buy', 'buys')} by the regime the day before · return against MSCI World after 90 days${today ? ` · today: ${today}` : ''}` : '';
        $('#rec-reg-table').innerHTML = regs.length ? `<thead><tr><th>Regime</th><th class="num">Buys</th><th class="num">Median</th><th class="num">Hit rate</th><th class="num">1st half</th><th class="num">2nd half</th><th>${wrTerm('Against your other regimes', 'The average return of your buys in this regime minus your buys in the other regimes, with the track record\'s t statistic and confidence.')}</th><th>${wrTerm('S&P 500 panel', 'Stocks at or above the entry level in this regime, next month annualised, from the market-wide study.')}</th><th>${wrTerm('Heaviest agents', 'The timing agents with the largest weight in this regime\'s profile (Weights tab).')}</th></tr></thead><tbody>${regs.map((r, i) => `
            <tr class="clickable tag-row ${r.regime === today ? 'is-current' : ''}" data-tag-row="${i}"><td><button type="button" class="co-rowbtn" aria-expanded="false"><strong>${esc(r.regime)}</strong></button>${r.regime === today ? ' <span class="badge accent">today</span>' : ''}</td>
                <td class="num">${esc(r.n)}</td><td class="num ${signCls(r.median)}">${esc(fmt.spct(r.median))}</td><td class="num">${esc(fmt.pct(r.hit_rate, 0))}</td>
                <td class="num ${signCls(r.first_half)}">${esc(fmt.spct(r.first_half))} <span class="muted small">${esc(r.n_first)}</span></td><td class="num ${signCls(r.second_half)}">${esc(fmt.spct(r.second_half))} <span class="muted small">${esc(r.n_second)}</span></td>
                <td class="wrap">${r.against_rest ? `${esc(fmt.spct(r.against_rest.difference))} <span class="muted small">t ${r.against_rest.t == null ? 'n/a' : esc(r.against_rest.t.toFixed(1))}</span> ${confBadge(r.against_rest.confidence)}` : '<span class="muted small">too few buys</span>'}</td>
                <td class="wrap">${r.panel ? `${esc(fmt.pct(r.panel.value, 1))}${isNum(r.panel.alt) ? ` <span class="muted small">${esc(fmt.pct(r.panel.alt, 1))} over 12 months</span>` : ''}` : '<span class="muted small">n/a</span>'}</td>
                <td class="wrap">${esc((r.heaviest_agents || []).map((h) => `${h.label.replace(/ (& .*)?Agent$/, '')} ${fmt.pct(h.weight, 0)}`).join(', '))}</td></tr>
            <tr class="detail" data-tag-detail="${i}" hidden><td colspan="9">${r.too_few ? `<p class="muted small">${plural(r.n, 'buy', 'buys')} in this regime: too few to cut by condition (at least 10 are needed). Keep what you do.</p>`
                : (r.inside || []).length ? `<table class="table compact"><thead><tr><th>Inside ${esc(r.regime)}</th><th>Best bucket</th><th>Worst bucket</th><th class="num">Gap</th><th class="num">1st half</th><th class="num">2nd half</th><th class="num">t</th><th>Confidence</th><th>S&amp;P 500 panel</th></tr></thead><tbody>${r.inside.map((x) => `<tr><td><strong>${esc(capFirst(x.tag))}</strong></td><td class="wrap">${esc(x.best)}</td><td class="wrap">${esc(x.worst)}</td><td class="num ${signCls(x.spread)}">${esc(fmt.spct(x.spread))}</td><td class="num">${esc(fmt.spct(x.first_half))}</td><td class="num">${esc(fmt.spct(x.second_half))}</td><td class="num">${x.t == null ? 'n/a' : esc(x.t.toFixed(1))}</td><td>${confBadge(x.confidence)}</td><td class="wrap muted small">${x.panel ? esc(x.panel.text) : 'not measured inside this regime'}</td></tr>`).join('')}</tbody></table><p class="muted small">Buckets with at least 3 buys; the market's own trend, VIX and drawdown are left out because they define the regime. Small groups: treat as a pattern to watch, not a rule.</p>`
                : '<p class="muted small">No condition has two buckets with at least 3 buys here yet.</p>'}</td></tr>`).join('')}</tbody>` : '<tbody><tr><td class="muted small">Regimes could not be added to your buys (the macro series did not load). Run the track record again; Yahoo sometimes refuses and works on a retry.</td></tr></tbody>';

        const now = w.now || {};
        const n = now.now;
        $('#rec-now-caption').textContent = n ? `today: ${n.regime}${isNum(n.vix) ? ` · VIX ${n.vix.toFixed(1)}` : ''}${isNum(n.spy_vs_200d) ? ` · S&P 500 ${fmt.spct(n.spy_vs_200d, 0)} against its 200-day mean` : ''}${isNum(n.market_from_high) ? ` · ${fmt.pct(Math.abs(n.market_from_high), 0)} off its high` : ''}` : 'the live market regime is not available';
        const item = (r) => `<div class="learning"><div class="learning-head static"><span class="learning-title">${esc(r.tag ? capFirst(r.tag) : 'Your history')}</span><span class="badge ${WR_EVIDENCE_TONE[r.label] || ''}">${esc(r.label)}</span></div>
            <p class="lesson-text">${esc(r.text)}</p>
            ${isNum(r.n) ? `<p class="muted small">${plural(r.n, 'case', 'cases')} · halves ${esc(fmt.spct(r.first_half))} / ${esc(fmt.spct(r.second_half))} · t ${r.t == null ? 'n/a' : esc(r.t.toFixed(1))}${r.confidence ? ` · ${esc(CONF_LABEL[r.confidence] || '')}` : ''}${r.link === 'weights' ? ' · <button class="link-btn" type="button" data-wr-goto="weights">Weights tab</button>' : ''}</p>` : ''}</div>`;
        const tests = (now.weight_tests || []).map((t) => `<div class="learning"><div class="learning-head static"><span class="learning-title">A weight question: ${esc(t.label)}</span><span class="badge ${WR_EVIDENCE_TONE[t.evidence] || ''}">${esc(t.evidence)}</span></div>
            <p class="lesson-text">${esc(t.text)}</p><p class="muted small"><button class="link-btn" type="button" data-wr-goto="lab">Backtest lab</button> · <button class="link-btn" type="button" data-wr-goto="weights">Weights tab</button></p></div>`);
        $('#rec-now-list').innerHTML = [...(now.items || []).map(item), ...tests].join('') || '<p class="muted small">Nothing to suggest yet.</p>';
    }

    function recRuleColor(key) {
        if (key === 'actual') return cssVar('--series-8');
        state.recRuleColors = state.recRuleColors || {};
        if (!state.recRuleColors[key]) {
            const used = new Set(Object.values(state.recRuleColors));
            state.recRuleColors[key] = REC_RULE_COLORS.find((c) => !used.has(c)) || REC_RULE_COLORS[0];
        }
        return cssVar(state.recRuleColors[key]);
    }

    function ruleChanges(c) {
        const items = [
            c.skipped_buys ? `${plural(c.skipped_buys, 'buy', 'buys')} skipped (${eur(c.skipped_eur)})` : '',
            c.scaled_buys ? `${plural(c.scaled_buys, 'buy', 'buys')} scaled down to the cap (${eur(c.scaled_eur)} less)` : '',
            c.unfunded_buys ? `${plural(c.unfunded_buys, 'buy', 'buys')} could not be funded (${eur(c.unfunded_eur)})` : '',
            c.funding_scaled ? `${plural(c.funding_scaled, 'buy', 'buys')} reduced for lack of cash` : '',
            c.deferred_sells ? `${plural(c.deferred_sells, 'sale', 'sales')} delayed` : '',
            c.unfilled_sells ? `${plural(c.unfilled_sells, 'sale', 'sales')} had nothing left to sell` : '',
            c.trims ? `${c.trims} month-end trims` : '', c.stops ? `${c.stops} stop-losses` : '', c.takes ? `${c.takes} take-profits` : '',
            c.core_buys ? `${plural(c.core_buys, 'buy', 'buys')} of the core fund` : '', c.core_sells ? `${plural(c.core_sells, 'sale', 'sales')} of the core fund to fund buys` : '',
            c.unpriced_deferrals ? `${c.unpriced_deferrals} delayed sales valued at cost (no price)` : '',
        ].filter(Boolean);
        return items.length ? items : ['Nothing changed.'];
    }

    function ruleDetailHtml(r) {
        return `<div class="rule-detail">
            <div><h4>Rule</h4>${esc(r.description || '')}${r.key === 'actual' ? '' : `<p class="muted small">Ends with ${esc(fmt.pct(r.end_cash_share, 0))} in cash${isNum(r.end_core_share) && r.end_core_share > 0 ? ` and ${esc(fmt.pct(r.end_core_share, 0))} in the core fund` : ''}; volatility ${esc(fmt.pct(r.volatility, 0))} a year, worst month ${esc(fmt.pct(r.worst_month, 1))}.</p>`}</div>
            <div><h4>What changed</h4><ul>${ruleChanges(r.changes || {}).map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div>
            <div><h4>Largest effects</h4>${(r.largest_effects || []).length ? `<ul>${r.largest_effects.map((x) => `<li>${esc(x.name)}: <span class="${signCls(x.eur)}">${esc(signedEur(x.eur))}</span></li>`).join('')}</ul>` : '<span class="muted small">none</span>'}
                <p class="muted small">Confidence: ${esc(r.why || '')}</p></div></div>`;
    }

    const RUN_TIP = 'Monthly imports in a row, this one included, in which the rule was ahead of what you did over the whole period and in each half alone. Re-run My track record after each month\'s export: a rule ahead in both halves three months running is a candidate to write down and follow on new trades. The replay alone adopts nothing.';

    function renderRecRules(lab) {
        const host = $('#rec-rules-table');
        if (!lab || !lab.scenarios) {
            host.innerHTML = '<tbody><tr><td class="muted small">The rules lab did not run; see the notes in Overview.</td></tr></tbody>';
            $('#rec-rules-caption').textContent = ''; $('#rec-rules-verdict').innerHTML = ''; $('#rec-rules-protocol').textContent = '';
            destroyChart('rec-rules-chart'); $('#rec-rules-legend').innerHTML = '';
            return;
        }
        const rows = lab.scenarios;
        $('#rec-rules-caption').textContent = `${plural(rows.length - 1, 'rule', 'rules')} · core fund ${lab.core} · halves split at ${lab.split} · ${fmt.money(lab.fee_per_order, 2, 'EUR')} per simulated order${isNum(lab.imports_recorded) ? ` · ${lab.imports_recorded} monthly import${lab.imports_recorded === 1 ? '' : 's'} on record` : ''}`;
        $('#rec-rules-verdict').innerHTML = `<strong>${(lab.adopted || []).length ? 'A rule passed the protocol' : 'Keep what you do'}</strong>${esc(lab.verdict || '')}`;
        $('#rec-rules-protocol').textContent = lab.protocol || '';
        if (!state.recRulesShown) state.recRulesShown = new Set(['actual', ...(lab.best ? [lab.best] : []), ...(lab.adopted || [])]);
        host.innerHTML = `<thead><tr><th><span class="muted small">Chart</span></th><th>Rule</th><th class="num">Against what you did</th><th class="num">A year (TWR)</th><th class="num">Max drawdown</th><th class="num">1st half alone</th><th class="num">2nd half alone</th><th class="num">t</th><th>Confidence</th><th class="num"><span class="term" tabindex="0" data-tip="${esc(RUN_TIP)}">In a row</span></th><th>Verdict</th></tr></thead><tbody>${rows.map((r, i) => {
            const ref = r.key === 'actual';
            return `<tr class="${ref ? 'is-current' : ''} ${r.adopt ? 'is-adopted' : ''}" data-rule="${esc(r.key)}">
                <td><input type="checkbox" data-rule-show="${esc(r.key)}" ${state.recRulesShown.has(r.key) ? 'checked' : ''} ${ref ? 'disabled' : ''} aria-label="Draw ${esc(r.label)}"></td>
                <td class="rule-cell"><button class="link-btn rule-name" type="button" data-rule-detail="${i}" aria-expanded="false">${esc(r.label)}</button>${r.preset ? '' : ' <span class="badge accent">yours</span>'}</td>
                <td class="num ${ref ? 'muted' : signCls(r.difference)}">${ref ? esc(eur(r.end_value)) : `${esc(signedEur(r.difference))} <span class="muted small">${esc(fmt.spct(r.difference_pct))}</span>`}</td>
                <td class="num">${esc(fmt.spct(r.twr_annual))}</td><td class="num">${esc(fmt.pct(r.max_drawdown))}</td>
                <td class="num ${ref ? '' : signCls(r.first_half)}">${ref ? '' : esc(signedEur(r.first_half))}</td>
                <td class="num ${ref ? '' : signCls(r.second_half)}">${ref ? '' : esc(signedEur(r.second_half))}</td>
                <td class="num">${r.t == null ? (ref ? '' : 'n/a') : esc(r.t.toFixed(1))}</td><td>${ref ? '' : confBadge(r.confidence)}</td>
                <td class="num">${ref || !isNum(r.run) ? '' : esc(r.run)}</td>
                <td class="wrap small">${esc(r.verdict || '')}</td></tr>
            <tr class="detail" data-rule-detail-row="${i}" hidden><td></td><td colspan="10">${ruleDetailHtml(r)}</td></tr>`;
        }).join('')}</tbody>`;
        renderRecRulesChart();
    }

    function renderRecRulesChart() {
        const lab = (state.rec || {}).rules_lab;
        const el = $('#rec-rules-chart');
        if (!lab || !lab.curves || el.closest('.rec-pane').hidden) return;
        const chart = makeChart('rec-rules-chart', MONEY_AXIS);
        const keys = ['actual', ...lab.scenarios.map((r) => r.key).filter((k) => k !== 'actual' && state.recRulesShown.has(k))];
        const items = [];
        keys.forEach((k) => {
            const data = lab.curves[k];
            if (!data) return;
            const row = lab.scenarios.find((r) => r.key === k);
            const color = recRuleColor(k);
            const line = chart.addLineSeries({ color, lineWidth: k === 'actual' ? 2.5 : 1.8, priceLineVisible: false, lastValueVisible: false, priceFormat: MONEY_FORMAT });
            line.setData(data);
            items.push({ key: k, label: row ? row.label : k, color, series: line });
        });
        legend('rec-rules-legend', items, { chart: 'rec-rules-chart' });
        chart.timeScale().fitContent();
    }

    function toggleRuleLine(key, on) {
        if (!state.recRulesShown) return;
        if (on) {
            if (state.recRulesShown.size - 1 >= MAX_RULE_LINES) { toast(`Up to ${MAX_RULE_LINES} rules in the chart: untick one first.`, 'error'); return false; }
            state.recRulesShown.add(key);
            legendForget('rec-rules-chart', key);   // a line put back on the chart shows
        } else {
            state.recRulesShown.delete(key);
        }
        renderRecRulesChart();
        return true;
    }

    function showRrProgress(on, p = {}) {
        setIf($('#rr-progress'), 'hidden', !on);
        if (!on) { $('#rr-progress [data-job-cancel]').disabled = false; return; }
        const share = p.total ? Math.min(1, p.done / p.total) : 0;
        $('#rr-progress-fill').style.width = `${(share * 100).toFixed(0)}%`;
        setIf($('#rr-progress-text'), 'textContent', p.stage || 'Working');
    }

    async function runCustomRule() {
        if (!state.rec) return;
        const btn = $('#rr-run');
        const rule = {
            label: $('#rr-label').value.trim() || 'My rule', min_hold_days: $('#rr-min-hold').value, low_vol_hold_days: $('#rr-low-vol-hold').value,
            max_weight: $('#rr-max-weight').value, min_buy: $('#rr-min-buy').value, cash_reserve: $('#rr-cash-reserve').value,
            stop_loss: $('#rr-stop').value, take_profit: $('#rr-take').value, no_turbos: $('#rr-no-turbos').checked,
            skip_discretionary: $('#rr-plans-only').checked, trim: $('#rr-trim').checked,
        };
        const form = $('#rec-rule-form');
        for (const input of $$('input[type=number]', form)) {
            if (input.value.trim() !== '' && notNumberIn(input, Number(input.min || -Infinity), Number(input.max || Infinity))) {
                const name = $(`label[for="${input.id}"]`).textContent;
                return formError(form, input.min !== '' && input.max !== '' ? `${name} goes from ${input.min} to ${input.max}.` : `${name} must be at least ${input.min}.`, input);
            }
        }
        clearFormError(form);
        setBusy(btn, true);
        showRrProgress(true, { stage: 'Starting' });
        try {
            const result = await runJob('/api/track-record/rules', { presets: false, method: $('#rec-method').value, custom: [rule] }, (p) => showRrProgress(true, p), 'rr');
            const lab = state.rec.rules_lab || (state.rec.rules_lab = { ...result, scenarios: result.scenarios.filter((r) => r.key === 'actual'), curves: { actual: result.curves.actual }, adopted: [] });
            state.recCustomCount = (state.recCustomCount || 0) + 1;
            result.scenarios.filter((r) => r.key !== 'actual').forEach((r) => {
                const key = `yours_${state.recCustomCount}`;
                lab.curves[key] = result.curves[r.key];
                r.key = key;
                r.label = `${r.label} (yours)`;
                lab.scenarios = lab.scenarios.filter((x) => x.key !== key).concat([r]);
                if (r.adopt) lab.adopted = [...(lab.adopted || []), key];
                if (state.recRulesShown && state.recRulesShown.size - 1 < MAX_RULE_LINES) state.recRulesShown.add(key);
                toast(`${r.label}: ${signedEur(r.difference)} against what you did. ${r.verdict}`, r.adopt ? 'success' : 'info', 9000);
            });
            lab.scenarios.sort((p, q) => (p.key === 'actual' ? -1 : q.key === 'actual' ? 1 : (q.weaker_half || 0) - (p.weaker_half || 0)));
            renderRecRules(lab);
        } catch (err) {
            if (err.stopped) toast('Stopped. Nothing was changed.', 'info');
            else formError(form, `Rule replay failed: ${err.message}`);
        } finally {
            setBusy(btn, false);
            showRrProgress(false);
        }
    }

    function renderRecTrips(d) {
        const trips = (d.round_trips || []).slice().sort((p, q) => (q.start || '').localeCompare(p.start || ''));
        $('#rec-trips-caption').textContent = `${plural(trips.length, 'round trip', 'round trips')}, newest first; worst and best are the price while held against your average buy`;
        const vol = { low: 'steady', mid: 'average', high: 'volatile' };
        $('#rec-trips').innerHTML = `<thead><tr><th>Holding</th><th>From</th><th>To</th><th class="num">Days</th><th class="num">Buys</th><th>Built over</th><th>Volatility</th><th class="num">Bought</th><th class="num">P&amp;L</th><th class="num">Return</th><th class="num">Worst</th><th class="num">Best</th></tr></thead><tbody>${
            trips.map((t) => `<tr><td>${esc(t.name)} <span class="muted mono small">${esc(t.ticker || '')}</span></td><td>${esc(fmtDay(t.start))}</td><td>${esc(fmtDay(t.end || t.outcome))}</td><td class="num">${esc(t.days)}</td><td class="num">${esc(t.buys ?? t.buy_orders)}</td><td>${esc(fmtDays(t.build_days))}</td><td>${esc(vol[t.vol_class] || '')}</td>
                <td class="num">${esc(eur(t.bought))}</td><td class="num ${signCls(t.total_pnl)}">${esc(signedEur(t.total_pnl))}</td><td class="num">${esc(fmt.spct(t.return_pct))}</td><td class="num">${esc(fmt.spct(t.mae))}</td><td class="num">${esc(fmt.spct(t.mfe))}</td></tr>`).join('')}</tbody>`;
    }

    async function shareRecord() {
        const btn = $('#rec-share');
        setBusy(btn, true);
        // The tab opens inside the click: a popup opened after the wait can be refused (Safari).
        const win = window.open('about:blank', '_blank');
        try {
            const res = await api('/api/track-record/share', { method: 'POST', body: { names: $('#rec-share-names').checked } });
            if (win) { win.opener = null; win.location = res.url; }
            $('#rec-share-download').hidden = false;
            toast('Your page without amounts is ready. Download it and send the file: the tab that just opened works only on this computer.', 'success', 12000);
        } catch (err) {
            if (win) win.close();
            toast(`Could not write the shareable page: ${err.message}`, 'error', 8000);
        } finally {
            setBusy(btn, false);
        }
    }

    const gridEntries = () => {
        const o = state.labOptions || {};
        return (o.grid_order || Object.keys(o.grid || {})).map((k) => [k, o.grid[k]]);
    };

    function renderGridEditor() {
        state.bkGrid = state.bkGrid || Object.fromEntries(gridEntries().map(([k, spec]) => [k, new Set(spec.values.map(gridKey))]));
        $('#bk-grid-editor').innerHTML = gridEntries().map(([key, spec]) => `
            <div class="grid-row">
                <span class="grid-label">${esc(spec.label)}</span>
                <div class="check-row">${spec.values.map((v) => `<label class="check"><input type="checkbox" data-grid="${esc(key)}" data-value="${esc(gridKey(v))}" ${state.bkGrid[key].has(gridKey(v)) ? 'checked' : ''}>${esc(bkValueLabel(key, v))}</label>`).join('')}</div>
            </div>`).join('');
        updateGridCount();
    }

    function selectedGrid() {
        // In the editor's order, so the server builds combinations in the order shown.
        return Object.fromEntries(gridEntries().filter(([k]) => state.bkGrid[k]).map(([k, spec]) => [k,
            spec.values.filter((v) => state.bkGrid[k].has(gridKey(v)))]));
    }

    const GRID_EMPTY = 'Pick at least one value in each row.';
    function updateGridCount() {
        const max = (state.labOptions || {}).max_setups || 1000;
        const n = Object.values(state.bkGrid || {}).reduce((acc, set) => acc * set.size, 1);
        const el = $('#bk-grid-count');
        el.textContent = `${n} setup${n === 1 ? '' : 's'}${n > max ? `: above the limit of ${max}, select fewer values` : ''}`;
        el.classList.toggle('neg', n > max || n === 0);
        // A grid with an empty row or too many setups cannot run: the button is off and the empty row is named.
        state.bkGridBad = n === 0 || n > max;
        const form = $('#bk-form');
        if (n === 0 && state.bkMode === 'search') formError(form, GRID_EMPTY);
        else if ($('#bk-error').textContent === GRID_EMPTY) clearFormError(form);
        syncBkRun();
    }

    async function runBk() {
        const search = state.bkMode === 'search';
        const form = $('#bk-form');
        const ticker = $('#bk-ticker').value.trim().toUpperCase();
        if (state.bkJobMode) return;
        if (search && state.bkGridBad) return;
        if (state.bkUniverse === 'ticker' && !TICKER_RE.test(ticker)) return formError(form, runErrorMessage({ message: 'Invalid ticker symbol' }, ticker), $('#bk-ticker'));
        if (notNumberIn($('#bk-cost'), 0, 200)) return formError(form, 'Cost per trade goes from 0 to 200 bps.', $('#bk-cost'));
        clearFormError(form);
        const body = {
            universe: state.bkUniverse,
            ticker,
            allocation: $('#bk-allocation').value,
            years: $('#bk-years').value || null,
            entry_gate: $('#bk-gate').value,
            start: $('#bk-start').value,
            rebalance: $('#bk-rebalance').value,
            cost_bps: Number($('#bk-cost').value),
        };
        if (search) body.grid = selectedGrid();
        else body.compare = $$('[data-compare]').filter((c) => c.checked).map((c) => c.dataset.compare);
        const jobMode = state.bkMode;
        state.bkJobMode = jobMode;
        syncBkRun();
        showBkProgress(true, { stage: 'Starting' });
        try {
            const result = await runJob(search ? '/api/lab/search' : '/api/lab/run', body, (p) => showBkProgress(true, p), 'bk');
            let headline;
            if (search) {
                state.bkSearch = result;
                state.bkSearchAll = false;
                setBkMode(state.bkMode);
                renderBkSearch(result);
                headline = `Search finished: ${result.summary.strong ? 'worth a closer look' : 'keep your setup'}.`;
            } else {
                state.bk = result;
                state.bkShown = new Set(result.rows.filter((r) => r.kind !== 'classic').map((r) => r.key)
                    .concat(result.rows.filter((r) => r.kind === 'classic').slice(0, 1).map((r) => r.key)));
                setBkMode(state.bkMode);
                renderBk(result);
                const ours = result.rows.find((r) => r.key === 'fastvalue');
                headline = `Compare finished: your setup ${fmt.spct(ours && ours.excess_cagr)} a year against buy and hold.`;
            }
            if (state.bkMode === jobMode && !$('#view-backtest').hidden) $(search ? '#bk-search-results' : '#bk-results').scrollIntoView({ block: 'start' });
            bkFinished(jobMode, headline);
        } catch (err) {
            if (err.stopped) toast('Stopped. Nothing was changed.', 'info');
            else formError(form, runErrorMessage(err, ticker), state.bkUniverse === 'ticker' && /ticker|price history|data available/i.test(err.message) ? $('#bk-ticker') : null);
        } finally {
            state.bkJobMode = null;
            syncBkRun();
            showBkProgress(false);
        }
    }

    const statHtml = (label, value, sub = '', cls = '', subCls = '') => `<div class="stat"><div class="stat-label">${statLabel(label)}</div><div class="stat-value ${cls}">${esc(value)}</div><div class="stat-sub ${subCls}">${esc(sub)}</div></div>`;
    // Gain or loss only; a change too small to show (0.0%) carries no tone.
    const signCls = (v) => (isNum(v) ? (Math.abs(v) < 0.0005 ? '' : v >= 0 ? 'pos' : 'neg') : '');

    function renderBk(d) {
        const ours = d.rows.find((r) => r.key === 'fastvalue');
        const bh = d.rows.find((r) => r.key === 'buy_hold');
        $('#bk-notes').innerHTML = d.notes.map((n) => `<div class="warning note"><strong>Note</strong>${esc(n)}</div>`).join('');
        $('#bk-stats').innerHTML = [
            statHtml('Your setup', `${fmt.pct(ours.stats.cagr)} a year`, `Sharpe ${fmt.num(ours.stats.sharpe)}`),
            statHtml('Buy and hold', `${fmt.pct(bh.stats.cagr)} a year`, `Sharpe ${fmt.num(bh.stats.sharpe)}`),
            statHtml('Difference', fmt.spct(ours.excess_cagr), `first half ${fmt.spct(ours.excess_first)}, second ${fmt.spct(ours.excess_second)}`, signCls(ours.excess_cagr)),
            statHtml([ 'Max drawdown', STAT_TIPS.drawdown], fmt.pct(ours.stats.max_drawdown), `buy and hold ${fmt.pct(bh.stats.max_drawdown)}`),
            statHtml('Time invested', fmt.pct(ours.invested_share, 0), `turnover ${fmt.pct(ours.turnover, 0)} a year`),
        ].join('');
        $('#bk-period').textContent = `${d.universe.label}: ${fmtDay(d.period.start)} to ${fmtDay(d.period.end)}, ${plural(d.period.months, 'month', 'months')}, halves split at ${fmtDay(d.period.split)}`;
        const activity = (r) => (r.kind === 'classic' ? `${fmt.num(r.trades_per_year, 1)} trades/yr` : r.turnover ? `${fmt.pct(r.turnover, 0)} turnover/yr` : 'none');
        const th = (label, tip) => `<th class="num">${tagTip(label, tip)}</th>`;
        $('#bk-table').innerHTML = `<thead><tr><th>Chart</th><th>Strategy</th>${th('CAGR', STAT_TIPS.cagr)}${th('Sharpe', STAT_TIPS.sharpeCash)}${th('Max drawdown', STAT_TIPS.drawdown)}${
            th('Volatility', 'How much the value swings: the yearly spread of its returns.')}${th('Activity', 'Trades a year for a classic strategy; the share of the portfolio traded each year for the others.')}${
            th('Invested', 'The share of the months spent in the stocks rather than in cash.')}${th('1st half', 'Compound annual growth over the first half of the period.')}${
            th('2nd half', 'Compound annual growth over the second half of the period.')}${th('vs buy and hold', 'Compound annual growth minus that of buy and hold. Positive means ahead.')}</tr></thead><tbody>${
            d.rows.map((r) => `<tr class="${r.key === 'fastvalue' ? 'is-current' : ''}">
                <td class="chk"><label class="cb-hit"><input type="checkbox" data-show="${esc(r.key)}" ${state.bkShown.has(r.key) ? 'checked' : ''} aria-label="Show ${esc(r.label)} in the chart"></label></td>
                <td class="span"><strong>${esc(r.label)}</strong></td>
                <td class="num col2 ${signCls(r.stats.cagr)}" data-label="CAGR">${esc(fmt.pct(r.stats.cagr))}</td>
                <td class="num hide-sm">${esc(fmt.num(r.stats.sharpe))}</td>
                <td class="num hide-sm">${esc(fmt.pct(r.stats.max_drawdown, 0))}</td>
                <td class="num hide-sm">${esc(fmt.pct(r.stats.volatility, 0))}</td>
                <td class="num hide-sm">${esc(activity(r))}</td>
                <td class="num hide-sm">${esc(fmt.pct(r.invested_share, 0))}</td>
                <td class="num" data-label="1st half">${esc(fmt.pct(r.first_half.cagr))}</td>
                <td class="num" data-label="2nd half">${esc(fmt.pct(r.second_half.cagr))}</td>
                <td class="num ${r.key === 'buy_hold' ? 'muted' : signCls(r.excess_cagr)}" data-label="vs buy and hold">${r.key === 'buy_hold' ? 'reference' : esc(fmt.spct(r.excess_cagr))}</td>
            </tr>`).join('')
        }</tbody>`;
        renderBkChart(d);
        renderBkTimeline(d);
    }

    // One colour per line. Your setup and buy and hold have their own; the three reference lines keep a fixed slot;
    // each classic strategy takes the next free slot in the order it was ticked. At most six extra lines, so none repeats.
    const BK_FIXED_COLOR = { fastvalue: '--accent', buy_hold: '--series-8', rebalanced: '--series-4', SPY: '--series-2', RSP: '--series-3' };
    const BK_FREE_COLORS = ['--series-5', '--series-6', '--series-7', '--series-4', '--series-3', '--series-2'];
    const MAX_EXTRA_LINES = 6;
    const bkExtraLines = (keys) => [...keys].filter((k) => k !== 'fastvalue' && k !== 'buy_hold');
    function bkColors(shownKeys) {
        const keys = [...shownKeys];
        const taken = new Set(keys.map((k) => BK_FIXED_COLOR[k]).filter(Boolean));
        const free = BK_FREE_COLORS.filter((token) => !taken.has(token));
        const out = {};
        keys.filter((k) => BK_FIXED_COLOR[k]).forEach((k) => { out[k] = cssVar(BK_FIXED_COLOR[k]); });
        keys.filter((k) => !BK_FIXED_COLOR[k]).forEach((k, n) => { out[k] = cssVar(free[n % free.length]); });
        return out;
    }

    function renderBkChart(d) {
        if (!d || $('#bk-results').hidden || $('#view-backtest').hidden) return;
        const shownRows = d.rows.filter((r) => state.bkShown.has(r.key) && d.equity[r.key]);
        const chart = makeChart('bk-chart', { rightPriceScale: { borderVisible: false, mode: 1 } }, chartSummary('Growth of 100, your setup against buy and hold and the other strategies', shownRows.map((r) => ({ label: r.label, data: d.equity[r.key] }))));
        const items = [];
        const colors = bkColors([...state.bkShown].filter((k) => d.equity[k]));
        d.rows.forEach((r) => {
            if (!state.bkShown.has(r.key) || !d.equity[r.key]) return;
            const color = colors[r.key];
            const line = chart.addLineSeries({
                color, lineWidth: r.key === 'fastvalue' ? 2.5 : 1.5, lineStyle: r.key === 'buy_hold' ? 2 : 0,
                priceLineVisible: false, lastValueVisible: false, priceFormat: MONEY_FORMAT,
            });
            line.setData(d.equity[r.key]);
            items.push({ key: r.key, label: r.label, color, dash: r.key === 'buy_hold', series: line });
        });
        legend('bk-legend', items, { chart: 'bk-chart' });
        chart.timeScale().fitContent();
    }

    function renderBkTimeline(d) {
        const dates = d.dates;
        const years = dates.map((x, i) => [x, i]).filter(([x], i) => i === 0 || x.slice(5, 7) === '01');
        // On a phone only every third year is labelled (the others are .minor), so the labels do not run into each other.
        const ticks = years.map(([x, i]) => `<span class="timeline-tick${Number(x.slice(0, 4)) % 3 ? ' minor' : ''}" style="left:${((i / dates.length) * 100).toFixed(2)}%">${esc(x.slice(0, 4))}</span>`).join('');
        $('#bk-timeline').innerHTML = d.per_ticker.map((p, n) => `
            <div class="timeline-row">
                <span class="timeline-label mono">${esc(p.ticker)}</span>
                <span class="timeline-strip" data-timeline="${n}">${p.decisions.map((c, i) => `<span class="tl d-${c || 'x'}${p.invested[i] ? '' : ' out'}"></span>`).join('')}</span>
            </div>`).join('') + `<div class="timeline-row"><span class="timeline-label"></span><span class="timeline-axis">${ticks}</span></div>`;
        $('#bk-timeline-readout').textContent = 'Point at a month to read it.';
        $('#bk-timeline-legend').innerHTML = [['At or above the entry level', 'd-B'], ['Between the levels', 'd-H'], ['At or below the exit level', 'd-S'], ['Not rated', 'd-N'], ['In cash (faded)', 'd-B out']]
            .map(([l, c]) => `<span class="legend-item"><span class="tl ${c}"></span>${esc(l)}</span>`).join('');
        $('#bk-holdings-table').innerHTML = `<thead><tr><th>Holding</th><th class="num">Weight</th><th class="num">Invested</th><th class="num">Entries</th><th class="num">Exits</th><th>Owner data</th><th>Prices from</th></tr></thead><tbody>${
            d.per_ticker.map((p) => `<tr><td class="mono"><strong>${esc(p.ticker)}</strong></td><td class="num">${esc(fmt.pct(p.weight, 1))}</td>
                <td class="num">${esc(fmt.pct(p.invested_share, 0))}</td><td class="num">${esc(p.entries)}</td><td class="num">${esc(p.exits)}</td>
                <td class="${p.owner_status === 'ok' ? '' : 'muted'}">${esc(p.owner_status === 'ok' ? 'SEC filings' : p.owner_status || 'n/a')}</td><td>${esc(p.first_date || 'n/a')}</td></tr>`).join('')
        }</tbody>`;
    }

    // Only what sets a row apart: settings with one value in the whole grid, or any value in this row, are left out.
    function describeValues(values, grid) {
        return gridEntries().map(([k]) => k).filter((k) => values[k] && grid[k] && grid[k].length > 1 && values[k].length < grid[k].length)
            .map((k) => (k === 'entry_gate' ? '' : `${GRID_SHORT[k] || k} `) + values[k].map((v) => bkValueLabel(k, v)).join(' / '))
            .join(' · ') || 'Every value tried gives this outcome';
    }

    function describeBaseline(b) {
        const s = b.setup || {};
        const ow = s.owner_weights || {};
        const pair = (ow.fundamentals || 0) + (ow.shareholder || 0);
        const parts = [];
        if (pair > 0) parts.push(`Shareholders ${fmt.pct((ow.shareholder || 0) / pair, 0)}`);
        parts.push(`Entry level ${fmt.signed(s.buy, 0)}`, `Exit level ${fmt.signed(s.sell, 0)}`, `Cap ${s.cap === null ? 'off' : `${s.cap}x`}`);
        const others = (s.deciding_agents || []).filter((k) => k !== 'fundamentals' && k !== 'shareholder');
        if (others.length) parts.push(`also deciding: ${others.join(', ')}`);
        parts.push(bkValueLabel('entry_gate', (b.rules || {}).entry_gate));
        return parts.join(' · ');
    }

    function renderBkSearch(d) {
        const s = d.summary;
        $('#bk-index').hidden = true;  // a new search: the previous S&P 500 check belongs to another table
        $('#bk-verdict').className = `verdict ${s.strong ? 'strong' : 'weak'}`;
        $('#bk-verdict').innerHTML = `<strong>${s.strong ? 'Worth a closer look' : 'Keep your setup'}</strong><span>${esc(s.verdict)}</span>`;
        $('#bk-search-stats').innerHTML = [
            statHtml('Setups tested', String(d.tested), `${d.outcomes} distinct outcomes${d.skipped ? `, ${d.skipped} invalid skipped` : ''}`),
            statHtml('Beat buy and hold in both halves', String(s.beat_both_halves), `of ${d.tested}`),
            statHtml('First half predicts second', isNum(s.rank_correlation) ? fmt.num(s.rank_correlation, 2) : 'n/a', 'rank correlation, 1 = same order'),
            statHtml('First-half winner, second half', fmt.spct(s.first_half_winner.excess_second), `your setup ${fmt.spct(s.baseline_excess_second)} (vs buy and hold)`, signCls(s.first_half_winner.excess_second)),
        ].join('');
        $('#bk-search-period').textContent = `${d.universe.label}: ${fmtDay(d.period.start)} to ${fmtDay(d.period.end)}, halves split at ${fmtDay(d.period.split)}`;
        const cells = (r) => `<td class="num hide-sm ${signCls(r.stats.cagr)}">${esc(fmt.pct(r.stats.cagr))}</td><td class="num hide-sm">${esc(fmt.num(r.stats.sharpe))}</td>
            <td class="num hide-sm">${esc(fmt.pct(r.stats.max_drawdown, 0))}</td>
            <td class="num ${signCls(r.excess_first)}" data-label="1st half vs buy and hold">${esc(fmt.spct(r.excess_first))}</td><td class="num ${signCls(r.excess_second)}" data-label="2nd half vs buy and hold">${esc(fmt.spct(r.excess_second))}</td>
            <td class="num ${signCls(r.robust)}" data-label="Weaker half"><strong>${esc(fmt.spct(r.robust))}</strong></td>`;
        const b = d.baseline;
        const th = (label, tip) => `<th class="num">${tagTip(label, tip)}</th>`;
        $('#bk-search-table').innerHTML = `<thead><tr><th>#</th><th>Setup</th>${th('Same outcome', STAT_TIPS.same)}${th('CAGR', STAT_TIPS.cagr)}${th('Sharpe', STAT_TIPS.sharpeCash)}${th('Max drawdown', STAT_TIPS.drawdown)}<th class="num">1st half vs buy and hold</th><th class="num">2nd half vs buy and hold</th>${th('Weaker half', STAT_TIPS.weaker)}<th><span class="sr-only">Actions</span></th></tr></thead><tbody>
            <tr class="is-current"><td class="hide-sm"></td><td class="wrap span"><strong>Your saved setup</strong><div class="muted small">${esc(describeBaseline(b))}</div></td><td class="num hide-sm"></td>${cells(b)}<td class="hide-sm"></td></tr>
            ${d.ranked.slice(0, state.bkSearchAll ? d.ranked.length : 20).map((r, i) => `<tr><td class="hide-sm">${i + 1}</td><td class="wrap span">${esc(describeValues(r.values, d.grid))}</td><td class="num hide-sm">${r.equivalent ? `+${esc(r.equivalent)}` : ''}</td>${cells(r)}
                <td class="nowrap span"><button class="btn btn-ghost" type="button" data-index-check="${i}">S&amp;P 500</button>
                <button class="btn btn-ghost" type="button" data-apply="${i}">Apply</button></td></tr>`).join('')}
        </tbody>`;
        $('#bk-search-more').hidden = state.bkSearchAll || d.ranked.length <= 20;
        $('#bk-search-more').textContent = `Show all ${d.ranked.length}`;
        $('#bk-search-notes').innerHTML = d.notes.map((n) => `<div class="warning note"><strong>Note</strong>${esc(n)}</div>`).join('');
    }

    async function checkOnIndex(i, btn) {
        const row = state.bkSearch && state.bkSearch.ranked[i];
        if (!row) return;
        const card = $('#bk-index');
        card.hidden = false;
        card.scrollIntoView({ block: 'start' });  // the answer lands below a long table: bring its card into view first
        $('#bk-index-caption').textContent = describeValues(row.values, state.bkSearch.grid);
        $('#bk-index-verdict').className = 'verdict';
        $('#bk-index-verdict').innerHTML = '<span>Replaying both setups on the S&amp;P 500...</span>';
        $('#bk-index-table').innerHTML = '';
        $('#bk-index-notes').innerHTML = '';
        setBusy(btn, true);
        try {
            const { job } = await api('/api/lab/index-check', { method: 'POST', body: { combo: row.combo, entry_gate: $('#bk-gate').value } });
            let shown = '';
            const d = await pollJob(job, (p) => {
                const html = `<span>${esc(p.stage || 'Working')}...</span>`;
                if (html !== shown) { $('#bk-index-verdict').innerHTML = html; shown = html; }
            });
            const strong = d.level === 'generalizes';
            $('#bk-index-verdict').className = `verdict ${strong ? 'strong' : 'weak'}`;
            $('#bk-index-verdict').innerHTML = `<strong>${strong ? 'Holds up out of sample' : d.level === 'same' ? 'Same trades' : 'Keep your setup'}</strong><span>${esc(d.verdict)}</span>`;
            const diff = d.difference || {};
            $('#bk-index-caption').textContent = `${plural(d.companies, 'company', 'companies')}, ${fmtDay(d.period.start)} to ${fmtDay(d.period.end)}, halves split at ${fmtDay(d.period.split)}`;
            $('#bk-index-table').innerHTML = `<thead><tr><th><span class="sr-only">Setup</span></th><th class="num">${tagTip('CAGR', STAT_TIPS.cagr)}</th><th class="num">${tagTip('Sharpe', STAT_TIPS.sharpeCash)}</th><th class="num">${tagTip('Max drawdown', STAT_TIPS.drawdown)}</th><th class="num">1st half</th><th class="num">2nd half</th><th class="num">Names held</th></tr></thead><tbody>${
                d.rows.map((r) => `<tr><td>${esc(r.label)}</td><td class="num">${esc(fmt.pct(r.stats.cagr))}</td><td class="num">${esc(fmt.num(r.stats.sharpe))}</td>
                    <td class="num">${esc(fmt.pct(r.stats.max_drawdown, 0))}</td><td class="num">${esc(fmt.pct(r.first_half.cagr))}</td><td class="num">${esc(fmt.pct(r.second_half.cagr))}</td>
                    <td class="num">${isNum(r.avg_names) ? esc(r.avg_names.toFixed(0)) : ''}</td></tr>`).join('')
            }<tr class="is-current"><td><strong>This setup minus yours</strong></td><td class="num ${signCls(diff.cagr)}">${esc(fmt.spct(diff.cagr))}</td><td class="num">t ${esc(isNum(diff.t) ? diff.t.toFixed(1) : 'n/a')}</td><td></td>
                <td class="num ${signCls(diff.first_half)}">${esc(fmt.spct(diff.first_half))}</td><td class="num ${signCls(diff.second_half)}">${esc(fmt.spct(diff.second_half))}</td><td></td></tr></tbody>`;
            $('#bk-index-notes').innerHTML = (d.notes || []).map((n) => `<div class="warning note"><strong>Note</strong>${esc(n)}</div>`).join('');
        } catch (err) {
            $('#bk-index-verdict').className = 'verdict weak';
            $('#bk-index-verdict').innerHTML = `<strong>Could not run the check</strong><span>${esc(err.message)}</span>`;
        } finally {
            setBusy(btn, false);
        }
    }

    function applySetup(i) {
        const row = state.bkSearch && state.bkSearch.ranked[i];
        if (!row) return;
        if (state.dirty && !window.confirm('Replace your unsaved changes in the Weights tab?')) return;
        state.draft = { ...clone(state.config), ...clone(row.apply) };
        markDirty(true);
        if (row.combo.entry_gate) $('#bk-gate').value = row.combo.entry_gate;
        if (row.combo.rebalance) $('#bk-rebalance').value = row.combo.rebalance;
        showView('weights');
        refreshPreview();
        toast('Setup loaded into the Weights tab. Check the preview, then save or discard.', 'success', 8000);
    }

    async function setupBacktest() {
        state.bkMode = 'compare';
        $('#bk-modes').addEventListener('click', (e) => { const m = e.target.closest('[data-mode]')?.dataset.mode; if (m) setBkMode(m); });
        $('#bk-universe').addEventListener('click', (e) => { const u = e.target.closest('[data-universe]')?.dataset.universe; if (u) setBkUniverse(u); });
        $('#bk-form').addEventListener('submit', (e) => { e.preventDefault(); runBk(); });
        $('#bk-grid-editor').addEventListener('change', (e) => {
            const t = e.target;
            if (!t.dataset.grid) return;
            const set = state.bkGrid[t.dataset.grid];
            if (t.checked) set.add(t.dataset.value); else set.delete(t.dataset.value);
            updateGridCount();
        });
        $('#bk-table').addEventListener('change', (e) => {
            const key = e.target.dataset.show;
            if (!key) return;
            if (e.target.checked) {
                if (bkExtraLines(state.bkShown).length >= MAX_EXTRA_LINES && key !== 'fastvalue' && key !== 'buy_hold') {
                    e.target.checked = false;
                    toast('Up to six extra lines at a time. Untick one to add another.', 'info');
                    return;
                }
                state.bkShown.add(key);
                legendForget('bk-chart', key);
            } else state.bkShown.delete(key);
            renderBkChart(state.bk);
        });
        // The month under the pointer or finger, written out: the cells are narrower than a fingertip (N-58).
        const readTimeline = (e) => {
            const strip = e.target.closest && e.target.closest('.timeline-strip');
            const d = state.bk;
            const p = strip && d && d.per_ticker[Number(strip.dataset.timeline)];
            if (!p) return;
            const rect = strip.getBoundingClientRect();
            const i = Math.min(p.decisions.length - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * p.decisions.length)));
            const c = p.decisions[i];
            $('#bk-timeline-readout').textContent = `${p.ticker}, ${d.dates[i]}: ${DECISION_NAMES[c] || c}, ${p.invested[i] ? 'invested' : 'in cash'}`;
        };
        $('#bk-timeline').addEventListener('pointermove', readTimeline);
        $('#bk-timeline').addEventListener('pointerdown', readTimeline);
        $('#bk-search-more').addEventListener('click', () => { state.bkSearchAll = true; renderBkSearch(state.bkSearch); });
        $('#rec-form').addEventListener('submit', (e) => { e.preventDefault(); runRecord(); });
        $('#rec-fastvalue').addEventListener('change', renderRecordForm);
        $('#rec-learnings-more').addEventListener('click', () => { state.recAllLearnings = true; renderRecord(state.rec); });
        $('#rec-holdings-more').addEventListener('click', () => { state.recAllHoldings = true; renderRecord(state.rec); });
        $('#rec-sections').addEventListener('click', (e) => { const b = e.target.closest('[data-rec-section]'); if (b) setRecSection(b.dataset.recSection); });
        $('#rec-chart-mode').addEventListener('click', (e) => {
            const b = e.target.closest('[data-rec-chart-mode]');
            if (!b) return;
            state.recChartMode = b.dataset.recChartMode;
            storageSet('fastvalue-rec-chart-mode', state.recChartMode);
            if (state.rec) renderRecChart(state.rec);
        });
        $('#rec-cut-dims').addEventListener('click', (e) => { const b = e.target.closest('[data-rec-cut]'); if (!b) return; state.recCut = b.dataset.recCut; segButtons('#rec-cut-dims', Object.keys(((state.rec || {}).style || {}).cuts || {}).map((k) => [k, capFirst(k)]), state.recCut, 'rec-cut'); renderRecCuts(); });
        $('#rec-share').addEventListener('click', shareRecord);
        $('#rec-share-names').addEventListener('change', () => { $('#rec-share-download').hidden = true; });  // the written page no longer matches
        $('#rec-rule-form').addEventListener('submit', (e) => { e.preventDefault(); runCustomRule(); });
        $('#rec-results').addEventListener('click', (e) => {
            const head = e.target.closest('[data-learning]');
            if (head) {
                const body = head.parentElement.querySelector('.learning-body');
                body.hidden = !body.hidden;
                head.setAttribute('aria-expanded', String(!body.hidden));
                return;
            }
            if (e.target.closest('[data-rec-rerun]')) { runRecord(); return; }
            const wrGo = e.target.closest('[data-wr-goto]');
            if (wrGo) { if (wrGo.dataset.wrGoto === 'weights') showView('weights'); else setBkMode('compare'); return; }
            const link = e.target.closest('[data-rec-goto]');
            if (link) { setRecSection(link.dataset.recGoto); return; }
            const tag = e.target.closest('[data-tag-row]');
            if (tag) {
                const detail = tag.parentElement.querySelector(`[data-tag-detail="${tag.dataset.tagRow}"]`);
                if (detail) {
                    detail.hidden = !detail.hidden;
                    tag.classList.toggle('open', !detail.hidden);
                    tag.querySelector('.co-rowbtn')?.setAttribute('aria-expanded', String(!detail.hidden));
                }
                return;
            }
            const rule = e.target.closest('[data-rule-detail]');
            if (rule) {
                const detail = $(`#rec-rules-table [data-rule-detail-row="${rule.dataset.ruleDetail}"]`);
                if (detail) { detail.hidden = !detail.hidden; rule.setAttribute('aria-expanded', String(!detail.hidden)); }
            }
        });
        $('#rec-rules-table').addEventListener('change', (e) => {
            const key = e.target.dataset.ruleShow;
            if (!key) return;
            if (toggleRuleLine(key, e.target.checked) === false) e.target.checked = false;
        });
        $('#view-backtest').addEventListener('click', (e) => {
            const stop = e.target.closest('[data-job-cancel]');
            if (stop) { stopJob(stop.dataset.jobCancel, stop); return; }
            // A tip inside the setup summary explains the term; it must not also fold the details block.
            if (e.target.closest('summary .term')) e.preventDefault();
            const edit = e.target.closest('[data-sec-edit]');
            if (edit) {
                const form = edit.closest('.sec-slot').querySelector('.sec-form');
                form.hidden = false;
                edit.hidden = true;  // the form is open: the button that opened it goes
                form.querySelector('input').focus();
                return;
            }
            if (e.target.closest('[data-sec-cancel]')) {
                const form = e.target.closest('.sec-form');
                form.hidden = true;
                secError(form, '');
                form.closest('.sec-slot').querySelectorAll('[data-sec-edit]').forEach((b) => { b.hidden = false; });
            }
            if (e.target.closest('[data-sec-save]')) saveSecContact(e.target.closest('.sec-form'));
        });
        $('#view-backtest').addEventListener('keydown', (e) => {
            if (e.key !== 'Enter' || !e.target.matches('[data-sec-input]')) return;
            e.preventDefault(); // Enter must save the contact, not run the backtest around it
            saveSecContact(e.target.closest('.sec-form'));
        });
        $('#bk-search-table').addEventListener('click', (e) => {
            const i = e.target.closest('[data-apply]')?.dataset.apply;
            if (i !== undefined) applySetup(Number(i));
            const check = e.target.closest('[data-index-check]');
            if (check) checkOnIndex(Number(check.dataset.indexCheck), check);
        });
        setBkMode('compare', { hash: false }); // the deep link is read later, in setupNav
        setBkUniverse('portfolio');
        await refreshBkOptions();
        if (!state.labOptions) return;
        const gates = state.labOptions.entry_gate_order || Object.keys(state.labOptions.entry_gates);
        $('#bk-gate').innerHTML = gates.map((k) => `<option value="${esc(k)}">${esc(state.labOptions.entry_gates[k])}</option>`).join('');
        $('#bk-gate').value = 'none';
        if (!state.labOptions.portfolio_tickers.length) setBkUniverse('ticker');
        renderGridEditor();
    }

    // ============================================================ TREND AND DIPS (core/systematic.py, D-48)

    const TR_HELP = {
        rule: 'One rule on one market, invested or in T-bills, against holding the same thing throughout. Hold the market itself, or a 2x or 3x version of it, the way the leveraged Nasdaq strategy does.',
        savings: 'Buying the dip with new money: the same saving each month, invested at once or kept in T-bills until the market is a set share below its high.',
        rotation: 'MSCI’s momentum method across a fixed list of funds: the 6- and 12-month return without the latest month, over three years of volatility; the strongest funds held, weighted by their score.',
    };
    const TR_YES = (v) => (v ? 'Yes' : 'No');
    // A worst fall: none for cash, one decimal near a total loss (a 3x fund in 2000 to 2002 lost 99.9%, not 100%).
    const trFall = (v) => (!isNum(v) ? 'n/a' : v > -0.005 ? 'none' : fmt.pct(v, v < -0.99 ? 1 : 0));
    // Grid settings in a few words, in the rule's own order.
    const TR_SHORT = {
        window: (v) => `${v}-day average`, buffer: (v) => `buffer ${fmt.pct(v, v * 100 % 1 ? 1 : 0)}`, vol_cap: (v) => `volatility cap ${fmt.pct(v, 0)}`,
        length: (v) => `EMA ${v}`, months: (v) => `${v} months`, skip: (v) => `skip ${v}`, entry: (v) => `RSI under ${v}`,
        exit_window: (v) => `sell above ${v}-day average`, dip: (v) => `${fmt.pct(v, 0)} below the high`, lookback: (v) => `high of ${v} days`,
    };

    function trVerdict(sel, v) {
        const el = $(sel);
        el.className = `verdict ${v.level === 'strong' ? 'strong' : v.level === 'weak' ? 'weak' : ''}`;
        el.innerHTML = `<strong>${esc(v.title)}</strong>${(v.lines || []).map((l) => `<span>${esc(l)}</span>`).join('')}`;
    }

    const trNotes = (sel, notes) => { $(sel).innerHTML = (notes || []).map((n) => `<div class="warning note"><strong>Note</strong>${esc(n)}</div>`).join(''); };

    function showTrProgress(on, p = {}) {
        setIf($('#tr-progress'), 'hidden', !on);
        if (!on) { $('#tr-progress [data-job-cancel]').disabled = false; return; }
        const share = p.total ? Math.min(1, p.done / p.total) : 0;
        $('#tr-progress-fill').style.width = `${(share * 100).toFixed(0)}%`;
        setIf($('#tr-progress-text'), 'textContent', p.stage || 'Working');
    }

    async function openTrend() {
        if (!state.trOptions) {
            try {
                state.trOptions = await api('/api/systematic/options');
            } catch (err) {
                toast(`Trend options unavailable: ${err.message}`, 'error');
                return;
            }
            const o = state.trOptions;
            $('#tr-rule').innerHTML = o.rule_order.map((k) => `<option value="${esc(k)}">${esc(o.rules[k].label)}</option>`).join('');
            $('#tr-rule').value = 'two_engine';
            $('#tr-execution').innerHTML = Object.entries(o.executions).map(([k, l]) => `<option value="${esc(k)}">${esc(l)}</option>`).join('');
            $('#tr-universe').innerHTML = Object.entries(o.universes).map(([k, l]) => `<option value="${esc(k)}">${esc(l)}</option>`).join('');
            $('#tr-universe').value = 'sectors';
            $('#tr-thresholds').innerHTML = o.savings_thresholds.map((t) => `<label class="check"><input type="checkbox" data-tr-threshold="${esc(t)}" checked>${esc(fmt.pct(t, 0))}</label>`).join('');
            renderTrParams();
            setTrStudy(state.trStudy || 'rule');
        }
        if (!state.trNow) loadTrendNow();
    }

    function setTrStudy(study) {
        state.trStudy = study;
        $$('#tr-studies .seg').forEach((s) => { const on = s.dataset.study === study; s.classList.toggle('active', on); s.setAttribute('aria-selected', String(on)); });
        $('#tr-study-help').textContent = TR_HELP[study];
        $$('[data-tr-for]').forEach((el) => { el.hidden = !el.dataset.trFor.split(' ').includes(study); });
        if ($('#tr-start').selectedOptions[0]?.hidden) $('#tr-start').value = '';
        $('#tr-results').hidden = study !== 'rule' || !state.trRule;
        $('#tr-savings-results').hidden = study !== 'savings' || !state.trSavings;
        $('#tr-rotation-results').hidden = study !== 'rotation' || !state.trRotation;
        renderTrMarketHint();
    }

    function renderTrParams() {
        const key = $('#tr-rule').value;
        const spec = state.trOptions.rules[key];
        $('#tr-rule-help').textContent = spec.description;
        $('#tr-params').innerHTML = (state.trOptions.param_order[key] || Object.keys(spec.params)).map((k) => [k, spec.params[k]]).map(([k, p]) => {
            const id = `tr-p-${k}`;
            if (p.type === 'choice') {
                return `<div class="field"><label for="${id}">${esc(p.label)}</label><select id="${id}" class="input" data-tr-param="${esc(k)}" data-type="choice">${
                    Object.entries(p.choices).map(([v, l]) => `<option value="${esc(v)}" ${v === p.default ? 'selected' : ''}>${esc(l)}</option>`).join('')}</select></div>`;
            }
            if (p.type === 'bool') return `<div class="field wide"><label class="check"><input type="checkbox" id="${id}" data-tr-param="${esc(k)}" data-type="bool" ${p.default ? 'checked' : ''}>${esc(p.label)}</label></div>`;
            const scale = p.type === 'pct' ? 100 : 1;
            const step = p.type === 'int' ? 1 : p.type === 'pct' ? 0.5 : 0.1;
            return `<div class="field"><label for="${id}">${esc(p.label)}${p.type === 'pct' ? ' (%)' : ''}</label><input id="${id}" class="input" type="number" data-tr-param="${esc(k)}" data-type="${esc(p.type)}"
                value="${esc(+(p.default * scale).toFixed(4))}" min="${esc(+(p.min * scale).toFixed(4))}" max="${esc(+(p.max * scale).toFixed(4))}" step="${step}"></div>`;
        }).join('');
    }

    function trParams() {
        const out = {};
        $$('[data-tr-param]').forEach((i) => {
            const t = i.dataset.type;
            out[i.dataset.trParam] = t === 'bool' ? i.checked : t === 'choice' ? i.value : t === 'pct' ? Number(i.value) / 100 : t === 'int' ? parseInt(i.value, 10) : Number(i.value);
        });
        return out;
    }

    function renderTrMarketHint() {
        const o = state.trOptions;
        if (!o || state.trStudy === 'rotation') { $('#tr-market-hint').textContent = ''; return; }
        const signal = $('#tr-signal').value.trim().toUpperCase();
        const lev = Number($('#tr-leverage').value);
        const vehicle = $('#tr-vehicle').value.trim().toUpperCase();
        const parts = [];
        if ($('#tr-extend').value) {
            const proxy = o.index_proxies[signal];
            parts.push(proxy ? `Before ${signal} existed, the ${proxy} price index stands in for it, without dividends.` : `No index is known for ${signal}, so the history starts on its first day.`);
        }
        if (lev > 1) parts.push(`${lev}x: ${lev} times the daily return, less ${lev - 1} times the T-bill rate plus ${fmt.pct(o.spread, 1)} for the borrowing and a ${fmt.pct(o.fee, 2)} yearly fee (the spread that matches TQQQ since 2010).`);
        if (vehicle) parts.push(`${vehicle} replaces it from its first day; the results show how closely the two agreed.`);
        $('#tr-market-hint').textContent = parts.join(' ');
    }

    async function loadTrendNow(force = false) {
        if (state.trNowLoading || (state.trNow && !force)) return;
        state.trNowLoading = true;
        const btn = $('#tr-now-refresh');
        setBusy(btn, true);
        $('#tr-now-caption').textContent = 'Reading the latest closes...';
        try {
            state.trNow = await api('/api/systematic/check');
            renderTrendNow(state.trNow);
        } catch (err) {
            $('#tr-now-caption').textContent = `Could not read the markets (${err.message}). Yahoo sometimes refuses for a minute; try Refresh.`;
        } finally {
            state.trNowLoading = false;
            setBusy(btn, false);
        }
    }

    function renderTrendNow(d) {
        const rules = (d.rule_order || Object.keys(d.rules || {})).map((k) => [k, d.rules[k]]);
        const dates = [...new Set(d.markets.filter((m) => m.date).map((m) => m.date))];
        $('#tr-now-table').innerHTML = `<thead><tr><th>Market</th><th class="num">Close</th><th class="num">Against 200-day average</th><th class="num">20-day volatility</th><th class="num">VIX against 3-month VIX</th><th class="num">12-month momentum</th><th class="num">From 1-year high</th>${
            rules.map(([k, label]) => `<th>${esc(label)}${(d.alert_rules || []).includes(k) ? ' <span class="badge accent">alerts</span>' : ''}</th>`).join('')}</tr></thead><tbody>${d.markets.map((m) => {
            if (m.error) return `<tr><td class="mono span"><strong>${esc(m.ticker)}</strong></td><td colspan="${6 + rules.length}" class="muted span">${esc(m.error)}</td></tr>`;
            const r = m.readings;
            const vsAvg = isNum(r.close) && isNum(r.sma200) ? r.close / r.sma200 - 1 : null;
            return `<tr><td class="mono span"><strong>${esc(m.ticker)}</strong></td><td class="num" data-label="Close">${esc(fmt.num(r.close))}</td>
                <td class="num ${signCls(vsAvg)}" data-label="Against 200-day average">${esc(fmt.spct(vsAvg))}</td><td class="num hide-sm">${esc(fmt.pct(r.vol20, 0))}</td>
                <td class="num hide-sm">${isNum(r.vix_ratio) ? esc(fmt.num(r.vix_ratio, 2)) : 'n/a'}</td><td class="num hide-sm ${signCls(r.momentum_12_1)}">${esc(fmt.spct(r.momentum_12_1))}</td>
                <td class="num hide-sm">${esc(fmt.pct(r.from_high))}</td>${rules.map(([k, label]) => {
                    const s = (m.rules || {})[k] || {};
                    return `<td data-label="${esc(label)}"><span class="badge ${s.in_market ? 'bullish' : 'warn'}${s.reason ? ' term quiet' : ''}"${s.reason ? ` tabindex="0" data-tip="${esc(s.reason)}"` : ''}>${s.in_market ? 'Invested' : 'Cash'}</span>${s.since ? `<div class="muted small">since ${esc(fmtDay(s.since))}</div>` : ''}</td>`;
                }).join('')}</tr>`;
        }).join('')}</tbody>`;
        const alerting = (d.alert_rules || []).some((k) => (d.rules || {})[k]);
        $('#tr-now-caption').textContent = `At the ${dates.join(' / ') || 'latest'} close. VIX against the 3-month VIX above 1 means an inverted curve.${alerting ? ' The daily check raises an alert when the rule marked "alerts" flips; the others are shown only.' : ''}`;
    }

    async function runTrend() {
        const study = state.trStudy;
        const form = $('#tr-form');
        const signal = $('#tr-signal').value.trim().toUpperCase();
        if (study !== 'rotation' && !TICKER_RE.test(signal)) return formError(form, runErrorMessage({ message: 'Invalid ticker symbol' }, signal), $('#tr-signal'));
        const costField = study === 'rotation' ? $('#tr-rot-cost') : study === 'rule' ? $('#tr-cost') : null;
        if (costField && notNumberIn(costField, 0, 200)) return formError(form, 'Cost per trade goes from 0 to 200 bps.', costField);
        clearFormError(form);
        const body = {
            study, signal: $('#tr-signal').value.trim().toUpperCase(), leverage: Number($('#tr-leverage').value),
            vehicle: $('#tr-vehicle').value.trim().toUpperCase() || null, extend: !!$('#tr-extend').value, start: $('#tr-start').value || null,
        };
        if (study === 'rule') Object.assign(body, { rule: $('#tr-rule').value, params: trParams(), execution: $('#tr-execution').value, cost_bps: Number($('#tr-cost').value) });
        if (study === 'savings') {
            body.thresholds = $$('[data-tr-threshold]').filter((c) => c.checked).map((c) => Number(c.dataset.trThreshold));
            if (!body.thresholds.length) return formError(form, 'Tick at least one fall to wait for.', $('[data-tr-threshold]'));
        }
        if (study === 'rotation') {
            Object.assign(body, {
                universe: $('#tr-universe').value, top: Number($('#tr-top').value), review: $('#tr-review').value, weighting: $('#tr-weighting').value,
                buffer: $('#tr-buffer').checked, adhoc: $('#tr-adhoc').checked, trend_filter: $('#tr-trend-filter').checked, cost_bps: Number($('#tr-rot-cost').value),
            });
        }
        const btn = $('#tr-run');
        setBusy(btn, true);
        showTrProgress(true, { stage: 'Starting' });
        try {
            const result = await runJob('/api/systematic/run', body, (p) => showTrProgress(true, p), 'tr');
            if (study === 'rule') { state.trRule = result; setTrStudy(study); renderTrRule(result); }
            if (study === 'savings') { state.trSavings = result; setTrStudy(study); renderTrSavings(result); }
            if (study === 'rotation') { state.trRotation = result; setTrStudy(study); renderTrRotation(result); }
            if (state.bkMode === 'trend' && !$('#view-backtest').hidden) $(study === 'rule' ? '#tr-results' : study === 'savings' ? '#tr-savings-results' : '#tr-rotation-results').scrollIntoView({ block: 'start' });
            bkFinished('trend', 'Trend and dips finished: the results are ready.');
        } catch (err) {
            if (err.stopped) toast('Stopped. Nothing was changed.', 'info');
            else formError(form, runErrorMessage(err, signal), /ticker|price history|data available/i.test(err.message) ? $('#tr-signal') : null);
        } finally {
            setBusy(btn, false);
            showTrProgress(false);
        }
    }

    const TR_LINES = {
        rule: { color: () => cssVar('--accent'), width: 2.5 },
        buy_hold: { color: () => cssVar('--series-8'), width: 1.5, dash: true },
        rule_1x: { color: () => cssVar('--series-3'), width: 1.5, hidden: true },
        signal_hold: { color: () => cssVar('--series-4'), width: 1.5, dash: true, hidden: true },
        cash: { color: () => cssVar('--text-3'), width: 1, hidden: true },
    };
    // Decades of weekly points: let bars get thin, or "fit" shows only the latest few thousand weeks.
    const TR_TIME = { timeScale: { borderVisible: false, minBarSpacing: 0.05 } };
    const TR_LOG = { ...TR_TIME, rightPriceScale: { borderVisible: false, mode: 1 } };
    const PCT_AXIS = { ...TR_TIME, localization: { priceFormatter: (v) => `${(v * 100).toFixed(0)}%` } };

    function trLines(chartId, legendId, rows, series, extra = {}) {
        const chart = makeChart(chartId, extra);
        const items = [];
        rows.forEach((r) => {
            const spec = TR_LINES[r.key];
            if (!spec || !series[r.key]) return;
            const color = spec.color();
            const line = chart.addLineSeries({ color, lineWidth: spec.width, lineStyle: spec.dash ? 2 : 0, priceLineVisible: false, lastValueVisible: false, ...(extra.localization ? {} : { priceFormat: MONEY_FORMAT }) });
            line.setData(series[r.key]);
            items.push({ key: r.key, label: r.label, color, dash: spec.dash, hidden: spec.hidden, series: line });
        });
        legend(legendId, items, { chart: chartId });
        chart.timeScale().fitContent();
        return chart;
    }

    function trParamText(values, rule) {
        const order = state.trOptions.param_order[rule] || Object.keys(values);
        return order.filter((k) => k in values).map((k) => (TR_SHORT[k] ? TR_SHORT[k](values[k]) : `${k} ${values[k]}`)).join(', ');
    }

    function renderTrRule(d) {
        const rule = d.rows.find((r) => r.key === 'rule');
        const bh = d.rows.find((r) => r.key === 'buy_hold');
        const ts = rule.trade_stats || {};
        trVerdict('#tr-verdict', d.verdict);
        $('#tr-stats').innerHTML = statCards([
            ['The rule', `${fmt.pct(rule.stats.cagr)} a year`, `holding throughout ${fmt.pct(bh.stats.cagr)}`],
            ['Worst fall', trFall(rule.stats.max_drawdown), `holding throughout ${trFall(bh.stats.max_drawdown)}`],
            ['Sharpe', fmt.num(rule.stats.sharpe), `holding throughout ${fmt.num(bh.stats.sharpe)}`],
            ['Time invested', fmt.pct(rule.exposure, 0), `${fmt.num(ts.per_year, 1)} trades a year`],
            ['At the latest close', d.now.in_market ? 'Invested' : 'Cash', d.now.since ? `since ${fmtDay(d.now.since)}` : ''],
        ]);
        $('#tr-period').textContent = `${d.label} on ${d.signal}, held as ${d.held}: ${fmtDay(d.period.start)} to ${fmtDay(d.period.end)}, halves split at ${fmtDay(d.period.split)}`;
        $('#tr-table').innerHTML = `<thead><tr><th>Strategy</th><th class="num">CAGR</th><th class="num">Volatility</th><th class="num">Sharpe</th><th class="num">Worst fall</th><th class="num">Worst year</th><th class="num">1st half</th><th class="num">2nd half</th><th class="num">Invested</th></tr></thead><tbody>${
            d.rows.map((r) => `<tr class="${r.key === 'rule' ? 'is-current' : ''}"><td><strong>${esc(r.label)}</strong></td>
                <td class="num ${signCls(r.stats.cagr)}">${esc(fmt.pct(r.stats.cagr))}</td><td class="num">${esc(fmt.pct(r.stats.volatility, 0))}</td>
                <td class="num">${esc(fmt.num(r.stats.sharpe))}</td><td class="num">${esc(trFall(r.stats.max_drawdown))}</td>
                <td class="num">${esc(fmt.pct(r.stats.worst_year, 0))}</td><td class="num">${esc(fmt.pct(r.first_half.cagr))}</td>
                <td class="num">${esc(fmt.pct(r.second_half.cagr))}</td><td class="num">${isNum(r.exposure) ? esc(fmt.pct(r.exposure, 0)) : ''}</td></tr>`).join('')
        }</tbody>`;
        renderTrCharts(d);
        renderTrRobustness(d);
        $('#tr-years').innerHTML = `<thead><tr><th>Year</th><th class="num">The rule</th><th class="num">Holding</th><th class="num">Difference</th></tr></thead><tbody>${
            d.yearly.slice().reverse().map((y) => `<tr><td>${esc(y.year)}</td><td class="num ${signCls(y.rule)}">${esc(fmt.pct(y.rule))}</td><td class="num ${signCls(y.buy_hold)}">${esc(fmt.pct(y.buy_hold))}</td>
                <td class="num ${signCls(y.rule - y.buy_hold)}">${esc(fmt.spct(y.rule - y.buy_hold))}</td></tr>`).join('')}</tbody>`;
        $('#tr-trades-caption').textContent = ts.trades ? `${plural(ts.trades, 'trade', 'trades')}; ${fmt.pct(ts.win_rate, 0)} won, average win ${fmt.pct(ts.avg_win)}, average loss ${fmt.pct(ts.avg_loss)}, ${fmt.spct(ts.expectancy)} a trade on average` : 'No trades';
        $('#tr-trades').innerHTML = `<thead><tr><th>Bought</th><th>Sold</th><th class="num">Days</th><th class="num">Return</th></tr></thead><tbody>${
            d.trades.slice().reverse().map((t) => `<tr><td>${esc(t.entry)}</td><td>${t.exit ? esc(t.exit) : '<span class="muted">still held</span>'}</td><td class="num">${esc(t.days)}</td>
                <td class="num ${signCls(t.return)}">${esc(fmt.pct(t.return))}</td></tr>`).join('')}</tbody>`;
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const yearly = Object.fromEntries(d.yearly.map((y) => [String(y.year), y.rule]));
        $('#tr-months').innerHTML = `<thead><tr><th>Year</th>${months.map((m) => `<th class="num">${m}</th>`).join('')}<th class="num">Year</th></tr></thead><tbody>${
            d.monthly.years.slice().reverse().map((y) => `<tr><td>${esc(y)}</td>${d.monthly.cells[y].map((v) => `<td class="num">${isNum(v) ? `<span class="heat" style="${heatStyle(v * 400)}">${esc(fmt.pct(v, 1))}</span>` : ''}</td>`).join('')}
                <td class="num"><strong>${esc(fmt.pct(yearly[y]))}</strong></td></tr>`).join('')}</tbody>`;
        trNotes('#tr-notes', d.notes);
    }

    function renderTrCharts(d) {
        if (!d || $('#tr-results').hidden) return;
        trLines('tr-equity-chart', 'tr-equity-legend', d.rows, d.equity, TR_LOG);
        trLines('tr-dd-chart', 'tr-dd-legend', d.rows.filter((r) => r.key === 'rule' || r.key === 'buy_hold'), d.drawdowns, PCT_AXIS);
        $('#tr-price-title').textContent = `${d.signal} and the rule`;
        const chart = makeChart('tr-price-chart', TR_LOG);
        const band = chart.addAreaSeries({ priceScaleId: 'held', lineVisible: false, topColor: cssVar('--accent-soft'), bottomColor: cssVar('--accent-soft'), priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        band.applyOptions({ autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 1 } }) });
        band.setData(d.price.held);
        chart.priceScale('held').applyOptions({ visible: false, scaleMargins: { top: 0, bottom: 0 } });
        const close = chart.addLineSeries({ color: cssVar('--text-2'), lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
        close.setData(d.price.close);
        const items = [{ key: 'close', label: d.signal, color: cssVar('--text-2'), series: close }];
        Object.entries(d.price.lines || {}).forEach(([name, data], i) => {
            const color = cssVar(`--series-${[2, 3, 7][i % 3]}`);
            const s = chart.addLineSeries({ color, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
            s.setData(data);
            items.push({ key: name, label: name, color, series: s });
        });
        items.push({ label: 'Invested', color: cssVar('--accent-soft'), block: true });
        legend('tr-price-legend', items, { chart: 'tr-price-chart' });
        chart.timeScale().fitContent();
    }

    function renderTrRobustness(d) {
        const sw = d.sweep || {};
        const s = sw.summary || {};
        if (sw.rows && sw.rows.length) {
            $('#tr-sweep-caption').textContent = `${s.settings} settings, same period`;
            $('#tr-sweep').innerHTML = `<thead><tr><th>Setting</th><th class="num">CAGR</th><th class="num">Sharpe</th><th class="num">Worst fall</th><th class="num">Invested</th><th>Sharpe above holding, both halves</th><th>Smaller worst fall, both halves</th></tr></thead><tbody>${
                sw.rows.map((r) => `<tr class="${r.is_default ? 'is-current' : ''}"><td>${esc(trParamText(r.values, d.rule))}${r.is_default ? ' <span class="muted small">(yours)</span>' : ''}</td>
                    <td class="num">${esc(fmt.pct(r.stats.cagr))}</td><td class="num">${esc(fmt.num(r.stats.sharpe))}</td><td class="num">${esc(trFall(r.stats.max_drawdown))}</td>
                    <td class="num">${esc(fmt.pct(r.exposure, 0))}</td><td>${TR_YES(r.sharpe_both)}</td><td>${TR_YES(r.drawdown_both)}</td></tr>`).join('')}</tbody>`;
            $('#tr-sweep-hint').textContent = `Sharpe above holding throughout in both halves: ${s.sharpe_both} of ${s.settings}; smaller worst fall in both halves: ${s.drawdown_both} of ${s.settings}; return from ${fmt.pct(s.cagr_range[0])} to ${fmt.pct(s.cagr_range[2])} a year.${s.default_rank ? ` Yours ranks ${s.default_rank} of ${s.settings} by Sharpe.` : ''}`;
        } else {
            $('#tr-sweep-caption').textContent = '';
            $('#tr-sweep').innerHTML = '';
            $('#tr-sweep-hint').textContent = 'This rule has no settings to vary.';
        }
        const wf = d.walk_forward || {};
        if (wf.walk_forward) {
            $('#tr-wf-caption').textContent = `Chosen on the previous ${wf.train_years} years, held the next year: ${wf.start.slice(0, 4)} to ${wf.end.slice(0, 4)}`;
            const row = (label, p, cls = '') => `<tr class="${cls}"><td>${esc(label)}</td><td class="num">${esc(fmt.pct(p.cagr))}</td><td class="num">${esc(fmt.num(p.sharpe))}</td><td class="num">${esc(trFall(p.max_drawdown))}</td></tr>`;
            const counts = {};
            wf.picks.forEach((p) => { const k = trParamText(p.values, d.rule); counts[k] = (counts[k] || 0) + 1; });
            const chosen = Object.entries(counts).sort((a, b) => b[1] - a[1]);
            const rest = chosen.slice(4).reduce((n, [, c]) => n + c, 0);
            $('#tr-wf').innerHTML = `<thead><tr><th></th><th class="num">CAGR</th><th class="num">Sharpe</th><th class="num">Worst fall</th></tr></thead><tbody>${
                row('Walk-forward', wf.walk_forward, 'is-current') + row('Your setting, fixed', wf.default) + row('Holding throughout', wf.buy_hold)}</tbody>
                <tfoot><tr><td colspan="4" class="muted small wrap">Settings chosen: ${esc(chosen.slice(0, 4).map(([k, n]) => `${k} (${n} year${n === 1 ? '' : 's'})`).join('; '))}${rest ? esc(`; ${chosen.length - 4} others in ${rest} year${rest === 1 ? '' : 's'}`) : ''}. ${esc(wf.distinct_settings)} different settings in ${plural(wf.picks.length, 'year', 'years')}: the more the choice moves, the less the past said about the next year.</td></tr></tfoot>`;
        } else {
            $('#tr-wf-caption').textContent = 'Not enough years, or no settings to choose between';
            $('#tr-wf').innerHTML = '';
        }
        const b = d.bootstrap || {};
        if (b.rule) {
            $('#tr-boot-caption').textContent = `${b.paths} reshuffles in blocks of about ${b.block_days} days: the rule fell less in ${fmt.pct(b.p_shallower_drawdown, 0)} and earned more in ${fmt.pct(b.p_higher_cagr, 0)}`;
            const row = (label, x) => `<tr><td>${esc(label)}</td>${['p5', 'p50', 'p95'].map((p) => `<td class="num">${esc(fmt.pct(x.cagr[p]))}</td>`).join('')}${['p5', 'p50', 'p95'].map((p) => `<td class="num">${esc(trFall(x.max_drawdown[p]))}</td>`).join('')}</tr>`;
            $('#tr-boot').innerHTML = `<thead><tr><th></th><th class="num">CAGR, bad 5%</th><th class="num">median</th><th class="num">good 5%</th><th class="num">Worst fall, bad 5%</th><th class="num">median</th><th class="num">good 5%</th></tr></thead><tbody>${
                row('The rule', b.rule) + row('Holding throughout', b.buy_hold)}</tbody>`;
        } else {
            $('#tr-boot-caption').textContent = '';
            $('#tr-boot').innerHTML = '';
        }
        const t = d.tracking;
        $('#tr-tracking').textContent = t && t.start ? `The synthetic ${d.leverage}x against the real ${d.vehicle}, ${fmtDay(t.start)} to ${fmtDay(t.end)}: ${fmt.pct(t.cagr_synthetic)} against ${fmt.pct(t.cagr_real)} a year, daily correlation ${fmt.num(t.correlation, 3)}, tracking error ${fmt.pct(t.tracking_error)} a year.` : '';
    }

    function renderTrSavings(d) {
        trVerdict('#tr-sv-verdict', d.verdict);
        $('#tr-sv-period').textContent = `Into ${d.held}: ${d.period.deposits} monthly savings, ${fmtDay(d.period.start)} to ${fmtDay(d.period.end)}, halves split at ${fmtDay(d.period.split)}`;
        $('#tr-sv-table').innerHTML = `<thead><tr><th>Strategy</th><th class="num">Return a year</th><th class="num">1st half</th><th class="num">2nd half</th><th class="num">Money multiple</th><th class="num">In cash on average</th><th class="num">Longest wait</th><th class="num">Dip buys</th></tr></thead><tbody>${
            d.rows.map((r) => `<tr class="${r.key === 'now' ? 'is-current' : ''}"><td><strong>${esc(r.label)}</strong></td><td class="num">${esc(fmt.pct(r.whole.irr))}</td>
                <td class="num">${esc(fmt.pct(r.first_half.irr))}</td><td class="num">${esc(fmt.pct(r.second_half.irr))}</td><td class="num">${esc(fmt.num(r.whole.multiple, 2))}x</td>
                <td class="num">${esc(fmt.pct(r.cash_share, 0))}</td><td class="num">${r.key === 'now' ? '' : `${esc(fmt.num(r.whole.longest_wait_months, 0))} months`}</td>
                <td class="num">${r.dip_buys === null ? '' : esc(r.dip_buys)}</td></tr>`).join('')}</tbody>`;
        const roll = Object.entries(d.rolling || {});
        const meta = d.rolling_meta || {};
        $('#tr-sv-rolling-caption').textContent = roll.length ? `${plural(roll[0][1].windows, 'window', 'windows')} of ${meta.horizon_years} years, starting ${meta.first_start} to ${meta.last_start}; earlier and later split at ${meta.split_start}` : `Too short for ${meta.horizon_years}-year windows`;
        const label = (k) => (d.rows.find((r) => r.key === k) || {}).label || k;
        $('#tr-sv-rolling').innerHTML = roll.length ? `<thead><tr><th>Strategy</th><th class="num">Beat investing at once</th><th class="num">Earlier windows</th><th class="num">Later windows</th><th class="num">Median difference</th><th class="num">Worst</th><th class="num">Best</th></tr></thead><tbody>${
            roll.map(([k, r]) => `<tr><td>${esc(label(k))}</td><td class="num"><strong>${esc(fmt.pct(r.beat_share, 0))}</strong></td><td class="num">${esc(fmt.pct(r.beat_share_first, 0))}</td>
                <td class="num">${esc(fmt.pct(r.beat_share_second, 0))}</td><td class="num ${signCls(r.median)}">${esc(fmt.spct(r.median, 2))}</td>
                <td class="num ${signCls(r.worst)}">${esc(fmt.spct(r.worst, 1))}</td><td class="num ${signCls(r.best)}">${esc(fmt.spct(r.best, 1))}</td></tr>`).join('')}</tbody>` : '';
        trNotes('#tr-sv-notes', d.notes);
        const chart = makeChart('tr-sv-chart', PCT_AXIS);
        const color = cssVar('--series-2');
        const line = chart.addLineSeries({ color, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
        line.setData(d.drawdown);
        const levels = d.thresholds.map((t) => line.createPriceLine({ price: -t, color: cssVar('--text-3'), lineWidth: 1, lineStyle: 2, axisLabelVisible: true }));
        legend('tr-sv-legend', [{ key: 'dd', label: `${d.signal} below its high`, color, series: line }, { key: 'levels', label: 'Levels waited for', color: cssVar('--text-3'), dash: true, lines: levels }], { chart: 'tr-sv-chart' });
        chart.timeScale().fitContent();
    }

    function renderTrRotation(d) {
        trVerdict('#tr-rot-verdict', d.verdict);
        $('#tr-rot-period').textContent = `${d.universe}: ${fmtDay(d.period.start)} to ${fmtDay(d.period.end)}, ${plural(d.period.months, 'month', 'months')}, halves split at ${fmtDay(d.period.split)}`;
        $('#tr-rot-table').innerHTML = `<thead><tr><th>Strategy</th><th class="num">CAGR</th><th class="num">Sharpe</th><th class="num">Worst fall</th><th class="num">1st half</th><th class="num">2nd half</th><th class="num">vs equal weight, 1st</th><th class="num">2nd</th><th class="num">Turnover a year</th></tr></thead><tbody>${
            d.rows.map((r, i) => `<tr class="${i === 0 ? 'is-current' : ''}"><td><strong>${esc(r.label)}</strong></td><td class="num">${esc(fmt.pct(r.stats.cagr))}</td><td class="num">${esc(fmt.num(r.stats.sharpe))}</td>
                <td class="num">${esc(trFall(r.stats.max_drawdown))}</td><td class="num">${esc(fmt.pct(r.first_half.cagr))}</td><td class="num">${esc(fmt.pct(r.second_half.cagr))}</td>
                <td class="num ${r.key === 'equal_weight' ? 'muted' : signCls(r.excess_first)}">${r.key === 'equal_weight' ? 'reference' : esc(fmt.spct(r.excess_first))}</td>
                <td class="num ${r.key === 'equal_weight' ? 'muted' : signCls(r.excess_second)}">${r.key === 'equal_weight' ? '' : esc(fmt.spct(r.excess_second))}</td>
                <td class="num">${isNum(r.turnover) ? esc(fmt.pct(r.turnover, 0)) : ''}</td></tr>`).join('')}</tbody>`;
        const chart = makeChart('tr-rot-chart', TR_LOG);
        const colors = [cssVar('--accent'), cssVar('--series-3'), cssVar('--series-8'), benchColor('sp500')];
        const items = d.rows.map((r, i) => {
            const s = chart.addLineSeries({ color: colors[i % colors.length], lineWidth: i === 0 ? 2.5 : 1.5, lineStyle: r.kind === 'benchmark' ? 2 : 0, priceLineVisible: false, lastValueVisible: false, priceFormat: MONEY_FORMAT });
            s.setData(d.equity[r.key] || []);
            return { key: r.key, label: r.label, color: colors[i % colors.length], dash: r.kind === 'benchmark', series: s };
        });
        legend('tr-rot-legend', items, { chart: 'tr-rot-chart' });
        chart.timeScale().fitContent();
        $('#tr-rot-latest-caption').textContent = `At the ${d.latest_date} close`;
        $('#tr-rot-latest').innerHTML = `<thead><tr><th>Fund</th><th class="num">6 months</th><th class="num">12 months</th><th class="num">Volatility</th><th class="num">Z</th><th class="num">Score</th></tr></thead><tbody>${
            d.latest.map((r, i) => `<tr class="${i < d.top ? 'is-current' : ''}"><td class="mono"><strong>${esc(r.ticker)}</strong></td><td class="num ${signCls(r.m6)}">${esc(fmt.spct(r.m6))}</td>
                <td class="num ${signCls(r.m12)}">${esc(fmt.spct(r.m12))}</td><td class="num">${esc(fmt.pct(r.sigma, 0))}</td><td class="num">${esc(fmt.num(r.z))}</td><td class="num">${esc(fmt.num(r.score))}</td></tr>`).join('')}</tbody>`;
        $('#tr-rot-sweep').innerHTML = `<thead><tr><th>Setting</th><th class="num">CAGR</th><th class="num">Worst fall</th><th>Ahead of equal weight, both halves</th></tr></thead><tbody>${
            d.sweep.map((r) => `<tr class="${r.is_default ? 'is-current' : ''}"><td>${esc(r.label)}</td><td class="num">${esc(fmt.pct(r.stats.cagr))}</td>
                <td class="num">${esc(trFall(r.stats.max_drawdown))}</td><td>${TR_YES(r.beats_both)}</td></tr>`).join('')}</tbody>`;
        $('#tr-rot-reviews').innerHTML = `<thead><tr><th>Date</th><th>Review</th><th>Held after it</th></tr></thead><tbody>${
            d.reviews.slice().reverse().map((r) => `<tr><td>${esc(fmtDay(r.date))}</td><td>${esc(capFirst(r.kind))}</td><td class="wrap">${esc(r.holdings.map((h) => `${h.ticker} ${fmt.pct(h.weight, 0)}`).join(', '))}</td></tr>`).join('')}</tbody>`;
        trNotes('#tr-rot-notes', d.notes);
    }

    function setupTrend() {
        state.trStudy = 'rule';
        $('#tr-studies').addEventListener('click', (e) => { const b = e.target.closest('[data-study]'); if (b) setTrStudy(b.dataset.study); });
        $('#tr-rule').addEventListener('change', renderTrParams);
        $('#tr-form').addEventListener('submit', (e) => { e.preventDefault(); runTrend(); });
        $('#tr-now-refresh').addEventListener('click', () => loadTrendNow(true));
        ['#tr-signal', '#tr-leverage', '#tr-vehicle', '#tr-extend'].forEach((sel) => $(sel).addEventListener('input', renderTrMarketHint));
        $('#tr-leverage').addEventListener('change', () => {
            // The usual pairing: 3x on QQQ is TQQQ's history, and the index takes the test back through 2000 to 2002.
            if ($('#tr-leverage').value === '3' && $('#tr-signal').value.trim().toUpperCase() === 'QQQ' && !$('#tr-vehicle').value.trim()) {
                $('#tr-vehicle').value = 'TQQQ';
                $('#tr-extend').value = '1';
            }
            renderTrMarketHint();
        });
    }

    async function init() {
        setupTheme();
        // The linked tab shows at once (class and hidden only, no data loads yet), and the forms stay inert until the
        // app is ready, so a click or Enter on a slow connection does not submit the bare form natively.
        const early = location.hash.replace('#', '').split('=')[0];
        if (early && document.getElementById(`view-${early}`)) showView(early, { push: false, load: false });
        try {
            await loadMeta();
        } catch (err) {
            if (err.message !== SERVER_DOWN) toast(`Could not reach the server: ${err.message}`, 'error', 15000);
            $$('[data-init-gate]').forEach((b) => { if (b.id !== 'rec-run') b.disabled = false; });
            return;
        }
        updateMarketPill();
        setInterval(updateMarketPill, 30000);
        setupAnalyze();
        setupCompany();
        setupWatchlist();
        renderChips();
        restoreWatchScan();
        setupPortfolio();
        setupWeights();
        markDirty(false);
        renderWeights();
        setupScoreBacktest();
        setupLab();
        setupBacktest();
        setupTrend();
        setupNav();
        setupSegs();
        setupTips();
        setupLegends();
        setupRegime();
        setupEdgeCues();
        setupScrollRegions();
        setupTipPlacement();
        setupStrips();
        appReady = true;
        // #rec-run is enabled by renderRecordForm, which knows whether there is an import to replay.
        $$('[data-init-gate]').forEach((b) => { if (b.id !== 'rec-run') b.disabled = false; });
        route(location.hash);  // a deep link runs what it names, now that the buttons it presses are live
    }

    // A row that scrolls sideways (tabs, section switchers, wide tables) fades at the edge that has more to show.
    function setupEdgeCues() {
        const SEL = '.tabs, .segmented, .table-wrap, .flow-scroll, .timeline, .heatmap';
        const mark = (el) => {
            const max = el.scrollWidth - el.clientWidth;
            const edge = max > 2 ? `${el.scrollLeft > 2 ? 'l' : ''}${el.scrollLeft < max - 2 ? 'r' : ''}` : '';
            if (el.dataset.edge !== edge) { if (edge) el.dataset.edge = edge; else delete el.dataset.edge; }
        };
        const markAll = () => $$(SEL).forEach(mark);
        let queued = false;
        const later = () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; markAll(); }); } };
        document.addEventListener('scroll', (e) => { if (e.target instanceof Element && e.target.matches(SEL)) mark(e.target); }, true);
        // The fades exist only on narrow screens (the [data-edge] rules in style.css sit inside max-width: 980px), so
        // the page is watched only there; on a wide screen every pass changed nothing on screen.
        const narrow = window.matchMedia('(max-width: 980px)');
        const observer = new MutationObserver(later);
        const watch = () => {
            if (!narrow.matches) { observer.disconnect(); return; }
            observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['hidden', 'class'] });
            later();
        };
        const whenNarrow = () => { if (narrow.matches) later(); };
        narrow.addEventListener('change', watch);
        window.addEventListener('resize', whenNarrow);
        // Layout changes that are not DOM mutations: a drawer opening, a late font face, a hover tip widening a table.
        document.addEventListener('toggle', whenNarrow, true);
        if (document.fonts) document.fonts.addEventListener('loadingdone', whenNarrow);
        document.addEventListener('pointerover', (e) => { if (e.target instanceof Element && e.target.closest('.term, .info, .peek')) whenNarrow(); });
        document.addEventListener('pointerout', (e) => { if (e.target instanceof Element && e.target.closest('.term, .info, .peek')) whenNarrow(); });
        watch();
    }

    // A scrolling table is a named region the keyboard can reach (Safari does not tab into it on its own, and no browser
    // names it). Tables written with tableHtml or detailTable carry their own label; every other .table-wrap, in the page
    // or rendered later, gets the heading above it.
    function setupScrollRegions() {
        const labelOf = (el) => {
            for (let n = el; n && !n.matches('.view'); n = n.parentElement) {
                for (let s = n.previousElementSibling; s; s = s.previousElementSibling) {
                    const h = s.matches('h2, h3, h4') ? s : s.querySelector('h2, h3, h4');
                    if (!h) continue;
                    const c = h.cloneNode(true);
                    c.querySelectorAll('.info, .muted, .badge').forEach((x) => x.remove());
                    const t = c.textContent.replace(/\s+/g, ' ').trim();
                    if (t) return t.slice(0, 80);
                }
                if (n.matches('.card')) break;
            }
            return 'Table';
        };
        const pass = () => $$('.table-wrap:not([role])').forEach((el) => {
            // Two regions with the same name are told apart by a number, so a screen reader can list them.
            const base = labelOf(el);
            let label = base;
            for (let n = 2; $$('[role="region"]').some((r) => r.getAttribute('aria-label') === label); n++) label = `${base} ${n}`;
            el.setAttribute('role', 'region');
            el.setAttribute('aria-label', label);
            el.setAttribute('tabindex', '0');
        });
        let queued = false;
        const later = () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; pass(); }); } };
        new MutationObserver(later).observe(document.body, { childList: true, subtree: true });
        later();
    }

    // ============================================================ MARKET REGIME
    // The Analyze-header badge opens a panel with why the market reads as it does today, the label and the Macro agent
    // score against SPY and QQQ since 2000, and what followed each regime. Loaded the first time it is opened
    // (GET /api/regime/history, cached for a day). Read only: nothing here feeds a score.

    const RG_NAMES = { BULL: 'Bull', CORRECTION: 'Correction', BEAR: 'Bear', CRISIS: 'Crisis' };
    const RG_ORDER = ['BULL', 'CORRECTION', 'BEAR', 'CRISIS'];
    const RG_METRICS = {
        vol_3m: { label: 'Volatility, next 3 months', note: 'How much the price swung (annualised), averaged over the months in each regime.', fmt: (v) => fmt.pct(v, 1) },
        mdd_3m: { label: 'Worst drop, next 3 months', note: 'The largest fall from a peak within the next 3 months, averaged.', fmt: (v) => fmt.pct(v, 1) },
        ret_12m: { label: 'Return, next 12 months', note: 'Price return over the next 12 months, averaged.', fmt: (v) => fmt.spct(v, 1) },
    };
    const RG_ERA_NAMES = { spy_trend: 'market trend', vix: 'VIX', vix_term: 'the 3-month VIX', fred_yields: 'yield curve', breadth: 'breadth', credit: 'credit', ten_year: '10 year yield' };
    const RG_ERA_MARK = { vix_term: 'VIX3M', breadth: 'Breadth', credit: 'Credit' };
    const TIP_REGIME = 'Market regime: how the whole stock market looks right now, read from the S&P 500 trend, the VIX (fear), the yield curve, breadth and credit. Each input adds or removes points and the sum gives Bull, Correction, Bear or Crisis. It describes risk (how bumpy the next months tend to be), not where prices go next.';
    const TIP_MACRO = 'Macro agent score: the same inputs turned into a score from -100 to +100. Unlike the regime it reads fear as an opportunity, so a high VIX adds points here. It never votes on the owner score; it only informs entry timing.';

    const rgInfo = (tip) => `<span class="info" tabindex="0" role="note" aria-label="${esc(tip)}" data-tip="${esc(tip)}">i</span>`;
    const rgName = (r) => RG_NAMES[r] || titleCase(r || '');
    const rgPts = (v) => (isNum(v) ? `<span class="${v > 0 ? 'pos' : v < 0 ? 'neg' : 'muted'}">${esc(fmt.signed(v, 1))}</span>` : '<span class="muted">n/a</span>');

    function hexAlpha(color, alpha) {
        const m = /^#([0-9a-f]{6})$/i.exec(String(color).trim());
        if (!m) return color;
        const n = parseInt(m[1], 16);
        return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
    }
    // One muted colour per regime; Bear and Crisis share a hue and differ in strength. The legend names each in words.
    const rgBand = (i) => [hexAlpha(cssVar('--pos'), 0.2), hexAlpha(cssVar('--warn'), 0.3), hexAlpha(cssVar('--neg'), 0.3), hexAlpha(cssVar('--neg'), 0.6)][i];

    function regimeCard() {
        let card = $('#regime-card');
        if (!card) {
            const anchor = $('#hero');
            if (!anchor) return null;
            card = document.createElement('div');
            card.className = 'card regime-card';
            card.id = 'regime-card';
            card.hidden = true;
            anchor.after(card);
        }
        return card;
    }

    function setupRegime() {
        state.regime = { asset: 'SPY', metric: 'vol_3m', data: null };
        document.addEventListener('keydown', (e) => {
            if (e.key !== 'Escape' || !state.regimeOpen) return;
            toggleRegime(false);
            const opener = $('[data-regime-open]');
            if (opener) opener.focus();
        });
        document.addEventListener('click', (e) => {
            if (e.target.closest('[data-regime-open]')) { toggleRegime(); return; }
            if (e.target.closest('[data-regime-close]')) { toggleRegime(false); return; }
            if (e.target.closest('[data-regime-retry]')) { loadRegime(true); return; }
            const asset = e.target.closest('[data-rg-asset]');
            const metric = e.target.closest('[data-rg-metric]');
            if (asset || metric) {
                if (asset) state.regime.asset = asset.dataset.rgAsset;
                if (metric) state.regime.metric = metric.dataset.rgMetric;
                renderRegimeStats();
            }
        });
    }

    function toggleRegime(force) {
        const card = regimeCard();
        if (!card) return;
        const open = typeof force === 'boolean' ? force : card.hidden;
        card.hidden = !open;
        state.regimeOpen = open;
        $$('[data-regime-open]').forEach((b) => b.setAttribute('aria-expanded', String(open)));
        if (!open) { destroyChart('rg-price'); destroyChart('rg-score'); return; }
        if (state.regime.data) renderRegimePanel();
        else loadRegime(false);
        card.scrollIntoView({ block: 'start' });  // next to the badge, below the sticky header (scroll-padding-top)
        const heading = card.querySelector('h3');
        if (heading) heading.focus({ preventScroll: true });
    }

    async function loadRegime(retry) {
        const card = regimeCard();
        card.innerHTML = `<div class="card-head"><h3 tabindex="-1">Market regime ${rgInfo(TIP_REGIME)}</h3></div>
            <p class="hint" role="status">Loading 25 years of market data. The first time can take a minute.</p>`;
        try {
            state.regime.data = await api(`/api/regime/history${retry ? '?refresh=1' : ''}`);
        } catch (err) {
            if (!state.regimeOpen) return;
            card.innerHTML = `<div class="card-head"><h3 tabindex="-1">Market regime ${rgInfo(TIP_REGIME)}</h3>
                <button type="button" class="btn btn-ghost btn-sm" data-regime-close>Close</button></div>
                <div class="warning"><strong>Check</strong>${esc(err.message)} Yahoo sometimes refuses a request, so waiting a minute usually works.</div>
                <p><button type="button" class="btn btn-sm" data-regime-retry>Try again</button></p>`;
            return;
        }
        if (state.regimeOpen) renderRegimePanel();
    }

    function rgCutoffText(why) {
        const c = why.cutoffs;
        return c.map((x, i) => {
            const name = rgName(x.regime);
            if (x.from === null) return `${name}: below ${fmt.num(c[i - 1].from, 0)}`;
            if (i === 0) return `${name}: ${fmt.num(x.from, 0)} or more`;
            return `${name}: ${fmt.num(x.from, 0)} up to ${fmt.num(c[i - 1].from, 0)}`;
        }).join('. ');
    }

    function rgWhyHtml(why) {
        const rows = why.inputs.map((i) => `<tr>
            <td class="wrap span"><strong>${esc(i.label)}</strong></td>
            <td class="wrap span" data-label="Reading now">${i.available ? esc(i.value_text) : '<span class="muted">not available</span>'}</td>
            <td class="wrap muted span" data-label="Rule">${esc(i.rule)}${i.agent_rule ? `<br>${esc(i.agent_rule)}` : ''}</td>
            <td class="num" data-label="Regime points">${rgPts(i.points)}</td>
            <td class="num" data-label="Macro agent points">${rgPts(i.agent_points)}</td></tr>`).join('');
        const c = why.cutoffs;
        const idx = Math.max(0, c.findIndex((x) => x.regime === why.regime));
        const above = idx === 0 ? null : c[idx - 1];
        const own = c[idx];
        let where;
        if (own.from === null) where = `${fmt.num(Math.abs(why.total - c[idx - 1].from), 1)} points below the ${rgName(c[idx - 1].regime)} cut-off`;
        else where = `${fmt.num(why.total - own.from, 1)} points above the ${rgName(own.regime)} cut-off${above ? `, ${fmt.num(above.from - why.total, 1)} short of ${rgName(above.regime)}` : ''}`;
        const ag = why.agent;
        return `<div class="table-wrap"><table class="table compact stack">
            <thead><tr><th>Input</th><th>Reading now</th><th>Rule</th><th class="num">Regime points</th><th class="num">Macro agent points</th></tr></thead>
            <tbody>${rows}
            <tr><td class="span"><strong>Sum</strong></td><td colspan="2" class="wrap span"><strong>${esc(rgName(why.regime))}</strong>: ${esc(where)}.</td>
                <td class="num" data-label="Regime points"><strong>${esc(fmt.signed(why.total, 1))}</strong></td><td class="num" data-label="Macro agent points"><strong>${esc(fmt.signed(ag.points, 1))}</strong></td></tr>
            </tbody></table></div>
            <p class="hint">Cut-offs on the regime sum. ${esc(rgCutoffText(why))}. The Macro agent score is its points divided by ${esc(fmt.num(ag.max_points, 1))}, times 100: <strong>${esc(fmt.signed(ag.score, 0))}</strong> today.</p>
            ${why.missing.length ? `<p class="hint">Not available today, so skipped: ${esc(why.missing.map((k) => RG_ERA_NAMES[k] || k).join(', '))}.</p>` : ''}`;
    }

    function rgStatsHtml(data) {
        const rs = state.regime;
        const stats = data.stats;
        const assets = Object.keys(stats.assets).sort((x, y) => (x === 'SPY' ? -1 : y === 'SPY' ? 1 : x.localeCompare(y)));
        if (!assets.includes(rs.asset)) rs.asset = assets[0];
        const a = stats.assets[rs.asset];
        const m = RG_METRICS[rs.metric];
        const one = (p, r) => {
            const s = a[p][r];
            const v = s[rs.metric];
            return `<td class="num">${isNum(v) ? esc(m.fmt(v)) : 'n/a'} <span class="muted small">${s[`${rs.metric}_n`]} months</span></td>`;
        };
        const body = RG_ORDER.map((r) => {
            const w = a.whole[r];
            return `<tr><td><strong>${esc(rgName(r))}</strong></td>
                <td class="num">${isNum(w.share) ? esc(fmt.pct(w.share, 0)) : 'n/a'} <span class="muted small">${plural(w.months, 'month', 'months')}</span></td>
                ${one('whole', r)}${one('first_half', r)}${one('second_half', r)}</tr>`;
        }).join('');
        const seg = (attr, cur, items) => `<div class="segmented seg-sm">${items.map(([k, label]) => `<button type="button" class="seg ${k === cur ? 'active' : ''}" ${attr}="${esc(k)}">${esc(label)}</button>`).join('')}</div>`;
        return `<div class="card-head sub"><h4>What followed each regime</h4>
                <div class="head-actions">${seg('data-rg-asset', rs.asset, assets.map((x) => [x, x]))}${seg('data-rg-metric', rs.metric, Object.entries(RG_METRICS).map(([k, v]) => [k, v.label]))}</div></div>
            <div class="table-wrap"><table class="table compact">
                <thead><tr><th>Regime</th><th class="num">Time in regime</th><th class="num">Whole period</th><th class="num">First half</th><th class="num">Second half</th></tr></thead>
                <tbody>${body}</tbody></table></div>
            <p class="hint">${esc(m.note)} Month-ends from ${esc(fmtDay(a.first_month || stats.first))}; the halves split at ${esc(fmtDay(stats.split))}. A regime with few months in a half is a weak sample.</p>`;
    }

    function renderRegimeStats() {
        const host = $('#rg-stats');
        if (host && state.regime.data) host.innerHTML = rgStatsHtml(state.regime.data);
    }

    function rgErasHtml(data) {
        const rows = data.eras.map((e) => `<tr><td>${esc(e.first.slice(0, 7))} to ${esc(e.last.slice(0, 7))}</td>
            <td class="wrap">${e.missing.length ? esc(e.missing.map((k) => RG_ERA_NAMES[k] || k).join(', ')) : '<span class="muted">none</span>'}</td>
            <td class="num">${e.months}</td><td class="num">${esc(fmt.pct(e.shares.BULL, 0))}</td><td class="num">${esc(fmt.pct(e.shares.BEAR + e.shares.CRISIS, 0))}</td></tr>`).join('');
        return `<details class="drawer"><summary>When each input began</summary>
            <p class="hint">Some inputs did not exist yet in the early years and were skipped, never filled in. The cut-offs stay the same, so with fewer inputs the same label means a slightly different thing.</p>
            <div class="table-wrap"><table class="table compact"><thead><tr><th>Period</th><th>Inputs missing</th><th class="num">Months</th><th class="num">Bull</th><th class="num">Bear or crisis</th></tr></thead><tbody>${rows}</tbody></table></div></details>`;
    }

    function renderRegimePanel() {
        const card = regimeCard();
        const data = state.regime.data;
        if (!card || !data) return;
        const why = data.today;
        const hadFocus = card.contains(document.activeElement);
        card.innerHTML = `
            <div class="card-head">
                <h3 tabindex="-1">Market regime ${rgInfo(TIP_REGIME)}</h3>
                <button type="button" class="btn btn-ghost btn-sm" data-regime-close>Close</button>
            </div>
            <p class="hint">The market reads as <strong>${esc(rgName(why.regime))}</strong> on ${esc(fmtDay(why.as_of))}, a sum of ${esc(fmt.signed(why.total, 1))} points. Every reading uses only data up to its own date.</p>
            ${data.warning ? `<div class="warning"><strong>Check</strong>${esc(data.warning)}</div>` : ''}
            <div class="card-head sub"><h4>Why today</h4></div>
            ${rgWhyHtml(why)}
            <div class="card-head sub"><h4>Since 2000 ${rgInfo(TIP_MACRO)}</h4><div class="legend" id="rg-legend"></div></div>
            <div class="hint" id="rg-readout"></div>
            <div class="chart chart-md" id="rg-price"></div>
            <div class="chart chart-sm" id="rg-score"></div>
            <p class="hint">Top: adjusted closing prices on a log scale, with the regime at that time as the colour behind them. Bottom: the Macro agent score, from -100 to +100; the squares mark when an input joined the score.</p>
            <div id="rg-stats">${rgStatsHtml(data)}</div>
            <p class="hint"><strong>How to read it.</strong> The regime sorts risk, not returns: swings and drops grow from Bull to Crisis, but the 12 months after were not reliably better or worse. The owner score works in every regime, so it does not change with it.</p>
            ${rgErasHtml(data)}`;
        if (hadFocus) card.querySelector('h3').focus({ preventScroll: true });  // the loading card held focus
        renderRegimeChart();
    }

    function renderRegimeChart() {
        const data = state.regime.data;
        if (!data || !$('#rg-price')) return;
        const w = data.weekly;
        const n = w.dates.length;
        const price = makeChart('rg-price', { rightPriceScale: { borderVisible: false, minimumWidth: 64, mode: 1 }, timeScale: { borderVisible: false, minBarSpacing: 0.05 } });
        // The regime as a coloured band: one area per regime on its own hidden scale (1 inside the regime, 0 outside).
        RG_ORDER.forEach((r, k) => {
            const area = price.addAreaSeries({ priceScaleId: 'rg-band', lineVisible: false, topColor: rgBand(k), bottomColor: rgBand(k), priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
            area.applyOptions({ autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 1 } }) });
            area.setData(w.dates.map((t, i) => ({ time: t, value: w.regime[i] === k ? 1 : 0 })));
        });
        price.priceScale('rg-band').applyOptions({ visible: false, scaleMargins: { top: 0, bottom: 0 } });
        const line = (key, color) => {
            const s = price.addLineSeries({ color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
            s.setData(w.dates.map((t, i) => (isNum(w[key][i]) ? { time: t, value: w[key][i] } : { time: t })));
            return s;
        };
        const spy = line('spy', benchColor('sp500'));
        const qqq = w.qqq.some(isNum) ? line('qqq', benchColor('nasdaq100')) : null;
        const items = [{ key: 'spy', label: 'SPY', color: benchColor('sp500'), series: spy }];
        if (qqq) items.push({ key: 'qqq', label: 'QQQ', color: benchColor('nasdaq100'), series: qqq });
        RG_ORDER.forEach((r, i) => items.push({ label: `${rgName(r)} regime`, color: rgBand(i), block: true }));
        legend('rg-legend', items, { chart: 'rg-price' });
        price.timeScale().fitContent();

        const score = makeChart('rg-score', { rightPriceScale: { borderVisible: false, minimumWidth: 64 }, timeScale: { borderVisible: false, minBarSpacing: 0.05 } });
        const sl = score.addLineSeries({ color: cssVar('--series-7'), lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
        sl.setData(w.dates.map((t, i) => (isNum(w.macro_score[i]) ? { time: t, value: w.macro_score[i] } : { time: t })));
        sl.createPriceLine({ price: 0, color: cssVar('--border-strong'), lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: '' });
        // Mark when an input joined: before that date the score was built from fewer inputs.
        const marks = [];
        const markPoints = [];
        // On a narrow chart the labels would collide: keep the squares, the eras table names them.
        const narrow = (document.getElementById('rg-score')?.clientWidth || 0) < 600;
        data.eras.slice(1).forEach((e, i) => {
            const added = e.inputs.filter((k) => !data.eras[i].inputs.includes(k));
            const label = added.map((k) => RG_ERA_MARK[k] || RG_ERA_NAMES[k] || k).join(', ');
            const t = w.dates.find((d) => d >= e.first);
            if (label && t) { marks.push({ time: t, position: 'inBar', shape: 'square', color: cssVar('--text-3'), text: narrow ? '' : label }); markPoints.push({ time: t, value: 0.92 }); }
        });
        // The markers ride on their own hidden scale, so they never stretch the score's axis.
        const marker = score.addLineSeries({ priceScaleId: 'rg-marks', lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        marker.applyOptions({ autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 1 } }) });
        score.priceScale('rg-marks').applyOptions({ visible: false, scaleMargins: { top: 0, bottom: 0 } });
        marker.setData(markPoints);
        marker.setMarkers(marks);
        score.timeScale().fitContent();

        // The two charts pan and zoom together.
        let busy = false;
        const link = (from, to) => from.timeScale().subscribeVisibleLogicalRangeChange((r) => {
            if (!r || busy) return;
            busy = true;
            to.timeScale().setVisibleLogicalRange(r);
            busy = false;
        });
        link(price, score);
        link(score, price);

        const readout = $('#rg-readout');
        const show = (i) => {
            const bits = [`${w.dates[i]}: ${rgName(RG_ORDER[w.regime[i]])}`, isNum(w.macro_score[i]) ? `Macro agent score ${fmt.signed(w.macro_score[i], 0)}` : null,
                isNum(w.spy[i]) ? `SPY ${fmt.num(w.spy[i], 2)}` : null, isNum(w.qqq[i]) ? `QQQ ${fmt.num(w.qqq[i], 2)}` : null];
            readout.textContent = bits.filter(Boolean).join(', ');
        };
        const at = new Map(w.dates.map((d, i) => [d, i]));
        const hover = (p) => show(at.has(p && p.time) ? at.get(p.time) : n - 1);
        price.subscribeCrosshairMove(hover);
        score.subscribeCrosshairMove(hover);
        show(n - 1);
    }

    // ============================================================ COMPANY (Analyze tab)
    // Reported figures, value checks, filings and industry sources for the analysed ticker. Every view is fetched on its
    // own, after the analysis has rendered, so a slow or failed fetch never touches the rest of the page. Read only:
    // nothing here feeds a score.

    const CO_PREF_KEY = 'fastvalue-company';
    const CO_VIEWS = ['financials', 'checklist', 'filings', 'industry'];
    const CO_PERIODS = ['annual', 'quarterly'];
    const CO = { ticker: null, view: 'financials', period: 'annual', section: null, row: null, kind: 'all', cache: {}, secOpen: false };

    const CO_KIND_LABELS = {
        annual: 'Annual reports', quarterly: 'Quarterly reports', earnings: 'Earnings', presentation: 'Presentations',
        transcript: 'Transcripts', proxy: 'Proxy', current: 'Current reports', foreign_annual: 'Annual reports (foreign)',
        foreign_current: 'Current reports (foreign)', other: 'Other',
    };
    const CO_LATEST = [
        ['annual', 'Annual report', 'annual'], ['quarterly', 'Quarterly report', 'quarterly'],
        ['earnings', 'Earnings release', 'earnings'], ['presentation', 'Presentation', 'presentation'], ['proxy', 'Proxy statement', 'proxy'],
    ];
    const CO_VERDICTS = {
        better: ['live', 'Better than the market average'], worse: ['unavailable', 'Worse than the market average'],
        'in line': ['', 'In line with the market average'],
    };
    const CO_SCOPES = { industry: 'This industry', sector: 'This sector', general: 'General' };
    const CO_MISSING_TIP = 'Not reported, or not available for this period. Shown as missing, never estimated.';
    const CO_DERIVED_TIP = 'Calculated by us from other reported figures. It is not a column the company filed.';

    const coLink = (url, text, cls = '') => {
        const u = safeUrl(url);
        return u ? `<a${cls ? ` class="${cls}"` : ''} href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(text)}</a>` : esc(text);
    };
    const coInfo = (tip) => (tip ? ` <span class="info" tabindex="0" role="note" aria-label="${esc(tip)}" data-tip="${esc(tip)}">i</span>` : '');

    function coPrefs() {
        try {
            const p = JSON.parse(storageGet(CO_PREF_KEY));
            if (p && CO_VIEWS.includes(p.view)) CO.view = p.view;
            if (p && CO_PERIODS.includes(p.period)) CO.period = p.period;
            if (p && typeof p.section === 'string') CO.section = p.section;
        } catch (_) { /* nothing saved yet, or private mode */ }
    }
    const coSavePrefs = () => storageSet(CO_PREF_KEY, JSON.stringify({ view: CO.view, period: CO.period, section: CO.section }));

    // Compact figures: 391.0B, 12.4%, 18.2x, 6.13 a share, 15.2B shares. A minus sign marks negatives; a dash marks missing.
    function coCompact(v) {
        const a = Math.abs(v);
        const [div, suffix] = a >= 1e12 ? [1e12, 'T'] : a >= 1e9 ? [1e9, 'B'] : a >= 1e6 ? [1e6, 'M'] : a >= 1e3 ? [1e3, 'K'] : [1, ''];
        return `${v < 0 ? '−' : ''}${(a / div).toFixed(suffix ? 1 : 0)}${suffix}`;
    }
    function coFmt(v, unit) {
        if (!isNum(v)) return '-';
        const sign = v < 0 ? '−' : '';
        const a = Math.abs(v);
        switch (unit) {
            case 'pct': return `${sign}${(a * 100).toFixed(1)}%`;
            case 'ratio': return `${sign}${a.toFixed(1)}x`;
            case 'per_share': return `${sign}${a.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            case 'years': return `${sign}${a.toFixed(1)} yrs`;
            case 'money': case 'shares': return coCompact(v);
            default: return `${sign}${a.toLocaleString('en-US', { maximumFractionDigits: 2 })}`;
        }
    }
    const coAxisLabel = (c) => String(c.label || c.key || '').replace(/^FY\s*/, '');

    // ---- loading, status, errors

    function coEnsure(key, path) {
        if (CO.cache[key]) return CO.cache[key];
        const ticker = CO.ticker;
        const entry = { loading: true };
        CO.cache[key] = entry;
        api(path).then((data) => { entry.loading = false; entry.data = data; }, (err) => { entry.loading = false; entry.error = err.message || 'The request failed.'; })
            .finally(() => { if (CO.ticker === ticker && CO.cache[key] === entry) renderCompany(); });
        return entry;
    }

    function coSecForm() {
        return `<div class="sec-form" role="group" aria-label="SEC contact email"${CO.secOpen ? '' : ' hidden'}>
                <input type="email" class="input input-compact" placeholder="you@example.com" aria-label="Contact email for the SEC" autocomplete="email" data-co-sec-input>
                <button class="btn btn-sm btn-primary" type="button" data-co-sec-save>Save</button>
                <button class="btn btn-sm btn-ghost" type="button" data-co-sec-cancel>Cancel</button>
                <span class="hint">Stored only on this machine and sent only to sec.gov, as the SEC's fair-access policy asks.</span>
            </div>`;
    }

    // Returns the html for a status that is not ok, or null when the data can be drawn.
    function coStatusHtml(d) {
        const status = d && d.status;
        if (d && d.announcements) return null;   // the announcements view draws its own status
        if (!status || status === 'ok') return null;
        const msg = esc(d.message || '');
        const fix = d.fix ? ` ${esc(d.fix)}` : '';
        if (status === 'no_access') {
            return `<div class="warning"><strong>The SEC needs a contact email</strong>${msg || 'Without one, this view cannot download filings.'}${fix}
                <div class="co-actions"><button class="btn btn-sm" type="button" data-co-sec-open>Set contact email</button></div></div>${coSecForm()}`;
        }
        if (d.reason === 'foreign_filer' || d.reason === 'no_us_gaap_tags' || d.reason === 'no_company_facts') {
            // Information, not a failure: no red, no retry. The filings list still works.
            const why = d.reason === 'foreign_filer' ? 'Files a 20-F or 40-F under IFRS' : d.reason === 'no_company_facts' ? 'No company statements' : 'No US GAAP statements to read';
            return `<div class="warning note"><strong>${why}</strong>${msg || 'Only US GAAP statements are read.'}${fix}
                <div class="co-actions"><button class="btn btn-sm" type="button" data-co-view="filings">Open filings</button></div></div>`;
        }
        if (status === 'not_sec_filer') {
            return `<div class="warning note"><strong>Not an SEC filer</strong>${msg || 'This company does not file with the SEC.'}${fix}${/SEC filings only/i.test(d.message || '') ? '' : ' FastValue reads SEC filings only.'}</div>`;
        }
        return `<div class="warning"><strong>This view could not load</strong>${msg || 'The data source did not answer.'}${fix}
            <div class="co-actions"><button class="btn btn-sm" type="button" data-co-retry>Try again</button></div></div>`;
    }

    function coFootnotes(notes) {
        const list = (notes || []).filter(Boolean);
        return list.length ? `<div class="notes">${list.map((n) => `<div class="warning note">${esc(n)}</div>`).join('')}</div>` : '';
    }

    // ---- financials

    // Bars or line for the reported figures; remembered per browser. Bars by default.
    const CO_MODE_KEY = 'fastvalue-co-chart-mode';
    let coMode = storageGet(CO_MODE_KEY) === 'line' ? 'line' : 'bars';
    // Runs of consecutive reported points: a missing period breaks the line, it is never bridged.
    function coRuns(pts) {
        const runs = [];
        let cur = [];
        pts.forEach((p) => { if (p) cur.push(p); else if (cur.length) { runs.push(cur); cur = []; } });
        if (cur.length) runs.push(cur);
        return runs;
    }

    function coSparkSvg(values) {
        const vals = values.slice(-12);
        const nums = vals.filter(isNum);
        if (nums.length < 2) return '';
        const w = 64, h = 18, gap = 1;
        if (coMode === 'line') {
            const hi = Math.max(0, ...nums), lo = Math.min(0, ...nums), span = hi - lo || 1;
            const pad = 2, step = vals.length > 1 ? (w - 2 * pad) / (vals.length - 1) : 0;
            const pts = vals.map((v, i) => (isNum(v) ? { x: pad + i * step, y: pad + ((hi - v) / span) * (h - 2 * pad), i } : null));
            const lines = coRuns(pts).filter((r) => r.length > 1).map((r) => `<polyline points="${r.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')}"/>`).join('');
            const last = [...pts].reverse().find(Boolean);
            return `<svg class="co-spark co-spark-line" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true" focusable="false">${lines}<circle cx="${last.x.toFixed(1)}" cy="${last.y.toFixed(1)}" r="1.8"${vals[last.i] < 0 ? ' class="neg"' : ''}/></svg>`;
        }
        const bw = Math.max(1.5, (w - gap * (vals.length - 1)) / vals.length);
        const hi = Math.max(0, ...nums), lo = Math.min(0, ...nums), span = hi - lo || 1;
        const y0 = (hi / span) * h;
        const bars = vals.map((v, i) => {
            if (!isNum(v)) return '';
            const bh = Math.max(1, (Math.abs(v) / span) * h);
            const y = v >= 0 ? y0 - bh : y0;
            return `<rect x="${(i * (bw + gap)).toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}"${i === vals.length - 1 ? ' class="last"' : ''}/>`;
        }).join('');
        return `<svg class="co-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true" focusable="false">${bars}</svg>`;
    }

    function coDefaultRow(section) {
        const rows = section.rows || [];
        return (rows.find((r) => r.key === 'revenue') || rows.find((r) => r.highlight) || rows[0] || {}).key || null;
    }

    function coChartHtml(d, row) {
        const cols = d.columns || [];
        const nums = row.values.filter(isNum);
        const modeSwitch = `<div class="segmented seg-sm co-mode" role="group" aria-label="Chart style"><button class="seg${coMode === 'bars' ? ' active' : ''}" type="button" data-co-mode="bars">Bars</button><button class="seg${coMode === 'line' ? ' active' : ''}" type="button" data-co-mode="line">Line</button></div>`;
        const head = `<div class="co-chart-head"><strong>${esc(row.label)}</strong><span class="muted small">${esc(row.tip || '')}</span>${modeSwitch}</div>`;
        if (!nums.length) return `${head}<p class="muted small">No values reported for this row in this period.</p>`;
        const hi = Math.max(0, ...nums), lo = Math.min(0, ...nums), span = hi - lo || 1;
        const zero = (hi / span) * 100;
        const n = cols.length;
        const cls = n > 12 ? 'dense' : n > 7 ? 'semi' : '';
        const step = n > 16 ? 3 : n > 10 ? 2 : 1;
        const lastIdx = row.values.reduce((a, v, i) => (isNum(v) ? i : a), -1);
        const linePlot = coMode === 'line';
        const bars = cols.map((c, i) => {
            const v = row.values[i];
            const when = c.label || c.key;
            if (linePlot && isNum(v)) {
                const y = ((hi - v) / span) * 100;
                const lab = i === lastIdx ? `<span class="co-bv co-lv" style="${v >= 0 ? `bottom:calc(${(100 - y).toFixed(2)}% + 7px)` : `top:calc(${y.toFixed(2)}% + 7px)`}">${esc(coFmt(v, row.unit))}</span>` : '';
                return `<div class="co-bc" data-ctip="${esc(`${when}: ${coFmt(v, row.unit)}`)}"><span class="co-dot${v < 0 ? ' neg' : ''}" style="top:${y.toFixed(2)}%"></span>${lab}</div>`;
            }
            if (!isNum(v)) return `<div class="co-bc" data-ctip="${esc(`${when}: not reported`)}"></div>`;
            const h = Math.max(0.8, (Math.abs(v) / span) * 100);
            const top = v >= 0 ? zero - h : zero;
            const val = v >= 0 ? `bottom:calc(${(100 - top).toFixed(2)}% + 3px)` : `top:calc(${(top + h).toFixed(2)}% + 3px)`;
            return `<div class="co-bc" data-ctip="${esc(`${when}: ${coFmt(v, row.unit)}`)}"><span class="co-b${v < 0 ? ' neg' : ''}" style="top:${top.toFixed(2)}%;height:${h.toFixed(2)}%"></span><span class="co-bv" style="${val}">${esc(coFmt(v, row.unit))}</span></div>`;
        }).join('');
        const axis = cols.map((c, i) => `<span>${(n - 1 - i) % step === 0 ? esc(coAxisLabel(c)) : ''}</span>`).join('');
        const label = `${row.label} by period: ${cols.map((c, i) => `${c.label || c.key} ${coFmt(row.values[i], row.unit)}`).join(', ')}`;
        let linesSvg = '';
        if (linePlot) {
            const pts = row.values.map((v, i) => (isNum(v) ? { x: ((i + 0.5) / n) * 100, y: ((hi - v) / span) * 100 } : null));
            linesSvg = `<svg class="co-line" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true" focusable="false">${coRuns(pts).filter((r) => r.length > 1).map((r) => `<polyline points="${r.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' ')}"/>`).join('')}</svg>`;
        }
        return `${head}<div class="co-chart ${cls}${linePlot ? ' line' : ''}${lo < 0 ? ' has-neg' : ''}" role="img" aria-label="${esc(label)}"><div class="co-plot"><span class="co-zero" style="top:${zero.toFixed(2)}%"></span>${linesSvg}${bars}</div><div class="co-axis">${axis}</div></div>`;
    }

    function coTableHtml(d, section, selected) {
        const cols = d.columns || [];
        const head = cols.map((c) => {
            const tips = [c.form ? `${c.form}${c.filed ? `, filed ${c.filed}` : ''}` : '', c.key === 'LTM' ? 'Last twelve months: the four most recent quarters added together.' : '', c.derived ? CO_DERIVED_TIP : ''].filter(Boolean).join(' ');
            const text = safeUrl(c.url) ? coLink(c.url, c.label || c.key) : esc(c.label || c.key);
            return `<th class="num${c.key === 'LTM' ? ' co-ltm' : ''}${c.derived ? ' co-derived' : ''}"${tips ? ` data-ctip="${esc(tips)}"` : ''}>${text}</th>`;
        }).join('');
        const body = section.rows.map((r) => {
            const cells = cols.map((c, i) => {
                const v = r.values[i];
                const cls = `num${c.key === 'LTM' ? ' co-ltm' : ''}${isNum(v) ? '' : ' na'}`;
                return `<td class="${cls}"${isNum(v) ? '' : ` data-ctip="${esc(CO_MISSING_TIP)}"`}>${esc(coFmt(v, r.unit))}</td>`;
            }).join('');
            return `<tr class="${r.key === selected ? 'is-current' : ''}${r.highlight ? ' hl' : ''}" data-co-row-tr="${esc(r.key)}">
                <td class="co-label"><button type="button" class="co-rowbtn" data-co-row="${esc(r.key)}"${r.tip ? ` data-ctip="${esc(r.tip)}"` : ''}${r.key === selected ? ' aria-pressed="true"' : ' aria-pressed="false"'}>${esc(r.label)}</button></td>
                <td class="co-trend">${coSparkSvg(r.values)}</td>${cells}</tr>`;
        }).join('');
        return `<div class="table-wrap co-wrap"${regionAttrs(section.label)}><table class="table compact co-table"><thead><tr><th class="co-label">${esc(section.label)}</th><th>Trend</th>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    }

    function coFinHtml(d) {
        const cols = d.columns || [];
        const sections = (d.sections || []).filter((s) => (s.rows || []).length);
        if (!cols.length || !sections.length) return '<p class="muted">No reported figures came back for this company.</p>';
        const section = sections.find((s) => s.key === CO.section) || sections[0];
        const row = section.rows.find((r) => r.key === CO.row) || section.rows.find((r) => r.key === coDefaultRow(section));
        CO.row = row.key;
        const picker = `<div class="segmented seg-sm" role="group" aria-label="Statement">${sections.map((s) => `<button class="seg${s.key === section.key ? ' active' : ''}" type="button" data-co-section="${esc(s.key)}">${esc(s.label)}</button>`).join('')}</div>`;
        const meta = [`Amounts in ${esc(d.currency || 'USD')}`, d.as_of ? `up to ${esc(fmtDay(d.as_of))}` : '', d.source ? `source: ${coLink(d.source.url, d.source.label || 'SEC')}` : ''].filter(Boolean).join(' · ');
        const fin = d.is_financial ? '<p class="hint">This is a financial company: rows that only fit industrial companies may be empty.</p>' : '';
        return `${picker}
            <div class="co-meta"><span>${meta}</span></div>
            <div id="co-chart-box">${coChartHtml(d, row)}</div>
            ${coTableHtml(d, section, row.key)}
            <p class="hint">Select a row to chart it. Each year's heading names its filing and opens it.${section.rows.some((r) => r.highlight) ? ' Rows in bold are the ones the value checklist uses.' : ''}</p>${fin}${coFootnotes(d.notes)}`;
    }

    function coSelectRow(key) {
        const entry = CO.cache[`fin:${CO.period}`];
        const d = entry && entry.data;
        if (!d) return;
        const section = (d.sections || []).find((s) => s.key === CO.section) || (d.sections || [])[0];
        const row = section && section.rows.find((r) => r.key === key);
        if (!row) return;
        CO.row = key;
        $('#co-chart-box').innerHTML = coChartHtml(d, row);
        $$('#company-body tr[data-co-row-tr]').forEach((tr) => {
            const on = tr.dataset.coRowTr === key;
            tr.classList.toggle('is-current', on);
            tr.querySelector('.co-rowbtn').setAttribute('aria-pressed', String(on));
        });
    }

    // ---- value checklist

    function coChecklistHtml(d) {
        const items = d.checklist || [];
        if (!items.length) return `<p class="muted">No value checks could be built for this company.</p>${coFootnotes(d.notes)}`;
        const rows = items.map((c) => {
            const bank = d.is_financial && !isNum(c.value) && !c.verdict;
            const [tone, generic] = CO_VERDICTS[c.verdict] || ['', bank ? 'Not applicable to banks' : isNum(c.value) ? 'For information' : 'Not applicable'];
            const text = c.verdict_text || generic;
            const ref = isNum(c.reference) ? coFmt(c.reference, c.unit) : '';
            const refText = [ref, c.reference_label].filter(Boolean).join(', ') || 'No reference';
            const dir = c.better === 'lower' ? 'Lower is better' : c.better === 'higher' ? 'Higher is better' : '';
            return `<div class="co-check-row" role="row">
                <div class="co-check-name" role="cell">${esc(c.label)}${coInfo(c.tip)}</div>
                <div class="co-check-value" role="cell">${c.value_text ? esc(c.value_text) : isNum(c.value) ? esc(coFmt(c.value, c.unit)) : `<span class="muted term quiet" tabindex="0" data-tip="${esc(bank ? (c.tip || 'Does not apply to banks.') : 'Not reported or not available. Shown as missing, never estimated.')}">-</span>`}</div>
                <div class="co-check-ref" role="cell">${esc(refText)}${dir ? `<span class="muted small"> · ${dir}</span>` : ''}</div>
                <div class="co-check-verdict" role="cell"><span class="badge ${tone}">${esc(text)}</span></div>
                ${c.detail ? `<div class="co-check-detail muted small">${esc(c.detail)}</div>` : ''}
            </div>`;
        }).join('');
        const src = d.source ? `<p class="hint">Source: ${coLink(d.source.url, d.source.label || 'SEC')}${d.as_of ? `, up to ${esc(fmtDay(d.as_of))}` : ''}.</p>` : '';
        const b = d.checklist_basis;
        // Which period and which close, so these ratios line up with the Fundamentals agent's (last 10-K year, live price).
        const basis = b ? `<p class="hint">Figures for the ${esc(b.label)}${isNum(b.price) ? `, ratios at the close of ${esc(fmtDay(b.price_date))} (${esc(fmt.money(b.price))})` : ''}.${b.kind === 'ltm' ? ' The Fundamentals agent scores the last 10-K year at the live price, so its EV/FCF, PER and ROIC can differ from these.' : ''}</p>` : '';
        return `${basis}<div class="co-check" role="table" aria-label="Value checklist">
                <div class="co-check-row co-check-head" role="row"><div role="columnheader">Check</div><div role="columnheader">This company</div><div role="columnheader">Reference</div><div role="columnheader">Verdict</div></div>${rows}</div>
            ${coFootnotes(d.notes)}${src}`;
    }

    // ---- filings

    function coBestUrl(f, kind) {
        const ex = (f.exhibits || []).find((e) => e.kind === kind && safeUrl(e.url));
        return safeUrl((ex && ex.url) || f.url) || safeUrl(f.index_url);
    }

    function coFilingsHtml(d) {
        const all = d.filings || [];
        const latest = d.latest || {};
        const latestRows = CO_LATEST.map(([key, label, kind]) => {
            const f = latest[key];
            if (!f) return `<tr><td>${esc(label)}</td><td class="muted" colspan="2">Not found in the recent filings</td></tr>`;
            const url = coBestUrl(f, kind);
            return `<tr><td>${esc(label)}</td><td>${esc(f.form || '')}, filed ${esc(fmtDay(f.filed))}${f.period ? `<span class="muted"> · period ending ${esc(fmtDay(f.period))}</span>` : ''}</td><td>${url ? coLink(url, 'Open') : ''}</td></tr>`;
        }).join('');
        const kinds = [...new Set(all.map((f) => f.kind || 'other'))];
        const press = all.filter((f) => (f.exhibits || []).some((e) => safeUrl(e.url)));   // 8-Ks that carry an exhibit 99: the press releases
        if (CO.kind !== 'all' && CO.kind !== 'press' && !kinds.includes(CO.kind)) CO.kind = 'all';
        if (CO.kind === 'press' && !press.length) CO.kind = 'all';
        const filter = kinds.length > 1
            ? `<div class="segmented seg-sm" role="group" aria-label="Filter filings by kind"><button class="seg${CO.kind === 'all' ? ' active' : ''}" type="button" data-co-kind="all">All (${all.length})</button>${press.length ? `<button class="seg${CO.kind === 'press' ? ' active' : ''}" type="button" data-co-kind="press">Press releases (${press.length})</button>` : ''}${kinds.map((k) => `<button class="seg${CO.kind === k ? ' active' : ''}" type="button" data-co-kind="${esc(k)}">${esc(CO_KIND_LABELS[k] || titleCase(k))}</button>`).join('')}</div>` : '';
        const shown = CO.kind === 'press' ? press : all.filter((f) => CO.kind === 'all' || (f.kind || 'other') === CO.kind);
        const list = shown.map((f) => {
            const ex = (f.exhibits || []).filter((e) => safeUrl(e.url));
            const items = Array.isArray(f.items) && f.items.length ? `<span class="muted small"> · items ${esc(f.items.join(', '))}</span>` : '';
            return `<tr><td>${esc(fmtDay(f.filed))}</td><td><span class="badge">${esc(f.form || '')}</span></td>
                <td class="wrap"><div>${coLink(f.url || f.index_url, f.title || f.form || 'Filing')}${items}</div>
                ${ex.length ? `<ul class="co-ex">${ex.map((e) => `<li>${coLink(e.url, e.label || 'Exhibit')}</li>`).join('')}</ul>` : ''}
                ${f.index_url && f.url ? `<div class="small">${coLink(f.index_url, 'All documents in this filing')}</div>` : ''}</td>
                <td class="muted">${esc(fmtDay(f.period))}</td></tr>`;
        }).join('');
        const links = (d.links || []).filter((l) => safeUrl(l.url)).map((l) => `<li>${coLink(l.url, l.label)}${l.note ? `<span class="muted small"> ${esc(l.note)}</span>` : ''}</li>`).join('');
        return `<div class="co-block"><h4>Latest ${coInfo('The most recent filing of each kind. Annual and quarterly reports hold the audited and reviewed statements; earnings releases and presentations are usually filed as exhibits to an 8-K.')}</h4>
                <div class="table-wrap"><table class="table compact co-latest"><tbody>${latestRows}</tbody></table></div></div>
            <div class="co-block"><h4>All filings</h4>${filter}
                ${shown.length ? `<div class="table-wrap"><table class="table compact"><thead><tr><th>Filed</th><th>Form</th><th>What</th><th>Period ending</th></tr></thead><tbody>${list}</tbody></table></div>` : '<p class="muted small">No filings of this kind in the list.</p>'}</div>
            ${links ? `<div class="co-block"><h4>Elsewhere</h4><ul class="co-links">${links}</ul></div>` : ''}${coFootnotes(d.notes)}`;
    }

    // ---- announcements (official sources outside the SEC: MFN, ASX, RNS; data/official_sources.py)

    const CO_ANN_KINDS = {
        regulatory: 'Regulatory', 'press release': 'Press releases', report: 'Reports', announcement: 'Announcements',
        presentation: 'Presentations', transcript: 'Transcripts', exhibit: 'Exhibits',
    };

    // A suffix other than a share class letter (.ST, .AX, .TO, .L, .V, .DE) means a listing outside the US.
    const coAbroad = (ticker) => /\.(L|V|[A-Za-z]{2,3})$/i.test(ticker || '');

    function coAnnouncementsHtml(d) {
        const items = d.items || [];
        const ok = d.status === 'ok';
        const where = d.market_name ? ` in ${esc(d.market_name)}` : '';
        const banner = ok ? (d.message ? `<div class="warning note">${esc(d.message)}</div>` : '')
            : `<div class="warning note"><strong>${d.status === 'not_readable' ? 'Not readable automatically' : d.status === 'no_adapter' ? 'Market not covered' : d.status === 'unreachable' ? 'Source not reachable' : d.status === 'ambiguous' ? 'More than one issuer matches' : 'Issuer not found'}</strong>${esc(d.message || '')}${d.fix ? ` ${esc(d.fix)}` : ''}${d.status === 'unreachable' ? '<div class="co-actions"><button class="btn btn-sm" type="button" data-co-retry>Try again</button></div>' : ''}</div>`;
        const kinds = [...new Set(items.map((i) => i.kind || 'announcement'))];
        if (CO.kind !== 'all' && !kinds.includes(CO.kind)) CO.kind = 'all';
        const filter = kinds.length > 1
            ? `<div class="segmented seg-sm" role="group" aria-label="Filter announcements by kind"><button class="seg${CO.kind === 'all' ? ' active' : ''}" type="button" data-co-kind="all">All (${items.length})</button>${kinds.map((k) => `<button class="seg${CO.kind === k ? ' active' : ''}" type="button" data-co-kind="${esc(k)}">${esc(CO_ANN_KINDS[k] || titleCase(k))}</button>`).join('')}</div>` : '';
        const shown = items.filter((i) => CO.kind === 'all' || (i.kind || 'announcement') === CO.kind);
        const rows = shown.map((i) => `<tr><td>${esc(fmtDay(i.date))}</td><td><span class="badge${i.kind === 'press release' ? ' accent' : ''}">${esc(titleCase(i.kind || 'announcement').replace('Press Release', 'Press release'))}</span></td>
            <td class="wrap">${coLink(i.url, i.title || 'Announcement')}${i.price_sensitive ? ' <span class="badge warn">Price sensitive</span>' : ''}</td></tr>`).join('');
        const list = ok
            ? `<div class="co-block"><h4>Latest announcements${where}${d.source ? `<span class="muted"> · source: ${esc(d.source)}</span>` : ''}${coInfo('Official announcements and press releases as the exchange, regulator or newswire lists them: date, headline and a link to the document. FastValue does not read figures from them.')}</h4>${filter}
                ${shown.length ? `<div class="table-wrap"><table class="table compact"><thead><tr><th>Date</th><th>Kind</th><th>Headline</th></tr></thead><tbody>${rows}</tbody></table></div>` : '<p class="muted small">No announcements of this kind in the list.</p>'}</div>` : '';
        const links = (d.links || []).filter((l) => safeUrl(l.url)).map((l) => `<li>${coLink(l.url, l.label)}${l.note ? `<span class="muted small"> ${esc(l.note)}</span>` : ''}</li>`).join('');
        const covered = d.status === 'no_adapter' ? `<p class="muted small">Covered markets: ${esc(d.covered || '')}.</p>` : '';
        return `${banner}${list}${covered}${links ? `<div class="co-block"><h4>Open it yourself</h4><ul class="co-links">${links}</ul></div>` : ''}${coFootnotes(d.notes)}`;
    }

    // ---- industry

    function coIndustryHtml(d) {
        const sources = d.sources || [];
        const groups = ['industry', 'sector', 'general'].map((scope) => {
            const items = sources.filter((s) => (s.scope || 'general') === scope);
            if (!items.length) return '';
            const where = scope === 'industry' ? d.industry : scope === 'sector' ? d.sector : '';
            const list = items.map((s) => `<li><div>${coLink(s.url, s.name || s.publisher || 'Source')}${s.conference ? ' <span class="badge accent">Conference</span>' : ''}</div>
                <div class="muted small">${esc([s.publisher, s.cadence].filter(Boolean).join(' · '))}</div>
                ${s.what ? `<div class="small">${esc(s.what)}</div>` : ''}</li>`).join('');
            return `<div class="co-block"><h4>${CO_SCOPES[scope]}${where ? `<span class="muted"> · ${esc(where)}</span>` : ''}</h4><ul class="co-list">${list}</ul></div>`;
        }).join('');
        const docs = (d.documents || []).filter((x) => safeUrl(x.url));
        const docRows = [...docs.filter((x) => !x.is_peer), ...docs.filter((x) => x.is_peer)].map((x) =>
            `<tr><td>${esc(x.name || x.ticker)}${x.is_peer ? ' <span class="muted small">peer</span>' : ''}</td><td>${esc(fmtDay(x.filed))}</td><td>${coLink(x.url, x.label || titleCase(x.kind || 'document'))}</td></tr>`).join('');
        const docsHtml = docRows
            ? `<div class="co-block"><h4>Latest documents${coInfo('The newest earnings release and investor presentation of this company and a few peers. Management often discusses market demand and its outlook there.')}</h4><div class="table-wrap"><table class="table compact"><thead><tr><th>Company</th><th>Filed</th><th>Document</th></tr></thead><tbody>${docRows}</tbody></table></div></div>`
            : '<div class="co-block"><h4>Latest documents</h4><p class="muted small">No earnings releases or presentations were found for this company or its peers.</p></div>';
        const head = d.industry || d.sector ? `<div class="co-meta"><span>${esc([d.sector, d.industry].filter(Boolean).join(' · '))}</span></div>` : '';
        return `${head}${groups || '<p class="muted">No industry sources are listed for this company yet.</p>'}${docsHtml}${coFootnotes(d.notes)}`;
    }

    // ---- shell

    const CO_SLOTS = {
        financials: () => [null, `fin:${CO.period}`, `/api/financials/${encodeURIComponent(CO.ticker)}?period=${CO.period}`, coFinHtml, 'Loading reported figures from the SEC...'],
        checklist: () => [null, 'fin:annual', `/api/financials/${encodeURIComponent(CO.ticker)}?period=annual`, coChecklistHtml, 'Loading reported figures from the SEC...'],
        filings: () => (coAbroad(CO.ticker)
            ? [null, 'ann', `/api/announcements/${encodeURIComponent(CO.ticker)}?limit=20`, coAnnouncementsHtml, 'Loading announcements from the official source...']
            : [null, 'filings', `/api/filings/${encodeURIComponent(CO.ticker)}?limit=40`, coFilingsHtml, 'Loading the filing list from the SEC...']),
        industry: () => [null, 'industry', `/api/industry/${encodeURIComponent(CO.ticker)}`, coIndustryHtml, 'Loading industry sources...'],
    };

    function renderCompany() {
        const card = $('#company-card');
        if (!CO.ticker || !card) return;
        card.hidden = false;
        $$('#company-views .seg').forEach((b) => { const on = b.dataset.coView === CO.view; b.classList.toggle('active', on); b.setAttribute('aria-selected', String(on)); });
        $$('#company-period .seg').forEach((b) => b.classList.toggle('active', b.dataset.coPeriod === CO.period));
        $('#company-period').hidden = CO.view !== 'financials';
        const [, key, path, draw, wait] = CO_SLOTS[CO.view]();
        const entry = coEnsure(key, path);
        const body = $('#company-body');
        const scroll = $('.co-wrap', body);
        const keepScroll = scroll ? scroll.scrollLeft : 0;
        let caption = '';
        if (entry.loading) body.innerHTML = `<p class="co-loading" role="status">${esc(wait)}</p>`;
        else if (entry.error) body.innerHTML = coStatusHtml({ status: 'error', message: entry.error });
        else {
            const d = entry.data;
            body.innerHTML = coStatusHtml(d) || draw(d);
            // A name the filing gives in capitals ("COCA COLA CO") reads as "Coca Cola Co", like the other names.
            if (d.status === 'ok' && d.name) caption = /^[A-Z0-9 .,&'-]+$/.test(d.name) ? d.name.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase()) : d.name;
            const again = $('.co-wrap', body);
            // Statements open on the latest year (the right-hand columns); a reload keeps where the reader scrolled to.
            if (again) again.scrollLeft = keepScroll || again.scrollWidth;
        }
        $('#company-caption').textContent = caption;
        if (CO.secOpen) { const f = $('.sec-form', body); if (f) { f.hidden = false; $('input', f).focus(); } }
    }

    // Called by renderAnalysis. The same ticker again (an auto refresh) keeps what was loaded.
    function loadCompany(ticker) {
        if (!ticker) return;
        if (ticker !== CO.ticker) {
            CO.ticker = ticker; CO.cache = {}; CO.row = null; CO.kind = 'all'; CO.secOpen = false;
        }
        renderCompany();
    }

    async function coSaveSec(form) {
        const email = $('input', form).value.trim();
        try {
            const res = await api('/api/settings/sec', { method: 'PUT', body: { email } });
            toast(res.configured ? 'Contact email saved. Loading the filings.' : 'Contact email removed.', 'success');
            CO.secOpen = false; CO.cache = {};
            renderCompany();
            refreshBkOptions();
        } catch (err) {
            toast(`Not saved: ${err.message}`, 'error');
        }
    }

    function setupCompany() {
        coPrefs();
        const card = $('#company-card');
        if (!card) return;
        card.addEventListener('click', (e) => {
            const t = e.target;
            let b;
            if ((b = t.closest('[data-co-view]'))) { CO.view = b.dataset.coView; coSavePrefs(); renderCompany(); }
            else if ((b = t.closest('[data-co-period]'))) { CO.period = b.dataset.coPeriod; CO.row = null; coSavePrefs(); renderCompany(); }
            else if ((b = t.closest('[data-co-section]'))) { CO.section = b.dataset.coSection; CO.row = null; coSavePrefs(); renderCompany(); }
            else if ((b = t.closest('[data-co-mode]'))) { coMode = b.dataset.coMode === 'line' ? 'line' : 'bars'; storageSet(CO_MODE_KEY, coMode); renderCompany(); }
            else if ((b = t.closest('[data-co-row]'))) coSelectRow(b.dataset.coRow);
            else if ((b = t.closest('[data-co-kind]'))) { CO.kind = b.dataset.coKind; renderCompany(); }
            else if (t.closest('[data-co-retry]')) { const [, key] = CO_SLOTS[CO.view](); delete CO.cache[key]; renderCompany(); }
            else if (t.closest('[data-co-sec-open]')) { CO.secOpen = true; renderCompany(); }
            else if (t.closest('[data-co-sec-cancel]')) { CO.secOpen = false; renderCompany(); }
            else if (t.closest('[data-co-sec-save]')) coSaveSec(t.closest('.sec-form'));
        });
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.target.matches('[data-co-sec-input]')) { e.preventDefault(); coSaveSec(e.target.closest('.sec-form')); }
        });
    }

    // On narrow screens a tip opens where it fits: shifted so it stays inside the window.
    function setupTipPlacement() {
        const place = (e) => {
            const el = e.target instanceof Element ? e.target.closest('.term[data-tip], .info[data-tip]') : null;
            if (!el) return;
            const vw = document.documentElement.clientWidth;
            if (vw > 980) { el.style.removeProperty('--tip-left'); el.style.removeProperty('--tip-right'); return; }
            const width = Math.min(340, vw * 0.8, el.dataset.tip.length * 6.6 + 24);
            const r = el.getClientRects()[0] || el.getBoundingClientRect();   // a wrapped term anchors its tip at its first line
            const left = Math.max(8, Math.min(r.left - (el.classList.contains('info') ? 8 : 0), vw - 8 - width));
            el.style.setProperty('--tip-left', `${(left - r.left).toFixed(0)}px`);
            el.style.setProperty('--tip-right', 'auto');
        };
        document.addEventListener('focusin', place);
        document.addEventListener('pointerover', place);
    }

    document.addEventListener('DOMContentLoaded', init);
})();
