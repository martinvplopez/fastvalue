/* The saved theme before the first paint (app.js applies it again and wires the toggle on DOMContentLoaded,
   which came one or two frames too late: the page flashed the system theme). Same keys as storageGet in app.js. */
try {
    const theme = localStorage.getItem('fastvalue-theme') ?? localStorage.getItem('quant-theme');
    if (theme) document.documentElement.dataset.theme = theme;
} catch (_) { /* private mode: the system theme, as before */ }
