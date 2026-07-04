// --- State ---
let ranges = [];
let source = "manual";
let validated = false;
let clickState = null; // null or { start: lineNum }
let isDragging = false;
let currentView = "markdown";
let markdownLines = [];   // text content only (no line numbers)
let markdownLineNums = []; // parallel array of actual line numbers

// --- Dark Mode ---
function toggleTheme() {
    const isDark = document.documentElement.classList.toggle("dark");
    localStorage.setItem("theme", isDark ? "dark" : "light");
    const btn = document.querySelector(".theme-toggle");
    if (btn) btn.textContent = isDark ? "Light Mode" : "Dark Mode";
}

// --- Column Resize ---
document.addEventListener("mousedown", function (e) {
    if (!e.target.classList.contains("resize-handle")) return;
    e.preventDefault();
    const th = e.target.parentElement;
    const startX = e.clientX;
    const startW = th.offsetWidth;

    function onMove(e2) {
        th.style.width = Math.max(40, startW + e2.clientX - startX) + "px";
    }
    function onUp() {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
    }
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
});

// --- Modal Init ---
function initModal(initialRanges, src, val, hasHtml) {
    ranges = initialRanges || [];
    source = src || "manual";
    validated = val || false;
    clickState = null;
    isDragging = false;
    currentView = hasHtml ? "structure" : "markdown";

    // Process line-viewer <pre> into clickable struct-lines (fallback when no HTML)
    var lvPre = document.getElementById("line-viewer-pre");
    if (lvPre) {
        processPreIntoStructLines(lvPre);
        extractMarkdownLines(lvPre.parentElement);
    }

    renderChips();

    if (hasHtml) {
        var sv = document.getElementById("structure-viewer");
        if (sv && !sv.dataset.loaded) {
            sv.innerHTML = "<p>Loading...</p>";
            fetch(sv.dataset.src)
                .then(function (r) { return r.text(); })
                .then(function (html) {
                    sv.innerHTML = html;
                    sv.dataset.loaded = "1";
                    processStructureLines();
                    extractMarkdownLines(sv);
                    renderStructureHighlights();
                    scrollToFirstRange();
                });
        }
    } else {
        renderStructureHighlights();
        scrollToFirstRange();
    }
}

function lazyLoadStructureViewer(viewer) {
    if (viewer.dataset.loaded) return;
    viewer.innerHTML = "<p>Loading...</p>";
    fetch(viewer.dataset.src)
        .then(function (r) { return r.text(); })
        .then(function (html) {
            viewer.innerHTML = html;
            viewer.dataset.loaded = "1";
            processStructureLines(viewer);
            renderStructureHighlights();
        });
}

// --- View Switching ---
function switchView(view) {
    currentView = view;
    clickState = null;
    clearClickHighlight();
    const lineViewer = document.getElementById("line-viewer");
    const htmlViewer = document.getElementById("html-viewer");
    const structureViewer = document.getElementById("structure-viewer");
    const structureLightViewer = document.getElementById("structure-light-viewer");
    const btnSetRange = document.getElementById("btn-set-range");

    lineViewer.style.display = "none";
    if (htmlViewer) htmlViewer.classList.remove("active");
    if (structureViewer) structureViewer.classList.remove("active");
    if (structureLightViewer) structureLightViewer.classList.remove("active");
    if (btnSetRange) btnSetRange.style.display = "none";

    if (view === "html" && htmlViewer) {
        htmlViewer.classList.add("active");
        if (btnSetRange) btnSetRange.style.display = "";
    } else if (view === "structure" && structureViewer) {
        structureViewer.classList.add("active");
        lazyLoadStructureViewer(structureViewer);
    } else if (view === "structure-light" && structureLightViewer) {
        structureLightViewer.classList.add("active");
        lazyLoadStructureViewer(structureLightViewer);
    } else {
        lineViewer.style.display = "";
    }

    var viewToLabel = { "structure": "structure", "structure-light": "light", "html": "html" };
    var activeLabel = viewToLabel[view] || view;
    document.querySelectorAll(".view-toggle-btn").forEach(function (btn) {
        btn.classList.toggle("active", btn.textContent.toLowerCase() === activeLabel);
    });
}

// --- Selection to Range Mapping ---
let _selectionTimeout = null;

function setRangeFromSelection() {
    const iframe = document.getElementById("html-iframe");
    if (!iframe || !iframe.contentWindow) return;

    iframe.contentWindow.postMessage("getSelection", "*");
    // Timeout fallback if iframe hasn't loaded or doesn't respond
    clearTimeout(_selectionTimeout);
    _selectionTimeout = setTimeout(function () {
        showSelectionWarning("Could not read selection");
    }, 500);
}

window.addEventListener("message", function (e) {
    if (e.data && e.data.type === "selection") {
        clearTimeout(_selectionTimeout);
        handleSelectionText(e.data.text);
    }
});

function handleSelectionText(selectedText) {
    if (!selectedText || !selectedText.trim()) {
        showSelectionWarning("No text selected");
        return;
    }

    selectedText = selectedText.trim();
    const selWords = normaliseText(selectedText);
    if (selWords.length === 0) {
        showSelectionWarning("Could not map selection to markdown lines");
        return;
    }

    const lineWords = [];
    const lineOffsets = [];
    let concat = "";
    for (let i = 0; i < markdownLines.length; i++) {
        const words = normaliseText(stripMarkdownLinks(markdownLines[i]));
        lineWords.push(words);
        lineOffsets.push(concat.length);
        if (words.length > 0) {
            concat += (concat ? " " : "") + words.join(" ");
        }
    }

    const selStart = selWords.slice(0, Math.min(5, selWords.length)).join(" ");
    const selEnd = selWords.slice(-Math.min(5, selWords.length)).join(" ");

    const startPos = concat.indexOf(selStart);
    const endPos = selEnd === selStart ? startPos : concat.lastIndexOf(selEnd);

    let startLine = -1;
    let endLine = -1;

    if (startPos !== -1) {
        for (let i = lineOffsets.length - 1; i >= 0; i--) {
            if (lineOffsets[i] <= startPos) {
                startLine = markdownLineNums[i];
                break;
            }
        }
    }

    if (endPos !== -1) {
        const endCharPos = endPos + selEnd.length;
        for (let i = lineOffsets.length - 1; i >= 0; i--) {
            if (lineOffsets[i] <= endCharPos) {
                endLine = markdownLineNums[i];
                break;
            }
        }
    }

    if (startLine === -1 || endLine === -1) {
        showSelectionWarning("Could not map selection to markdown lines");
        return;
    }

    if (startLine > endLine) {
        const tmp = startLine;
        startLine = endLine;
        endLine = tmp;
    }

    if (!overlapsExisting(startLine, endLine)) {
        ranges.push({ start: startLine, end: endLine });
        ranges.sort(function (a, b) { return a.start - b.start; });
        source = "manual";
        renderLines();
        renderChips();
    }
}

function stripMarkdownLinks(text) {
    return text.replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1");
}

function normaliseText(text) {
    return text.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(Boolean);
}

function showSelectionWarning(msg) {
    const existing = document.querySelector(".selection-warning");
    if (existing) existing.remove();

    const el = document.createElement("div");
    el.className = "selection-warning";
    el.textContent = msg;
    const modal = document.querySelector(".modal");
    if (modal) {
        modal.style.position = "relative";
        modal.appendChild(el);
        setTimeout(function () { el.remove(); }, 3000);
    }
}

// --- Line Click ---
function handleLineClick(lineNum) {
    // Check if clicking a range boundary — delete that range
    var boundaryIdx = findBoundaryRange(lineNum);
    if (boundaryIdx !== -1) {
        ranges.splice(boundaryIdx, 1);
        source = "manual";
        clickState = null;
        renderLines();
        renderChips();
        return;
    }

    if (clickState === null) {
        clickState = { start: lineNum };
        highlightClickStart(lineNum);
    } else {
        var s = Math.min(clickState.start, lineNum);
        var e2 = Math.max(clickState.start, lineNum);
        if (!overlapsExisting(s, e2)) {
            ranges.push({ start: s, end: e2 });
            ranges.sort(function (a, b) { return a.start - b.start; });
            source = "manual";
        }
        clickState = null;
        renderLines();
        renderChips();
    }
}

// Line click (works in both structure-viewer and line-viewer)
document.addEventListener("click", function (e) {
    if (isDragging) return;
    var structLine = e.target.closest(".struct-line[data-md-line]");
    if (!structLine) return;
    handleLineClick(parseInt(structLine.dataset.mdLine));
});

function findBoundaryRange(lineNum) {
    for (let i = 0; i < ranges.length; i++) {
        if (ranges[i].start === lineNum || ranges[i].end === lineNum) {
            return i;
        }
    }
    return -1;
}

function overlapsExisting(s, e) {
    for (const r of ranges) {
        if (s <= r.end && e >= r.start) return true;
    }
    return false;
}

function highlightClickStart(lineNum) {
    clearClickHighlight();
    document.querySelectorAll('.struct-line[data-md-line="' + lineNum + '"]').forEach(function (el) {
        el.classList.add("click-start");
    });
}

function clearClickHighlight() {
    document.querySelectorAll(".click-start").forEach(function (el) { el.classList.remove("click-start"); });
}

// --- Drag Range Boundaries & Move Whole Range (works in both viewers) ---
document.addEventListener("mousedown", function (e) {
    var structLine = e.target.closest(".struct-line[data-md-line]");
    if (!structLine) return;

    var isBoundary = structLine.classList.contains("range-start") || structLine.classList.contains("range-end");
    var isInRange = structLine.classList.contains("in-range");
    if (!isBoundary && !isInRange) return;

    e.preventDefault();
    var lineNum = parseInt(structLine.dataset.mdLine);

    // Find which range this line belongs to
    var rangeIdx = -1;
    if (isBoundary) {
        rangeIdx = findBoundaryRange(lineNum);
    } else {
        for (var ri = 0; ri < ranges.length; ri++) {
            if (lineNum >= ranges[ri].start && lineNum <= ranges[ri].end) {
                rangeIdx = ri;
                break;
            }
        }
    }
    if (rangeIdx === -1) return;

    var isStart = isBoundary && ranges[rangeIdx].start === lineNum;
    var viewer = structLine.closest("#structure-viewer, #line-viewer");
    if (!viewer) return;
    var allStructLines = Array.from(viewer.querySelectorAll(".struct-line[data-md-line]"));
    var hasMoved = false;
    var lastLineNum = lineNum;

    function getLineFromY(y) {
        for (var i = 0; i < allStructLines.length; i++) {
            var rect = allStructLines[i].getBoundingClientRect();
            if (y >= rect.top && y <= rect.bottom) {
                return parseInt(allStructLines[i].dataset.mdLine);
            }
        }
        return null;
    }

    function onMove(e2) {
        hasMoved = true;
        isDragging = true;
        var newLn = getLineFromY(e2.clientY);
        if (newLn === null) return;

        var r = ranges[rangeIdx];

        if (isBoundary) {
            // Resize boundary (existing behaviour)
            if (isStart) {
                if (newLn > r.end) return;
                if (rangeIdx > 0 && newLn <= ranges[rangeIdx - 1].end) return;
                r.start = newLn;
            } else {
                if (newLn < r.start) return;
                if (rangeIdx < ranges.length - 1 && newLn >= ranges[rangeIdx + 1].start) return;
                r.end = newLn;
            }
        } else {
            // Move whole range
            var delta = newLn - lastLineNum;
            if (delta === 0) return;
            var newStart = r.start + delta;
            var newEnd = r.end + delta;
            if (newStart < 1) return;
            // Check overlap with adjacent ranges
            if (rangeIdx > 0 && newStart <= ranges[rangeIdx - 1].end) return;
            if (rangeIdx < ranges.length - 1 && newEnd >= ranges[rangeIdx + 1].start) return;
            // Check we don't go past the last line
            var maxLine = parseInt(allStructLines[allStructLines.length - 1].dataset.mdLine);
            if (newEnd > maxLine) return;
            r.start = newStart;
            r.end = newEnd;
            lastLineNum = newLn;
        }
        source = "manual";
        renderLines();
        renderChips();
    }

    function onUp() {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        if (hasMoved) {
            setTimeout(function () { isDragging = false; }, 50);
        }
    }

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
});

// --- Render ---
function renderLines() {
    clearClickHighlight();
    renderStructureHighlights();
}

// --- Process <pre> into clickable struct-lines ---
function processPreIntoStructLines(pre) {
    var text = pre.textContent;
    var lines = text.split("\n");
    var html = "";
    for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        var match = line.match(/^\s*(\d+)\s*\|/);
        var mdLine = match ? match[1] : "";
        var escaped = line.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        html += '<div class="struct-line"' + (mdLine ? ' data-md-line="' + mdLine + '"' : '') + '>' + escaped + '</div>';
    }
    pre.innerHTML = html;
}

function extractMarkdownLines(container) {
    markdownLines = [];
    markdownLineNums = [];
    var structLines = container.querySelectorAll(".struct-line[data-md-line]");
    structLines.forEach(function (el) {
        var text = el.textContent.replace(/^\s*\d+\s*\|\s*/, "");
        markdownLines.push(text);
        markdownLineNums.push(parseInt(el.dataset.mdLine));
    });
}

// --- Structure View ---
function processStructureLines(viewer) {
    if (!viewer) viewer = document.getElementById("structure-viewer");
    if (!viewer) return;
    var pre = viewer.querySelector("pre");
    if (!pre) return;

    var text = pre.textContent;
    var lines = text.split("\n");
    var html = "";
    for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        // Extract line number from "  N | content" format (blank for structural labels)
        var match = line.match(/^\s*(\d+)\s*\|/);
        var mdLine = match ? match[1] : "";
        var escaped = line.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        html += '<div class="struct-line"' + (mdLine ? ' data-md-line="' + mdLine + '"' : '') + '>' + escaped + '</div>';
    }
    pre.innerHTML = html;
}

function scrollToFirstRange() {
    requestAnimationFrame(function () {
        var viewer = document.getElementById("structure-viewer");
        if (!viewer || viewer.style.display === "none") {
            viewer = document.getElementById("line-viewer");
        }
        if (!viewer) return;
        var el = viewer.querySelector(".struct-line.in-range");
        if (!el) return;
        var viewerRect = viewer.getBoundingClientRect();
        var elRect = el.getBoundingClientRect();
        viewer.scrollTop += elRect.top - viewerRect.top - viewerRect.height / 2;
    });
}

function renderStructureHighlights() {
    // Highlight in both structure-viewer and line-viewer (both use struct-line elements)
    var viewers = [
        document.getElementById("structure-viewer"),
        document.getElementById("structure-light-viewer"),
        document.getElementById("line-viewer"),
    ];

    viewers.forEach(function (viewer) {
        if (!viewer) return;
        var allLines = viewer.querySelectorAll(".struct-line[data-md-line]");

        allLines.forEach(function (el) {
            var mdLine = parseInt(el.dataset.mdLine);
            el.classList.remove("in-range", "manual-range", "llm-range", "range-start", "range-end");

            for (var i = 0; i < ranges.length; i++) {
                if (mdLine >= ranges[i].start && mdLine <= ranges[i].end) {
                    el.classList.add("in-range");
                    if (source === "llm" && !validated) {
                        el.classList.add("llm-range");
                    } else {
                        el.classList.add("manual-range");
                    }
                    if (mdLine === ranges[i].start) el.classList.add("range-start");
                    if (mdLine === ranges[i].end) el.classList.add("range-end");
                    break;
                }
            }
        });
    });
}

function renderChips() {
    const container = document.getElementById("range-chips");
    if (!container) return;

    if (ranges.length === 0) {
        container.innerHTML = '<span class="none-label">none</span>';
        return;
    }

    container.innerHTML = ranges
        .map((r, i) =>
            `<span class="chip">[${r.start}:${r.end}] <button class="chip-remove" onclick="removeRange(${i})">&times;</button></span>`
        )
        .join(" ");
}

function removeRange(idx) {
    ranges.splice(idx, 1);
    source = "manual";
    renderLines();
    renderChips();
}

function clearRanges() {
    ranges = [];
    source = "manual";
    clickState = null;
    renderLines();
    renderChips();
}

// --- Actions ---
async function savePage(pageId, andNext) {
    const body = JSON.stringify({ ranges: ranges, source: source });
    const resp = await fetch(`/annotate/${pageId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
    });
    if (!resp.ok) {
        alert("Save failed: " + resp.status);
        return;
    }

    if (andNext) {
        htmx.ajax("GET", `/annotate/next/${pageId}`, { target: "#modal-container", swap: "innerHTML" });
    } else {
        closeModal();
    }
    refreshTable();
}

async function toggleCookie(pageId) {
    const resp = await fetch(`/annotate/${pageId}/cookie`, { method: "POST" });
    if (!resp.ok) {
        alert("Toggle cookie failed: " + resp.status);
        return;
    }
    const isOn = (await resp.text()) === "1";
    var btn = document.getElementById("btn-cookie");
    if (btn) btn.classList.toggle("active", isOn);
}

async function skipPage(pageId) {
    const resp = await fetch(`/annotate/${pageId}/skip`, { method: "POST" });
    if (!resp.ok) {
        alert("Skip failed: " + resp.status);
        return;
    }
    closeModal();
    refreshTable();
}

async function acceptPage(pageId) {
    const resp = await fetch(`/annotate/${pageId}/accept`, { method: "POST" });
    if (!resp.ok) {
        alert("Accept failed: " + resp.status);
        return;
    }
    closeModal();
    refreshTable();
}

function closeModal() {
    const container = document.getElementById("modal-container");
    if (container) container.innerHTML = "";
}

function refreshTable() {
    const tableArea = document.getElementById("table-area");
    if (!tableArea) return;

    // Build URL from current state
    const activeTab = document.querySelector(".tab.active");
    if (activeTab) {
        const url = activeTab.getAttribute("hx-get");
        if (url) {
            // Append current page number from pagination
            const pageSpan = document.querySelector(".page-controls span");
            let currentPage = 1;
            if (pageSpan) {
                const match = pageSpan.textContent.match(/Page (\d+)/);
                if (match) currentPage = parseInt(match[1]);
            }
            const separator = url.includes("?") ? "&" : "?";
            htmx.ajax("GET", url + separator + "page=" + currentPage, { target: "#table-area", swap: "innerHTML" });
            return;
        }
    }
    htmx.ajax("GET", "/pages", { target: "#table-area", swap: "innerHTML" });
}

// Close modal on Escape
document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
        closeModal();
    }
});

// Close modal on overlay click (not modal content)
document.addEventListener("click", function (e) {
    if (e.target.classList.contains("modal-overlay")) {
        closeModal();
    }
});
