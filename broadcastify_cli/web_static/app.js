"use strict";

const token = document.querySelector('meta[name="app-token"]').content;
const byId = (id) => document.getElementById(id);
const html = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const DEFAULT_SETTINGS = {
  whisperModel: "turbo",
  asrEngine: "auto",
  device: "auto",
  diarizationDevice: "auto",
  batchSize: 8,
  asrModelPath: "",
  analysisProvider: "local",
  analysisModel: "ggml-org/gemma-4-12B-it-GGUF:Q4_K_M",
  analysisEndpoint: "",
  apiKeyEnvironment: "OPENAI_API_KEY",
  codexPath: "",
  allowExternal: false,
};

const state = {
  bootstrap: { days: [], profiles: [], summary: {}, runtime: {} },
  selectedDay: null,
  selectedDayDetail: null,
  incidentExpanded: false,
  transcript: { segments: [], offset: 0, total: 0, hasMore: false, query: "" },
  selectedFeed: null,
  areaDiscovered: [],
  areaBriefResult: null,
  areaStoriesExpanded: false,
  settings: loadSettings(),
  activeJob: null,
  jobTimer: null,
  jobCallback: null,
  analysisQueue: [],
};

function loadSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem("radioArchiveSettings") || "{}") };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function saveSettings() {
  readSettingsForm();
  localStorage.setItem("radioArchiveSettings", JSON.stringify(state.settings));
  toast("Non-secret settings saved in this browser profile.");
}

function readSettingsForm() {
  state.settings = {
    whisperModel: byId("settingWhisperModel").value,
    asrEngine: byId("settingAsrEngine").value,
    device: byId("settingDevice").value,
    diarizationDevice: byId("settingDiarizationDevice").value,
    batchSize: Math.max(1, Number(byId("settingBatchSize").value) || 8),
    asrModelPath: byId("settingAsrModelPath").value.trim(),
    analysisProvider: byId("settingAnalysisProvider").value,
    analysisModel: byId("settingAnalysisModel").value.trim(),
    analysisEndpoint: byId("settingAnalysisEndpoint").value.trim(),
    apiKeyEnvironment: byId("settingApiKeyEnvironment").value.trim() || "OPENAI_API_KEY",
    codexPath: byId("settingCodexPath").value.trim(),
    allowExternal: byId("settingAllowExternal").checked,
  };
  updateProviderNotice();
}

function applySettingsForm() {
  byId("settingWhisperModel").value = state.settings.whisperModel;
  byId("settingAsrEngine").value = state.settings.asrEngine;
  byId("settingDevice").value = state.settings.device;
  byId("settingDiarizationDevice").value = state.settings.diarizationDevice;
  byId("settingBatchSize").value = state.settings.batchSize;
  byId("settingAsrModelPath").value = state.settings.asrModelPath;
  byId("settingAnalysisProvider").value = state.settings.analysisProvider;
  byId("settingAnalysisModel").value = state.settings.analysisModel;
  byId("settingAnalysisEndpoint").value = state.settings.analysisEndpoint;
  byId("settingApiKeyEnvironment").value = state.settings.apiKeyEnvironment;
  byId("settingCodexPath").value = state.settings.codexPath;
  byId("settingAllowExternal").checked = Boolean(state.settings.allowExternal);
  updateProviderNotice();
}

function providerPayload() {
  readSettingsForm();
  const key = byId("settingAnalysisApiKey").value;
  return {
    analysis_provider: state.settings.analysisProvider,
    analysis_model: state.settings.analysisModel,
    analysis_endpoint: state.settings.analysisEndpoint,
    analysis_api_key: key || undefined,
    analysis_api_key_env: state.settings.apiKeyEnvironment,
    codex_cli_path: state.settings.codexPath,
    allow_external_analysis: Boolean(state.settings.allowExternal),
  };
}

function processingPayload() {
  readSettingsForm();
  const huggingFaceToken = byId("settingHuggingFaceToken").value;
  return {
    model: state.settings.whisperModel,
    asr_engine: state.settings.asrEngine,
    device: state.settings.device,
    device_index: 0,
    compute_type: "auto",
    asr_model_path: state.settings.asrModelPath || undefined,
    diarization_device: state.settings.diarizationDevice,
    batch_size: state.settings.batchSize,
    huggingface_token: huggingFaceToken || undefined,
  };
}

async function api(path, options = {}) {
  const request = { credentials: "same-origin", ...options };
  if (request.body) {
    request.headers = {
      "Content-Type": "application/json",
      "X-Radio-Archive-Token": token,
      ...(request.headers || {}),
    };
  }
  const response = await fetch(path, request);
  const payload = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function toast(message, error = false) {
  const item = document.createElement("div");
  item.className = `toast${error ? " error" : ""}`;
  item.textContent = message;
  byId("toastRegion").append(item);
  setTimeout(() => item.remove(), 5000);
}

function setBusyLabel(message) {
  byId("topbarStatus").innerHTML = `<span class="status-dot"></span>${html(message)}`;
}

function setView(name) {
  document.querySelectorAll(".view").forEach((value) => value.classList.toggle("active", value.id === `view-${name}`));
  document.querySelectorAll(".nav-item[data-view]").forEach((value) => {
    const active = value.dataset.view === name;
    value.classList.toggle("active", active);
    if (active) value.setAttribute("aria-current", "page"); else value.removeAttribute("aria-current");
  });
  const labels = {
    library: ["Local library", "Retained evidence stays on this computer"],
    archive: ["New archive", "Quota-safe website archive acquisition"],
    review: ["Review & ask", "Evidence-grounded summaries and questions"],
    area: ["Area watch", "Regional profiles and local story leads"],
    settings: ["Settings", "Local processing and analysis defaults"],
  };
  byId("topbarTitle").textContent = labels[name][0];
  byId("topbarSubtitle").textContent = labels[name][1];
  document.querySelector(".sidebar").classList.remove("open");
  history.replaceState(null, "", `#${name}`);
}

function bytes(value) {
  let number = Number(value) || 0;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let index = 0;
  while (number >= 1024 && index < units.length - 1) { number /= 1024; index += 1; }
  return `${number.toFixed(index < 2 ? 0 : 1)} ${units[index]}`;
}

function clock(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = Math.floor(value % 60);
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}` : `${minutes}:${String(secs).padStart(2, "0")}`;
}

function words(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

async function refreshBootstrap({ preserveSelection = true } = {}) {
  setBusyLabel("Refreshing local records…");
  try {
    state.bootstrap = await api("/api/bootstrap");
    renderMetrics();
    renderLibrary();
    renderProfiles();
    renderRuntime();
    if (preserveSelection && state.selectedDay) {
      const replacement = state.bootstrap.days.find((value) => value.feed_id === state.selectedDay.feed_id && value.archive_date === state.selectedDay.archive_date);
      if (replacement) await selectDay(replacement, false);
    }
    setBusyLabel("Ready");
  } catch (error) {
    setBusyLabel("Service error");
    toast(error.message, true);
  }
}

function renderMetrics() {
  const summary = state.bootstrap.summary || {};
  const metrics = [
    ["Feeds", summary.feed_count || 0],
    ["Local days", summary.day_count || 0],
    ["Need a next step", summary.attention_count || 0],
    ["Ready to review", summary.complete_count || 0],
  ];
  byId("libraryMetrics").innerHTML = metrics.map(([label, value]) => `<div class="metric-card"><span>${html(label)}</span><strong>${html(value)}</strong></div>`).join("");
}

function filteredDays() {
  const search = byId("librarySearch").value.trim().toLowerCase();
  const filter = byId("libraryFilter").value;
  return state.bootstrap.days.filter((day) => {
    const matchesSearch = !search || `${day.feed_name} ${day.feed_id} ${day.archive_date} ${day.status}`.toLowerCase().includes(search);
    const matchesFilter = filter === "all" || (filter === "complete" ? day.is_complete : !day.is_complete);
    return matchesSearch && matchesFilter;
  });
}

function renderLibrary() {
  const days = filteredDays();
  if (!days.length) {
    byId("dayList").innerHTML = '<div class="empty-compact">No retained days match this filter.</div>';
    return;
  }
  byId("dayList").innerHTML = days.map((day) => {
    const selected = state.selectedDay && state.selectedDay.feed_id === day.feed_id && state.selectedDay.archive_date === day.archive_date;
    return `<button class="day-row${selected ? " selected" : ""}" role="option" aria-selected="${selected}" data-feed-id="${html(day.feed_id)}" data-date="${html(day.archive_date)}">
      <strong>${html(day.feed_name)}</strong><span class="status-chip${day.is_complete ? " ready" : ""}">${html(day.status)}</span>
      <span class="date">${html(day.archive_date)} · ${bytes(day.storage_bytes)}</span>
      <span class="next">${html(day.pipeline_percent)}% · Next: ${html(day.next_step)}</span>
    </button>`;
  }).join("");
}

async function selectDay(day, updateInputs = true) {
  state.selectedDay = day;
  state.incidentExpanded = false;
  state.transcript = { segments: [], offset: 0, total: 0, hasMore: false, query: "" };
  renderLibrary();
  byId("dayDetail").innerHTML = '<div class="empty-state"><div class="job-spinner running"></div><h2>Loading retained evidence…</h2></div>';
  try {
    state.selectedDayDetail = await api(`/api/day?feed_id=${encodeURIComponent(day.feed_id)}&date=${encodeURIComponent(day.archive_date)}`);
    if (updateInputs) syncReviewInputs(day);
    renderDayDetail("incidents");
    await loadTranscript(false);
  } catch (error) {
    byId("dayDetail").innerHTML = `<div class="notice danger"><strong>Unable to open day</strong><span>${html(error.message)}</span></div>`;
  }
}

function syncReviewInputs(day) {
  byId("askFeedId").value = day.feed_id;
  byId("askStartDate").value = day.archive_date;
  byId("askEndDate").value = day.archive_date;
  byId("weekFeedId").value = day.feed_id;
  byId("weekEnding").value = day.archive_date;
}

function stageCards(day) {
  const stages = [
    [Boolean(day.raw_file_count || day.has_combined), "Archive audio", day.raw_file_count ? `${day.raw_file_count} source blocks` : "retained audio"],
    [day.has_combined, "Combine", day.has_combined ? "continuous timeline" : "not ready"],
    [day.has_transcript, "Transcription", day.has_transcript ? `${day.segment_count || 0} segments` : "not ready"],
    [day.has_diarization, "Speaker labels", day.has_diarization ? "attached" : "not ready"],
    [day.has_analysis, "Event analysis", day.has_analysis ? `${day.incident_count || 0} incidents` : "not ready"],
  ];
  return stages.map(([done, label, detail], index) => `<div class="stage${done ? " done" : ""}"><span class="stage-index">${done ? "✓" : index + 1}</span><strong>${html(label)}</strong><small>${html(detail)}</small></div>`).join("");
}

function renderDayDetail(activeTab = "incidents") {
  const detail = state.selectedDayDetail;
  if (!detail) return;
  const day = detail.state;
  const primaryLabel = day.primary_action === "open_review" ? "Review evidence" : day.next_step;
  const actionDisabled = day.primary_action === "resume_download" ? "" : "";
  byId("dayDetail").innerHTML = `
    <div class="detail-head"><div><h2>${html(day.feed_name)}</h2><p>Feed ${html(day.feed_id)} · ${html(day.archive_date)} · ${bytes(day.storage_bytes)}</p></div>
      <button class="button ${day.is_complete ? "secondary" : "primary"}" data-action="primary-day" ${actionDisabled}>${html(primaryLabel)}</button></div>
    <div class="notice ${day.is_complete ? "success" : day.needs_network ? "warning" : "success"}"><strong>${html(day.status)}</strong><span>${html(day.status_detail)}. Next: ${html(day.next_step)}.</span></div>
    <div class="pipeline">${stageCards(day)}</div>
    ${detail.audio_url ? `<div class="audio-block"><audio id="dayAudio" controls preload="metadata" src="${html(detail.audio_url)}"></audio><small>Retained continuous recording. Incident play buttons jump to the cited time without contacting Broadcastify.</small></div>` : ""}
    ${detail.summary ? `<div class="notice success"><strong>Daily brief</strong><span>${html(detail.summary)}</span></div>` : ""}
    <div class="detail-tabs"><button class="detail-tab${activeTab === "incidents" ? " active" : ""}" data-detail-tab="incidents">Incidents (${detail.incidents.length})</button><button class="detail-tab${activeTab === "transcript" ? " active" : ""}" data-detail-tab="transcript">Transcript (${day.segment_count || state.transcript.total || 0})</button></div>
    <div class="detail-panel" id="detailPanel">${activeTab === "incidents" ? incidentMarkup(detail.incidents) : transcriptMarkup()}</div>`;
}

function incidentMarkup(incidents) {
  if (!incidents.length) return '<div class="empty-compact">No extracted incidents are saved for this day yet.</div>';
  const ordered = incidents.slice().sort((a, b) => b.priority - a.priority || a.start_seconds - b.start_seconds);
  const visible = state.incidentExpanded ? ordered : ordered.slice(0, 12);
  return `<div class="incident-list">${visible.map((item) => `<article class="incident-card">
    <div class="incident-title"><div><h3>${html(item.title)}</h3><div class="incident-meta">${clock(item.start_seconds)} · ${html(words(item.event_type))}${item.location ? ` · ${html(item.location)}` : ""} · ${Math.round(item.confidence * 100)}% extraction confidence · I${item.id}</div></div><span class="priority${item.priority >= 4 ? " high" : ""}">P${item.priority}</span></div>
    <p>${html(item.summary)}</p>${item.quote ? `<p class="quote">“${html(item.quote)}”</p>` : ""}
    <div class="button-row"><button class="button secondary small" data-action="play-incident" data-start="${Number(item.start_seconds)}">Play cited time</button><button class="button secondary small" data-action="export-incident" data-incident-id="${item.id}">Prepare exact clip</button></div>
  </article>`).join("")}</div>${ordered.length > 12 ? `<button class="button secondary wide load-more" data-action="toggle-incidents">${state.incidentExpanded ? "Show highest-priority only" : `Show all ${ordered.length} incidents`}</button>` : ""}`;
}

async function loadTranscript(append, query = state.transcript.query) {
  if (!state.selectedDay) return;
  const offset = append ? state.transcript.segments.length : 0;
  const payload = await api(`/api/transcript?feed_id=${encodeURIComponent(state.selectedDay.feed_id)}&date=${encodeURIComponent(state.selectedDay.archive_date)}&offset=${offset}&limit=250&q=${encodeURIComponent(query)}`);
  state.transcript = {
    segments: append ? [...state.transcript.segments, ...payload.segments] : payload.segments,
    offset,
    total: payload.total,
    hasMore: payload.has_more,
    query,
  };
  const panel = byId("detailPanel");
  if (panel && document.querySelector('.detail-tab.active')?.dataset.detailTab === "transcript") panel.innerHTML = transcriptMarkup();
}

function transcriptMarkup() {
  const content = state.transcript.segments.length ? state.transcript.segments.map((segment) => `<div class="transcript-row"><time>${clock(segment.start_seconds)}</time><span class="speaker">${html(segment.speaker || "Unknown speaker")}</span><p>${html(segment.text)}</p></div>`).join("") : '<div class="empty-compact">No transcript segments match this search.</div>';
  return `<div class="transcript-tools"><input id="transcriptSearch" type="search" value="${html(state.transcript.query)}" placeholder="Search this transcript"><button class="button secondary small" data-action="search-transcript">Search</button></div><div class="transcript-list">${content}</div>${state.transcript.hasMore ? '<button class="button secondary wide load-more" data-action="load-transcript">Load more transcript</button>' : ""}`;
}

function renderProfiles() {
  const profiles = state.bootstrap.profiles || [];
  const current = byId("areaProfileSelect").value;
  byId("areaProfileSelect").innerHTML = '<option value="">Choose a profile</option>' + profiles.map((profile) => `<option value="${html(profile.name)}">${html(profile.name)} · ${profile.feeds.length} feeds</option>`).join("");
  if (profiles.some((profile) => profile.name === current)) byId("areaProfileSelect").value = current;
}

function renderRuntime() {
  const runtime = state.bootstrap.runtime || {};
  byId("runtimeDescription").textContent = `${runtime.platform || "Unknown"} ${runtime.platform_release || ""} · Python ${runtime.python || ""}`;
  byId("runtimeFacts").innerHTML = [
    ["Access", runtime.loopback_only ? "This computer only" : "Network"],
    ["Archive root", runtime.output_dir || "archives"],
    ["Evidence database", runtime.database_path || ""],
    ["Default behavior", runtime.platform === "Windows" ? "Tested Windows automatic profile" : "Portable automatic detection"],
  ].map(([label, value]) => `<div class="runtime-fact"><span>${html(label)}</span><strong>${html(value)}</strong></div>`).join("");
}

function updateProviderNotice(message = "") {
  const provider = byId("settingAnalysisProvider")?.value || state.settings.analysisProvider;
  const external = provider === "openai-responses" || provider === "codex-cli" || (provider === "openai-compatible" && !/^https?:\/\/(127\.0\.0\.1|localhost|\[::1\])(?::\d+)?(?:\/|$)/i.test(byId("settingAnalysisEndpoint")?.value || ""));
  const allowed = byId("settingAllowExternal")?.checked;
  const box = byId("providerNotice");
  if (!box) return;
  if (!external) {
    box.className = "notice success";
    box.innerHTML = `<strong>Private local analysis</strong><span>${html(message || "Transcript text stays on this computer.")}</span>`;
  } else if (!allowed) {
    box.className = "notice warning";
    box.innerHTML = `<strong>External analysis is off</strong><span>${html(message || "Enable the explicit transcript-sharing switch before using this provider.")}</span>`;
  } else {
    box.className = "notice warning";
    box.innerHTML = `<strong>External transcript sharing allowed</strong><span>${html(message || "Prompts may contain sensitive or unverified radio text; raw audio is not sent.")}</span>`;
  }
}

async function startJob(command, payload = {}, options = {}) {
  if (state.activeJob) {
    toast("Another local job is already active.", true);
    return null;
  }
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ command, payload }) });
    state.activeJob = job;
    state.jobCallback = options.onComplete || null;
    byId("jobDrawer").classList.add("open");
    byId("jobDrawerToggle").setAttribute("aria-expanded", "true");
    byId("jobTitle").textContent = options.label || command.replaceAll("-", " ");
    byId("jobMessage").textContent = "Starting local worker…";
    byId("cancelJobButton").disabled = false;
    renderJob(job);
    pollJob();
    return job;
  } catch (error) {
    toast(error.message, true);
    return null;
  }
}

async function pollJob() {
  if (!state.activeJob) return;
  try {
    const job = await api(`/api/jobs/${state.activeJob.id}`);
    state.activeJob = job;
    renderJob(job);
    if (["completed", "failed", "canceled"].includes(job.status)) {
      clearTimeout(state.jobTimer);
      const callback = state.jobCallback;
      state.activeJob = null;
      state.jobCallback = null;
      byId("cancelJobButton").disabled = true;
      byId("jobDrawer").classList.remove("open");
      byId("jobDrawerToggle").setAttribute("aria-expanded", "false");
      if (job.status === "completed") {
        toast(`${job.command.replaceAll("-", " ")} completed.`);
        if (callback) await callback(job);
      } else {
        toast(job.error || `${job.command} ${job.status}.`, true);
      }
      return;
    }
  } catch (error) {
    toast(error.message, true);
  }
  state.jobTimer = setTimeout(pollJob, 900);
}

function renderJob(job) {
  const events = job.events || [];
  const last = events[events.length - 1];
  byId("jobSpinner").classList.toggle("running", ["queued", "running", "canceling"].includes(job.status));
  byId("jobMessage").textContent = last?.message || job.error || `${words(job.status)}.`;
  const progress = [...events].reverse().find((event) => event.type === "progress" && Number(event.total) > 0);
  const percentage = progress ? Math.max(0, Math.min(100, Number(progress.current) / Number(progress.total) * 100)) : job.status === "completed" ? 100 : 0;
  byId("jobProgressBar").style.width = `${percentage}%`;
  byId("jobLog").innerHTML = events.slice(-80).map((event) => `<div>${html(event.message || `${event.type}: ${event.stage || "updated"}`)}</div>`).join("") || '<div>Waiting for the first progress update…</div>';
  byId("jobLog").scrollTop = byId("jobLog").scrollHeight;
}

function eventOf(job, type) {
  return [...(job.events || [])].reverse().find((event) => event.type === type);
}

async function queueAnalyses(feedId, dates) {
  state.analysisQueue = dates.map((archiveDate) => ({ feedId, archiveDate }));
  await runNextAnalysis();
}

async function runNextAnalysis() {
  const next = state.analysisQueue.shift();
  if (!next) {
    await refreshBootstrap();
    toast("Automatic analysis finished for every completed transcript.");
    return;
  }
  await startJob("analyze-day", { feed_id: next.feedId, archive_date: next.archiveDate, ...providerPayload() }, {
    label: `Analyzing ${next.archiveDate}`,
    onComplete: runNextAnalysis,
  });
}

function renderAnswer(result) {
  const evidence = result.evidence_ids || [];
  const limitations = result.limitations || [];
  byId("answerPanel").innerHTML = `<h3>Evidence-grounded answer</h3><p>${html(result.answer || "No answer text was returned.")}</p>${evidence.length ? `<div class="citation-list">${evidence.map((value) => `<div class="citation">Evidence ${html(value)}</div>`).join("")}</div>` : ""}${limitations.length ? `<div class="notice warning"><strong>Limitations</strong><span>${html(limitations.join("; "))}</span></div>` : ""}`;
}

function renderWeek(result) {
  if (!result) {
    byId("weekPanel").innerHTML = '<div class="empty-compact">No saved summary covers this exact seven-day window.</div>';
    return;
  }
  const missing = result.missing_dates || [];
  const notable = result.notable_incident_ids || [];
  const coverageText = missing.length ? ` · Missing ${missing.join(", ")}` : " · Complete date coverage";
  byId("weekPanel").innerHTML = `<h3>${html(result.start_date)} through ${html(result.end_date)}</h3><p>${html(result.summary || "")}</p><div class="notice ${missing.length ? "warning" : "success"}"><strong>${Number(result.days_available) || 0}/7 days available</strong><span>${Number(result.incident_count) || 0} retained incidents${html(coverageText)}</span></div>${notable.length ? `<div class="citation-list">${notable.map((value) => `<div class="citation">Notable record I${html(value)}</div>`).join("")}</div>` : ""}`;
}

function renderAreaBrief(result) {
  state.areaBriefResult = result;
  state.areaStoriesExpanded = false;
  renderAreaBriefContent();
}

function renderAreaBriefContent() {
  const result = state.areaBriefResult;
  if (!result) {
    byId("areaBriefPanel").innerHTML = '<div class="empty-compact">No saved story-lead brief exists for this profile yet.</div>';
    return;
  }
  const coverage = result.coverage || {};
  const stories = result.stories || [];
  const visible = state.areaStoriesExpanded ? stories : stories.slice(0, 10);
  byId("areaBriefPanel").innerHTML = `<div class="notice ${Number(coverage.feeds_with_data) < Number(coverage.feed_count) ? "warning" : "success"}"><strong>${html(result.start_date)} through ${html(result.end_date)}</strong><span>${Number(coverage.feeds_with_data) || 0}/${Number(coverage.feed_count) || 0} feeds with data · ${Number(coverage.feed_days_available) || 0}/${Number(coverage.feed_days_expected) || 0} feed-days · ${Number(coverage.incident_count) || 0} incidents</span></div><p>${html(result.summary || "")}</p>${visible.map((story) => `<article class="story-card"><span class="story-score">${html(story.interest_level || "Lead")} · score ${Number(story.newsworthiness_score) || 0} · P${Number(story.priority) || 0}</span><h3>${html(story.headline || "Untitled story lead")}</h3><p>${html(story.summary || "")}</p><p><strong>Why interesting:</strong> ${html(story.why_interesting || "")}</p><div class="tag-list">${[...(story.neighborhood_tags || []), ...(story.topic_tags || [])].map((tag) => `<span class="tag">${html(words(tag))}</span>`).join("")}</div></article>`).join("") || '<div class="empty-compact">No story leads met the saved threshold.</div>'}${stories.length > 10 ? `<button class="button secondary wide" data-action="toggle-stories">${state.areaStoriesExpanded ? "Show highest-ranked only" : `Show all ${stories.length} story leads`}</button>` : ""}`;
}

function renderFeedResults(results) {
  if (!results.length) {
    byId("feedSearchResults").innerHTML = '<div class="empty-compact">No matching feeds were returned by the website search.</div>';
    return;
  }
  byId("feedSearchResults").innerHTML = results.map((feed) => `<button class="feed-result selectable${state.selectedFeed?.feed_id === feed.feed_id ? " selected" : ""}" data-select-feed="${html(feed.feed_id)}"><div></div><div><h3>${html(feed.name || `Feed ${feed.feed_id}`)}</h3><p>Feed ${html(feed.feed_id)}${feed.location ? ` · ${html(feed.location)}` : ""}${feed.genre ? ` · ${html(feed.genre)}` : ""}</p><p>${html(feed.description || "")}</p></div><span class="listener-count">${Number(feed.listeners) || 0} listeners</span></button>`).join("");
  byId("feedSearchResults").dataset.results = JSON.stringify(results);
}

function renderAreaResults(results) {
  state.areaDiscovered = results;
  if (!results.length) {
    byId("areaSearchResults").innerHTML = '<div class="empty-compact">No public-safety feeds were found for those ZIPs.</div>';
    return;
  }
  byId("areaSearchResults").innerHTML = results.map((feed, index) => `<label class="feed-result"><input type="checkbox" data-area-feed-index="${index}" checked><div><h3>${html(feed.name || `Feed ${feed.feed_id}`)}</h3><p>Feed ${html(feed.feed_id)} · ${html(feed.location || "")}</p></div><span class="listener-count">${Number(feed.listeners) || 0} listeners</span></label>`).join("");
}

async function openSavedWeek() {
  const feedId = byId("weekFeedId").value.trim();
  const weekEnding = byId("weekEnding").value;
  if (!feedId || !weekEnding) return toast("Enter a feed ID and week ending date.", true);
  try {
    const payload = await api(`/api/saved-week?feed_id=${encodeURIComponent(feedId)}&week_ending=${encodeURIComponent(weekEnding)}`);
    renderWeek(payload.result);
  } catch (error) { toast(error.message, true); }
}

async function openSavedArea() {
  const profileName = byId("areaProfileSelect").value;
  if (!profileName) return toast("Choose a saved area profile.", true);
  try {
    const payload = await api(`/api/saved-area-digest?profile_name=${encodeURIComponent(profileName)}`);
    renderAreaBrief(payload.result);
  } catch (error) { toast(error.message, true); }
}

document.addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.view) return setView(button.dataset.view);
  if (button.classList.contains("day-row")) {
    const day = state.bootstrap.days.find((value) => value.feed_id === button.dataset.feedId && value.archive_date === button.dataset.date);
    if (day) await selectDay(day);
    return;
  }
  if (button.dataset.detailTab) {
    renderDayDetail(button.dataset.detailTab);
    return;
  }
  if (button.dataset.selectFeed) {
    const results = JSON.parse(byId("feedSearchResults").dataset.results || "[]");
    state.selectedFeed = results.find((value) => String(value.feed_id) === button.dataset.selectFeed);
    if (state.selectedFeed) {
      byId("archiveFeedId").value = state.selectedFeed.feed_id;
      byId("selectedFeedLabel").textContent = `${state.selectedFeed.name} · feed ${state.selectedFeed.feed_id}`;
      renderFeedResults(results);
    }
    return;
  }
  const action = button.dataset.action;
  if (action === "play-incident") {
    const audio = byId("dayAudio");
    if (!audio) return toast("No retained combined audio is available for this day.", true);
    audio.currentTime = Number(button.dataset.start) || 0;
    await audio.play().catch(() => toast("Use the audio play control to allow playback.", true));
  }
  if (action === "export-incident") {
    await startJob("incident-clip", { incident_id: Number(button.dataset.incidentId) }, {
      label: `Preparing incident I${button.dataset.incidentId} clip`,
      onComplete: async (job) => {
        const result = eventOf(job, "incident_clip")?.clip;
        if (result?.media_url) {
          const anchor = document.createElement("a");
          anchor.href = result.media_url;
          anchor.download = result.filename || `incident-I${result.incident_id}.mp3`;
          anchor.click();
        } else toast("The clip was prepared, but it is outside the selected archive root.", true);
      },
    });
  }
  if (action === "load-transcript") await loadTranscript(true);
  if (action === "search-transcript") await loadTranscript(false, byId("transcriptSearch").value);
  if (action === "toggle-incidents") {
    state.incidentExpanded = !state.incidentExpanded;
    const panel = byId("detailPanel");
    if (panel) panel.innerHTML = incidentMarkup(state.selectedDayDetail?.incidents || []);
  }
  if (action === "toggle-stories") {
    state.areaStoriesExpanded = !state.areaStoriesExpanded;
    renderAreaBriefContent();
  }
  if (action === "primary-day") {
    const day = state.selectedDayDetail?.state;
    if (!day) return;
    if (day.primary_action === "resume_download") {
      setView("archive");
      byId("archiveFeedId").value = day.feed_id;
      byId("selectedFeedLabel").textContent = `${day.feed_name} · feed ${day.feed_id}`;
      byId("archiveStartDate").value = day.archive_date;
      byId("archiveEndDate").value = day.archive_date;
    } else if (day.primary_action === "continue_local") {
      await startJob("continue-local", { feed_id: day.feed_id, archive_date: day.archive_date, diarize: true, analyze: true, ...processingPayload(), ...providerPayload() }, { label: `Finishing ${day.archive_date}`, onComplete: refreshBootstrap });
    } else {
      renderDayDetail("incidents");
    }
  }
});

byId("menuButton").addEventListener("click", () => document.querySelector(".sidebar").classList.toggle("open"));
byId("refreshLibraryButton").addEventListener("click", () => refreshBootstrap());
byId("librarySearch").addEventListener("input", renderLibrary);
byId("libraryFilter").addEventListener("change", renderLibrary);
byId("saveSettingsButton").addEventListener("click", saveSettings);
byId("settingAnalysisProvider").addEventListener("change", () => {
  const defaults = { local: "ggml-org/gemma-4-12B-it-GGUF:Q4_K_M", "openai-responses": "gpt-5.6-luna", "openai-compatible": "", "codex-cli": "" };
  byId("settingAnalysisModel").value = defaults[byId("settingAnalysisProvider").value] || "";
  updateProviderNotice();
});
byId("settingAnalysisEndpoint").addEventListener("input", () => updateProviderNotice());
byId("settingAllowExternal").addEventListener("change", () => updateProviderNotice());
byId("jobDrawerToggle").addEventListener("click", () => {
  const open = byId("jobDrawer").classList.toggle("open");
  byId("jobDrawerToggle").setAttribute("aria-expanded", String(open));
});
byId("cancelJobButton").addEventListener("click", async () => {
  if (!state.activeJob) return;
  try { await api(`/api/jobs/${state.activeJob.id}/cancel`, { method: "POST", body: "{}" }); } catch (error) { toast(error.message, true); }
});

byId("feedSearchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  await startJob("search", { query: byId("feedSearchInput").value }, { label: "Searching website feeds", onComplete: (job) => renderFeedResults(eventOf(job, "result")?.results || []) });
});

byId("archiveDiarize").addEventListener("change", () => {
  if (byId("archiveDiarize").checked) {
    byId("archiveCombine").checked = true;
    byId("archiveTranscribe").checked = true;
  }
});
byId("archiveForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const feedId = byId("archiveFeedId").value;
  if (!feedId) return toast("Select a feed from the search results first.", true);
  const analyze = byId("archiveAnalyze").checked;
  await startJob("run", {
    feed_id: feedId,
    start_date: byId("archiveStartDate").value,
    end_date: byId("archiveEndDate").value,
    combine: byId("archiveCombine").checked,
    transcribe: byId("archiveTranscribe").checked,
    diarize: byId("archiveDiarize").checked,
    ...processingPayload(),
  }, {
    label: `Archiving feed ${feedId}`,
    onComplete: async (job) => {
      await refreshBootstrap({ preserveSelection: false });
      const result = eventOf(job, "complete")?.result;
      const transcriptDates = (result?.days || []).filter((day) => (day.transcripts || []).length).map((day) => day.date);
      if (analyze && transcriptDates.length) await queueAnalyses(feedId, transcriptDates);
    },
  });
});

byId("askForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  await startJob("ask", { feed_id: byId("askFeedId").value, start_date: byId("askStartDate").value, end_date: byId("askEndDate").value, question: byId("askQuestion").value, ...providerPayload() }, { label: "Answering from retained evidence", onComplete: (job) => renderAnswer(eventOf(job, "answer")?.result || {}) });
});
byId("weekForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  await startJob("summarize-week", { feed_id: byId("weekFeedId").value, week_ending: byId("weekEnding").value, ...providerPayload() }, { label: "Building seven-day summary", onComplete: (job) => renderWeek(eventOf(job, "weekly_summary")?.result) });
});
byId("openSavedWeekButton").addEventListener("click", openSavedWeek);

byId("areaSearchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const zipCodes = byId("areaZipCodes").value.split(/[\s,;]+/).filter(Boolean);
  await startJob("area-search", { zip_codes: zipCodes }, { label: "Discovering nearby feeds", onComplete: (job) => renderAreaResults(eventOf(job, "area_search")?.results || []) });
});
byId("saveAreaProfileForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const feeds = [...document.querySelectorAll("[data-area-feed-index]:checked")].map((value) => state.areaDiscovered[Number(value.dataset.areaFeedIndex)]).filter(Boolean);
  const zipCodes = byId("areaZipCodes").value.split(/[\s,;]+/).filter(Boolean);
  await startJob("save-area-profile", { name: byId("areaProfileName").value, zip_codes: zipCodes, feeds }, { label: "Saving area profile", onComplete: refreshBootstrap });
});
byId("buildAreaBriefButton").addEventListener("click", async () => {
  const profileName = byId("areaProfileSelect").value;
  if (!profileName) return toast("Choose a saved area profile.", true);
  await startJob("summarize-area", { profile_name: profileName, start_date: byId("areaStartDate").value, end_date: byId("areaEndDate").value, ...providerPayload() }, { label: `Ranking ${profileName} story leads`, onComplete: (job) => renderAreaBrief(eventOf(job, "area_digest")?.result) });
});
byId("openSavedAreaButton").addEventListener("click", openSavedArea);
byId("areaProfileSelect").addEventListener("change", openSavedArea);

byId("providerCheckButton").addEventListener("click", async () => {
  await startJob("analysis-provider-diagnostics", providerPayload(), { label: "Checking analysis provider", onComplete: (job) => {
    const result = eventOf(job, "analysis_provider_diagnostics")?.result;
    if (result) updateProviderNotice(`${result.verified ? "Verified" : result.ready ? "Configured" : "Setup needed"}: ${result.message}`);
  } });
});
byId("runtimeCheckButton").addEventListener("click", async () => {
  await startJob("diagnostics", {}, { label: "Checking local hardware", onComplete: (job) => {
    const result = eventOf(job, "diagnostics");
    if (!result) return;
    const accelerators = result.accelerators || {};
    const summary = [result.cuda_available ? `CUDA: ${(result.cuda_devices || []).join(", ")}` : "CUDA unavailable", result.ffmpeg ? "FFmpeg ready" : "FFmpeg missing", result.llama_server ? "llama.cpp ready" : "llama.cpp missing", `Profiles: ${Object.keys(accelerators.profiles || accelerators).length || "checked"}`].join(" · ");
    byId("runtimeDescription").textContent = summary;
  } });
});
byId("asrSelfTestButton").addEventListener("click", async () => {
  await startJob("asr-self-test", processingPayload(), { label: "Testing selected transcription engine", onComplete: (job) => {
    const result = eventOf(job, "asr_self_test")?.result;
    if (!result) return;
    const notice = byId("asrSelfTestNotice");
    notice.className = "notice success";
    notice.querySelector("strong").textContent = "Transcription ready";
    const fallback = result.fallback_reason ? ` Fallback: ${result.fallback_reason}` : "";
    notice.querySelector("span").textContent = `${result.message}${fallback}`;
  } });
});
byId("loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const username = byId("loginUsername").value;
  const password = byId("loginPassword").value;
  byId("loginPassword").value = "";
  await startJob("authenticate", { username, password }, { label: "Signing in to Broadcastify", onComplete: () => toast("Broadcastify website session refreshed.") });
});

const today = new Date();
const localToday = new Date(today.getTime() - today.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
const weekAgo = new Date(today.getTime() - 6 * 86400000 - today.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
byId("archiveStartDate").value = localToday;
byId("archiveEndDate").value = localToday;
byId("areaStartDate").value = weekAgo;
byId("areaEndDate").value = localToday;
byId("weekEnding").value = localToday;
applySettingsForm();
setView(location.hash.replace("#", "") in { library: 1, archive: 1, review: 1, area: 1, settings: 1 } ? location.hash.replace("#", "") : "library");
refreshBootstrap({ preserveSelection: false }).then(async () => {
  if (state.bootstrap.days.length) await selectDay(state.bootstrap.days[0]);
});
