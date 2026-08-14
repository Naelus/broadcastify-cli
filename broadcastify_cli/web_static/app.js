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
  hardwareProfile: "auto",
  whisperModel: "turbo",
  asrEngine: "auto",
  device: "auto",
  diarizationEngine: "community-1",
  diarizationDevice: "auto",
  batchSize: 8,
  asrModelPath: "",
  analysisProvider: "local",
  analysisModel: "ggml-org/gemma-4-12B-it-GGUF:Q4_0",
  analysisDevice: "auto",
  analysisEndpoint: "",
  apiKeyEnvironment: "OPENAI_API_KEY",
  codexPath: "",
  allowExternal: false,
  lanSyncEnabled: true,
  lanDiscoveryEnabled: true,
  lanPeerUrls: "",
};

const SETTINGS_SECTIONS = {
  setup: "First-run readiness and execution checks",
  processing: "Transcription, speaker labels, and hardware profiles",
  analysis: "Local or explicitly consented analysis providers",
  account: "Premium website session and private credentials",
};

const HARDWARE_PROFILE_DESCRIPTIONS = {
  auto: "Automatic keeps the tested Windows behavior and chooses safe per-stage fallbacks elsewhere.",
  cuda: "NVIDIA CUDA keeps transcription and speaker labels on the GPU for the fastest validated Windows path.",
  vulkan: "Cross-vendor Vulkan uses whisper.cpp plus the fast sherpa-onnx CPU speaker preview. Community-1 remains the later accuracy upgrade.",
  openvino: "OpenVINO uses Intel's runtime plus the fast sherpa-onnx CPU speaker preview. Community-1 remains the later accuracy upgrade.",
  metal: "Apple Metal uses whisper.cpp plus the fast sherpa-onnx CPU speaker preview. Community-1 remains the later accuracy upgrade.",
  windowsml: "Windows ML uses the validated managed Whisper graph plus the fast sherpa-onnx CPU speaker preview. Community-1 remains the later accuracy upgrade.",
  qwen: "Fast CPU preview uses managed Qwen3-ASR 0.6B INT8 and sherpa-onnx speaker regions. Neither preview replaces the validated evidence defaults.",
  cpu: "CPU only avoids GPU dependencies for every stage; expect lower throughput on long archive ranges.",
  custom: "Custom overrides are active. Test transcription, speaker labels, and analysis before an unattended run.",
};

const state = {
  bootstrap: { days: [], feed_coverage: [], catchups: [], profiles: [], schedules: [], summary: {}, runtime: {} },
  selectedDay: null,
  selectedDayDetail: null,
  editingSchedule: null,
  incidentExpanded: false,
  transcript: { segments: [], offset: 0, total: 0, hasMore: false, query: "" },
  selectedFeed: null,
  areaDiscovered: [],
  areaSelectedFeedIds: new Set(),
  areaCoverage: { mode: "radius", center_zip: "", radius_miles: 25, max_zip_codes: 12, searched_zip_codes: [] },
  areaBriefResult: null,
  areaSelectedStoryIndex: -1,
  settings: loadSettings(),
  settingsSection: loadSettingsSection(),
  activeJob: null,
  jobTimer: null,
  jobCallback: null,
  jobFailureCallback: null,
  analysisQueue: [],
  hardwareDiagnostics: null,
  asrSelfTest: null,
  diarizationSelfTest: null,
  analysisProviderStatus: null,
  analysisSelfTest: null,
  profileSelfTest: null,
  accountVerified: false,
};

let applyingHardwareProfile = false;

function loadSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem("radioArchiveSettings") || "{}") };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function loadSettingsSection() {
  const saved = localStorage.getItem("radioArchiveSettingsSection") || "setup";
  return Object.hasOwn(SETTINGS_SECTIONS, saved) ? saved : "setup";
}

function saveSettings() {
  readSettingsForm();
  localStorage.setItem("radioArchiveSettings", JSON.stringify(state.settings));
  toast("Non-secret settings saved in this browser profile.");
}

function readSettingsForm() {
  state.settings = {
    hardwareProfile: byId("settingHardwareProfile").value,
    whisperModel: byId("settingWhisperModel").value,
    asrEngine: byId("settingAsrEngine").value,
    device: byId("settingDevice").value,
    diarizationEngine: byId("settingDiarizationEngine").value,
    diarizationDevice: byId("settingDiarizationDevice").value,
    batchSize: Math.max(1, Number(byId("settingBatchSize").value) || 8),
    asrModelPath: byId("settingAsrModelPath").value.trim(),
    analysisProvider: byId("settingAnalysisProvider").value,
    analysisModel: byId("settingAnalysisModel").value.trim(),
    analysisDevice: byId("settingAnalysisDevice").value,
    analysisEndpoint: byId("settingAnalysisEndpoint").value.trim(),
    apiKeyEnvironment: byId("settingApiKeyEnvironment").value.trim() || "OPENAI_API_KEY",
    codexPath: byId("settingCodexPath").value.trim(),
    allowExternal: byId("settingAllowExternal").checked,
    lanSyncEnabled: byId("settingLanSyncEnabled").checked,
    lanDiscoveryEnabled: byId("settingLanDiscoveryEnabled").checked,
    lanPeerUrls: byId("settingLanPeerUrls").value.trim(),
  };
  updateHardwareProfileDescription();
  updateProviderNotice();
  renderSetupReadiness();
}

function updateHardwareProfileDescription() {
  const description = byId("processingProfileDescription");
  const profile = byId("settingHardwareProfile")?.value || state.settings.hardwareProfile;
  if (description) {
    description.textContent = HARDWARE_PROFILE_DESCRIPTIONS[profile]
      || HARDWARE_PROFILE_DESCRIPTIONS.custom;
  }
}

function applySettingsForm() {
  byId("settingHardwareProfile").value = state.settings.hardwareProfile;
  byId("settingWhisperModel").value = state.settings.whisperModel;
  byId("settingAsrEngine").value = state.settings.asrEngine;
  byId("settingDevice").value = state.settings.device;
  byId("settingDiarizationEngine").value = state.settings.diarizationEngine;
  byId("settingDiarizationDevice").value = state.settings.diarizationDevice;
  byId("settingBatchSize").value = state.settings.batchSize;
  byId("settingAsrModelPath").value = state.settings.asrModelPath;
  byId("settingAnalysisProvider").value = state.settings.analysisProvider;
  byId("settingAnalysisModel").value = state.settings.analysisModel;
  byId("settingAnalysisDevice").value = state.settings.analysisDevice;
  byId("settingAnalysisEndpoint").value = state.settings.analysisEndpoint;
  byId("settingApiKeyEnvironment").value = state.settings.apiKeyEnvironment;
  byId("settingCodexPath").value = state.settings.codexPath;
  byId("settingAllowExternal").checked = Boolean(state.settings.allowExternal);
  byId("settingLanSyncEnabled").checked = Boolean(state.settings.lanSyncEnabled);
  byId("settingLanDiscoveryEnabled").checked = Boolean(state.settings.lanDiscoveryEnabled);
  byId("settingLanPeerUrls").value = state.settings.lanPeerUrls;
  updateHardwareProfileDescription();
  updateProviderNotice();
  updateAsrModelPreparationUi();
  renderSetupReadiness();
}

function applyHardwareProfile(profile, notify = true) {
  const deploymentDefaults = profile === "auto"
    ? (state.bootstrap.runtime?.processing_defaults || {})
    : {};
  const hasDeploymentDefaults = Boolean(deploymentDefaults.hardware_profile);
  const choices = {
    auto: hasDeploymentDefaults
      ? [
          deploymentDefaults.asr_engine || "auto",
          deploymentDefaults.device || "auto",
          deploymentDefaults.diarization_device || "auto",
          "auto",
          deploymentDefaults.diarization_engine || "community-1",
        ]
      : ["auto", "auto", "auto", "auto", "community-1"],
    cuda: ["faster-whisper", "cuda", "cuda", "auto", "community-1"],
    vulkan: ["whisper.cpp", "vulkan", "cpu", "auto", "sherpa-onnx"],
    openvino: ["openvino", "openvino-auto", "cpu", "auto", "sherpa-onnx"],
    metal: ["whisper.cpp", "metal", "cpu", "auto", "sherpa-onnx"],
    windowsml: ["windows-ml", "windows-ml", "cpu", "auto", "sherpa-onnx"],
    qwen: ["qwen3-asr", "cpu", "cpu", "auto", "sherpa-onnx"],
    cpu: ["faster-whisper", "cpu", "cpu", "cpu", "community-1"],
  };
  const choice = choices[profile];
  if (!choice) {
    resetProfileVerification();
    resetAsrVerification();
    resetDiarizationVerification();
    resetAnalysisVerification();
    state.analysisProviderStatus = null;
    readSettingsForm();
    updateAsrModelPreparationUi();
    renderHardwareProfiles();
    renderSetupReadiness();
    return;
  }
  const resetWhisperModel = ["vulkan", "metal"].includes(profile)
    && byId("settingWhisperModel").value === "distil-large-v3";
  const useWindowsMlStarter = profile === "windowsml"
    && !["base", "base.en"].includes(byId("settingWhisperModel").value);
  const useQwenStarter = profile === "qwen"
    && byId("settingWhisperModel").value !== "qwen3-asr-0.6b-int8";
  const resetQwenModel = profile !== "qwen"
    && byId("settingWhisperModel").value === "qwen3-asr-0.6b-int8";
  resetProfileVerification();
  resetAsrVerification();
  resetDiarizationVerification();
  resetAnalysisVerification();
  state.analysisProviderStatus = null;
  applyingHardwareProfile = true;
  byId("settingHardwareProfile").value = profile;
  byId("settingAsrEngine").value = choice[0];
  byId("settingDevice").value = choice[1];
  byId("settingDiarizationDevice").value = choice[2];
  byId("settingAnalysisDevice").value = choice[3];
  byId("settingDiarizationEngine").value = choice[4];
  if (hasDeploymentDefaults && deploymentDefaults.model) {
    byId("settingWhisperModel").value = deploymentDefaults.model;
  }
  if (hasDeploymentDefaults && Number(deploymentDefaults.batch_size) > 0) {
    byId("settingBatchSize").value = Number(deploymentDefaults.batch_size);
  }
  if (resetWhisperModel) byId("settingWhisperModel").value = "turbo";
  if (useWindowsMlStarter) byId("settingWhisperModel").value = "base.en";
  if (useQwenStarter) byId("settingWhisperModel").value = "qwen3-asr-0.6b-int8";
  if (resetQwenModel && !useWindowsMlStarter) byId("settingWhisperModel").value = "turbo";
  applyingHardwareProfile = false;
  readSettingsForm();
  updateAsrModelPreparationUi();
  renderHardwareProfiles();
  renderSetupReadiness();
  if (notify) {
    const modelMessage = resetWhisperModel
      ? " The Whisper model was reset to turbo because whisper.cpp has no managed distil-large-v3 mapping."
      : useWindowsMlStarter
      ? " The radio-tested Base CPU starter was selected. Tiny is faster, but it missed important words in retained scanner audio."
      : useQwenStarter
      ? " The optional Qwen3-ASR 0.6B INT8 fast-CPU model was selected; speech-region timestamps remain attached."
      : "";
    const deploymentMessage = hasDeploymentDefaults
      ? ` The installed ${words(deploymentDefaults.hardware_profile)} deployment preset is active.`
      : "";
    toast(`${byId("settingHardwareProfile").selectedOptions[0].textContent} defaults applied.${deploymentMessage}${modelMessage}`);
    void runHardwareCheck();
  }
}

function resetAsrVerification() {
  state.asrSelfTest = null;
  const notice = byId("asrSelfTestNotice");
  if (!notice) return;
  notice.className = "notice";
  notice.querySelector("strong").textContent = "Transcription not tested for these settings";
  notice.querySelector("span").textContent = "Run the synthetic decode after changing the engine, model, device, path, or batch.";
}

function resetDiarizationVerification() {
  state.diarizationSelfTest = null;
  const notice = byId("diarizationSelfTestNotice");
  if (!notice) return;
  notice.className = "notice";
  notice.querySelector("strong").textContent = "Speakers not tested for these settings";
  notice.querySelector("span").textContent = "Run the generated-audio test after changing its engine, device, token, or batch.";
}

function resetAnalysisVerification() {
  state.analysisSelfTest = null;
  const notice = byId("analysisSelfTestNotice");
  if (!notice) return;
  notice.className = "notice";
  notice.querySelector("strong").textContent = "Analysis not tested for these settings";
  notice.querySelector("span").textContent = "Run Test model after changing the provider, model, endpoint, credential, or device.";
}

function resetProfileVerification() {
  state.profileSelfTest = null;
  const notice = byId("profileSelfTestNotice");
  if (!notice) return;
  notice.className = "notice";
  notice.querySelector("strong").textContent = "Profile not verified for these settings";
  notice.querySelector("span").textContent = "Verify profile runs transcription, speaker labels, and analysis in sequence with generated input.";
}

function providerPayload() {
  readSettingsForm();
  const key = byId("settingAnalysisApiKey").value;
  return {
    analysis_provider: state.settings.analysisProvider,
    analysis_model: state.settings.analysisModel,
    analysis_device: state.settings.analysisDevice,
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
    hardware_profile: state.settings.hardwareProfile,
    model: state.settings.whisperModel,
    asr_engine: state.settings.asrEngine,
    device: state.settings.device,
    device_index: 0,
    compute_type: "auto",
    asr_model_path: state.settings.asrModelPath || undefined,
    diarization_engine: state.settings.diarizationEngine,
    diarization_device: state.settings.diarizationDevice,
    batch_size: state.settings.batchSize,
    huggingface_token: huggingFaceToken || undefined,
    lan_sync_enabled: Boolean(state.settings.lanSyncEnabled),
    lan_discovery_enabled: Boolean(state.settings.lanDiscoveryEnabled),
    lan_peer_urls: state.settings.lanPeerUrls
      .split(/[\s,;]+/)
      .map((value) => value.trim())
      .filter(Boolean),
  };
}

function applyRuntimeProcessingDefaults() {
  const defaults = state.bootstrap.runtime?.processing_defaults || {};
  if (!defaults.hardware_profile || state.settings.hardwareProfile !== "auto") {
    return false;
  }
  const mapped = {
    whisperModel: defaults.model,
    asrEngine: defaults.asr_engine,
    device: defaults.device,
    diarizationEngine: defaults.diarization_engine,
    diarizationDevice: defaults.diarization_device,
    batchSize: defaults.batch_size,
  };
  Object.entries(mapped).forEach(([field, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      state.settings[field] = value;
    }
  });
  // Keep Automatic as the user's selection. The server resolves it again at
  // every job boundary, so a deployment preset can be upgraded without a
  // browser silently turning the prior resolved preset into an explicit one.
  state.settings.hardwareProfile = "auto";
  localStorage.setItem("radioArchiveSettings", JSON.stringify(state.settings));
  applySettingsForm();
  toast(`${words(defaults.hardware_profile)} deployment defaults are active while Automatic remains selected.`);
  return true;
}

function effectiveAsrEngine() {
  const engine = byId("settingAsrEngine")?.value || state.settings.asrEngine || "auto";
  if (engine !== "auto") return engine;
  const device = byId("settingDevice")?.value || state.settings.device || "auto";
  if (["vulkan", "metal"].includes(device)) return "whisper.cpp";
  if (["windows-ml", "directml"].includes(device)) return "windows-ml";
  if (device.startsWith("openvino") || ["gpu", "npu"].includes(device)) return "openvino";
  return "faster-whisper";
}

function updateAsrModelPreparationUi() {
  const button = byId("asrPrepareButton");
  if (!button) return;
  const engine = effectiveAsrEngine();
  const nextAction = currentProfileAction();
  const runtimeBlocked = nextAction?.stage === "transcription"
    && nextAction.kind === "configure-transcription";
  button.hidden = runtimeBlocked
    || !["windows-ml", "whisper.cpp", "qwen3-asr"].includes(engine);
  button.disabled = runtimeBlocked;
  if (engine === "windows-ml") {
    button.textContent = "Build & test model";
    button.title = "Explicitly build the selected ONNX Runtime GenAI CPU model, retain its path, then prove a local decode.";
  } else if (engine === "whisper.cpp") {
    button.textContent = "Download & test model";
    button.title = "Explicitly download the selected public GGML model, retain its path, then prove the configured whisper.cpp runtime.";
  } else if (engine === "qwen3-asr") {
    button.textContent = "Download & test model";
    button.title = "Explicitly download and checksum-verify Qwen3-ASR 0.6B INT8 plus Silero VAD, retain their path, then prove local CPU execution.";
  }
  if (runtimeBlocked) {
    button.title = nextAction.message;
  }
}

function ensureAsrModelCompatibility() {
  const engine = byId("settingAsrEngine").value;
  const model = byId("settingWhisperModel").value;
  if (engine === "qwen3-asr" && model !== "qwen3-asr-0.6b-int8") {
    byId("settingWhisperModel").value = "qwen3-asr-0.6b-int8";
  } else if (engine !== "qwen3-asr" && model === "qwen3-asr-0.6b-int8") {
    byId("settingWhisperModel").value = "turbo";
  }
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

function closeNavigation() {
  document.querySelector(".sidebar").classList.remove("open");
  byId("menuButton").setAttribute("aria-expanded", "false");
}

function setSettingsSection(name, { focusTab = false, updateHistory = true } = {}) {
  const section = Object.hasOwn(SETTINGS_SECTIONS, name) ? name : "setup";
  state.settingsSection = section;
  localStorage.setItem("radioArchiveSettingsSection", section);
  document.querySelectorAll("[data-settings-section]").forEach((tab) => {
    const active = tab.dataset.settingsSection === section;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focusTab) tab.focus();
  });
  document.querySelectorAll("[data-settings-panel]").forEach((panel) => {
    const active = panel.dataset.settingsPanel === section;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
  if (byId("view-settings").classList.contains("active")) {
    byId("topbarSubtitle").textContent = SETTINGS_SECTIONS[section];
    if (updateHistory) history.replaceState(null, "", `#settings/${section}`);
  }
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
    about: ["About", "Installed version and shared runtime identity"],
  };
  byId("topbarTitle").textContent = labels[name][0];
  byId("topbarSubtitle").textContent = labels[name][1];
  closeNavigation();
  if (name === "settings") {
    setSettingsSection(state.settingsSection, { updateHistory: false });
    history.replaceState(null, "", `#settings/${state.settingsSection}`);
    renderSetupReadiness();
  } else {
    history.replaceState(null, "", `#${name}`);
  }
}

function credentialStatus() {
  return state.bootstrap.runtime?.credentials || {
    storage: "",
    broadcastify: { saved: false, configured: false, username: "", password_preview: "", source: "" },
    huggingface: { saved: false, configured: false, token_preview: "", source: "" },
  };
}

function accountPool() {
  return state.bootstrap.runtime?.account_pool || {
    authorized: false,
    profiles: [],
    configured_profile_ids: [],
    quota: state.bootstrap.runtime?.archive_quota || {},
  };
}

function configuredAccountProfiles() {
  return (accountPool().profiles || []).filter((value) => value.configured);
}

function selectedArchiveProfile() {
  return byId("archiveAccountProfile")?.value || "automatic";
}

function replaceAccountOptions(select, { preserve = true } = {}) {
  if (!select) return;
  const previous = preserve ? select.value : "automatic";
  const profiles = configuredAccountProfiles();
  select.innerHTML = `<option value="automatic">Automatic — use every available account</option>${profiles.map((profile) => `<option value="${html(profile.id)}">${html(profile.label || profile.id)} · ${html(profile.id)}</option>`).join("")}`;
  select.value = [...select.options].some((option) => option.value === previous)
    ? previous
    : "automatic";
}

function renderAccountProfiles() {
  replaceAccountOptions(byId("archiveAccountProfile"));
  replaceAccountOptions(byId("scheduleAccountProfile"));
  const target = byId("accountProfileList");
  if (!target) return;
  const profiles = accountPool().profiles || [];
  if (!profiles.length) {
    target.innerHTML = '<div class="empty-compact">No account profiles are configured on this app server.</div>';
    return;
  }
  target.innerHTML = profiles.map((profile) => {
    const quota = profile.quota || {};
    const source = profile.saved ? "encrypted" : profile.session_available ? "saved session" : profile.source || "not configured";
    return `<button class="result-row selectable" type="button" data-account-profile="${html(profile.id)}"><div><strong>${html(profile.label || profile.id)}</strong><small>${html(profile.username || "No username shown")} · ${html(profile.id)} · ${html(source)}</small><small>${Number(quota.remaining) || 0}/${Number(quota.automated_limit) || 240} automated requests available</small></div><span class="status-chip${profile.configured ? " ready" : ""}">${profile.configured ? "Ready" : "Needs sign-in"}</span></button>`;
  }).join("");
}

function renderAbout() {
  const runtime = state.bootstrap.runtime || {};
  if (!byId("aboutVersion")) return;
  byId("aboutVersion").textContent = runtime.version || "Unknown";
  byId("aboutSourceCommit").textContent = runtime.source_commit || "Development source";
  byId("aboutRuntime").textContent = `${runtime.platform || "Unknown"} ${runtime.platform_release || ""} · Python ${runtime.python || "Unknown"}`;
  byId("aboutAccessScope").textContent = runtime.access_scope || "Unknown";
  byId("aboutStorage").querySelector("span").textContent = `Library: ${runtime.output_dir || "Unknown"} · Database: ${runtime.database_path || "Unknown"}`;
  const lan = runtime.lan_sync || {};
  const accounts = runtime.account_pool?.quota?.account_count || 0;
  byId("aboutLan").className = `notice${lan.sharing_enabled && lan.acquisition_queue_available ? " success" : " warning"}`;
  byId("aboutLan").querySelector("span").textContent = `${lan.sharing_enabled ? "Original-block sharing enabled" : "Original-block sharing disabled"} · ${lan.acquisition_queue_available ? "shared acquisition queue ready" : "shared queue unavailable"} · ${accounts} configured account profile${accounts === 1 ? "" : "s"}.`;
}

function catchUpFeed() {
  const feedId = byId("catchUpFeedSelect")?.value || "";
  return (state.bootstrap.feed_coverage || []).find((value) => String(value.feed_id) === feedId) || null;
}

function updateCatchUpStatus({ resetStart = false } = {}) {
  const feed = catchUpFeed();
  const notice = byId("catchUpStatus");
  if (!feed || !notice) {
    if (notice) {
      notice.className = "notice";
      notice.querySelector("strong").textContent = "Select an existing feed";
      notice.querySelector("span").textContent = "No website request is made while evaluating coverage.";
    }
    byId("clearCatchUpButton").disabled = true;
    return;
  }
  const retainedDates = (state.bootstrap.days || [])
    .filter((day) => String(day.feed_id) === String(feed.feed_id))
    .map((day) => String(day.archive_date || ""))
    .filter((value) => /^\d{4}-\d{2}-\d{2}$/.test(value))
    .sort();
  if (resetStart || !byId("catchUpStartDate").value) {
    byId("catchUpStartDate").value = feed.catch_up_start_date || feed.target_start_date || retainedDates[0] || localToday;
  }
  const start = byId("catchUpStartDate").value;
  const targetDays = start && start <= localToday
    ? Math.floor((new Date(`${localToday}T12:00:00`) - new Date(`${start}T12:00:00`)) / 86400000) + 1
    : 0;
  const readyDates = new Set((state.bootstrap.days || [])
    .filter((day) => String(day.feed_id) === String(feed.feed_id) && day.is_complete && String(day.archive_date) >= start && String(day.archive_date) <= localToday)
    .map((day) => String(day.archive_date)));
  const workDays = Math.max(0, targetDays - readyDates.size);
  const saved = Boolean(feed.catch_up_saved);
  notice.className = `notice${workDays ? " warning" : " success"}`;
  notice.querySelector("strong").textContent = workDays
    ? `${workDays} of ${targetDays} calendar days need work`
    : `${targetDays} calendar days already complete`;
  notice.querySelector("span").textContent = `${feed.feed_name} · ${start || "choose a start"} through ${localToday} · ${readyDates.size} complete locally${saved ? " · resumable catch-up saved" : ""}.`;
  byId("clearCatchUpButton").disabled = !saved;
}

function renderCatchUpFeeds() {
  const select = byId("catchUpFeedSelect");
  if (!select) return;
  const previous = select.value;
  const feeds = state.bootstrap.feed_coverage || [];
  select.innerHTML = `<option value="">Choose a retained or scheduled feed</option>${feeds.map((feed) => `<option value="${html(feed.feed_id)}">${html(feed.feed_name)} · feed ${html(feed.feed_id)}</option>`).join("")}`;
  select.value = feeds.some((feed) => String(feed.feed_id) === previous)
    ? previous
    : feeds.length === 1
      ? String(feeds[0].feed_id)
      : "";
  updateCatchUpStatus({ resetStart: true });
}

function renderCredentials() {
  const credentials = credentialStatus();
  const requestedProfileId = String(byId("loginProfileId")?.value || "default").trim().toLowerCase();
  const poolProfile = (accountPool().profiles || []).find((value) => value.id === requestedProfileId) || {};
  const savedProfile = requestedProfileId === "default"
    ? credentials.broadcastify || {}
    : (credentials.broadcastify_profiles || []).find((value) => value.id === requestedProfileId) || {};
  const broadcastify = { ...poolProfile, ...savedProfile };
  const huggingface = credentials.huggingface || {};
  if (byId("loginUsername") && !byId("loginUsername").value) {
    byId("loginUsername").value = broadcastify.username || "";
  }
  byId("forgetBroadcastifyLoginButton").disabled = !poolProfile.saved;
  byId("forgetHuggingFaceTokenButton").disabled = !huggingface.saved;

  const loginNotice = byId("loginNotice");
  if (loginNotice) {
    const ready = Boolean(state.accountVerified || poolProfile.configured || poolProfile.session_available);
    loginNotice.className = `notice${ready ? " success" : " warning"}`;
    loginNotice.querySelector("strong").textContent = state.accountVerified
      ? "Session verified"
      : poolProfile.saved
        ? "Encrypted login saved"
        : ready
          ? "Archive access configured"
          : "Sign-in needed";
    loginNotice.querySelector("span").textContent = poolProfile.saved
      ? `${broadcastify.username} · password ${broadcastify.password_preview || "saved"} · encrypted with ${credentials.storage}`
      : poolProfile.configured
        ? `${broadcastify.username || "Broadcastify login"} · configured in the server environment`
        : poolProfile.session_available
          ? "A saved website session is available, but no refresh login is stored."
          : `Enter a premium Broadcastify website login for profile ${requestedProfileId}.`;
  }

  const huggingFaceNotice = byId("huggingFaceCredentialNotice");
  if (huggingFaceNotice) {
    huggingFaceNotice.className = `notice${huggingface.configured ? " success" : " warning"}`;
    huggingFaceNotice.querySelector("strong").textContent = huggingface.saved
      ? "Encrypted read token saved"
      : huggingface.configured
        ? "Read token configured"
        : "Read token needed for gated models";
    huggingFaceNotice.querySelector("span").textContent = huggingface.saved
      ? `${huggingface.token_preview} · encrypted with ${credentials.storage}`
      : huggingface.configured
        ? "HUGGINGFACE_TOKEN or HF_TOKEN is configured in the server environment."
        : "Create a read token and accept the Community-1 model terms before its first download.";
  }

  const processingNotice = byId("processingCredentialNotice");
  if (processingNotice) {
    processingNotice.className = `notice${huggingface.configured ? " success" : " warning"}`;
    processingNotice.querySelector("strong").textContent = huggingface.configured
      ? "Hugging Face model access configured"
      : "Gated model access needs a token";
    processingNotice.querySelector("span").textContent = huggingface.configured
      ? `${huggingface.token_preview || "Token available"}; saved credentials are supplied automatically to local model jobs.`
      : "Manage an encrypted Hugging Face read token before downloading Community-1 speaker labels.";
  }
}

function setViewFromLocation() {
  const [requestedView, requestedSettingsSection] = location.hash.replace("#", "").split("/", 2);
  if (Object.hasOwn(SETTINGS_SECTIONS, requestedSettingsSection)) {
    state.settingsSection = requestedSettingsSection;
  }
  const view = requestedView in { library: 1, archive: 1, review: 1, area: 1, settings: 1, about: 1 }
    ? requestedView
    : "library";
  setView(view);
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
    applyRuntimeProcessingDefaults();
    renderMetrics();
    renderLibrary();
    renderProfiles();
    renderRuntime();
    renderArchiveQuota();
    renderAccountProfiles();
    renderCatchUpFeeds();
    renderFeedSchedules();
    renderCredentials();
    renderAbout();
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

function dayStorage(day) {
  const retained = `${bytes(day.storage_bytes)} retained`;
  const working = Number(day.working_storage_bytes || 0);
  return working > 0
    ? `${retained} · ${bytes(working)} temporary`
    : retained;
}

function renderFeedSchedules() {
  const target = byId("feedScheduleList");
  if (!target) return;
  const schedules = state.bootstrap.schedules || [];
  if (!schedules.length) {
    target.innerHTML = '<div class="empty-compact">No feed schedules yet.</div>';
    return;
  }
  target.innerHTML = schedules.map((schedule) => {
    const catchUp = schedule.backfill_start_date
      ? ` · ${schedule.recurring_catch_up ? "recurring catch-up" : "catching up once"} from ${html(schedule.backfill_start_date)}`
      : "";
    return `<div class="result-row">
      <div><strong>${html(schedule.feed_name)}</strong><small>Feed ${html(schedule.feed_id)} · daily ${html(schedule.run_time_local)} · latest ${html(schedule.lookback_days)} day${Number(schedule.lookback_days) === 1 ? "" : "s"}${catchUp}</small><small>${schedule.enabled ? "Enabled" : "Disabled"} · ${html(words(schedule.state))} · ${schedule.account_profile_id === "automatic" ? "all authorized accounts, sequentially" : `account ${html(schedule.account_profile_id || "default")}`}${schedule.message ? ` · ${html(schedule.message)}` : ""}</small></div>
      <div class="button-row"><button class="button subtle small" type="button" data-edit-schedule="${html(schedule.id)}">Edit</button><button class="button subtle small" type="button" data-delete-schedule="${html(schedule.id)}">Remove</button></div>
    </div>`;
  }).join("");
}

function renderArchiveQuota() {
  const notice = byId("archiveQuotaNotice");
  if (!notice) return;
  const quota = state.bootstrap.runtime?.archive_quota;
  if (!quota) {
    notice.className = "notice warning";
    notice.querySelector("strong").textContent = "Archive budget status unavailable";
    notice.querySelector("span").textContent = "Do not start unattended acquisition until the installation ledger is available.";
    return;
  }
  const available = Boolean(quota.available);
  notice.className = available ? "notice success" : "notice warning";
  notice.querySelector("strong").textContent = available
    ? `${quota.remaining} of ${quota.automated_limit} pooled automated archive requests available`
    : "Archive requests are paused for this installation";
  const instance = String(quota.instance_id || "").slice(0, 8);
  const next = quota.next_request_at
    ? ` Next safe request: ${new Date(quota.next_request_at).toLocaleString()}.`
    : "";
  const reason = quota.blocked_reason ? ` ${quota.blocked_reason}` : "";
  const accounts = Number(quota.account_count) || 1;
  const profileSummary = (quota.profiles || []).filter((profile) => profile.configured).map((profile) => `${profile.label || profile.id}: ${profile.quota?.remaining ?? 0}/${profile.quota?.automated_limit ?? 240}`).join(" · ");
  notice.querySelector("span").textContent = `Rolling 24 hours · ${accounts} authorized account${accounts === 1 ? "" : "s"} · ${quota.used} used · ${quota.user_reserve} total held for manual use${instance ? ` · ledger ${instance}` : ""}.${next}${reason}${profileSummary ? ` ${profileSummary}.` : ""}`;
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
      <span class="date">${html(day.archive_date)} · ${dayStorage(day)}</span>
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
  const analysisDetail = day.has_analysis
    ? `${day.incident_count || 0} incidents`
    : day.has_stale_analysis
      ? "update required"
      : "not ready";
  const stages = [
    [Boolean(day.raw_file_count || day.has_combined), "Archive audio", day.raw_file_count ? `${day.raw_file_count} source blocks` : "retained audio"],
    [day.has_combined, "Combine", day.has_combined ? "continuous timeline" : "not ready"],
    [
      day.has_transcript,
      "Transcription",
      day.has_transcript
        ? day.has_imported_transcript
          ? `${day.segment_count || 0} segments`
          : "awaiting database import"
        : day.has_stale_transcript
          ? "update required"
          : "not ready",
    ],
    [
      day.has_diarization,
      "Speaker labels",
      day.speaker_upgrade_available
        ? "fast preview · accuracy upgrade available"
        : day.has_diarization
          ? "Community-1 accuracy labels"
          : "not ready",
    ],
    [day.has_analysis, "Event analysis", analysisDetail],
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
    <div class="detail-head"><div><h2>${html(day.feed_name)}</h2><p>Feed ${html(day.feed_id)} · ${html(day.archive_date)} · ${dayStorage(day)}</p></div>
      <div class="button-row">${day.speaker_upgrade_available ? '<button class="button secondary" data-action="upgrade-speakers" title="Replace fast preview labels with Community-1 without repeating transcription.">Improve speakers</button>' : ""}
      <button class="button ${day.is_complete ? "secondary" : "primary"}" data-action="primary-day" ${actionDisabled}>${html(primaryLabel)}</button></div></div>
    <div class="notice ${day.is_complete ? "success" : day.needs_network || day.has_stale_combined || day.has_stale_transcript || day.has_stale_analysis ? "warning" : "success"}"><strong>${html(day.status)}</strong><span>${html(day.status_detail)}. Next: ${html(day.next_step)}.</span></div>
    <div class="pipeline">${stageCards(day)}</div>
    ${detail.audio_url ? `<div class="audio-block"><audio id="dayAudio" controls preload="metadata" src="${html(detail.audio_url)}"></audio><small>Retained continuous recording. Incident play buttons jump to the cited time without contacting Broadcastify.</small></div>` : ""}
    ${day.has_stale_transcript ? '<div class="notice warning"><strong>Transcript update required</strong><span>The combined recording changed. The previous transcript and its derived incidents are preserved locally but hidden until local processing updates them.</span></div>' : ""}
    ${day.has_stale_analysis ? '<div class="notice warning"><strong>Analysis update required</strong><span>Older incident claims are hidden. Finish this day to apply the current evidence rules using the retained transcript—no archive download is needed.</span></div>' : detail.summary ? `<div class="notice success"><strong>Daily brief</strong><span>${html(detail.summary)}</span></div>` : ""}
    <div class="detail-tabs"><button class="detail-tab${activeTab === "incidents" ? " active" : ""}" data-detail-tab="incidents">Incidents (${detail.incidents.length})</button><button class="detail-tab${activeTab === "transcript" ? " active" : ""}" data-detail-tab="transcript">Transcript (${day.segment_count || state.transcript.total || 0})</button></div>
    <div class="detail-panel" id="detailPanel">${activeTab === "incidents" ? incidentMarkup(detail.incidents) : transcriptMarkup()}</div>`;
}

function incidentMarkup(incidents) {
  if (!incidents.length) {
    const day = state.selectedDayDetail?.state;
    if (day?.has_stale_transcript || day?.has_stale_analysis) return '<div class="empty-compact">Previous incidents are preserved but hidden until this day is updated locally.</div>';
    return '<div class="empty-compact">No extracted incidents are saved for this day yet.</div>';
  }
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
  const day = state.selectedDayDetail?.state;
  const empty = day?.has_stale_transcript
    ? "The previous transcript is preserved but hidden until this day is updated locally."
    : "No transcript segments match this search.";
  const content = state.transcript.segments.length ? state.transcript.segments.map((segment) => `<div class="transcript-row"><time>${clock(segment.start_seconds)}</time><span class="speaker">${html(segment.speaker || "Unknown speaker")}</span><p>${html(segment.text)}</p></div>`).join("") : `<div class="empty-compact">${html(empty)}</div>`;
  return `<div class="transcript-tools"><input id="transcriptSearch" type="search" value="${html(state.transcript.query)}" placeholder="Search this transcript"><button class="button secondary small" data-action="search-transcript">Search</button></div><div class="transcript-list">${content}</div>${state.transcript.hasMore ? '<button class="button secondary wide load-more" data-action="load-transcript">Load more transcript</button>' : ""}`;
}

function renderProfiles() {
  const profiles = state.bootstrap.profiles || [];
  const current = byId("areaProfileSelect").value;
  byId("areaProfileSelect").innerHTML = '<option value="">Choose a profile</option>' + profiles.map((profile) => `<option value="${html(profile.name)}">${html(profile.name)} · ${profile.feeds.length} feeds</option>`).join("");
  if (profiles.some((profile) => profile.name === current)) byId("areaProfileSelect").value = current;
  renderAreaQueue();
}

function renderAreaQueue() {
  const profileName = byId("areaProfileSelect").value;
  const run = (state.bootstrap.area_runs || []).find((value) => value.profile_name === profileName);
  if (!profileName) {
    byId("areaQueuePanel").innerHTML = '<div class="empty-compact">Choose a saved profile to inspect its last acquisition queue.</div>';
    return;
  }
  if (!run) {
    byId("areaQueuePanel").innerHTML = '<div class="notice"><strong>No retained queue yet</strong><span>Starting it will process the saved feed order and persist every stop/resume point.</span></div>';
    return;
  }
  const items = run.items || [];
  const complete = items.filter((value) => value.status === "complete").length;
  const warning = run.status === "quota_limited" || run.status === "failed" || run.status === "partial";
  byId("areaQueuePanel").innerHTML = `<div class="notice ${warning ? "warning" : "success"}"><strong>Queue ${Number(run.id)} · ${html(words(run.status))}</strong><span>${complete}/${items.length} feeds complete · ${html(run.start_date)} through ${html(run.end_date)}${run.stop_reason ? ` · ${html(run.stop_reason)}` : ""}</span></div><div class="queue-items">${items.map((item) => `<span><b>${Number(item.priority_rank)}. ${html(item.feed_name)}</b> · ${html(words(item.status))} · ${Number(item.completed_days)}/${Number(item.requested_days)} days</span>`).join("")}</div>`;
}

function syncPlatformProfileOptions(platform) {
  const select = byId("settingHardwareProfile");
  if (!select || !platform) return;
  const isWindows = platform === "Windows";
  const isMac = platform === "Darwin" || platform === "macOS";
  const availability = {
    windowsml: isWindows,
    metal: isMac,
  };
  Object.entries(availability).forEach(([value, available]) => {
    const option = [...select.options].find((candidate) => candidate.value === value);
    if (!option) return;
    option.hidden = !available;
    option.disabled = !available;
  });
  const selected = select.selectedOptions[0];
  if (selected?.disabled) applyHardwareProfile("auto", false);
}

function renderRuntime() {
  const runtime = state.bootstrap.runtime || {};
  syncPlatformProfileOptions(runtime.platform);
  const accessStatus = byId("accessScopeStatus");
  if (accessStatus) {
    accessStatus.innerHTML = `<span class="status-dot"></span>${runtime.loopback_only ? "Loopback only" : "Trusted LAN"}`;
    accessStatus.title = runtime.loopback_only
      ? "Only this computer can open the app."
      : "The app is available to devices on the trusted local network.";
  }
  byId("runtimeDescription").textContent = `${runtime.platform || "Unknown"} ${runtime.platform_release || ""} · Python ${runtime.python || ""}`;
  byId("runtimeFacts").innerHTML = [
    ["Access", runtime.loopback_only ? "This computer only" : "Network"],
    ["Archive root", runtime.output_dir || "archives"],
    ["Evidence database", runtime.database_path || ""],
    ["Default behavior", runtime.platform === "Windows" ? "Tested Windows automatic profile" : "Portable automatic detection"],
  ].map(([label, value]) => `<div class="runtime-fact"><span>${html(label)}</span><strong>${html(value)}</strong></div>`).join("");
  const lan = runtime.lan_sync || {};
  const lanNotice = byId("lanSyncNotice");
  if (lanNotice) {
    const serving = Boolean(lan.sharing_enabled);
    const coordinating = Boolean(lan.acquisition_queue_available);
    lanNotice.className = `notice ${serving ? "success" : ""}`.trim();
    lanNotice.querySelector("strong").textContent = serving
      ? "This app is a LAN archive peer and queue coordinator"
      : "LAN reuse is client-side only here";
    lanNotice.querySelector("span").textContent = serving
      ? `Original source blocks are available to trusted-LAN clients${lan.key_required ? " that have the shared key" : ""}. ${coordinating ? "Feed/day leases prevent duplicate upstream downloads. " : ""}${lan.discovery_available ? "Automatic discovery is active." : "Use this app URL as an explicit peer."}`
      : "This client can still reuse blocks from discovered or configured peers. Serving local blocks requires BROADCASTIFY_LAN_SHARING=true when the app starts.";
  }
  renderHardwareProfiles();
  renderSetupReadiness();
}

function selectedHardwareProfile() {
  const profiles = state.hardwareDiagnostics?.accelerators?.profiles || [];
  const selected = byId("settingHardwareProfile")?.value || state.settings.hardwareProfile;
  return profiles.find((profile) => profile.id === selected) || null;
}

function currentProfileAction() {
  if (state.profileSelfTest && !state.profileSelfTest.ready) {
    return state.profileSelfTest.recovery || null;
  }
  return selectedHardwareProfile()?.next_action || null;
}

function renderProfileNextAction() {
  const button = byId("profileNextActionButton");
  if (!button) return;
  const action = currentProfileAction();
  const show = Boolean(action?.kind && action.kind !== "verify-profile");
  button.hidden = !show;
  button.textContent = show ? (action.label || "Complete setup") : "Complete setup";
  button.title = show ? (action.message || "") : "";
  if (show && !state.profileSelfTest) {
    const notice = byId("profileSelfTestNotice");
    notice.className = "notice warning";
    notice.querySelector("strong").textContent = action.label || "Profile setup is incomplete";
    notice.querySelector("span").textContent = action.message || "Complete the next setup step, then verify this profile.";
  }
}

function renderSetupReadiness() {
  const region = byId("setupReadinessGrid");
  if (!region) return;
  const runtime = state.bootstrap.runtime || {};
  const account = runtime.account || {};
  const profile = selectedHardwareProfile();
  const profileAction = currentProfileAction();
  const speakerLabels = state.hardwareDiagnostics?.accelerators?.speaker_labels || {};
  const savedModelAccess = Boolean(runtime.credentials?.huggingface?.configured);
  const tokenInTab = Boolean(byId("settingHuggingFaceToken")?.value);
  const selectedDiarization = byId("settingDiarizationDevice")?.value || state.settings.diarizationDevice;
  const selectedDiarizationEngine = byId("settingDiarizationEngine")?.value || state.settings.diarizationEngine;
  const tokenConfigured = Boolean(speakerLabels.token_configured || savedModelAccess || tokenInTab);
  const speakerAccessConfigured = Boolean(speakerLabels.access_configured || tokenConfigured);
  const speakerConfigured = selectedDiarizationEngine === "sherpa-onnx"
    ? Boolean(speakerLabels.portable?.ready)
    : Boolean(
      speakerLabels.package_installed
      && speakerAccessConfigured
      && (selectedDiarization !== "cuda" || speakerLabels.cuda_available)
    );
  const accountReady = Boolean(state.accountVerified || account.configured);
  const storageReady = Boolean(runtime.storage_ready);
  const transcriptionReady = Boolean(state.asrSelfTest?.ready || profile?.transcription_ready);
  const diarizationReady = Boolean(
    state.diarizationSelfTest?.ready || speakerConfigured
  );
  const provider = byId("settingAnalysisProvider")?.value || state.settings.analysisProvider;
  const localAnalysisDetected = provider === "local" && Boolean(profile?.analysis_ready);
  const analysisConfigured = state.analysisProviderStatus
    ? Boolean(state.analysisProviderStatus.ready)
    : localAnalysisDetected;
  const analysisVerified = Boolean(
    state.analysisSelfTest?.ready && state.analysisSelfTest?.verified
  );

  const items = [
    {
      id: "account", icon: "◉", title: "Archive account", available: accountReady,
      state: state.accountVerified ? "Verified" : accountReady ? "Configured" : "Sign in",
      detail: state.accountVerified
        ? "Premium website session refreshed in this browser session."
        : accountReady
          ? "An encrypted login, environment login, or saved website session is available."
          : "A premium Broadcastify website login is needed for archives.",
      action: accountReady ? "Review" : "Sign in",
    },
    {
      id: "storage", icon: "▣", title: "Storage", available: storageReady,
      state: storageReady ? "Ready" : "Needs attention",
      detail: storageReady ? `Writable library: ${runtime.output_dir || "archives"}` : "The archive library path is not writable.",
      action: "Review path",
    },
    {
      id: "transcription", icon: "≋", title: "Transcription", available: transcriptionReady,
      state: state.asrSelfTest?.ready ? "Verified" : transcriptionReady ? "Detected" : state.hardwareDiagnostics ? "Setup needed" : "Check",
      detail: state.asrSelfTest?.message || (!transcriptionReady && profileAction?.stage === "transcription" ? profileAction.message : profile?.transcription) || "Check this computer to choose a local transcription path.",
      action: state.asrSelfTest?.ready ? "Retest" : transcriptionReady ? "Test" : profileAction?.stage === "transcription" ? profileAction.label : "Check",
    },
    {
      id: "diarization", icon: "◎", title: "Speaker labels", available: diarizationReady,
      state: state.diarizationSelfTest?.ready ? "Verified" : diarizationReady ? "Configured" : state.hardwareDiagnostics ? "Setup needed" : "Check",
      detail: state.diarizationSelfTest?.message || (!diarizationReady && profileAction?.stage === "diarization" ? profileAction.message : profile?.diarization) || "Check the selected accuracy or fast-preview speaker engine.",
      action: state.diarizationSelfTest?.ready ? "Retest" : diarizationReady ? "Test" : profileAction?.stage === "diarization" ? profileAction.label : "Configure",
    },
    {
      id: "analysis", icon: "✦", title: "Event analysis", available: analysisConfigured,
      state: analysisVerified ? "Verified" : analysisConfigured ? "Configured" : state.hardwareDiagnostics ? "Check provider" : "Check",
      detail: state.analysisSelfTest?.message || state.analysisProviderStatus?.message || (!analysisConfigured && profileAction?.stage === "analysis" ? profileAction.message : localAnalysisDetected ? profile.analysis : "Check the selected local, API, endpoint, or Codex provider."),
      action: analysisVerified ? "Retest" : analysisConfigured ? "Test" : profileAction?.stage === "analysis" ? profileAction.label : "Check",
    },
  ];
  const available = items.filter((item) => item.available).length;
  const verified = [
    state.accountVerified,
    state.asrSelfTest?.ready,
    state.diarizationSelfTest?.ready,
    analysisVerified,
  ].filter(Boolean).length;
  byId("setupReadinessSummary").textContent = `${available} of 5 setup steps available · ${verified} execution check${verified === 1 ? "" : "s"} verified this session`;
  byId("setupProgressBar").style.width = `${available * 20}%`;
  region.innerHTML = items.map((item) => `
    <article class="setup-item${item.available ? " available" : ""}">
      <span class="setup-icon" aria-hidden="true">${item.icon}</span>
      <div class="setup-copy"><strong>${html(item.title)}</strong><span>${html(item.state)}</span></div>
      <p>${html(item.detail)}</p>
      <button class="button secondary small" data-setup-action="${html(item.id)}">${html(item.action)}</button>
    </article>`).join("");

  renderCredentials();
}

function renderHardwareProfiles() {
  const profiles = state.hardwareDiagnostics?.accelerators?.profiles || [];
  const region = byId("runtimeProfiles");
  if (!profiles.length) {
    region.innerHTML = '<div class="runtime-profile-empty">Run the hardware check to compare stage-by-stage profiles.</div>';
    renderProfileNextAction();
    return;
  }
  const selectedId = byId("settingHardwareProfile")?.value || state.settings.hardwareProfile;
  const configured = profiles.filter((profile) => Boolean(
    profile.configured ?? (
      profile.transcription_ready && profile.diarization_ready && profile.analysis_ready
    )
  )).length;
  const executionVerified = Boolean(
    state.asrSelfTest?.ready
    && state.diarizationSelfTest?.ready
    && state.analysisSelfTest?.ready
    && state.analysisSelfTest?.verified
  );
  region.innerHTML = `<details class="runtime-profile-details"><summary><span>Compare hardware profiles</span><strong>${configured} of ${profiles.length} detected</strong></summary><div class="runtime-profile-grid">${profiles.map((profile) => {
    const isConfigured = Boolean(profile.configured ?? (
      profile.transcription_ready && profile.diarization_ready && profile.analysis_ready
    ));
    const isSelected = profile.id === selectedId;
    const isVerified = isSelected && isConfigured && executionVerified;
    const profileState = isVerified ? "Verified now" : isConfigured ? "Detected" : "Setup needed";
    const nextSteps = isVerified
      ? []
      : Array.isArray(profile.next_steps) ? profile.next_steps : [];
    return `
    <article class="runtime-profile-card${isConfigured ? " configured" : ""}${isVerified ? " ready" : ""}${isSelected ? " selected" : ""}">
      <div class="runtime-profile-head"><strong>${html(profile.name)}</strong><span>${profileState}</span></div>
      <div class="runtime-stage"><b>Transcribe</b><span>${html(profile.transcription)}</span></div>
      <div class="runtime-stage"><b>Speakers</b><span>${html(profile.diarization)}</span></div>
      <div class="runtime-stage"><b>Analyze</b><span>${html(profile.analysis)}</span></div>
      ${profile.note ? `<p>${html(profile.note)}</p>` : ""}
      ${nextSteps.length ? `<ul class="runtime-next-steps">${nextSteps.map((step) => `<li>${html(step)}</li>`).join("")}</ul>` : ""}
      <button class="button secondary small" data-use-hardware-profile="${html(profile.id)}">Use this profile</button>
    </article>`;
  }).join("")}</div></details>`;
  renderProfileNextAction();
  updateAsrModelPreparationUi();
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
    state.jobFailureCallback = options.onFailure || null;
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
      const failureCallback = state.jobFailureCallback;
      state.activeJob = null;
      state.jobCallback = null;
      state.jobFailureCallback = null;
      byId("cancelJobButton").disabled = true;
      byId("jobDrawer").classList.remove("open");
      byId("jobDrawerToggle").setAttribute("aria-expanded", "false");
      if (job.status === "completed") {
        const handled = callback ? await callback(job) : undefined;
        if (handled !== false) toast(`${job.command.replaceAll("-", " ")} completed.`);
      } else {
        const handled = failureCallback ? await failureCallback(job) : undefined;
        if (handled !== false) toast(job.error || `${job.command} ${job.status}.`, true);
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

async function queueAreaAnalyses(feedResults) {
  state.analysisQueue = feedResults.flatMap((value) =>
    (value.result?.days || [])
      .filter((day) => (day.transcripts || []).length)
      .map((day) => ({ feedId: value.feed?.feed_id || value.result?.feed_id, archiveDate: day.date })))
    .filter((value) => value.feedId && value.archiveDate);
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
  const coverage = result.coverage || {};
  const coverageMarkup = Number(coverage.requested_day_count)
    ? `<div class="notice ${coverage.complete_coverage ? "success" : "warning"}"><strong>${Number(coverage.question_ready_day_count) || 0}/${Number(coverage.requested_day_count) || 0} days question-ready</strong><span>${html(coverage.summary || "")}</span></div>`
    : "";
  byId("answerPanel").innerHTML = `<h3>Evidence-grounded answer</h3>${coverageMarkup}<p>${html(result.answer || "No answer text was returned.")}</p>${evidence.length ? `<div class="citation-list">${evidence.map((value) => `<div class="citation">Evidence ${html(value)}</div>`).join("")}</div>` : ""}${limitations.length ? `<div class="notice warning"><strong>Limitations</strong><span>${html(limitations.join("; "))}</span></div>` : ""}`;
}

function archiveJobResult(job) {
  return eventOf(job, "complete")?.result
    || eventOf(job, "area_complete")?.result
    || eventOf(job, "scheduled_complete")?.result
    || {};
}

function hasUntriedAccount(attempted) {
  return configuredAccountProfiles().some((profile) => !attempted.includes(profile.id));
}

async function startPooledArchiveJob(command, payload, options = {}, attempted = []) {
  const requested = payload.account_profile_id || selectedArchiveProfile();
  const automatic = requested === "automatic";
  const runPayload = {
    ...payload,
    account_profile_id: requested,
    ...(automatic ? { exclude_account_profile_ids: attempted } : {}),
  };
  return startJob(command, runPayload, {
    ...options,
    onComplete: async (job) => {
      const nextAttempted = [...new Set([...attempted, job.account_profile_id].filter(Boolean))];
      const result = archiveJobResult(job);
      if (automatic && result.download_limited && hasUntriedAccount(nextAttempted)) {
        toast(`Account ${job.account_profile_id} reached its rolling boundary; continuing sequentially with the next authorized account.`);
        await startPooledArchiveJob(command, payload, options, nextAttempted);
        return false;
      }
      return options.onComplete ? options.onComplete(job) : undefined;
    },
    onFailure: async (job) => {
      const nextAttempted = [...new Set([...attempted, job.account_profile_id].filter(Boolean))];
      const credentialFailure = /auth|credential|forbidden|login|premium|unauthori[sz]ed/i.test(job.error || "");
      if (automatic && credentialFailure && hasUntriedAccount(nextAttempted)) {
        toast(`Account ${job.account_profile_id} could not authenticate; trying the next authorized account.`);
        await startPooledArchiveJob(command, payload, options, nextAttempted);
        return false;
      }
      return options.onFailure ? options.onFailure(job) : undefined;
    },
  });
}

function renderWeek(result) {
  if (!result) {
    byId("weekPanel").innerHTML = '<div class="empty-compact">No saved summary covers this exact seven-day window.</div>';
    return;
  }
  const missing = result.missing_dates || [];
  const updates = result.analysis_update_dates || [];
  const notable = result.notable_incident_ids || [];
  const coverageText = missing.length ? ` · Missing ${missing.join(", ")}` : " · Complete date coverage";
  const updateText = updates.length ? ` · Reanalysis needed ${updates.join(", ")}` : "";
  byId("weekPanel").innerHTML = `<h3>${html(result.start_date)} through ${html(result.end_date)}</h3><p>${html(result.summary || "")}</p><div class="notice ${missing.length || updates.length ? "warning" : "success"}"><strong>${Number(result.days_available) || 0}/7 days available</strong><span>${Number(result.incident_count) || 0} retained incidents${html(coverageText)}${html(updateText)}</span></div>${notable.length ? `<div class="citation-list">${notable.map((value) => `<div class="citation">Notable record I${html(value)}</div>`).join("")}</div>` : ""}`;
}

function renderAreaBrief(result) {
  state.areaBriefResult = result;
  state.areaSelectedStoryIndex = result?.stories?.length ? 0 : -1;
  renderAreaBriefContent();
}

function areaEvidenceMarkup(story, open = false) {
  const references = story.incident_references || [];
  if (!references.length) return '<div class="notice warning"><strong>No source package</strong><span>This lead should not be published until retained evidence is available.</span></div>';
  const clipCount = references.filter((reference) => reference.media_url).length;
  const referenceMarkup = references.map((reference) => {
    const provenance = reference.source_audio_sha256 ? ` · audio ${String(reference.source_audio_sha256).slice(0, 12)}…` : "";
    const clipMarkup = reference.media_url
      ? `<audio controls preload="none" src="${html(reference.media_url)}"></audio><a class="button secondary small" href="${html(reference.media_url)}" download="${html(reference.filename || `incident-I${reference.incident_id}.mp3`)}">Download exact clip</a>`
      : '<span class="evidence-unavailable">Transcript evidence only; no retained clip is available.</span>';
    return `<div class="evidence-reference">
      <div class="evidence-source">${html(reference.feed_name || `Feed ${reference.feed_id}`)} · I${html(reference.incident_id)} · ${html(reference.archive_time || reference.archive_date || "")} · ${Math.round((Number(reference.confidence) || 0) * 100)}% extraction confidence${html(provenance)}</div>
      ${reference.quote ? `<blockquote class="evidence-quote">“${html(reference.quote)}”</blockquote>` : '<div class="evidence-unavailable">No display quote is available.</div>'}
      <div class="evidence-actions">${clipMarkup}</div>
    </div>`;
  }).join("");
  return `<details class="evidence-package"${open ? " open" : ""}><summary><span>Evidence package</span><small>${references.length} source record${references.length === 1 ? "" : "s"} · ${clipCount} exact clip${clipCount === 1 ? "" : "s"}</small></summary><div class="evidence-body">${referenceMarkup}</div></details>`;
}

function areaStoryIndexMarkup(stories) {
  return stories.map((story, index) => {
    const selected = index === state.areaSelectedStoryIndex;
    const evidenceCount = (story.incident_references || []).length;
    const reported = story.first_reported || story.location || "Time or place unavailable";
    return `<button type="button" class="story-index-item${selected ? " active" : ""}" data-area-story-index="${index}" aria-pressed="${selected}">
      <span class="story-rank">#${index + 1}</span>
      <span class="story-index-copy">
        <strong>${html(story.headline || "Untitled story lead")}</strong>
        <small>${html(story.interest_level || "Lead")} · score ${Number(story.newsworthiness_score) || 0} · P${Number(story.priority) || 0}</small>
        <small>${html(reported)} · ${evidenceCount} source record${evidenceCount === 1 ? "" : "s"}</small>
      </span>
    </button>`;
  }).join("");
}

function areaStoryDetailMarkup(story, index) {
  if (!story) return '<div class="empty-compact">Choose a ranked lead to inspect its complete evidence package.</div>';
  const tags = [...(story.neighborhood_tags || []), ...(story.topic_tags || [])];
  const meta = [story.first_reported, story.location].filter(Boolean).join(" · ");
  return `<button class="story-back button secondary small" type="button" data-action="focus-story-index">← Ranked leads</button>
    <div class="story-detail-head">
      <div>
        <span class="story-score">${html(story.interest_level || "Lead")} · score ${Number(story.newsworthiness_score) || 0} · P${Number(story.priority) || 0}</span>
        <h3>${html(story.headline || "Untitled story lead")}</h3>
      </div>
      <span class="story-detail-rank">#${index + 1}</span>
    </div>
    ${meta ? `<p class="story-detail-meta">${html(meta)}</p>` : ""}
    <p>${html(story.summary || "")}</p>
    <div class="story-why"><strong>Why this surfaced</strong><span>${html(story.why_interesting || "")}</span></div>
    <div class="tag-list">${tags.map((tag) => `<span class="tag">${html(words(tag))}</span>`).join("")}</div>
    <p class="story-audience">${story.subscription_eligible ? "Neighborhood-ready after editor verification." : "Not eligible for neighborhood alerts without additional location or confidence."}</p>
    ${areaEvidenceMarkup(story, true)}`;
}

function renderAreaStorySelection({ reveal = false } = {}) {
  const stories = state.areaBriefResult?.stories || [];
  if (!stories.length) return;
  state.areaSelectedStoryIndex = Math.max(0, Math.min(
    Number(state.areaSelectedStoryIndex) || 0,
    stories.length - 1,
  ));
  document.querySelectorAll("[data-area-story-index]").forEach((button) => {
    const selected = Number(button.dataset.areaStoryIndex) === state.areaSelectedStoryIndex;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  const detail = byId("areaStoryDetail");
  if (!detail) return;
  detail.innerHTML = areaStoryDetailMarkup(
    stories[state.areaSelectedStoryIndex],
    state.areaSelectedStoryIndex,
  );
  detail.scrollTop = 0;
  if (reveal) detail.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderAreaBriefContent() {
  const result = state.areaBriefResult;
  if (!result) {
    byId("areaBriefPanel").innerHTML = '<div class="empty-compact">No saved story-lead brief exists for this profile yet.</div>';
    return;
  }
  const coverage = result.coverage || {};
  const stories = result.stories || [];
  const staleFeedDays = coverage.stale_feed_days || [];
  state.areaSelectedStoryIndex = stories.length
    ? Math.max(0, Math.min(Number(state.areaSelectedStoryIndex) || 0, stories.length - 1))
    : -1;
  const staleText = staleFeedDays.length ? ` · ${staleFeedDays.length} retained feed-days need reanalysis` : "";
  const browser = stories.length ? `<div class="story-browser">
      <aside class="story-index" aria-label="Ranked story leads">
        <div class="story-index-head"><strong>${stories.length} ranked leads</strong><span>Select one to audit</span></div>
        <div class="story-index-list">${areaStoryIndexMarkup(stories)}</div>
      </aside>
      <article class="story-detail" id="areaStoryDetail" aria-live="polite">${areaStoryDetailMarkup(stories[state.areaSelectedStoryIndex], state.areaSelectedStoryIndex)}</article>
    </div>` : '<div class="empty-compact">No story leads met the saved threshold.</div>';
  byId("areaBriefPanel").innerHTML = `
    <div class="notice ${Number(coverage.feeds_with_data) < Number(coverage.feed_count) || staleFeedDays.length ? "warning" : "success"}">
      <strong>${html(result.start_date)} through ${html(result.end_date)}</strong>
      <span>${Number(coverage.feeds_with_data) || 0}/${Number(coverage.feed_count) || 0} feeds with data · ${Number(coverage.feed_days_available) || 0}/${Number(coverage.feed_days_expected) || 0} feed-days · ${Number(coverage.incident_count) || 0} incidents${html(staleText)}</span>
    </div>
    ${result.summary ? `<details class="area-narrative"><summary>Generated assignment brief</summary><p>${html(result.summary)}</p></details>` : ""}
    ${browser}`;
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
  if (results !== state.areaDiscovered) {
    state.areaDiscovered = results;
    const available = new Set(results.map((feed) => String(feed.feed_id)));
    state.areaSelectedFeedIds = new Set([...state.areaSelectedFeedIds].filter((feedId) => available.has(feedId)));
  }
  const visible = state.areaDiscovered
    .map((feed, index) => ({ feed, index }))
    .filter(({ feed }) => !byId("areaPublicSafetyOnly").checked || String(feed.genre || "").toLowerCase() === "public safety");
  if (!visible.length) {
    byId("areaSearchResults").innerHTML = '<div class="empty-compact">No public-safety feeds were found for those ZIPs.</div>';
    return;
  }
  const selectedCount = state.areaSelectedFeedIds.size;
  byId("areaSearchResults").innerHTML = `<div class="result-summary"><span id="areaSelectionSummary">Showing ${visible.length} of ${state.areaDiscovered.length} discovered feeds · ${selectedCount} selected · no archive audio requested</span><span class="result-summary-actions"><button type="button" data-area-selection="nearest">Nearest 3</button><button type="button" data-area-selection="clear">Clear</button></span></div>` + visible.map(({ feed, index }) => {
    const priority = Number(feed.priority_rank) || index + 1;
    const distance = feed.distance_miles == null ? `ZIP ${html(feed.nearest_zip_code || "priority")}` : `about ${Number(feed.distance_miles).toFixed(1)} mi`;
    const checked = state.areaSelectedFeedIds.has(String(feed.feed_id)) ? " checked" : "";
    return `<label class="feed-result"><input type="checkbox" data-area-feed-index="${index}"${checked}><div><h3>${html(feed.name || `Feed ${feed.feed_id}`)}</h3><p>Priority ${priority} · ${distance} · Feed ${html(feed.feed_id)}${feed.location ? ` · ${html(feed.location)}` : ""}</p></div><span class="listener-count">${Number(feed.listeners) || 0} listeners</span></label>`;
  }).join("");
}

function updateAreaCoverageControls() {
  const radiusMode = byId("areaCoverageMode").value === "radius";
  byId("areaRadiusMiles").disabled = !radiusMode;
  byId("areaMaxZipCodes").disabled = !radiusMode;
  byId("areaZipLabel").textContent = radiusMode ? "Center ZIP" : "ZIPs in priority order";
  byId("areaZipCodes").placeholder = radiusMode ? "5-digit ZIP" : "Comma-separated ZIPs";
  byId("areaSearchForm").querySelector('button[type="submit"]').textContent = radiusMode ? "Discover nearest feeds" : "Discover feeds";
}

function applySelectedAreaProfile() {
  const profile = (state.bootstrap.profiles || []).find((value) => value.name === byId("areaProfileSelect").value);
  if (!profile) return;
  const coverage = profile.coverage || { mode: "zip-list" };
  state.areaCoverage = coverage;
  byId("areaCoverageMode").value = coverage.mode === "radius" ? "radius" : "zip-list";
  byId("areaZipCodes").value = coverage.mode === "radius"
    ? (coverage.center_zip || profile.zip_codes?.[0] || "")
    : (profile.zip_codes || []).join(", ");
  byId("areaRadiusMiles").value = coverage.radius_miles || 25;
  byId("areaMaxZipCodes").value = coverage.max_zip_codes || Math.min(20, profile.zip_codes?.length || 12);
  byId("areaProfileName").value = profile.name;
  state.areaDiscovered = profile.feeds || [];
  state.areaSelectedFeedIds = new Set(state.areaDiscovered.map((feed) => String(feed.feed_id)));
  renderAreaResults(state.areaDiscovered);
  updateAreaCoverageControls();
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
    if (payload.stale) {
      byId("areaBriefPanel").innerHTML = '<div class="notice warning"><strong>Saved brief needs an evidence update</strong><span>This brief predates the current incident evidence rules, so its story claims are hidden. Reanalyze retained days, then find story leads again.</span></div>';
    }
  } catch (error) { toast(error.message, true); }
}

document.addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.view) {
    if (button.dataset.settingsSectionTarget) {
      state.settingsSection = button.dataset.settingsSectionTarget;
    }
    return setView(button.dataset.view);
  }
  if (button.dataset.accountProfile) {
    const profile = (accountPool().profiles || []).find((value) => value.id === button.dataset.accountProfile);
    if (!profile) return;
    byId("loginProfileId").value = profile.id;
    byId("loginProfileLabel").value = profile.label || profile.id;
    byId("loginUsername").value = profile.username || "";
    state.accountVerified = false;
    renderCredentials();
    byId("loginPassword").focus();
    return;
  }
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
      state.editingSchedule = null;
      byId("saveFeedScheduleButton").textContent = "Schedule selected feed daily";
      byId("archiveFeedId").value = state.selectedFeed.feed_id;
      byId("selectedFeedLabel").textContent = `${state.selectedFeed.name} · feed ${state.selectedFeed.feed_id}`;
      renderFeedResults(results);
    }
    return;
  }
  if (button.dataset.areaStoryIndex !== undefined) {
    state.areaSelectedStoryIndex = Number(button.dataset.areaStoryIndex) || 0;
    renderAreaStorySelection({
      reveal: window.matchMedia("(max-width: 560px)").matches,
    });
    return;
  }
  const action = button.dataset.action;
  if (action === "manage-huggingface") {
    state.settingsSection = "account";
    setView("settings");
    requestAnimationFrame(() => byId("settingHuggingFaceToken").focus());
    return;
  }
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
  if (action === "focus-story-index") {
    const selected = document.querySelector(
      `[data-area-story-index="${state.areaSelectedStoryIndex}"]`,
    );
    selected?.scrollIntoView({ behavior: "smooth", block: "center" });
    selected?.focus({ preventScroll: true });
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
  if (action === "upgrade-speakers") {
    const day = state.selectedDayDetail?.state;
    if (!day?.speaker_upgrade_available) return;
    await startJob(
      "continue-local",
      {
        feed_id: day.feed_id,
        archive_date: day.archive_date,
        diarize: true,
        analyze: true,
        ...processingPayload(),
        diarization_engine: "community-1",
        ...providerPayload(),
      },
      {
        label: `Improving speaker labels for ${day.archive_date}`,
        onComplete: refreshBootstrap,
      },
    );
  }
});

byId("menuButton").addEventListener("click", () => {
  const open = document.querySelector(".sidebar").classList.toggle("open");
  byId("menuButton").setAttribute("aria-expanded", String(open));
});
byId("sidebarScrim").addEventListener("click", closeNavigation);
byId("refreshLibraryButton").addEventListener("click", () => refreshBootstrap());
byId("librarySearch").addEventListener("input", renderLibrary);
byId("libraryFilter").addEventListener("change", renderLibrary);
byId("saveSettingsButton").addEventListener("click", saveSettings);
byId("settingAnalysisProvider").addEventListener("change", () => {
  const defaults = { local: "ggml-org/gemma-4-12B-it-GGUF:Q4_0", "openai-responses": "gpt-5.6-luna", "openai-compatible": "", "codex-cli": "" };
  byId("settingAnalysisModel").value = defaults[byId("settingAnalysisProvider").value] || "";
  state.analysisProviderStatus = null;
  resetProfileVerification();
  resetAnalysisVerification();
  updateProviderNotice();
  renderHardwareProfiles();
  renderSetupReadiness();
});
["settingAnalysisModel", "settingAnalysisEndpoint", "settingApiKeyEnvironment", "settingCodexPath", "settingAnalysisApiKey"].forEach((id) => byId(id).addEventListener("input", () => {
  state.analysisProviderStatus = null;
  resetProfileVerification();
  resetAnalysisVerification();
  updateProviderNotice();
  renderHardwareProfiles();
  renderSetupReadiness();
}));
byId("settingAllowExternal").addEventListener("change", () => {
  state.analysisProviderStatus = null;
  resetProfileVerification();
  resetAnalysisVerification();
  updateProviderNotice();
  renderHardwareProfiles();
  renderSetupReadiness();
});
byId("settingHuggingFaceToken").addEventListener("input", () => {
  resetProfileVerification();
  resetAsrVerification();
  resetDiarizationVerification();
  renderHardwareProfiles();
  renderSetupReadiness();
});
byId("settingWhisperModel").addEventListener("change", () => {
  resetProfileVerification();
  resetAsrVerification();
  updateAsrModelPreparationUi();
  renderHardwareProfiles();
  renderSetupReadiness();
});
byId("settingAsrModelPath").addEventListener("input", () => {
  resetProfileVerification();
  resetAsrVerification();
  renderHardwareProfiles();
  renderSetupReadiness();
});
byId("settingBatchSize").addEventListener("input", () => {
  resetProfileVerification();
  resetAsrVerification();
  resetDiarizationVerification();
  renderHardwareProfiles();
  renderSetupReadiness();
});
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
  await startJob("search", { query: byId("feedSearchInput").value, account_profile_id: selectedArchiveProfile() }, { label: "Searching website feeds", onComplete: (job) => renderFeedResults(eventOf(job, "result")?.results || []) });
});

byId("archiveDiarize").addEventListener("change", () => {
  if (byId("archiveDiarize").checked) {
    byId("archiveCombine").checked = true;
    byId("archiveTranscribe").checked = true;
  }
});
byId("catchUpFeedSelect").addEventListener("change", () => updateCatchUpStatus({ resetStart: true }));
byId("catchUpStartDate").addEventListener("input", () => updateCatchUpStatus());
byId("catchUpMissingDaysButton").addEventListener("click", async () => {
  const feed = catchUpFeed();
  const startDate = byId("catchUpStartDate").value;
  if (!feed) return toast("Choose an existing retained or scheduled feed first.", true);
  if (!startDate || startDate > localToday) return toast("Choose a catch-up start date no later than today.", true);
  try {
    if (byId("catchUpSaveForResume").checked) {
      await api("/api/catchups", {
        method: "POST",
        body: JSON.stringify({ action: "save", feed_id: feed.feed_id, feed_name: feed.feed_name, start_date: startDate }),
      });
    } else if (feed.catch_up_saved) {
      await api("/api/catchups", {
        method: "POST",
        body: JSON.stringify({ action: "clear", feed_id: feed.feed_id }),
      });
    }
  } catch (error) {
    toast(error.message, true);
    return;
  }
  const analyze = byId("archiveAnalyze").checked;
  await startPooledArchiveJob("run", {
    feed_id: feed.feed_id,
    feed_name: feed.feed_name,
    start_date: startDate,
    end_date: localToday,
    combine: byId("archiveCombine").checked,
    transcribe: byId("archiveTranscribe").checked,
    diarize: byId("archiveDiarize").checked,
    account_profile_id: selectedArchiveProfile(),
    ...processingPayload(),
  }, {
    label: `Catching up ${feed.feed_name} through today`,
    onComplete: async (job) => {
      await refreshBootstrap({ preserveSelection: false });
      const result = archiveJobResult(job);
      const transcriptDates = (result.days || []).filter((day) => (day.transcripts || []).length).map((day) => day.date);
      if (analyze && transcriptDates.length) await queueAnalyses(feed.feed_id, transcriptDates);
    },
  });
});
byId("clearCatchUpButton").addEventListener("click", async () => {
  const feed = catchUpFeed();
  if (!feed?.catch_up_saved) return;
  try {
    await api("/api/catchups", {
      method: "POST",
      body: JSON.stringify({ action: "clear", feed_id: feed.feed_id }),
    });
    await refreshBootstrap();
    toast(`Cleared the saved one-time catch-up for ${feed.feed_name}.`);
  } catch (error) {
    toast(error.message, true);
  }
});
byId("archiveForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const feedId = byId("archiveFeedId").value;
  if (!feedId) return toast("Select a feed from the search results first.", true);
  const analyze = byId("archiveAnalyze").checked;
  await startPooledArchiveJob("run", {
    feed_id: feedId,
    feed_name: state.selectedFeed?.name || "",
    start_date: byId("archiveStartDate").value,
    end_date: byId("archiveEndDate").value,
    combine: byId("archiveCombine").checked,
    transcribe: byId("archiveTranscribe").checked,
    diarize: byId("archiveDiarize").checked,
    account_profile_id: selectedArchiveProfile(),
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

byId("saveFeedScheduleButton").addEventListener("click", async () => {
  const feedId = byId("archiveFeedId").value;
  if (!feedId) return toast("Select the feed to schedule first.", true);
  const lookback = Math.max(1, Math.min(14, Number(byId("scheduleLookbackDays").value) || 2));
  const editing = state.editingSchedule?.feed_id === feedId
    ? state.editingSchedule
    : null;
  const job = editing
    ? {
        ...(editing.job || {}),
        combine: byId("archiveCombine").checked,
        transcribe: byId("archiveTranscribe").checked,
        diarize: byId("archiveDiarize").checked,
      }
    : {
        combine: byId("archiveCombine").checked,
        transcribe: byId("archiveTranscribe").checked,
        diarize: byId("archiveDiarize").checked,
        ...processingPayload(),
        ...providerPayload(),
      };
  try {
    await api("/api/schedules", {
      method: "POST",
      body: JSON.stringify({
        feed_id: feedId,
        feed_name: state.selectedFeed?.name || `Feed ${feedId}`,
        run_time_local: byId("scheduleRunTime").value || "02:00",
        lookback_days: lookback,
        backfill_start_date: byId("scheduleBackfillStartDate").value || "",
        recurring_catch_up: byId("scheduleRecurringCatchUp").checked,
        account_profile_id: byId("scheduleAccountProfile").value || "automatic",
        analyze: byId("archiveAnalyze").checked,
        enabled: byId("scheduleEnabled").checked,
        job,
      }),
    });
    state.editingSchedule = null;
    byId("saveFeedScheduleButton").textContent = "Schedule selected feed daily";
    await refreshBootstrap();
    toast(`Daily schedule saved for ${state.selectedFeed?.name || `feed ${feedId}`}.`);
  } catch (error) {
    toast(error.message, true);
  }
});

byId("feedScheduleList").addEventListener("click", async (event) => {
  const editButton = event.target.closest("[data-edit-schedule]");
  if (editButton) {
    const schedule = (state.bootstrap.schedules || []).find(
      (value) => String(value.id) === editButton.dataset.editSchedule,
    );
    if (!schedule) return toast("That schedule is no longer available.", true);
    state.editingSchedule = schedule;
    state.selectedFeed = { feed_id: schedule.feed_id, name: schedule.feed_name };
    byId("archiveFeedId").value = schedule.feed_id;
    byId("selectedFeedLabel").textContent = `${schedule.feed_name} · feed ${schedule.feed_id}`;
    byId("scheduleRunTime").value = schedule.run_time_local || "02:00";
    byId("scheduleLookbackDays").value = Number(schedule.lookback_days) || 2;
    byId("scheduleBackfillStartDate").value = schedule.backfill_start_date || "";
    byId("scheduleRecurringCatchUp").checked = Boolean(schedule.recurring_catch_up);
    byId("scheduleAccountProfile").value = [...byId("scheduleAccountProfile").options].some((option) => option.value === (schedule.account_profile_id || "automatic"))
      ? schedule.account_profile_id || "automatic"
      : "automatic";
    syncRecurringCatchUpInput();
    byId("scheduleEnabled").checked = Boolean(schedule.enabled);
    byId("archiveCombine").checked = Boolean(schedule.job?.combine);
    byId("archiveTranscribe").checked = Boolean(schedule.job?.transcribe);
    byId("archiveDiarize").checked = Boolean(schedule.job?.diarize);
    byId("archiveAnalyze").checked = Boolean(schedule.analyze);
    byId("saveFeedScheduleButton").textContent = "Update selected feed schedule";
    toast(`Editing the saved schedule for ${schedule.feed_name}.`);
    return;
  }
  const button = event.target.closest("[data-delete-schedule]");
  if (!button) return;
  try {
    await api(`/api/schedules/${encodeURIComponent(button.dataset.deleteSchedule)}/delete`, {
      method: "POST",
      body: "{}",
    });
    if (String(state.editingSchedule?.id || "") === button.dataset.deleteSchedule) {
      state.editingSchedule = null;
      byId("saveFeedScheduleButton").textContent = "Schedule selected feed daily";
    }
    await refreshBootstrap();
  } catch (error) {
    toast(error.message, true);
  }
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
  const radiusMode = byId("areaCoverageMode").value === "radius";
  if (radiusMode && zipCodes.length !== 1) return toast("Radius coverage needs one five-digit center ZIP.", true);
  const payload = radiusMode
    ? { center_zip: zipCodes[0], radius_miles: Number(byId("areaRadiusMiles").value), max_zip_codes: Number(byId("areaMaxZipCodes").value) }
    : { zip_codes: zipCodes };
  await startJob("area-search", { ...payload, account_profile_id: selectedArchiveProfile() }, { label: "Discovering nearby feeds", onComplete: (job) => {
    const result = eventOf(job, "area_search") || {};
    state.areaCoverage = result.coverage || state.areaCoverage;
    renderAreaResults(result.results || []);
  } });
});
byId("areaSearchResults").addEventListener("change", (event) => {
  const checkbox = event.target.closest("[data-area-feed-index]");
  if (!checkbox) return;
  const feed = state.areaDiscovered[Number(checkbox.dataset.areaFeedIndex)];
  if (!feed) return;
  const feedId = String(feed.feed_id);
  if (checkbox.checked) state.areaSelectedFeedIds.add(feedId);
  else state.areaSelectedFeedIds.delete(feedId);
  const summary = byId("areaSelectionSummary");
  if (summary) summary.textContent = `Showing ${document.querySelectorAll("[data-area-feed-index]").length} of ${state.areaDiscovered.length} discovered feeds · ${state.areaSelectedFeedIds.size} selected · no archive audio requested`;
});
byId("areaSearchResults").addEventListener("click", (event) => {
  const action = event.target.closest("[data-area-selection]")?.dataset.areaSelection;
  if (!action) return;
  if (action === "clear") {
    state.areaSelectedFeedIds.clear();
  } else if (action === "nearest") {
    state.areaSelectedFeedIds.clear();
    state.areaDiscovered
      .filter((feed) => !byId("areaPublicSafetyOnly").checked || String(feed.genre || "").toLowerCase() === "public safety")
      .slice(0, 3)
      .forEach((feed) => state.areaSelectedFeedIds.add(String(feed.feed_id)));
  }
  renderAreaResults(state.areaDiscovered);
});
byId("saveAreaProfileForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const feeds = state.areaDiscovered.filter((feed) => state.areaSelectedFeedIds.has(String(feed.feed_id)));
  if (!feeds.length) return toast("Select at least one discovered feed.", true);
  const searched = state.areaCoverage?.searched_zip_codes || [];
  const zipCodes = searched.map((value) => value.zip_code).filter(Boolean).length
    ? searched.map((value) => value.zip_code).filter(Boolean)
    : byId("areaZipCodes").value.split(/[\s,;]+/).filter(Boolean);
  const profileName = byId("areaProfileName").value;
  await startJob("save-area-profile", { name: profileName, zip_codes: zipCodes, feeds, coverage: state.areaCoverage }, { label: "Saving area profile", onComplete: async () => {
    await refreshBootstrap();
    byId("areaProfileSelect").value = profileName;
    applySelectedAreaProfile();
    renderAreaQueue();
  } });
});
byId("buildAreaBriefButton").addEventListener("click", async () => {
  const profileName = byId("areaProfileSelect").value;
  if (!profileName) return toast("Choose a saved area profile.", true);
  await startJob("summarize-area", { profile_name: profileName, start_date: byId("areaStartDate").value, end_date: byId("areaEndDate").value, ...providerPayload() }, { label: `Ranking ${profileName} story leads`, onComplete: (job) => renderAreaBrief(eventOf(job, "area_digest")?.result) });
});
byId("areaRunDiarize").addEventListener("change", () => {
  if (byId("areaRunDiarize").checked) {
    byId("areaRunCombine").checked = true;
    byId("areaRunTranscribe").checked = true;
  }
});
byId("runAreaQueueButton").addEventListener("click", async () => {
  const profileName = byId("areaProfileSelect").value;
  if (!profileName) return toast("Choose and save an area profile first.", true);
  const analyze = byId("areaRunAnalyze").checked;
  await startPooledArchiveJob("run-area", {
    profile_name: profileName,
    account_profile_id: selectedArchiveProfile(),
    job: {
      feed_id: "0",
      start_date: byId("areaStartDate").value,
      end_date: byId("areaEndDate").value,
      combine: byId("areaRunCombine").checked,
      transcribe: byId("areaRunTranscribe").checked,
      diarize: byId("areaRunDiarize").checked,
      ...processingPayload(),
    },
  }, { label: `Running ${profileName} nearest first`, onComplete: async (job) => {
    const result = eventOf(job, "area_complete")?.result;
    await refreshBootstrap();
    if (analyze && result?.feed_results?.length) await queueAreaAnalyses(result.feed_results);
  } });
});
byId("openSavedAreaButton").addEventListener("click", openSavedArea);
byId("areaProfileSelect").addEventListener("change", async () => {
  applySelectedAreaProfile();
  renderAreaQueue();
  await openSavedArea();
});
byId("areaCoverageMode").addEventListener("change", updateAreaCoverageControls);
byId("areaPublicSafetyOnly").addEventListener("change", () => renderAreaResults(state.areaDiscovered));

async function runProviderCheck() {
  await startJob("analysis-provider-diagnostics", providerPayload(), { label: "Checking analysis provider", onComplete: (job) => {
    const result = eventOf(job, "analysis_provider_diagnostics")?.result;
    if (!result) return;
    state.analysisProviderStatus = result;
    updateProviderNotice(`${result.ready ? "Provider configured" : "Setup needed"}: ${result.message}`);
    renderSetupReadiness();
  } });
}

async function runHardwareCheck() {
  await startJob("diagnostics-selected", processingPayload(), { label: "Checking local hardware", onComplete: (job) => {
    const result = eventOf(job, "diagnostics");
    if (!result) return;
    state.hardwareDiagnostics = result;
    const accelerators = result.accelerators || {};
    const profiles = Array.isArray(accelerators.profiles) ? accelerators.profiles : [];
    const summary = [result.cuda_available ? `CUDA: ${(result.cuda_devices || []).join(", ")}` : "CUDA unavailable", result.ffmpeg ? "FFmpeg ready" : "FFmpeg missing", result.llama_server ? "llama.cpp ready" : "llama.cpp missing", `Profiles: ${profiles.length || "checked"}`].join(" · ");
    byId("runtimeDescription").textContent = summary;
    renderHardwareProfiles();
    renderSetupReadiness();
  } });
}

async function runAsrSelfTest() {
  resetProfileVerification();
  await startJob("asr-self-test", processingPayload(), { label: "Testing selected transcription engine", onComplete: (job) => {
    const result = eventOf(job, "asr_self_test")?.result;
    if (!result) return;
    applyAsrSelfTestResult(result);
  } });
}

async function runAsrModelPreparation() {
  const notice = byId("asrSelfTestNotice");
  notice.className = "notice";
  notice.querySelector("strong").textContent = "Preparing the selected model";
  notice.querySelector("span").textContent = "This explicit action may download model files. It does not use archive audio or Broadcastify quota.";
  await startJob("prepare-asr-model", processingPayload(), {
    label: "Preparing the selected transcription model",
    onComplete: async (job) => {
      const result = eventOf(job, "asr_model_prepared")?.result;
      if (!result?.ready || !result.path) {
        notice.className = "notice warning";
        notice.querySelector("strong").textContent = "Model preparation needs attention";
        notice.querySelector("span").textContent = "The worker did not return a usable managed model path.";
        return false;
      }
      byId("settingAsrModelPath").value = result.path;
      readSettingsForm();
      localStorage.setItem("radioArchiveSettings", JSON.stringify(state.settings));
      notice.querySelector("strong").textContent = result.reused
        ? "Matching model found; proving a decode"
        : "Model prepared; proving a decode";
      notice.querySelector("span").textContent = result.message;
      await runAsrSelfTest();
      return false;
    },
  });
}

function applyAsrSelfTestResult(result) {
  state.asrSelfTest = result;
  const notice = byId("asrSelfTestNotice");
  notice.className = "notice success";
  notice.querySelector("strong").textContent = "Transcription engine executed locally";
  const fallback = result.fallback_reason ? ` Fallback: ${result.fallback_reason}` : "";
  notice.querySelector("span").textContent = `${result.message}${fallback}`;
  renderHardwareProfiles();
  renderSetupReadiness();
}

async function runDiarizationSelfTest() {
  resetProfileVerification();
  await startJob("diarization-self-test", processingPayload(), { label: "Testing selected speaker-label engine", onComplete: (job) => {
    const result = eventOf(job, "diarization_self_test")?.result;
    if (!result) return;
    applyDiarizationSelfTestResult(result);
  } });
}

function applyDiarizationSelfTestResult(result) {
  state.diarizationSelfTest = result;
  const notice = byId("diarizationSelfTestNotice");
  notice.className = "notice success";
  notice.querySelector("strong").textContent = "Speaker labels ready";
  notice.querySelector("span").textContent = result.message;
  renderHardwareProfiles();
  renderSetupReadiness();
}

async function runAnalysisSelfTest() {
  resetProfileVerification();
  resetAnalysisVerification();
  const notice = byId("analysisSelfTestNotice");
  notice.querySelector("strong").textContent = "Loading and generating synthetic output";
  notice.querySelector("span").textContent = "No archive evidence is used. A hosted provider may record minimal model usage.";
  await startJob("analysis-self-test", providerPayload(), { label: "Testing selected analysis model", onComplete: (job) => {
    const result = eventOf(job, "analysis_self_test")?.result;
    if (!result) return;
    applyAnalysisSelfTestResult(result);
  } });
}

async function runProfileNextAction() {
  const action = currentProfileAction();
  if (!action) return;
  const notice = byId("profileSelfTestNotice");
  notice.className = "notice warning";
  notice.querySelector("strong").textContent = action.label || "Complete profile setup";
  notice.querySelector("span").textContent = action.message || "Complete this setup step, then verify the profile.";
  if (action.kind === "verify-profile") {
    await runProfileSelfTest();
  } else if (action.kind === "prepare-asr-model") {
    setSettingsSection("processing");
    await runAsrModelPreparation();
  } else if (action.kind === "test-transcription") {
    setSettingsSection("processing");
    await runAsrSelfTest();
  } else if (action.kind === "configure-speakers") {
    setSettingsSection("account");
    requestAnimationFrame(() => byId("settingHuggingFaceToken").focus());
  } else if (action.kind === "configure-analysis") {
    setSettingsSection("analysis");
    requestAnimationFrame(() => byId("settingAnalysisProvider").focus());
  } else {
    setSettingsSection("processing");
    requestAnimationFrame(() => byId("settingAsrEngine").focus());
  }
}

function applyAnalysisSelfTestResult(result) {
  state.analysisSelfTest = result;
  state.analysisProviderStatus = {
    ...(state.analysisProviderStatus || {}),
    provider: result.provider,
    model: result.model,
    device: result.device,
    external: result.external,
    ready: true,
    message: result.message,
  };
  const notice = byId("analysisSelfTestNotice");
  notice.className = "notice success";
  notice.querySelector("strong").textContent = "Analysis model ready";
  notice.querySelector("span").textContent = result.message;
  updateProviderNotice();
  renderHardwareProfiles();
  renderSetupReadiness();
}

async function runProfileSelfTest() {
  resetProfileVerification();
  resetAsrVerification();
  resetDiarizationVerification();
  resetAnalysisVerification();
  const notice = byId("profileSelfTestNotice");
  notice.className = "notice";
  notice.querySelector("strong").textContent = "Verifying all three model stages";
  notice.querySelector("span").textContent = "The check stops at the first stage that needs setup. Archive audio and quota are not used.";
  await startJob(
    "profile-self-test",
    { ...processingPayload(), ...providerPayload() },
    {
      label: "Verifying selected hardware profile",
      onComplete: (job) => {
        const transcription = eventOf(job, "asr_self_test")?.result;
        const diarization = eventOf(job, "diarization_self_test")?.result;
        const analysis = eventOf(job, "analysis_self_test")?.result;
        if (transcription) applyAsrSelfTestResult(transcription);
        if (diarization) applyDiarizationSelfTestResult(diarization);
        if (analysis) applyAnalysisSelfTestResult(analysis);
        const result = eventOf(job, "profile_self_test")?.result;
        if (!result) {
          notice.className = "notice warning";
          notice.querySelector("strong").textContent = "Profile verification returned no summary";
          notice.querySelector("span").textContent = "Review the job log and rerun the individual stage tests.";
          toast("Profile verification returned no summary.", true);
          return false;
        }
        if (!result.ready || !result.verified) {
          const failedNoticeId = {
            transcription: "asrSelfTestNotice",
            diarization: "diarizationSelfTestNotice",
            analysis: "analysisSelfTestNotice",
          }[result.failed_stage];
          const failedTitle = {
            transcription: "Transcription needs setup",
            diarization: "Speaker labels need setup",
            analysis: "Analysis model needs setup",
          }[result.failed_stage] || "Profile needs setup";
          const failedNotice = failedNoticeId ? byId(failedNoticeId) : null;
          if (failedNotice) {
            failedNotice.className = "notice warning";
            failedNotice.querySelector("strong").textContent = failedTitle;
            failedNotice.querySelector("span").textContent = result.message;
          }
        }
        state.profileSelfTest = result;
        notice.className = `notice ${result.ready && result.verified ? "success" : "warning"}`;
        notice.querySelector("strong").textContent = result.ready && result.verified
          ? "Selected profile verified"
          : `${words(result.failed_stage || "profile")} needs setup`;
        notice.querySelector("span").textContent = result.message;
        renderHardwareProfiles();
        renderSetupReadiness();
        if (result.ready && result.verified) {
          toast("Selected profile passed all three execution checks.");
        } else {
          toast(result.message, true);
          if (result.failed_stage === "analysis") setSettingsSection("analysis");
          else setSettingsSection("processing");
        }
        return false;
      },
    },
  );
}

byId("providerCheckButton").addEventListener("click", runProviderCheck);
byId("analysisSelfTestButton").addEventListener("click", runAnalysisSelfTest);
byId("runtimeCheckButton").addEventListener("click", runHardwareCheck);
byId("setupCheckButton").addEventListener("click", runHardwareCheck);
byId("setupProfileSelfTestButton").addEventListener("click", runProfileSelfTest);
byId("profileSelfTestButton").addEventListener("click", runProfileSelfTest);
byId("profileNextActionButton").addEventListener("click", runProfileNextAction);
byId("settingHardwareProfile").addEventListener("change", (event) => applyHardwareProfile(event.target.value));
["settingAsrEngine", "settingDevice"].forEach((id) => byId(id).addEventListener("change", () => {
  if (id === "settingAsrEngine") ensureAsrModelCompatibility();
  if (!applyingHardwareProfile) {
    byId("settingHardwareProfile").value = "custom";
  }
  resetProfileVerification();
  resetAsrVerification();
  updateAsrModelPreparationUi();
  renderHardwareProfiles();
  renderSetupReadiness();
}));
["settingDiarizationEngine", "settingDiarizationDevice"].forEach((id) => byId(id).addEventListener("change", () => {
  if (byId("settingDiarizationEngine").value === "sherpa-onnx"
      && byId("settingDiarizationDevice").value === "cuda") {
    byId("settingDiarizationDevice").value = "cpu";
    toast("Fast portable speaker preview runs on CPU; the speaker device was reset to CPU.");
  }
  if (!applyingHardwareProfile) byId("settingHardwareProfile").value = "custom";
  resetProfileVerification();
  resetDiarizationVerification();
  renderHardwareProfiles();
  renderSetupReadiness();
}));
byId("settingAnalysisDevice").addEventListener("change", () => {
  if (!applyingHardwareProfile) byId("settingHardwareProfile").value = "custom";
  state.analysisProviderStatus = null;
  resetProfileVerification();
  resetAnalysisVerification();
  renderHardwareProfiles();
  renderSetupReadiness();
});
byId("runtimeProfiles").addEventListener("click", (event) => {
  const button = event.target.closest("[data-use-hardware-profile]");
  if (!button) return;
  applyHardwareProfile(button.dataset.useHardwareProfile);
});
byId("settingsSectionTabs").addEventListener("click", (event) => {
  const tab = event.target.closest("[data-settings-section]");
  if (!tab) return;
  setSettingsSection(tab.dataset.settingsSection);
});
byId("settingsSectionTabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = [...byId("settingsSectionTabs").querySelectorAll("[data-settings-section]")];
  const current = Math.max(0, tabs.indexOf(document.activeElement));
  let next = current;
  if (event.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
  if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
  if (event.key === "Home") next = 0;
  if (event.key === "End") next = tabs.length - 1;
  event.preventDefault();
  setSettingsSection(tabs[next].dataset.settingsSection, { focusTab: true });
});
byId("setupReadinessGrid").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-setup-action]");
  if (!button) return;
  const action = button.dataset.setupAction;
  if (action === "account") {
    setSettingsSection("account");
    requestAnimationFrame(() => byId("loginUsername").focus());
  } else if (action === "storage") {
    setSettingsSection("processing");
    requestAnimationFrame(() => byId("runtimeCheckButton").focus());
  } else if (action === "transcription") {
    setSettingsSection("processing");
    const next = currentProfileAction();
    if (!state.hardwareDiagnostics) await runHardwareCheck();
    else if (!selectedHardwareProfile()?.transcription_ready && next?.stage === "transcription") await runProfileNextAction();
    else await runAsrSelfTest();
  } else if (action === "diarization") {
    const next = currentProfileAction();
    const speaker = state.hardwareDiagnostics?.accelerators?.speaker_labels || {};
    const selectedEngine = byId("settingDiarizationEngine").value;
    const accessReady = speaker.access_configured || speaker.token_configured || byId("settingHuggingFaceToken").value;
    const selectedDevice = byId("settingDiarizationDevice").value;
    const readyToTest = selectedEngine === "sherpa-onnx"
      ? Boolean(speaker.portable?.runtime_installed)
      : Boolean(speaker.package_installed && accessReady && (selectedDevice !== "cuda" || speaker.cuda_available));
    if (readyToTest) {
      await runDiarizationSelfTest();
    } else if (next?.stage === "diarization") {
      await runProfileNextAction();
    } else if (!state.hardwareDiagnostics) {
      setSettingsSection("processing");
      await runHardwareCheck();
    } else {
      setSettingsSection("account");
      byId("settingHuggingFaceToken").focus();
    }
  } else if (action === "analysis") {
    setSettingsSection("analysis");
    const profile = selectedHardwareProfile();
    const next = currentProfileAction();
    const provider = byId("settingAnalysisProvider").value;
    const configured = Boolean(
      state.analysisProviderStatus?.ready
      || (provider === "local" && profile?.analysis_ready)
    );
    if (configured || state.analysisSelfTest?.ready) {
      await runAnalysisSelfTest();
    } else if (next?.stage === "analysis") {
      await runProfileNextAction();
    } else {
      await runProviderCheck();
    }
  }
});
byId("asrSelfTestButton").addEventListener("click", runAsrSelfTest);
byId("asrPrepareButton").addEventListener("click", runAsrModelPreparation);
byId("diarizationSelfTestButton").addEventListener("click", runDiarizationSelfTest);
byId("loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const profileId = byId("loginProfileId").value.trim().toLowerCase();
  const label = byId("loginProfileLabel").value.trim() || profileId;
  if (!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(profileId)) {
    toast("Account profile IDs may use letters, numbers, underscores, and hyphens.", true);
    return;
  }
  const saved = (accountPool().profiles || []).find((value) => value.id === profileId) || {};
  const username = byId("loginUsername").value.trim() || saved.username || "";
  const password = byId("loginPassword").value;
  const remember = byId("rememberBroadcastifyLogin").checked;
  const canReuse = Boolean(
    !password
    && saved.configured
    && (!saved.username || saved.username === username)
  );
  if (!username || (!password && !canReuse)) {
    toast("Enter a username and password, or keep the username matching the saved login.", true);
    return;
  }
  byId("loginPassword").value = "";
  const payload = password ? { username, password, account_profile_id: profileId } : { account_profile_id: profileId };
  await startJob("authenticate", payload, { label: "Signing in to Broadcastify", onComplete: async () => {
    if (password && remember) {
      await api("/api/credentials", {
        method: "POST",
        body: JSON.stringify({ kind: "broadcastify", action: "save", profile_id: profileId, label, username, secret: password }),
      });
    } else if (!remember) {
      await api("/api/credentials", {
        method: "POST",
        body: JSON.stringify({ kind: "broadcastify", action: "clear", profile_id: profileId }),
      });
    }
    state.accountVerified = true;
    await refreshBootstrap();
    renderSetupReadiness();
    toast(remember ? "Broadcastify session refreshed; encrypted login is ready." : "Broadcastify session refreshed without saving the login.");
  } });
});
byId("loginProfileId").addEventListener("input", () => {
  state.accountVerified = false;
  renderCredentials();
});
byId("useAskMonthButton").addEventListener("click", () => {
  const month = byId("askMonth").value;
  if (!/^\d{4}-\d{2}$/.test(month)) return toast("Choose a calendar month first.", true);
  const [year, monthNumber] = month.split("-").map(Number);
  const start = `${month}-01`;
  const last = new Date(year, monthNumber, 0);
  const end = new Date(last.getTime() - last.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  if (start > localToday) return toast("Choose the current month or an earlier month.", true);
  byId("askStartDate").value = start;
  byId("askEndDate").value = end > localToday ? localToday : end;
  toast("Month range selected. The answer will report retained coverage gaps.");
});
function useEntireDownloadedFeed() {
  const feedId = byId("askFeedId").value.trim();
  if (!/^\d+$/.test(feedId)) {
    toast("Enter or select a numeric feed ID first.", true);
    return false;
  }
  const dates = [...new Set((state.bootstrap.days || [])
    .filter((day) => String(day.feed_id) === feedId)
    .filter((day) => Number(day.raw_file_count || 0) > 0 || Boolean(day.has_combined) || Boolean(day.has_transcript))
    .map((day) => String(day.archive_date || ""))
    .filter((value) => /^\d{4}-\d{2}-\d{2}$/.test(value)))].sort();
  if (!dates.length) {
    toast(`No locally retained days exist for feed ${feedId}.`, true);
    return false;
  }
  byId("askStartDate").value = dates[0];
  byId("askEndDate").value = dates[dates.length - 1];
  toast(`Entire downloaded span selected: ${dates[0]} through ${dates[dates.length - 1]}. Gaps inside the span remain visible.`);
  return true;
}
byId("useAskEntireFeedButton").addEventListener("click", () => {
  useEntireDownloadedFeed();
});
byId("askFeedHotspotsButton").addEventListener("click", () => {
  if (!useEntireDownloadedFeed()) return;
  byId("askQuestion").value = "Across the entire downloaded feed, where and when do supported incident records cluster? Rank repeated extracted locations, categories, weekdays, and six-hour time windows using exact aggregate counts and citations. Include exact archive dates and times for representative events. Treat missing or unprocessed dates as coverage limits, and do not claim population-normalized crime rates or trends.";
  byId("askQuestion").focus();
});
function syncRecurringCatchUpInput() {
  const recurring = byId("scheduleRecurringCatchUp");
  recurring.disabled = !byId("scheduleBackfillStartDate").value;
  if (recurring.disabled) recurring.checked = false;
}
byId("scheduleBackfillStartDate").addEventListener("input", syncRecurringCatchUpInput);
byId("forgetBroadcastifyLoginButton").addEventListener("click", async () => {
  const profileId = byId("loginProfileId").value.trim().toLowerCase() || "default";
  try {
    await api("/api/credentials", {
      method: "POST",
      body: JSON.stringify({ kind: "broadcastify", action: "clear", profile_id: profileId }),
    });
    await refreshBootstrap();
    toast(`Encrypted Broadcastify login ${profileId} removed; its current website session remains.`);
  } catch (error) {
    toast(error.message, true);
  }
});
byId("huggingFaceCredentialForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const secret = byId("settingHuggingFaceToken").value.trim();
  if (!secret) {
    toast("Enter a Hugging Face read token to replace the saved token.", true);
    return;
  }
  try {
    await api("/api/credentials", {
      method: "POST",
      body: JSON.stringify({ kind: "huggingface", action: "save", secret }),
    });
    byId("settingHuggingFaceToken").value = "";
    resetProfileVerification();
    resetAsrVerification();
    resetDiarizationVerification();
    await refreshBootstrap();
    toast("Hugging Face token encrypted and saved.");
  } catch (error) {
    toast(error.message, true);
  }
});
byId("forgetHuggingFaceTokenButton").addEventListener("click", async () => {
  try {
    await api("/api/credentials", {
      method: "POST",
      body: JSON.stringify({ kind: "huggingface", action: "clear" }),
    });
    byId("settingHuggingFaceToken").value = "";
    await refreshBootstrap();
    toast("Encrypted Hugging Face token removed.");
  } catch (error) {
    toast(error.message, true);
  }
});

const today = new Date();
const localToday = new Date(today.getTime() - today.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
const weekAgo = new Date(today.getTime() - 6 * 86400000 - today.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
byId("archiveStartDate").value = localToday;
byId("archiveEndDate").value = localToday;
byId("areaStartDate").value = weekAgo;
byId("areaEndDate").value = localToday;
byId("weekEnding").value = localToday;
byId("askMonth").value = localToday.slice(0, 7);
byId("askStartDate").value = localToday;
byId("askEndDate").value = localToday;
syncRecurringCatchUpInput();
applySettingsForm();
updateAreaCoverageControls();
window.addEventListener("hashchange", setViewFromLocation);
setViewFromLocation();
refreshBootstrap({ preserveSelection: false }).then(async () => {
  if (state.bootstrap.days.length) await selectDay(state.bootstrap.days[0]);
});
