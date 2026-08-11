using System.Text.Json.Serialization;

namespace BroadcastifyCli.WinUI;

public sealed record FeedSearchResult
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("name")]
    public string Name { get; init; } = "";

    [JsonPropertyName("location")]
    public string Location { get; init; } = "";

    [JsonPropertyName("description")]
    public string Description { get; init; } = "";

    [JsonPropertyName("genre")]
    public string Genre { get; init; } = "";

    [JsonPropertyName("listeners")]
    public int Listeners { get; init; }

    [JsonPropertyName("status")]
    public string Status { get; init; } = "";

    [JsonPropertyName("matched_zip_codes")]
    public List<string> MatchedZipCodes { get; init; } = [];

    [JsonPropertyName("nearest_zip_code")]
    public string NearestZipCode { get; init; } = "";

    [JsonPropertyName("distance_miles")]
    public double? DistanceMiles { get; init; }

    [JsonPropertyName("priority_rank")]
    public int PriorityRank { get; init; }

    public string FeedLabel => $"Feed {FeedId}";
    public string LocationAndGenre => string.Join(" · ", new[] { Location, Genre }.Where(value => !string.IsNullOrWhiteSpace(value)));
    public string ListenerSummary => $"{Listeners:N0} listener{(Listeners == 1 ? "" : "s")}";
    public string AreaMatchSummary => MatchedZipCodes.Count == 0
        ? LocationAndGenre
        : string.Join(" · ", new[]
        {
            DistanceMiles is null
                ? $"Priority {PriorityRank}: ZIP {NearestZipCode}"
                : $"Priority {PriorityRank}: about {DistanceMiles:0.#} mi via ZIP {NearestZipCode}",
            LocationAndGenre,
        }.Where(value => !string.IsNullOrWhiteSpace(value)));
}

public sealed record AreaZipCandidate
{
    [JsonPropertyName("zip_code")]
    public string ZipCode { get; init; } = "";

    [JsonPropertyName("distance_miles")]
    public double? DistanceMiles { get; init; }
}

public sealed record AreaCoverage
{
    [JsonPropertyName("mode")]
    public string Mode { get; init; } = "zip-list";

    [JsonPropertyName("center_zip")]
    public string CenterZip { get; init; } = "";

    [JsonPropertyName("radius_miles")]
    public double? RadiusMiles { get; init; }

    [JsonPropertyName("max_zip_codes")]
    public int MaxZipCodes { get; init; } = 12;

    [JsonPropertyName("searched_zip_codes")]
    public List<AreaZipCandidate> SearchedZipCodes { get; init; } = [];

    [JsonPropertyName("distance_basis")]
    public string DistanceBasis { get; init; } = "";
}

internal sealed record AreaSearchResponse
{
    public List<FeedSearchResult> Results { get; init; } = [];
    public AreaCoverage Coverage { get; init; } = new();
}

public sealed record AreaProfile
{
    [JsonPropertyName("id")]
    public long Id { get; init; }

    [JsonPropertyName("name")]
    public string Name { get; init; } = "";

    [JsonPropertyName("zip_codes")]
    public List<string> ZipCodes { get; init; } = [];

    [JsonPropertyName("feeds")]
    public List<FeedSearchResult> Feeds { get; init; } = [];

    [JsonPropertyName("feed_ids")]
    public List<string> FeedIds { get; init; } = [];

    [JsonPropertyName("coverage")]
    public AreaCoverage Coverage { get; init; } = new();

    public string DisplayName => $"{Name} · {Feeds.Count} feed{(Feeds.Count == 1 ? "" : "s")}";
    public string CoverageArea => $"ZIPs {string.Join(", ", ZipCodes)}";
}

internal sealed record AreaProfileSaveRequest
{
    [JsonPropertyName("name")]
    public string Name { get; init; } = "";

    [JsonPropertyName("zip_codes")]
    public List<string> ZipCodes { get; init; } = [];

    [JsonPropertyName("feeds")]
    public List<FeedSearchResult> Feeds { get; init; } = [];

    [JsonPropertyName("coverage")]
    public AreaCoverage Coverage { get; init; } = new();
}

internal abstract record AnalysisProviderRequest
{
    [JsonPropertyName("analysis_provider")]
    public string AnalysisProvider { get; set; } = "local";

    [JsonPropertyName("analysis_model")]
    public string AnalysisModel { get; set; } = "";

    [JsonPropertyName("analysis_device")]
    public string AnalysisDevice { get; set; } = "auto";

    [JsonPropertyName("analysis_endpoint")]
    public string AnalysisEndpoint { get; set; } = "";

    [JsonPropertyName("analysis_api_key")]
    public string? AnalysisApiKey { get; set; }

    [JsonPropertyName("analysis_api_key_env")]
    public string AnalysisApiKeyEnvironment { get; set; } = "OPENAI_API_KEY";

    [JsonPropertyName("codex_cli_path")]
    public string CodexCliPath { get; set; } = "";

    [JsonPropertyName("allow_external_analysis")]
    public bool AllowExternalAnalysis { get; set; }
}

internal sealed record AnalysisProviderDiagnosticsRequest : AnalysisProviderRequest;
internal sealed record AnalysisSelfTestRequest : AnalysisProviderRequest;

public sealed record AnalysisProviderStatus
{
    [JsonPropertyName("provider")]
    public string Provider { get; init; } = "";

    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("device")]
    public string Device { get; init; } = "";

    [JsonPropertyName("external")]
    public bool External { get; init; }

    [JsonPropertyName("ready")]
    public bool Ready { get; init; }

    [JsonPropertyName("configured")]
    public bool Configured { get; init; }

    [JsonPropertyName("verified")]
    public bool Verified { get; init; }

    [JsonPropertyName("elapsed_seconds")]
    public double ElapsedSeconds { get; init; }

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";
}

internal sealed record AreaDigestRequest : AnalysisProviderRequest
{
    [JsonPropertyName("profile_name")]
    public string ProfileName { get; init; } = "";

    [JsonPropertyName("start_date")]
    public string StartDate { get; init; } = "";

    [JsonPropertyName("end_date")]
    public string EndDate { get; init; } = "";

    [JsonPropertyName("force")]
    public bool Force { get; init; }
}

public sealed record AreaStoryReference
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("feed_name")]
    public string FeedName { get; init; } = "";

    [JsonPropertyName("incident_id")]
    public long IncidentId { get; init; }

    [JsonPropertyName("archive_time")]
    public string ArchiveTime { get; init; } = "";

    [JsonPropertyName("priority")]
    public int Priority { get; init; }

    [JsonPropertyName("confidence")]
    public double Confidence { get; init; }

    [JsonPropertyName("quote")]
    public string Quote { get; init; } = "";

    [JsonPropertyName("quote_redacted")]
    public bool QuoteRedacted { get; init; }

    [JsonPropertyName("speaker_labels")]
    public List<string> SpeakerLabels { get; init; } = [];

    [JsonPropertyName("evidence_segment_count")]
    public int EvidenceSegmentCount { get; init; }

    [JsonPropertyName("source_evidence_segment_count")]
    public int SourceEvidenceSegmentCount { get; init; }

    [JsonPropertyName("has_diarization")]
    public bool HasDiarization { get; init; }

    [JsonPropertyName("clip_path")]
    public string ClipPath { get; init; } = "";

    [JsonPropertyName("clip_available")]
    public bool ClipAvailable { get; init; }

    [JsonPropertyName("clip_status")]
    public string ClipStatus { get; init; } = "";

    [JsonPropertyName("clip_sha256")]
    public string ClipSha256 { get; init; } = "";

    [JsonPropertyName("source_audio_sha256")]
    public string SourceAudioSha256 { get; init; } = "";

    [JsonPropertyName("transcript_sha256")]
    public string TranscriptSha256 { get; init; } = "";

    public string SourceAndTime => $"{FeedName} · {ArchiveTime} · incident I{IncidentId}";
    public string QuoteDisplay => string.IsNullOrWhiteSpace(Quote)
        ? "No transcript excerpt was retained for this extraction."
        : $"“{Quote}”{(QuoteRedacted ? " (obvious identifier redacted)" : "")}";
    public string ProvenanceSummary
    {
        get
        {
            var speaker = HasDiarization && SpeakerLabels.Count > 0
                ? $" · speakers {string.Join(", ", SpeakerLabels)}"
                : " · speaker not established";
            var integrity = string.IsNullOrWhiteSpace(ClipSha256)
                ? ""
                : $" · clip SHA-256 {ClipSha256[..Math.Min(12, ClipSha256.Length)]}…";
            var excerpt = SourceEvidenceSegmentCount > EvidenceSegmentCount
                ? $" · {EvidenceSegmentCount}/{SourceEvidenceSegmentCount} cited segments in this clip"
                : $" · {EvidenceSegmentCount} cited segment{(EvidenceSegmentCount == 1 ? "" : "s")}";
            return $"P{Priority} · {Confidence:P0} extraction confidence{speaker}{excerpt}{integrity}";
        }
    }
}

public sealed record AreaStory
{
    [JsonPropertyName("story_id")]
    public string StoryId { get; init; } = "";

    [JsonPropertyName("headline")]
    public string Headline { get; init; } = "";

    [JsonPropertyName("summary")]
    public string Summary { get; init; } = "";

    [JsonPropertyName("event_type")]
    public string EventType { get; init; } = "";

    [JsonPropertyName("location")]
    public string Location { get; init; } = "";

    [JsonPropertyName("first_reported")]
    public string FirstReported { get; init; } = "";

    [JsonPropertyName("newsworthiness_score")]
    public int NewsworthinessScore { get; init; }

    [JsonPropertyName("interest_level")]
    public string InterestLevel { get; init; } = "";

    [JsonPropertyName("priority")]
    public int Priority { get; init; }

    [JsonPropertyName("feed_count")]
    public int FeedCount { get; init; }

    [JsonPropertyName("why_interesting")]
    public string WhyInteresting { get; init; } = "";

    [JsonPropertyName("incident_references")]
    public List<AreaStoryReference> IncidentReferences { get; init; } = [];

    [JsonPropertyName("evidence_clip_count")]
    public int EvidenceClipCount { get; init; }

    [JsonPropertyName("quote_count")]
    public int QuoteCount { get; init; }

    [JsonPropertyName("neighborhood_tags")]
    public List<string> NeighborhoodTags { get; init; } = [];

    [JsonPropertyName("topic_tags")]
    public List<string> TopicTags { get; init; } = [];

    [JsonPropertyName("subscription_eligible")]
    public bool SubscriptionEligible { get; init; }

    [JsonPropertyName("publication_status")]
    public string PublicationStatus { get; init; } = "review_required";

    public string ScoreSummary => $"{InterestLevel} · score {NewsworthinessScore} · P{Priority} · {EventType.Replace('_', ' ')}";
    public string TimeAndPlace => string.IsNullOrWhiteSpace(Location)
        ? FirstReported
        : $"{FirstReported} · {Location}";
    public string SourcesSummary => string.Join(
        " · ",
        IncidentReferences.Select(value => $"{value.FeedName} I{value.IncidentId}"));
    public string EvidenceSummary => $"Evidence: {EvidenceClipCount} clip{(EvidenceClipCount == 1 ? "" : "s")} · "
        + $"{QuoteCount} transcript quote{(QuoteCount == 1 ? "" : "s")} · {IncidentReferences.Count} source record{(IncidentReferences.Count == 1 ? "" : "s")}";
    public string AudienceSummary => SubscriptionEligible
        ? $"Neighborhood-ready after editor verification · {string.Join(", ", NeighborhoodTags)} · {string.Join(", ", TopicTags.Select(value => value.Replace('_', ' ')))}"
        : "Not eligible for neighborhood alerts without additional location or confidence.";
}

public sealed record AreaDigestCoverage
{
    [JsonPropertyName("feed_count")]
    public int FeedCount { get; init; }

    [JsonPropertyName("feeds_with_data")]
    public int FeedsWithData { get; init; }

    [JsonPropertyName("feed_days_available")]
    public int FeedDaysAvailable { get; init; }

    [JsonPropertyName("feed_days_expected")]
    public int FeedDaysExpected { get; init; }

    [JsonPropertyName("incident_count")]
    public int IncidentCount { get; init; }

    [JsonPropertyName("stale_feed_days")]
    public List<string> StaleFeedDays { get; init; } = [];
}

public sealed record AreaDigestReport
{
    [JsonPropertyName("profile_name")]
    public string ProfileName { get; init; } = "";

    [JsonPropertyName("start_date")]
    public string StartDate { get; init; } = "";

    [JsonPropertyName("end_date")]
    public string EndDate { get; init; } = "";

    [JsonPropertyName("summary")]
    public string Summary { get; init; } = "";

    [JsonPropertyName("stories")]
    public List<AreaStory> Stories { get; init; } = [];

    [JsonPropertyName("coverage")]
    public AreaDigestCoverage Coverage { get; init; } = new();

    [JsonPropertyName("cached")]
    public bool Cached { get; init; }

    public string CoverageSummary =>
        $"{StartDate} through {EndDate} · {Coverage.FeedDaysAvailable}/{Coverage.FeedDaysExpected} feed-days · "
        + $"{Coverage.FeedsWithData}/{Coverage.FeedCount} feeds with data · {Coverage.IncidentCount} extracted incidents · "
        + $"{Stories.Count} ranked leads"
        + (Coverage.StaleFeedDays.Count == 0
            ? ""
            : $" · {Coverage.StaleFeedDays.Count} retained feed-days need reanalysis");
}

internal sealed record JobRequest : AnalysisProviderRequest
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("feed_name")]
    public string FeedName { get; init; } = "";

    [JsonPropertyName("start_date")]
    public string StartDate { get; init; } = "";

    [JsonPropertyName("end_date")]
    public string EndDate { get; init; } = "";

    [JsonPropertyName("output_dir")]
    public string OutputDirectory { get; init; } = "archives";

    [JsonPropertyName("combine")]
    public bool Combine { get; init; }

    [JsonPropertyName("keep_originals")]
    public bool KeepOriginals { get; init; } = true;

    [JsonPropertyName("transcribe")]
    public bool Transcribe { get; init; }

    [JsonPropertyName("diarize")]
    public bool Diarize { get; init; }

    [JsonPropertyName("model")]
    public string Model { get; init; } = "turbo";

    [JsonPropertyName("asr_engine")]
    public string AsrEngine { get; init; } = "auto";

    [JsonPropertyName("device")]
    public string Device { get; init; } = "auto";

    [JsonPropertyName("device_index")]
    public int DeviceIndex { get; init; }

    [JsonPropertyName("compute_type")]
    public string ComputeType { get; init; } = "auto";

    [JsonPropertyName("asr_model_path")]
    public string? AsrModelPath { get; init; }

    [JsonPropertyName("diarization_engine")]
    public string DiarizationEngine { get; init; } = "community-1";

    [JsonPropertyName("diarization_device")]
    public string DiarizationDevice { get; init; } = "auto";

    [JsonPropertyName("download_jobs")]
    public int DownloadJobs { get; init; } = 1;

    [JsonPropertyName("batch_size")]
    public int BatchSize { get; init; } = 8;

    [JsonPropertyName("min_speakers")]
    public int? MinimumSpeakers { get; init; }

    [JsonPropertyName("max_speakers")]
    public int? MaximumSpeakers { get; init; }

    [JsonPropertyName("huggingface_token")]
    public string? HuggingFaceToken { get; init; }

    [JsonPropertyName("lan_sync_enabled")]
    public bool LanSyncEnabled { get; init; } = true;

    [JsonPropertyName("lan_discovery_enabled")]
    public bool LanDiscoveryEnabled { get; init; } = true;

    [JsonPropertyName("lan_peer_urls")]
    public List<string> LanPeerUrls { get; init; } = [];
}

internal sealed record AsrSelfTestRequest
{
    [JsonPropertyName("model")]
    public string Model { get; init; } = "turbo";

    [JsonPropertyName("asr_engine")]
    public string AsrEngine { get; init; } = "auto";

    [JsonPropertyName("device")]
    public string Device { get; init; } = "auto";

    [JsonPropertyName("device_index")]
    public int DeviceIndex { get; init; }

    [JsonPropertyName("compute_type")]
    public string ComputeType { get; init; } = "auto";

    [JsonPropertyName("asr_model_path")]
    public string? AsrModelPath { get; init; }

    [JsonPropertyName("diarization_engine")]
    public string DiarizationEngine { get; init; } = "community-1";

    [JsonPropertyName("diarization_device")]
    public string DiarizationDevice { get; init; } = "auto";

    [JsonPropertyName("batch_size")]
    public int BatchSize { get; init; } = 8;

    [JsonPropertyName("huggingface_token")]
    public string? HuggingFaceToken { get; init; }
}

public sealed record AsrSelfTestStatus
{
    [JsonPropertyName("ready")]
    public bool Ready { get; init; }

    [JsonPropertyName("engine")]
    public string Engine { get; init; } = "";

    [JsonPropertyName("backend")]
    public string Backend { get; init; } = "";

    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("requested_model")]
    public string RequestedModel { get; init; } = "";

    [JsonPropertyName("model_path")]
    public string ModelPath { get; init; } = "";

    [JsonPropertyName("provider")]
    public string Provider { get; init; } = "";

    [JsonPropertyName("precision")]
    public string Precision { get; init; } = "";

    [JsonPropertyName("device")]
    public string Device { get; init; } = "";

    [JsonPropertyName("elapsed_seconds")]
    public double ElapsedSeconds { get; init; }

    [JsonPropertyName("fallback_reason")]
    public string FallbackReason { get; init; } = "";

    [JsonPropertyName("fallback_stage")]
    public string FallbackStage { get; init; } = "";

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";
}

public sealed record AsrModelPreparationStatus
{
    [JsonPropertyName("ready")]
    public bool Ready { get; init; }

    [JsonPropertyName("engine")]
    public string Engine { get; init; } = "";

    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("source_model")]
    public string SourceModel { get; init; } = "";

    [JsonPropertyName("provider")]
    public string Provider { get; init; } = "";

    [JsonPropertyName("precision")]
    public string Precision { get; init; } = "";

    [JsonPropertyName("path")]
    public string Path { get; init; } = "";

    [JsonPropertyName("reused")]
    public bool Reused { get; init; }

    [JsonPropertyName("bytes")]
    public long Bytes { get; init; }

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";
}

internal sealed record DiarizationSelfTestRequest
{
    [JsonPropertyName("diarization_engine")]
    public string DiarizationEngine { get; init; } = "community-1";

    [JsonPropertyName("diarization_device")]
    public string DiarizationDevice { get; init; } = "auto";

    [JsonPropertyName("device_index")]
    public int DeviceIndex { get; init; }

    [JsonPropertyName("batch_size")]
    public int BatchSize { get; init; } = 8;

    [JsonPropertyName("min_speakers")]
    public int? MinimumSpeakers { get; init; }

    [JsonPropertyName("max_speakers")]
    public int? MaximumSpeakers { get; init; }

    [JsonPropertyName("huggingface_token")]
    public string? HuggingFaceToken { get; init; }
}

public sealed record DiarizationSelfTestStatus
{
    [JsonPropertyName("ready")]
    public bool Ready { get; init; }

    [JsonPropertyName("engine")]
    public string Engine { get; init; } = "";

    [JsonPropertyName("model")]
    public string Model { get; init; } = "";

    [JsonPropertyName("device")]
    public string Device { get; init; } = "";

    [JsonPropertyName("provider")]
    public string Provider { get; init; } = "";

    [JsonPropertyName("quality")]
    public string Quality { get; init; } = "";

    [JsonPropertyName("elapsed_seconds")]
    public double ElapsedSeconds { get; init; }

    [JsonPropertyName("turn_count")]
    public int TurnCount { get; init; }

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";
}

public sealed record ProfileSelfTestResults
{
    [JsonPropertyName("transcription")]
    public AsrSelfTestStatus? Transcription { get; init; }

    [JsonPropertyName("diarization")]
    public DiarizationSelfTestStatus? Diarization { get; init; }

    [JsonPropertyName("analysis")]
    public AnalysisProviderStatus? Analysis { get; init; }
}

public sealed record ProfileSelfTestStatus
{
    [JsonPropertyName("ready")]
    public bool Ready { get; init; }

    [JsonPropertyName("verified")]
    public bool Verified { get; init; }

    [JsonPropertyName("failed_stage")]
    public string FailedStage { get; init; } = "";

    [JsonPropertyName("elapsed_seconds")]
    public double ElapsedSeconds { get; init; }

    [JsonPropertyName("results")]
    public ProfileSelfTestResults Results { get; init; } = new();

    [JsonPropertyName("recovery")]
    public ProfileSetupAction? Recovery { get; init; }

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";
}

internal sealed record JobRunResult
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("days")]
    public List<JobDayResult> Days { get; init; } = [];

    [JsonPropertyName("requested_days")]
    public int RequestedDays { get; init; }

    [JsonPropertyName("completed_days")]
    public int CompletedDays { get; init; }

    [JsonPropertyName("download_limited")]
    public bool DownloadLimited { get; init; }

    [JsonPropertyName("missing_days")]
    public List<string> MissingDays { get; init; } = [];
}

internal sealed record FeedSchedule
{
    [JsonPropertyName("id")]
    public long Id { get; init; }

    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("feed_name")]
    public string FeedName { get; init; } = "";

    [JsonPropertyName("run_time_local")]
    public string RunTimeLocal { get; init; } = "02:00";

    [JsonPropertyName("lookback_days")]
    public int LookbackDays { get; init; } = 2;

    [JsonPropertyName("backfill_start_date")]
    public string BackfillStartDate { get; init; } = "";

    [JsonPropertyName("job")]
    public JobRequest Job { get; init; } = new();

    [JsonPropertyName("analyze")]
    public bool Analyze { get; init; } = true;

    [JsonPropertyName("enabled")]
    public bool Enabled { get; init; } = true;

    [JsonPropertyName("state")]
    public string State { get; init; } = "scheduled";

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";

    [JsonPropertyName("last_run_date")]
    public string LastRunDate { get; init; } = "";

    [JsonPropertyName("next_run_at")]
    public string NextRunAt { get; init; } = "";

    [JsonPropertyName("due_date")]
    public string DueDate { get; init; } = "";

    public string ScheduleSummary => string.IsNullOrWhiteSpace(BackfillStartDate)
        ? $"Daily at {RunTimeLocal} · revisit {LookbackDays} day{(LookbackDays == 1 ? "" : "s")}"
        : $"Daily at {RunTimeLocal} · revisit {LookbackDays} day{(LookbackDays == 1 ? "" : "s")} · catch up from {BackfillStartDate}";

    public string StateSummary => string.IsNullOrWhiteSpace(Message)
        ? State.Replace('_', ' ')
        : $"{State.Replace('_', ' ')} · {Message}";
}

internal sealed record FeedScheduleSaveRequest
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("feed_name")]
    public string FeedName { get; init; } = "";

    [JsonPropertyName("run_time_local")]
    public string RunTimeLocal { get; init; } = "02:00";

    [JsonPropertyName("lookback_days")]
    public int LookbackDays { get; init; } = 2;

    [JsonPropertyName("backfill_start_date")]
    public string BackfillStartDate { get; init; } = "";

    [JsonPropertyName("job")]
    public JobRequest Job { get; init; } = new();

    [JsonPropertyName("analyze")]
    public bool Analyze { get; init; } = true;

    [JsonPropertyName("enabled")]
    public bool Enabled { get; init; } = true;
}

internal sealed record FeedScheduleFinishRequest
{
    [JsonPropertyName("schedule_id")]
    public long ScheduleId { get; init; }

    [JsonPropertyName("due_date")]
    public string DueDate { get; init; } = "";

    [JsonPropertyName("status")]
    public string Status { get; init; } = "";

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";

    [JsonPropertyName("next_request_at")]
    public string NextRequestAt { get; init; } = "";
}

internal sealed record AreaAcquisitionRequest
{
    [JsonPropertyName("profile_name")]
    public string ProfileName { get; init; } = "";

    [JsonPropertyName("feed_ids")]
    public List<string> FeedIds { get; init; } = [];

    [JsonPropertyName("job")]
    public JobRequest Job { get; init; } = new();
}

internal sealed record AreaFeedJobResult
{
    [JsonPropertyName("feed")]
    public FeedSearchResult Feed { get; init; } = new();

    [JsonPropertyName("result")]
    public JobRunResult Result { get; init; } = new();

    [JsonPropertyName("status")]
    public string Status { get; init; } = "";
}

internal sealed record AreaAcquisitionResult
{
    [JsonPropertyName("id")]
    public long Id { get; init; }

    [JsonPropertyName("status")]
    public string Status { get; init; } = "";

    [JsonPropertyName("start_date")]
    public string StartDate { get; init; } = "";

    [JsonPropertyName("end_date")]
    public string EndDate { get; init; } = "";

    [JsonPropertyName("stop_reason")]
    public string StopReason { get; init; } = "";

    [JsonPropertyName("download_limited")]
    public bool DownloadLimited { get; init; }

    [JsonPropertyName("feed_results")]
    public List<AreaFeedJobResult> FeedResults { get; init; } = [];

    [JsonPropertyName("items")]
    public List<AreaAcquisitionItemState> Items { get; init; } = [];

    public string Summary =>
        $"Queue {Id} · {Status.Replace('_', ' ')} · {Items.Count(value => value.Status == "complete")}/{Items.Count} feeds complete";
}

internal sealed record AreaAcquisitionItemState
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("feed_name")]
    public string FeedName { get; init; } = "";

    [JsonPropertyName("priority_rank")]
    public int PriorityRank { get; init; }

    [JsonPropertyName("status")]
    public string Status { get; init; } = "";

    [JsonPropertyName("requested_days")]
    public int RequestedDays { get; init; }

    [JsonPropertyName("completed_days")]
    public int CompletedDays { get; init; }
}

internal sealed record JobDayResult
{
    [JsonPropertyName("date")]
    public string ArchiveDate { get; init; } = "";

    [JsonPropertyName("transcripts")]
    public List<string> Transcripts { get; init; } = [];
}

public sealed record LibraryDay
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("feed_name")]
    public string FeedName { get; init; } = "";

    [JsonPropertyName("archive_date")]
    public string ArchiveDate { get; init; } = "";

    [JsonPropertyName("day_directory")]
    public string DayDirectory { get; init; } = "";

    [JsonPropertyName("combined_path")]
    public string CombinedPath { get; init; } = "";

    [JsonPropertyName("transcript_path")]
    public string TranscriptPath { get; init; } = "";

    [JsonPropertyName("manifest_path")]
    public string ManifestPath { get; init; } = "";

    [JsonPropertyName("raw_file_count")]
    public int RawFileCount { get; init; }

    [JsonPropertyName("has_combined")]
    public bool HasCombined { get; init; }

    [JsonPropertyName("has_transcript")]
    public bool HasTranscript { get; init; }

    [JsonPropertyName("has_stale_transcript")]
    public bool HasStaleTranscript { get; init; }

    [JsonPropertyName("has_imported_transcript")]
    public bool HasImportedTranscript { get; init; }

    [JsonPropertyName("has_diarization")]
    public bool HasDiarization { get; init; }

    [JsonPropertyName("diarization_engine")]
    public string DiarizationEngine { get; init; } = "";

    [JsonPropertyName("diarization_quality")]
    public string DiarizationQuality { get; init; } = "";

    [JsonPropertyName("speaker_upgrade_available")]
    public bool SpeakerUpgradeAvailable { get; init; }

    [JsonPropertyName("has_analysis")]
    public bool HasAnalysis { get; init; }

    [JsonPropertyName("has_stale_analysis")]
    public bool HasStaleAnalysis { get; init; }

    [JsonPropertyName("incident_count")]
    public int IncidentCount { get; init; }

    [JsonPropertyName("segment_count")]
    public int SegmentCount { get; init; }

    [JsonPropertyName("storage_bytes")]
    public long StorageBytes { get; init; }

    [JsonPropertyName("working_storage_bytes")]
    public long WorkingStorageBytes { get; init; }

    [JsonPropertyName("pipeline_percent")]
    public int PipelinePercent { get; init; }

    [JsonPropertyName("pipeline_summary")]
    public string PipelineSummary { get; init; } = "";

    [JsonPropertyName("status")]
    public string Status { get; init; } = "";

    [JsonPropertyName("status_detail")]
    public string StatusDetail { get; init; } = "";

    [JsonPropertyName("next_step")]
    public string NextStep { get; init; } = "";

    [JsonPropertyName("primary_action")]
    public string PrimaryAction { get; init; } = "";

    [JsonPropertyName("can_open_review")]
    public bool CanOpenReview { get; init; }

    [JsonPropertyName("is_complete")]
    public bool IsComplete { get; init; }

    [JsonPropertyName("needs_network")]
    public bool NeedsNetwork { get; init; }

    [JsonPropertyName("needs_local_processing")]
    public bool NeedsLocalProcessing { get; init; }

    [JsonPropertyName("known_source_count")]
    public int KnownSourceCount { get; init; }

    [JsonPropertyName("retained_source_count")]
    public int RetainedSourceCount { get; init; }

    [JsonPropertyName("missing_source_count")]
    public int MissingSourceCount { get; init; }

    [JsonPropertyName("source_checked_at")]
    public string SourceCheckedAt { get; init; } = "";

    [JsonPropertyName("source_snapshot_complete")]
    public bool SourceSnapshotComplete { get; init; }

    [JsonPropertyName("source_check_due")]
    public bool SourceCheckDue { get; init; }

    [JsonPropertyName("scheduled_missing")]
    public bool ScheduledMissing { get; init; }

    public string FeedAndDate => $"{FeedName} · feed {FeedId} · {ArchiveDate}";
    public string DateAndStatus => $"{ArchiveDate} · {Status}";
    public string StatusAndNext => $"{Status} · Next: {NextStep}";
    public string ProgressSummary => $"{PipelinePercent}% · {NextStep}";
    public string SourceCoverageSummary => SourceSnapshotComplete
        ? $"Last source check: {RetainedSourceCount:N0}/{KnownSourceCount:N0} blocks retained"
            + (SourceCheckDue ? " · refresh due" : "")
        : SourceCheckDue
            ? "Current source availability has not been checked recently"
            : "No complete source-list snapshot is retained";
    public string PrimaryButtonLabel => PrimaryAction switch
    {
        "resume_download" => "Verify & resume",
        "continue_local" => "Finish locally",
        "open_review" => "Open review",
        _ => "Continue",
    };
    public string StorageSummary => WorkingStorageBytes > 0
        ? $"{FormatBytes(StorageBytes, "retained")} · "
            + $"{FormatBytes(WorkingStorageBytes, "temporary")}"
        : FormatBytes(StorageBytes, "retained");

    private static string FormatBytes(long bytes, string label)
    {
        string[] units = ["B", "KB", "MB", "GB", "TB"];
        var value = Math.Max(0, bytes);
        var display = (double)value;
        var unit = 0;
        while (display >= 1024 && unit < units.Length - 1)
        {
            display /= 1024;
            unit++;
        }
        return $"{display:0.#} {units[unit]} {label}";
    }
}

public sealed record LibraryFeedCoverage
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("feed_name")]
    public string FeedName { get; init; } = "";

    [JsonPropertyName("scheduled")]
    public bool Scheduled { get; init; }

    [JsonPropertyName("catch_up_saved")]
    public bool CatchUpSaved { get; init; }

    [JsonPropertyName("catch_up_through_current")]
    public bool CatchUpThroughCurrent { get; init; }

    [JsonPropertyName("catch_up_start_date")]
    public string CatchUpStartDate { get; init; } = "";

    [JsonPropertyName("catch_up_end_date")]
    public string CatchUpEndDate { get; init; } = "";

    [JsonPropertyName("target_start_date")]
    public string TargetStartDate { get; init; } = "";

    [JsonPropertyName("target_end_date")]
    public string TargetEndDate { get; init; } = "";

    [JsonPropertyName("target_day_count")]
    public int TargetDayCount { get; init; }

    [JsonPropertyName("retained_day_count")]
    public int RetainedDayCount { get; init; }

    [JsonPropertyName("ready_day_count")]
    public int ReadyDayCount { get; init; }

    [JsonPropertyName("incomplete_day_count")]
    public int IncompleteDayCount { get; init; }

    [JsonPropertyName("missing_day_count")]
    public int MissingDayCount { get; init; }

    [JsonPropertyName("missing_dates")]
    public List<string> MissingDates { get; init; } = [];

    [JsonPropertyName("source_check_due_count")]
    public int SourceCheckDueCount { get; init; }

    [JsonPropertyName("network_day_count")]
    public int NetworkDayCount { get; init; }

    [JsonPropertyName("local_processing_day_count")]
    public int LocalProcessingDayCount { get; init; }

    [JsonPropertyName("backlog_count")]
    public int BacklogCount { get; init; }

    [JsonPropertyName("latest_local_date")]
    public string LatestLocalDate { get; init; } = "";

    [JsonPropertyName("days_behind_today")]
    public int? DaysBehindToday { get; init; }

    [JsonPropertyName("last_source_check_at")]
    public string LastSourceCheckAt { get; init; } = "";

    [JsonPropertyName("known_source_block_count")]
    public int KnownSourceBlockCount { get; init; }

    [JsonPropertyName("retained_source_block_count")]
    public int RetainedSourceBlockCount { get; init; }

    [JsonPropertyName("missing_source_block_count")]
    public int MissingSourceBlockCount { get; init; }

    [JsonPropertyName("progress_percent")]
    public int ProgressPercent { get; init; }

    [JsonPropertyName("status")]
    public string Status { get; init; } = "";

    public string FeedLabel => $"{FeedName} · feed {FeedId}";
    public string TargetSummary => CatchUpSaved
        ? CatchUpThroughCurrent
            ? $"Saved catch-up {CatchUpStartDate} through today ({TargetEndDate})"
                + (Scheduled ? " · daily schedule also active" : "")
                + $" · {RetainedDayCount}/{TargetDayCount} target days retained"
            : $"Saved catch-up {CatchUpStartDate} through {CatchUpEndDate}"
            + (Scheduled ? " · daily schedule also active" : "")
            + $" · {RetainedDayCount}/{TargetDayCount} target days retained"
        : Scheduled
            ? $"Scheduled {TargetStartDate} through {TargetEndDate} · {RetainedDayCount}/{TargetDayCount} days retained"
            : $"{RetainedDayCount} retained day{(RetainedDayCount == 1 ? "" : "s")} · no active target range";
    public string WorkSummary => BacklogCount == 0
        ? Status
        : $"{Status} · {LocalProcessingDayCount} local · {NetworkDayCount} source/network";
    public string SourceSummary => KnownSourceBlockCount > 0
        ? $"Last-known provider blocks: {RetainedSourceBlockCount:N0}/{KnownSourceBlockCount:N0} retained"
            + (MissingSourceBlockCount > 0
                ? $" · {MissingSourceBlockCount:N0} missing"
                : "")
            + (string.IsNullOrWhiteSpace(LastSourceCheckDisplay)
                ? ""
                : $" · checked {LastSourceCheckDisplay}")
        : "No authenticated source-list snapshot is retained for this target range";
    private string LastSourceCheckDisplay =>
        DateTimeOffset.TryParse(LastSourceCheckAt, out var parsed)
            ? parsed.ToLocalTime().ToString("g")
            : LastSourceCheckAt;
}

internal sealed record LibrarySummary
{
    [JsonPropertyName("feed_count")]
    public int FeedCount { get; init; }

    [JsonPropertyName("day_count")]
    public int DayCount { get; init; }

    [JsonPropertyName("complete_count")]
    public int CompleteCount { get; init; }

    [JsonPropertyName("attention_count")]
    public int AttentionCount { get; init; }

    [JsonPropertyName("storage_bytes")]
    public long StorageBytes { get; init; }

    [JsonPropertyName("working_storage_bytes")]
    public long WorkingStorageBytes { get; init; }

    [JsonPropertyName("backlog_count")]
    public int BacklogCount { get; init; }

    [JsonPropertyName("missing_day_count")]
    public int MissingDayCount { get; init; }

    [JsonPropertyName("network_day_count")]
    public int NetworkDayCount { get; init; }
}

internal sealed record LibraryResponse
{
    [JsonPropertyName("days")]
    public List<LibraryDay> Days { get; init; } = [];

    [JsonPropertyName("feeds")]
    public List<LibraryFeedCoverage> Feeds { get; init; } = [];

    [JsonPropertyName("summary")]
    public LibrarySummary Summary { get; init; } = new();
}

internal sealed record LibraryResumePlan
{
    [JsonPropertyName("days")]
    public List<LibraryDay> Days { get; init; } = [];

    [JsonPropertyName("feeds")]
    public List<LibraryFeedCoverage> Feeds { get; init; } = [];

    [JsonPropertyName("local_count")]
    public int LocalCount { get; init; }

    [JsonPropertyName("network_count")]
    public int NetworkCount { get; init; }

    [JsonPropertyName("quota")]
    public ArchiveQuotaStatus Quota { get; init; } = new();

    [JsonPropertyName("scope_feed_id")]
    public string ScopeFeedId { get; init; } = "";

    [JsonPropertyName("scope_start_date")]
    public string ScopeStartDate { get; init; } = "";

    [JsonPropertyName("scope_end_date")]
    public string ScopeEndDate { get; init; } = "";

    [JsonPropertyName("scope_through_current")]
    public bool ScopeThroughCurrent { get; init; }
}

internal sealed record LibraryFeedDeleteResult
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("directory_deleted")]
    public bool DirectoryDeleted { get; init; }

    [JsonPropertyName("cleanup_pending")]
    public bool CleanupPending { get; init; }

    [JsonPropertyName("days_deleted")]
    public int DaysDeleted { get; init; }

    [JsonPropertyName("segments_deleted")]
    public int SegmentsDeleted { get; init; }

    [JsonPropertyName("incidents_deleted")]
    public int IncidentsDeleted { get; init; }

    [JsonPropertyName("area_digests_invalidated")]
    public int AreaDigestsInvalidated { get; init; }

    [JsonPropertyName("schedules_deleted")]
    public int SchedulesDeleted { get; init; }

    [JsonPropertyName("catchups_deleted")]
    public int CatchUpsDeleted { get; init; }
}

internal sealed record ManagedRuntimeStatus
{
    [JsonPropertyName("profile")]
    public string Profile { get; init; } = "";

    [JsonPropertyName("display_name")]
    public string DisplayName { get; init; } = "";

    [JsonPropertyName("revision")]
    public string Revision { get; init; } = "";

    [JsonPropertyName("ready")]
    public bool Ready { get; init; }

    [JsonPropertyName("partial")]
    public bool Partial { get; init; }

    [JsonPropertyName("python_path")]
    public string PythonPath { get; init; } = "";

    [JsonPropertyName("storage_path")]
    public string StoragePath { get; init; } = "";

    [JsonPropertyName("cache_path")]
    public string CachePath { get; init; } = "";

    [JsonPropertyName("estimated_installed_bytes")]
    public long EstimatedInstalledBytes { get; init; }

    [JsonPropertyName("installed_bytes")]
    public long InstalledBytes { get; init; }

    [JsonPropertyName("packages")]
    public List<string> Packages { get; init; } = [];

    [JsonPropertyName("source_urls")]
    public List<string> SourceUrls { get; init; } = [];

    [JsonPropertyName("licenses")]
    public List<string> Licenses { get; init; } = [];

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";

    [JsonPropertyName("installed_at")]
    public string InstalledAt { get; init; } = "";

    [JsonPropertyName("cuda_available_at_install")]
    public bool CudaAvailableAtInstall { get; init; }

    public string EstimatedStorage => FormatBytes(EstimatedInstalledBytes);
    public string InstalledStorage => FormatBytes(InstalledBytes);

    private static string FormatBytes(long bytes)
    {
        string[] units = ["B", "KB", "MB", "GB", "TB"];
        var display = (double)Math.Max(0, bytes);
        var unit = 0;
        while (display >= 1024 && unit < units.Length - 1)
        {
            display /= 1024;
            unit++;
        }
        return $"{display:0.#} {units[unit]}";
    }
}

public sealed record ProfileSetupAction
{
    [JsonPropertyName("stage")]
    public string Stage { get; init; } = "";

    [JsonPropertyName("kind")]
    public string Kind { get; init; } = "";

    [JsonPropertyName("label")]
    public string Label { get; init; } = "";

    [JsonPropertyName("message")]
    public string Message { get; init; } = "";
}

public sealed record HardwareProfileStatus
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = "";

    [JsonPropertyName("name")]
    public string Name { get; init; } = "";

    [JsonPropertyName("ready")]
    public bool Ready { get; init; }

    [JsonPropertyName("configured")]
    public bool Configured { get; init; }

    [JsonPropertyName("verified")]
    public bool Verified { get; init; }

    [JsonPropertyName("transcription_ready")]
    public bool TranscriptionReady { get; init; }

    [JsonPropertyName("diarization_ready")]
    public bool DiarizationReady { get; init; }

    [JsonPropertyName("analysis_ready")]
    public bool AnalysisReady { get; init; }

    [JsonPropertyName("transcription")]
    public string Transcription { get; init; } = "";

    [JsonPropertyName("diarization")]
    public string Diarization { get; init; } = "";

    [JsonPropertyName("analysis")]
    public string Analysis { get; init; } = "";

    [JsonPropertyName("note")]
    public string Note { get; init; } = "";

    [JsonPropertyName("next_steps")]
    public List<string> NextSteps { get; init; } = [];

    [JsonPropertyName("next_action")]
    public ProfileSetupAction? NextAction { get; init; }

    public string StateText => Verified || Ready
        ? "Verified"
        : Configured
            ? "Detected"
            : "Setup needed";
    public string StageSummary =>
        $"Transcription: {Transcription}\nDiarization: {Diarization}\nAnalysis: {Analysis}";
    public string NextStepSummary => Verified || Ready || NextSteps.Count == 0
        ? ""
        : "Next:\n• " + string.Join("\n• ", NextSteps);
}

internal sealed record LocalProcessingRequest : AnalysisProviderRequest
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("archive_date")]
    public string ArchiveDate { get; init; } = "";

    [JsonPropertyName("output_dir")]
    public string OutputDirectory { get; init; } = "archives";

    [JsonPropertyName("model")]
    public string Model { get; init; } = "turbo";

    [JsonPropertyName("asr_engine")]
    public string AsrEngine { get; init; } = "auto";

    [JsonPropertyName("device")]
    public string Device { get; init; } = "auto";

    [JsonPropertyName("device_index")]
    public int DeviceIndex { get; init; }

    [JsonPropertyName("compute_type")]
    public string ComputeType { get; init; } = "auto";

    [JsonPropertyName("asr_model_path")]
    public string? AsrModelPath { get; init; }

    [JsonPropertyName("diarization_engine")]
    public string DiarizationEngine { get; init; } = "community-1";

    [JsonPropertyName("diarization_device")]
    public string DiarizationDevice { get; init; } = "auto";

    [JsonPropertyName("batch_size")]
    public int BatchSize { get; init; } = 8;

    [JsonPropertyName("diarize")]
    public bool Diarize { get; init; } = true;

    [JsonPropertyName("analyze")]
    public bool Analyze { get; init; } = true;

    [JsonPropertyName("min_speakers")]
    public int? MinimumSpeakers { get; init; }

    [JsonPropertyName("max_speakers")]
    public int? MaximumSpeakers { get; init; }

    [JsonPropertyName("huggingface_token")]
    public string? HuggingFaceToken { get; init; }
}

public sealed record AnalysisDay
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("feed_name")]
    public string FeedName { get; init; } = "";

    [JsonPropertyName("archive_date")]
    public string ArchiveDate { get; init; } = "";

    [JsonPropertyName("duration_seconds")]
    public double DurationSeconds { get; init; }

    [JsonPropertyName("segment_count")]
    public int SegmentCount { get; init; }

    [JsonPropertyName("speaker_count")]
    public int SpeakerCount { get; init; }

    [JsonPropertyName("incident_count")]
    public int IncidentCount { get; init; }

    [JsonPropertyName("has_summary")]
    public int HasSummaryValue { get; init; }

    [JsonPropertyName("analysis_current")]
    public bool AnalysisCurrent { get; init; }

    [JsonPropertyName("analysis_update_required")]
    public bool AnalysisUpdateRequired { get; init; }

    [JsonPropertyName("transcript_import_required")]
    public bool TranscriptImportRequired { get; init; }

    [JsonPropertyName("has_diarization")]
    public int HasDiarizationValue { get; init; }

    public string FeedAndDate => $"{(string.IsNullOrWhiteSpace(FeedName) ? $"Feed {FeedId}" : FeedName)} · {ArchiveDate}";
    public string ProcessingSummary => TranscriptImportRequired
        ? $"Current transcript awaits database import · saved analysis hidden · {DurationSeconds / 3600:0.0} hours"
        : $"{SegmentCount:N0} segments · {IncidentCount:N0} incidents · {DurationSeconds / 3600:0.0} hours"
          + (AnalysisUpdateRequired ? " · analysis update required" : "");
    public string SpeakerSummary => HasDiarizationValue != 0
        ? $"Diarized · {SpeakerCount} transcript clusters"
        : "Not diarized";
    public bool HasSummary => HasSummaryValue != 0;
}

public sealed record IncidentRecord
{
    [JsonPropertyName("id")]
    public long Id { get; init; }

    [JsonPropertyName("event_type")]
    public string EventType { get; init; } = "";

    [JsonPropertyName("title")]
    public string Title { get; init; } = "";

    [JsonPropertyName("summary")]
    public string Summary { get; init; } = "";

    [JsonPropertyName("location")]
    public string? Location { get; init; }

    [JsonPropertyName("priority")]
    public int Priority { get; init; }

    [JsonPropertyName("confidence")]
    public double Confidence { get; init; }

    [JsonPropertyName("start_seconds")]
    public double StartSeconds { get; init; }

    [JsonPropertyName("end_seconds")]
    public double EndSeconds { get; init; }

    [JsonPropertyName("archive_time")]
    public string ArchiveTime { get; init; } = "";

    [JsonPropertyName("evidence_quote")]
    public string EvidenceQuote { get; init; } = "";

    public string TypeAndPriority => $"P{Priority} · {EventType.Replace('_', ' ')}";
    public string TimeAndLocation => string.IsNullOrWhiteSpace(Location)
        ? ArchiveTime
        : $"{ArchiveTime} · {Location}";
    public string ConfidenceSummary => $"{Confidence:P0} extraction confidence · I{Id}";
    public string EvidenceQuoteDisplay => string.IsNullOrWhiteSpace(EvidenceQuote)
        ? ""
        : $"Cited radio: “{EvidenceQuote}”";
}

public sealed record IncidentClip
{
    [JsonPropertyName("incident_id")]
    public long IncidentId { get; init; }

    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("archive_date")]
    public string ArchiveDate { get; init; } = "";

    [JsonPropertyName("archive_time")]
    public string ArchiveTime { get; init; } = "";

    [JsonPropertyName("start_seconds")]
    public double StartSeconds { get; init; }

    [JsonPropertyName("end_seconds")]
    public double EndSeconds { get; init; }

    [JsonPropertyName("duration_seconds")]
    public double DurationSeconds { get; init; }

    [JsonPropertyName("clip_kind")]
    public string ClipKind { get; init; } = "evidence";

    [JsonPropertyName("path")]
    public string Path { get; init; } = "";

    [JsonPropertyName("sha256")]
    public string Sha256 { get; init; } = "";
}

public sealed record DayReport
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("archive_date")]
    public string ArchiveDate { get; init; } = "";

    [JsonPropertyName("summary")]
    public string Summary { get; init; } = "";

    [JsonPropertyName("incidents")]
    public List<IncidentRecord> Incidents { get; init; } = [];

    [JsonPropertyName("audio_path")]
    public string AudioPath { get; init; } = "";

    [JsonPropertyName("has_diarization")]
    public bool HasDiarization { get; init; }

    [JsonPropertyName("analysis_current")]
    public bool AnalysisCurrent { get; init; }

    [JsonPropertyName("analysis_update_required")]
    public bool AnalysisUpdateRequired { get; init; }
}

internal sealed record AnalysisRequest : AnalysisProviderRequest
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("archive_date")]
    public string ArchiveDate { get; init; } = "";

    [JsonPropertyName("output_dir")]
    public string OutputDirectory { get; init; } = "archives";

    [JsonPropertyName("force")]
    public bool Force { get; init; }

    [JsonPropertyName("force_summary")]
    public bool ForceSummary { get; init; }
}

internal sealed record WeeklySummaryRequest : AnalysisProviderRequest
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("week_ending")]
    public string WeekEnding { get; init; } = "";

    [JsonPropertyName("force")]
    public bool Force { get; init; }
}

public sealed record WeeklyReport
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("start_date")]
    public string StartDate { get; init; } = "";

    [JsonPropertyName("end_date")]
    public string EndDate { get; init; } = "";

    [JsonPropertyName("summary")]
    public string Summary { get; init; } = "";

    [JsonPropertyName("notable_incident_ids")]
    public List<long> NotableIncidentIds { get; init; } = [];

    [JsonPropertyName("days_available")]
    public int DaysAvailable { get; init; }

    [JsonPropertyName("days_expected")]
    public int DaysExpected { get; init; } = 7;

    [JsonPropertyName("missing_dates")]
    public List<string> MissingDates { get; init; } = [];

    [JsonPropertyName("analysis_update_dates")]
    public List<string> AnalysisUpdateDates { get; init; } = [];

    [JsonPropertyName("incident_count")]
    public int IncidentCount { get; init; }

    [JsonPropertyName("priority_4_5_count")]
    public int SeriousIncidentCount { get; init; }

    [JsonPropertyName("cached")]
    public bool Cached { get; init; }

    public string CoverageSummary =>
        $"{StartDate} through {EndDate} · {DaysAvailable}/{DaysExpected} days available · "
        + $"{IncidentCount} incidents · {SeriousIncidentCount} priority 4–5"
        + (MissingDates.Count == 0 ? "" : $" · Missing: {string.Join(", ", MissingDates)}")
        + (AnalysisUpdateDates.Count == 0
            ? ""
            : $" · Reanalysis needed: {string.Join(", ", AnalysisUpdateDates)}");

    public string NotableRecordsSummary => NotableIncidentIds.Count == 0
        ? ""
        : "Notable incident records: " + string.Join(", ", NotableIncidentIds.Select(value => $"I{value}"));
}

internal sealed record ArchiveQuestionRequest : AnalysisProviderRequest
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("start_date")]
    public string StartDate { get; init; } = "";

    [JsonPropertyName("end_date")]
    public string EndDate { get; init; } = "";

    [JsonPropertyName("output_dir")]
    public string OutputDirectory { get; init; } = "archives";

    [JsonPropertyName("question")]
    public string Question { get; init; } = "";

    [JsonPropertyName("history")]
    public List<ArchiveConversationTurn> History { get; init; } = [];
}

public sealed record ArchiveQuestionCoverage
{
    [JsonPropertyName("feed_id")]
    public string FeedId { get; init; } = "";

    [JsonPropertyName("start_date")]
    public string StartDate { get; init; } = "";

    [JsonPropertyName("end_date")]
    public string EndDate { get; init; } = "";

    [JsonPropertyName("requested_day_count")]
    public int RequestedDayCount { get; init; }

    [JsonPropertyName("audio_day_count")]
    public int AudioDayCount { get; init; }

    [JsonPropertyName("question_ready_day_count")]
    public int QuestionReadyDayCount { get; init; }

    [JsonPropertyName("analyzed_day_count")]
    public int AnalyzedDayCount { get; init; }

    [JsonPropertyName("question_ready_dates")]
    public List<string> QuestionReadyDates { get; init; } = [];

    [JsonPropertyName("local_processing_dates")]
    public List<string> LocalProcessingDates { get; init; } = [];

    [JsonPropertyName("missing_audio_dates")]
    public List<string> MissingAudioDates { get; init; } = [];

    [JsonPropertyName("unavailable_dates")]
    public List<string> UnavailableDates { get; init; } = [];

    [JsonPropertyName("complete_coverage")]
    public bool CompleteCoverage { get; init; }

    [JsonPropertyName("summary")]
    public string Summary { get; init; } = "";

    public string DisplaySummary => string.IsNullOrWhiteSpace(Summary)
        ? $"{QuestionReadyDayCount:N0}/{RequestedDayCount:N0} days question-ready · "
            + $"audio retained for {AudioDayCount:N0}/{RequestedDayCount:N0}"
        : Summary;
}

internal sealed record ArchiveConversationTurn
{
    [JsonPropertyName("role")]
    public string Role { get; init; } = "";

    [JsonPropertyName("content")]
    public string Content { get; init; } = "";
}

public sealed record ArchiveChatMessage
{
    public string Role { get; init; } = "";
    public string Content { get; init; } = "";
    public List<string> EvidenceIds { get; init; } = [];
    public List<string> Limitations { get; init; } = [];
    public ArchiveQuestionCoverage Coverage { get; init; } = new();
    public string SpeakerLabel => Role == "user" ? "You" : "Archive assistant";
    public string EvidenceSummary => EvidenceIds.Count == 0
        ? ""
        : "Evidence: " + string.Join(", ", EvidenceIds);
    public string LimitationSummary => Limitations.Count == 0
        ? ""
        : "Limitations: " + string.Join("; ", Limitations);
    public string CoverageSummary => Coverage.RequestedDayCount == 0
        ? ""
        : "Coverage: " + Coverage.DisplaySummary;
}

public sealed record ArchiveAnswer
{
    [JsonPropertyName("answer")]
    public string Answer { get; init; } = "";

    [JsonPropertyName("evidence_ids")]
    public List<string> EvidenceIds { get; init; } = [];

    [JsonPropertyName("limitations")]
    public List<string> Limitations { get; init; } = [];

    [JsonPropertyName("coverage")]
    public ArchiveQuestionCoverage Coverage { get; init; } = new();

    public string LimitationsSummary => Limitations.Count == 0
        ? ""
        : "Limitations: " + string.Join("; ", Limitations);
}

public sealed record ArchiveQuotaStatus
{
    [JsonPropertyName("instance_id")]
    public string InstanceId { get; init; } = "";

    [JsonPropertyName("provider_limit")]
    public int ProviderLimit { get; init; }

    [JsonPropertyName("automated_limit")]
    public int AutomatedLimit { get; init; }

    [JsonPropertyName("user_reserve")]
    public int UserReserve { get; init; }

    [JsonPropertyName("used")]
    public int Used { get; init; }

    [JsonPropertyName("remaining")]
    public int Remaining { get; init; }

    [JsonPropertyName("available")]
    public bool Available { get; init; }

    [JsonPropertyName("blocked")]
    public bool Blocked { get; init; }

    [JsonPropertyName("blocked_reason")]
    public string BlockedReason { get; init; } = "";

    [JsonPropertyName("blocked_until")]
    public string BlockedUntil { get; init; } = "";

    [JsonPropertyName("next_request_at")]
    public string NextRequestAt { get; init; } = "";
}
