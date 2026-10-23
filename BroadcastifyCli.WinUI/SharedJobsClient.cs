using System.Net.Http.Json;
using System.Text;
using System.Text.Json;

namespace BroadcastifyCli.WinUI;

internal sealed partial class WorkerClient
{
    private readonly AsyncLocal<SharedJobObservation?> _sharedObservation = new();

    internal async Task<JsonElement> SharedJobsRequestAsync(string path, object? value = null)
    {
        var port = Volatile.Read(ref _lanNodePort);
        if (port == 0) throw new InvalidOperationException("Enable Windows master sharing to use shared Jobs.");
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
        var key = ReadPrivateSetting("BROADCASTIFY_LAN_SYNC_KEY");
        if (!string.IsNullOrWhiteSpace(key)) client.DefaultRequestHeaders.Add("X-Radio-Archive-LAN-Key", key);
        var url = $"http://127.0.0.1:{port}/api/lan/v1/jobs{path}";
        // The bounded LAN reader requires Content-Length; PostAsJsonAsync
        // streams chunked JSON, which that endpoint deliberately does not read.
        using var body = value is null ? null : new StringContent(
            JsonSerializer.Serialize(value), Encoding.UTF8, "application/json");
        using var response = body is null ? await client.GetAsync(url).ConfigureAwait(false)
            : await client.PostAsync(url, body).ConfigureAwait(false);
        var result = await response.Content.ReadFromJsonAsync<JsonElement>().ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw new InvalidOperationException(result.TryGetProperty("error", out var error)
                ? error.GetString() : "Shared jobs are unavailable.");
        return result;
    }

    internal async Task<SharedJobObservation?> BeginSharedJobAsync(JobRequest request, bool analyze,
        string accountProfile, bool automaticProfilePolicy, CancellationToken cancellationToken)
    {
        if (Volatile.Read(ref _lanNodePort) == 0 || IsPreview) return null;
        var observation = new SharedJobObservation(this, request, analyze, accountProfile,
            automaticProfilePolicy, cancellationToken);
        await observation.PublishAsync("running").ConfigureAwait(false);
        observation.Start();
        return observation;
    }

    // A synchronous scope is intentional: AsyncLocal changes made inside an
    // awaited method would not flow back into the caller's worker operations.
    internal IDisposable ObserveSharedJob(SharedJobObservation? observation)
    {
        var previous = _sharedObservation.Value;
        _sharedObservation.Value = observation;
        return new SharedObservationScope(() => _sharedObservation.Value = previous);
    }

    private sealed class SharedObservationScope(Action close) : IDisposable
    {
        public void Dispose() => close();
    }

    internal sealed class SharedJobObservation
    {
        private readonly WorkerClient _client;
        private readonly object _request;
        private readonly object _gate = new();
        private readonly List<JsonElement> _events = [];
        private readonly CancellationTokenSource _heartbeat = new();
        private readonly CancellationTokenSource _work;
        private Task _task = Task.CompletedTask;
        private long _eventIndex;
        private readonly string _id = Guid.NewGuid().ToString("N")[..16];
        private readonly string _created = DateTimeOffset.UtcNow.ToString("O");
        private readonly string _profile;
        private readonly string _feed;
        public string Control { get; private set; } = "";
        public CancellationToken Token => _work.Token;

        internal SharedJobObservation(WorkerClient client, JobRequest request, bool analyze,
            string profile, bool automaticProfilePolicy, CancellationToken cancellationToken)
        {
            _client = client;
            // The server persists only non-secret workflow settings.
            var persistedRequest = request with { AnalysisApiKey = null, HuggingFaceToken = null };
            _request = new { job = JsonSerializer.SerializeToElement(persistedRequest, JsonOptions), analyze,
                account_profile_id = automaticProfilePolicy ? "automatic" : profile };
            _profile = profile;
            _feed = request.FeedId;
            _work = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        }

        internal void Report(JsonElement message)
        {
            lock (_gate)
            {
                var allowed = new[] { "type", "stage", "phase", "current", "total", "message", "feed_id", "date", "elapsed_seconds", "estimated_remaining_seconds" };
                var fields = message.EnumerateObject().Where(p => allowed.Contains(p.Name))
                    .ToDictionary(p => p.Name, p => p.Value.ValueKind == JsonValueKind.String
                        ? (object)(p.Value.GetString() ?? "")[..Math.Min(1024, (p.Value.GetString() ?? "").Length)] : p.Value.Clone());
                fields["received_at"] = DateTimeOffset.UtcNow.ToString("O");
                fields["event_index"] = _eventIndex++;
                _events.Add(JsonSerializer.SerializeToElement(fields));
                if (_events.Count > 100) _events.RemoveRange(0, _events.Count - 100);
            }
        }

        internal async Task PublishAsync(string status, string error = "")
        {
            JsonElement[] events;
            lock (_gate) events = _events.ToArray();
            var response = await _client.SharedJobsRequestAsync("/observe", new
            {
                id = _id, command = "run-scheduled", request = _request,
                owner_pid = Environment.ProcessId, account_profile_id = _profile,
                feed_id = _feed, status, created_at = _created, started_at = _created,
                finished_at = status == "running" ? "" : DateTimeOffset.UtcNow.ToString("O"),
                events, error,
            }).ConfigureAwait(false);
            var job = response.GetProperty("job");
            Control = job.TryGetProperty("control", out var control) ? control.GetString() ?? "" : "";
            if (Control is "pause" or "cancel") _work.Cancel();
        }

        internal void Start() => _task = Task.Run(async () =>
        {
            try
            {
                while (!_heartbeat.IsCancellationRequested)
                {
                    await Task.Delay(2000, _heartbeat.Token).ConfigureAwait(false);
                    try { await PublishAsync("running").ConfigureAwait(false); }
                    catch (HttpRequestException) { /* Preserve work while the companion reconnects. */ }
                    catch (TaskCanceledException) { /* The request timeout must not end the job observer. */ }
                    catch (InvalidOperationException) { /* The companion may be stopping or restarting. */ }
                }
            }
            catch (OperationCanceledException) { }
        });

        internal async Task FinishAsync(string status, string error)
        {
            _heartbeat.Cancel();
            await _task.ConfigureAwait(false);
            try { await PublishAsync(status, error).ConfigureAwait(false); }
            catch (Exception exception) when (exception is HttpRequestException or TaskCanceledException or InvalidOperationException)
            { /* The durable record remains recoverable if the service is shutting down. */ }
            _work.Dispose();
            _heartbeat.Dispose();
        }
    }
}
