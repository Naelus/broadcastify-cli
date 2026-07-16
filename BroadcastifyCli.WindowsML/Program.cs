using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace BroadcastifyCli.WindowsML;

internal static class Program
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    public static async Task<int> Main(string[] args)
    {
        try
        {
            var options = ParseArguments(args);
            string[] registeredProviders = [];
            if (options.ContainsKey("ensure-winml"))
            {
#if WINDOWS_ML_CATALOG
                registeredProviders = await EnsureWindowsMlProviders();
#else
                throw new PlatformNotSupportedException(
                    "This helper was not built with the Windows ML provider catalog.");
#endif
            }
            else if (options.ContainsKey("register-winml"))
            {
#if WINDOWS_ML_CATALOG
                registeredProviders = await RegisterReadyWindowsMlProviders();
#else
                throw new PlatformNotSupportedException(
                    "This helper was not built with the Windows ML provider catalog.");
#endif
            }

            if (options.ContainsKey("providers"))
            {
#if WINDOWS_ML_CATALOG
                var providers = GetWindowsMlProviders();
                WriteJson(new
                {
                    ready = true,
                    providers,
                    registered_providers = registeredProviders,
                });
                return 0;
#else
                throw new PlatformNotSupportedException(
                    "This helper was not built with the Windows ML provider catalog.");
#endif
            }

            if (options.ContainsKey("probe"))
            {
                var configuredModel = options.GetValueOrDefault("model");
                var decodeReady = false;
                var configuredProvider = "";
                var backend = "Windows ML (ONNX Runtime GenAI)";
                if (!string.IsNullOrWhiteSpace(configuredModel))
                {
                    if (!Directory.Exists(configuredModel))
                    {
                        throw new DirectoryNotFoundException(
                            $"Windows ML model directory not found: {configuredModel}");
                    }
                    configuredProvider = GetConfiguredProvider(configuredModel);
                    backend = DescribeBackend(configuredProvider);
                    decodeReady = RunDecodeSelfTest(configuredModel);
                }
                WriteJson(new
                {
                    ready = true,
                    decode_ready = decodeReady,
                    backend,
                    configured_provider = configuredProvider,
                    architecture = Environment.Is64BitProcess ? "x64" : "x86",
                    registered_providers = registeredProviders,
                });
                return 0;
            }

            var modelPath = Required(options, "model");
            if (!Directory.Exists(modelPath))
            {
                throw new DirectoryNotFoundException($"Windows ML model directory not found: {modelPath}");
            }

            if (options.ContainsKey("stream"))
            {
                return TranscribeStream(modelPath);
            }

            var audioPath = Required(options, "audio");
            if (!File.Exists(audioPath))
            {
                throw new FileNotFoundException("Audio chunk not found.", audioPath);
            }

            var text = Transcribe(modelPath, audioPath);
            var modelProvider = GetConfiguredProvider(modelPath);
            WriteJson(new
            {
                text,
                backend = DescribeBackend(modelProvider),
                configured_provider = modelProvider,
            });
            return 0;
        }
        catch (Exception exception)
        {
            WriteJson(new
            {
                error = exception.Message,
                type = exception.GetType().Name,
            });
            return 1;
        }
    }

    private static string Transcribe(string modelPath, string audioPath)
    {
        using var runtime = new OgaHandle();
        using var config = new Config(modelPath);
        using var model = new Model(config);
        using var processor = new MultiModalProcessor(model);
        return Transcribe(model, processor, audioPath);
    }

    private static string Transcribe(
        Model model,
        MultiModalProcessor processor,
        string audioPath)
    {
        using var audios = Audios.Load([audioPath]);
        const string prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>";
        using var inputs = processor.ProcessImagesAndAudios([prompt], null, audios);
        using var parameters = new GeneratorParams(model);
        parameters.SetSearchOption("do_sample", false);
        parameters.SetSearchOption("num_beams", 1d);
        parameters.SetSearchOption("num_return_sequences", 1d);
        parameters.SetSearchOption("max_length", 448d);
        parameters.SetSearchOption("batch_size", 1d);
        using var generator = new Generator(model, parameters);
        generator.SetInputs(inputs);
        while (!generator.IsDone())
        {
            generator.GenerateNextToken();
        }
        return processor.Decode(generator.GetSequence(0)).Trim();
    }

    private static int TranscribeStream(string modelPath)
    {
        using var runtime = new OgaHandle();
        using var config = new Config(modelPath);
        using var model = new Model(config);
        using var processor = new MultiModalProcessor(model);
        var configuredProvider = GetConfiguredProvider(modelPath);
        var backend = DescribeBackend(configuredProvider);
        while (Console.In.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }
            using var request = JsonDocument.Parse(line);
            var root = request.RootElement;
            if (root.TryGetProperty("command", out var command)
                && string.Equals(command.GetString(), "stop", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }
            var audioPath = root.GetProperty("path").GetString()
                ?? throw new ArgumentException("Stream request path is required.");
            if (!File.Exists(audioPath))
            {
                throw new FileNotFoundException("Stream audio chunk not found.", audioPath);
            }
            WriteJson(new
            {
                text = Transcribe(model, processor, audioPath),
                backend,
                configured_provider = configuredProvider,
            });
        }
        return 0;
    }

    private static bool RunDecodeSelfTest(string modelPath)
    {
        var audioPath = Path.Combine(
            Path.GetTempPath(), $"broadcastify-winml-self-test-{Guid.NewGuid():N}.wav");
        try
        {
            WriteSilentWave(audioPath, sampleRate: 16_000, seconds: 1);
            _ = Transcribe(modelPath, audioPath);
            return true;
        }
        finally
        {
            try
            {
                File.Delete(audioPath);
            }
            catch (IOException)
            {
                // A failed cleanup must not turn a successful decode into a failed probe.
            }
        }
    }

    private static void WriteSilentWave(string path, int sampleRate, int seconds)
    {
        const short channels = 1;
        const short bitsPerSample = 16;
        var dataLength = sampleRate * seconds * channels * (bitsPerSample / 8);
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write("RIFF"u8.ToArray());
        writer.Write(36 + dataLength);
        writer.Write("WAVE"u8.ToArray());
        writer.Write("fmt "u8.ToArray());
        writer.Write(16);
        writer.Write((short)1);
        writer.Write(channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * (bitsPerSample / 8));
        writer.Write((short)(channels * (bitsPerSample / 8)));
        writer.Write(bitsPerSample);
        writer.Write("data"u8.ToArray());
        writer.Write(dataLength);
        writer.Write(new byte[dataLength]);
    }

    private static string GetConfiguredProvider(string modelPath)
    {
        var configPath = Path.Combine(modelPath, "genai_config.json");
        using var document = JsonDocument.Parse(File.ReadAllText(configPath));
        var decoder = document.RootElement.GetProperty("model").GetProperty("decoder");
        if (!decoder.TryGetProperty("session_options", out var sessionOptions)
            || !sessionOptions.TryGetProperty("provider_options", out var providerOptions)
            || providerOptions.ValueKind != JsonValueKind.Array)
        {
            return "CPU";
        }
        foreach (var item in providerOptions.EnumerateArray())
        {
            foreach (var provider in item.EnumerateObject())
            {
                return provider.Name;
            }
        }
        return "CPU";
    }

    private static string DescribeBackend(string configuredProvider) =>
        configuredProvider.Trim().ToLowerInvariant() switch
        {
            "cpu" => "Windows ML / ONNX Runtime GenAI CPU",
            "dml" or "directml" => "ONNX Runtime GenAI DirectML",
            "nvtensorrtrtx" or "trt-rtx" => "Windows ML TensorRT RTX",
            var provider => $"Windows ML / ONNX Runtime GenAI ({provider})",
        };

    private static Dictionary<string, string?> ParseArguments(IEnumerable<string> args)
    {
        var values = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        var input = args.ToArray();
        for (var index = 0; index < input.Length; index++)
        {
            var value = input[index];
            if (!value.StartsWith("--", StringComparison.Ordinal))
            {
                throw new ArgumentException($"Unexpected argument: {value}");
            }
            var key = value[2..];
            if (string.Equals(key, "probe", StringComparison.OrdinalIgnoreCase)
                || string.Equals(key, "stream", StringComparison.OrdinalIgnoreCase)
                || string.Equals(key, "providers", StringComparison.OrdinalIgnoreCase)
                || string.Equals(key, "register-winml", StringComparison.OrdinalIgnoreCase)
                || string.Equals(key, "ensure-winml", StringComparison.OrdinalIgnoreCase))
            {
                values[key] = null;
                continue;
            }
            if (++index >= input.Length)
            {
                throw new ArgumentException($"Missing value for --{key}.");
            }
            values[key] = input[index];
        }
        return values;
    }

    private static string Required(IReadOnlyDictionary<string, string?> values, string key) =>
        values.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value)
            ? value
            : throw new ArgumentException($"--{key} is required.");

    private static void WriteJson(object value) =>
        WriteAndFlush(JsonSerializer.Serialize(value, JsonOptions));

    private static void WriteAndFlush(string value)
    {
        Console.Out.WriteLine(value);
        Console.Out.Flush();
    }

#if WINDOWS_ML_CATALOG
    private static object[] GetWindowsMlProviders()
    {
        var catalog = Microsoft.Windows.AI.MachineLearning.ExecutionProviderCatalog.GetDefault();
        return catalog.FindAllProviders()
            .Select(provider => (object)new
            {
                name = provider.Name,
                ready_state = provider.ReadyState.ToString(),
                certification = provider.Certification.ToString(),
                installed = provider.ReadyState
                    != Microsoft.Windows.AI.MachineLearning.ExecutionProviderReadyState.NotPresent,
                library_path = provider.LibraryPath,
                package_name = provider.PackageId?.Name ?? "",
            })
            .ToArray();
    }

    private static async Task<string[]> RegisterReadyWindowsMlProviders()
    {
        var catalog = Microsoft.Windows.AI.MachineLearning.ExecutionProviderCatalog.GetDefault();
        var providers = catalog.FindAllProviders();
        foreach (var provider in providers)
        {
            if (provider.ReadyState
                != Microsoft.Windows.AI.MachineLearning.ExecutionProviderReadyState.NotReady)
            {
                continue;
            }
            var result = await provider.EnsureReadyAsync();
            if (result.Status
                != Microsoft.Windows.AI.MachineLearning.ExecutionProviderReadyResultState.Success)
            {
                throw new InvalidOperationException(
                    $"Windows ML could not activate {provider.Name}: "
                    + $"{result.Status}; {result.DiagnosticText}");
            }
        }
        return RegisterProviderLibraries(
            providers.Where(provider =>
                provider.ReadyState
                == Microsoft.Windows.AI.MachineLearning.ExecutionProviderReadyState.Ready));
    }

    private static async Task<string[]> EnsureWindowsMlProviders()
    {
        var catalog = Microsoft.Windows.AI.MachineLearning.ExecutionProviderCatalog.GetDefault();
        var providers = catalog.FindAllProviders();
        foreach (var provider in providers)
        {
            if (provider.ReadyState
                == Microsoft.Windows.AI.MachineLearning.ExecutionProviderReadyState.Ready)
            {
                continue;
            }
            var result = await provider.EnsureReadyAsync();
            if (result.Status
                != Microsoft.Windows.AI.MachineLearning.ExecutionProviderReadyResultState.Success)
            {
                throw new InvalidOperationException(
                    $"Windows ML could not prepare {provider.Name}: "
                    + $"{result.Status}; {result.DiagnosticText}");
            }
        }
        return RegisterProviderLibraries(providers);
    }

    private static string[] RegisterProviderLibraries(
        IEnumerable<Microsoft.Windows.AI.MachineLearning.ExecutionProvider> providers)
    {
        var registered = new List<string>();
        foreach (var provider in providers)
        {
            if (string.IsNullOrWhiteSpace(provider.LibraryPath))
            {
                continue;
            }
            OgaRegisterExecutionProviderLibrary(provider.Name, provider.LibraryPath);
            registered.Add(provider.Name);
        }
        return registered.ToArray();
    }

    [DllImport("onnxruntime-genai", CallingConvention = CallingConvention.Winapi)]
    private static extern void OgaRegisterExecutionProviderLibrary(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string registrationName,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string libraryPath);
#endif
}
