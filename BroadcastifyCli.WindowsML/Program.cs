using System.Text.Json;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace BroadcastifyCli.WindowsML;

internal static class Program
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    public static int Main(string[] args)
    {
        try
        {
            var options = ParseArguments(args);
            if (options.ContainsKey("probe"))
            {
                var configuredModel = options.GetValueOrDefault("model");
                var decodeReady = false;
                if (!string.IsNullOrWhiteSpace(configuredModel))
                {
                    if (!Directory.Exists(configuredModel))
                    {
                        throw new DirectoryNotFoundException(
                            $"Windows ML model directory not found: {configuredModel}");
                    }
                    decodeReady = RunDecodeSelfTest(configuredModel);
                }
                WriteJson(new
                {
                    ready = true,
                    decode_ready = decodeReady,
                    backend = "Windows ML (ONNX Runtime GenAI)",
                    architecture = Environment.Is64BitProcess ? "x64" : "x86",
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
            WriteJson(new
            {
                text,
                backend = "Windows ML (ONNX Runtime GenAI)",
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
        using var model = new Model(modelPath);
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
        using var inputs = processor.ProcessImagesAndAudios(prompt, null, audios);
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
        using var model = new Model(modelPath);
        using var processor = new MultiModalProcessor(model);
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
                backend = "Windows ML (ONNX Runtime GenAI)",
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
                || string.Equals(key, "stream", StringComparison.OrdinalIgnoreCase))
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
}
