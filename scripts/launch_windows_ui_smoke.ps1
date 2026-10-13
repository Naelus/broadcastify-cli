[CmdletBinding()]
param(
    [string]$Executable = "",
    [string]$SessionRoot = "",
    [string]$FeedId = "999991",
    [Alias("Python")][string]$PythonExecutable = "",
    [switch]$WorkflowPreview
)

$ErrorActionPreference = "Stop"
if (-not $FeedId -or $FeedId -notmatch '^\d+$') {
    throw "FeedId must contain only digits."
}
if ([string]::IsNullOrWhiteSpace($Executable)) {
    $Executable = Join-Path $env:LOCALAPPDATA (
        "Programs\Broadcastify Desktop\Broadcastify Desktop.exe"
    )
}
$Executable = [System.IO.Path]::GetFullPath($Executable)
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Installed desktop executable not found: $Executable"
}
if ([string]::IsNullOrWhiteSpace($SessionRoot)) {
    $SessionRoot = Join-Path (
        [System.IO.Path]::GetTempPath()
    ) ("broadcastify-desktop-ui-smoke-" + [Guid]::NewGuid().ToString("N"))
}
$SessionRoot = [System.IO.Path]::GetFullPath($SessionRoot)
if (Test-Path -LiteralPath $SessionRoot) {
    throw "The isolated UI smoke root already exists: $SessionRoot"
}

$dataRoot = Join-Path $SessionRoot "app-data"
$libraryRoot = Join-Path $SessionRoot "library"
$today = [DateTime]::Today
$startDate = $today.AddDays(-9)
if ($WorkflowPreview) { $startDate = $today.AddDays(-3) }
$dayRoot = Join-Path (
    (Join-Path $libraryRoot $FeedId)
) $startDate.ToString("yyyyMMdd")
New-Item -ItemType Directory -Path $dataRoot, $dayRoot -Force | Out-Null

$settings = [ordered]@{
    Version = 9
    OutputDirectory = $libraryRoot
    LanSyncEnabled = $false
    LanDiscoveryEnabled = $false
    LanShareEnabled = $false
    LastReviewFeedId = $FeedId
    LastReviewDate = $startDate.ToString("yyyy-MM-dd")
}
$settings | ConvertTo-Json -Depth 3 |
    Set-Content -LiteralPath (Join-Path $dataRoot "settings.json") -Encoding UTF8

$installRoot = Split-Path -Parent $Executable
$python = Join-Path $installRoot "runtime\python\python.exe"
if (-not [string]::IsNullOrWhiteSpace($PythonExecutable)) { $python = [System.IO.Path]::GetFullPath($PythonExecutable) }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Installed portable Python runtime not found: $python"
}
$database = Join-Path $libraryRoot "broadcastify-analysis.sqlite3"
$ledger = Join-Path $dataRoot "archive-quota.sqlite3"
$settings.PythonRuntimePath = [System.IO.Path]::GetFullPath($python)
$settings | ConvertTo-Json -Depth 3 |
    Set-Content -LiteralPath (Join-Path $dataRoot "settings.json") -Encoding UTF8
$payload = [ordered]@{
    feed_id = $FeedId
    feed_name = "Isolated UI smoke feed"
    start_date = $startDate.ToString("yyyy-MM-dd")
    through_current = $true
} | ConvertTo-Json -Compress

$savedEnvironment = @{}
$workerEnvironment = [ordered]@{
    BROADCASTIFY_DESKTOP_TEST_DATA_ROOT = $dataRoot
    BROADCASTIFY_LIBRARY_ROOT = $libraryRoot
    BROADCASTIFY_SECURE_ANALYSIS_DB = $database
    BROADCASTIFY_QUOTA_LEDGER = $ledger
    BROADCASTIFY_DESKTOP_E2E_ISOLATED = "1"
    BROADCASTIFY_ENV_FILE = ""
}
if ($WorkflowPreview) { $workerEnvironment.BROADCASTIFY_DESKTOP_PREVIEW = "1" }
try {
    foreach ($entry in $workerEnvironment.GetEnumerator()) {
        $savedEnvironment[$entry.Key] = [Environment]::GetEnvironmentVariable(
            $entry.Key,
            [EnvironmentVariableTarget]::Process
        )
        [Environment]::SetEnvironmentVariable(
            $entry.Key,
            $entry.Value,
            [EnvironmentVariableTarget]::Process
        )
    }
    $workerOutput = $payload |
        & $python -m broadcastify_cli.worker save-library-catch-up 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated saved catch-up fixture: $workerOutput"
    }
    if ($WorkflowPreview) {
        # Seed real retained records. Only the sample audio/incident is synthetic;
        # Library, resume planning, review and acknowledgements use production code.
        $seed = @'
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
from datetime import date,timedelta
from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.analysis import DEFAULT_EMBEDDING_MODEL,PROMPT_VERSION
root,db,feed=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
today=date.today()
ffmpeg=shutil.which('ffmpeg')
if not ffmpeg: raise RuntimeError('FFmpeg is required to create playable preview audio.')
sample=root/'preview-silence.mp3'
subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i','anullsrc=r=16000:cl=mono','-t','2',str(sample)],check=True)
audio_bytes=sample.read_bytes()
sample.unlink()
with AnalysisStore(db) as store:
 for offset in (3,2):
  day=today-timedelta(days=offset)
  folder=root/feed/day.strftime('%Y%m%d')
  (folder/'transcripts').mkdir(parents=True,exist_ok=True)
  stem=f'combined_{feed}_{day:%Y%m%d}'
  audio=folder/(stem+'.mp3');audio.write_bytes(audio_bytes)
  transcript=folder/'transcripts'/(stem+'.json')
  text='Preview example: dispatch reports a traffic incident near Main and First.'
  transcript.write_text(json.dumps({'duration':2,'model':'preview','diarization_completed':True,'segments':[{'start':0,'end':2,'text':text,'speaker':'SPEAKER_00'}]}),encoding='utf-8')
  imported=store.import_transcript(feed,day,transcript,audio)
  ids=store.replace_incidents(imported.day_id,[{'fingerprint':f'preview-{day}','event_type':'traffic','title':'Preview traffic report','summary':'Synthetic sample for reviewing the interface. Audio is two seconds of silence.','location':'Main and First','start_seconds':0,'end_seconds':2,'priority':2,'confidence':1,'evidence':[{'segment_index':0,'start_seconds':0,'end_seconds':2,'text':text}],'attributes':{}}],model='preview',prompt_version=PROMPT_VERSION)
  store.save_daily_summary(imported.day_id,'Synthetic preview report; not an actual incident.',ids,'preview',PROMPT_VERSION,imported.transcript_sha256)
  if offset==3:
   store.save_embeddings('passage',[(p['id'],hashlib.sha256(f"passage: {p['text']}".encode()).hexdigest(),1,b'\0'*4) for p in store.passages_missing_embeddings(DEFAULT_EMBEDDING_MODEL)],DEFAULT_EMBEDDING_MODEL)
 other='999992' if feed!='999992' else '999993'
 day=today-timedelta(days=1);folder=root/other/day.strftime('%Y%m%d');folder.mkdir(parents=True)
 (folder/f'combined_{other}_{day:%Y%m%d}.mp3').write_bytes(audio_bytes)
 store.save_feed_catalog([{'feed_id':feed,'name':'Preview — ready and index-pending days'},{'feed_id':other,'name':'Preview — audio awaiting transcription'}])
 folder=root/feed/today.strftime('%Y%m%d');folder.mkdir(parents=True)
 (folder/'.broadcastify-archive-complete.json').write_text(json.dumps({'schema_version':1,'feed_id':feed,'archive_date':today.isoformat(),'archive_ids':[],'completed_at_unix':time.time()}),encoding='utf-8')
'@
        & $python -c $seed $libraryRoot $database $FeedId
        if ($LASTEXITCODE -ne 0) { throw "Could not seed workflow preview." }
    }
}
finally {
    foreach ($entry in $savedEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable(
            $entry.Key,
            $entry.Value,
            [EnvironmentVariableTarget]::Process
        )
    }
}

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $Executable
$startInfo.UseShellExecute = $false
if ($WorkflowPreview) {
    foreach ($name in @($startInfo.Environment.Keys)) {
        if ($name -match '^(BROADCASTIFY_|HUGGINGFACE_|HF_TOKEN$|OPENAI_API_KEY$)') {
            $startInfo.Environment.Remove($name) | Out-Null
        }
    }
}
foreach ($entry in $workerEnvironment.GetEnumerator()) {
    $startInfo.Environment[$entry.Key] = $entry.Value
}
$process = [System.Diagnostics.Process]::Start($startInfo)
if ($null -eq $process) {
    throw "The isolated desktop smoke process did not start."
}

$session = [ordered]@{
    schema_version = 1
    process_id = $process.Id
    executable = $Executable
    session_root = $SessionRoot
    data_root = $dataRoot
    library_root = $libraryRoot
    database = $database
    feed_id = $FeedId
    catch_up_start_date = $startDate.ToString("yyyy-MM-dd")
    catch_up_end_date = $today.ToString("yyyy-MM-dd")
    catch_up_through_current = $true
}
$session | ConvertTo-Json -Depth 3 |
    Set-Content -LiteralPath (Join-Path $SessionRoot "smoke-session.json") -Encoding UTF8
$session | ConvertTo-Json -Depth 3
