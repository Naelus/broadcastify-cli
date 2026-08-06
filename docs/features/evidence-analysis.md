# Incident analysis and evidence

## Responsibilities

The analysis layer imports timestamped transcripts into SQLite, indexes passages
for retrieval, extracts structured incidents, writes daily and weekly briefs,
answers date-range questions, and prepares exact evidence clips.

The default local stack uses:

- BGE Small through FastEmbed for inexpensive semantic retrieval;
- quantized Gemma 4 12B through llama.cpp for structured extraction and prose;
- SQLite/FTS5 as the durable source of truth.

Dense radio is bounded by both elapsed time and transcript-prompt size before it
reaches the model. Each chunk stays within a conservative llama.cpp context
budget, carries a small segment overlap across artificial boundaries, and is
checkpointed independently. A failed model request therefore resumes from the
first incomplete chunk without repeating acquisition, ASR, or diarization.

Embeddings retrieve likely passages; they do not replace the generative model.
The model never becomes the evidence store.

## Incident acceptance

Schema-valid JSON is necessary but insufficient. A retained incident must:

- cite exact transcript segment IDs;
- keep evidence within a bounded time gap;
- share meaningful claim concepts with those citations;
- support critical claims such as shots, weapons, assault, fire, pursuit,
  collision, overdose, theft, welfare, or trespass in the cited ASR; and
- avoid unsupported guilt, identity, confirmation, resolution, or outcome
  language.

Category/priority contradictions are normalized from evidence. Unsupported
outcome sentences are removed or the incident is rejected. A narrow deterministic
critical-phrase pass can recover an explicitly spoken high-salience event that
the model omitted, but it must cite the exact matching segment.

## Summaries and questions

Daily and weekly briefs are derived only from current versioned incidents.
Grounding checks reject unknown incident IDs, impossible counts, unsupported
outcomes, and invented category rankings. Deterministic fallbacks remain
available when the model fails those checks.

Range questions combine structured incidents with retrieved transcript evidence
and require E/I citations. Before Review, questions, weekly/area summaries, or
clip export can use a saved row, the retained audio, transcript import hash,
analysis hash, and prompt revision must still agree. Older rows remain
recoverable but cannot be presented against newer retained audio. Missing or
stale dates are reported explicitly. Saved weekly and area briefs also carry an
exact source fingerprint and are hidden as stale when a current daily summary,
incident set, or area profile changes.

The native **Ask the archive** surface selects feeds by friendly name and keeps
a bounded recent conversation so follow-up phrases can refer to the prior turn.
Earlier answers are context, never evidence: retrieval runs again for the
current question and the response must cite the newly supplied E/I records.
Questions and answers continue to be written to the local Q&A audit table.
Archive chat shares a single model-operation gate with background incident
analysis so download/transcription can continue while llama.cpp work is safely
serialized.

Local Gemma is the default. OpenAI Responses, compatible `/v1` endpoints, and a
saved-login Codex harness are optional and require explicit consent before
transcript excerpts leave the machine. See [model providers](../model-providers.md).

## Evidence clips

Each surfaced incident can produce a compact MP3 cut from the retained combined
audio around its strongest cited segment. The clip records source and clip
hashes, offsets, feed/day identity, confidence, and speaker context. A separate
surrounding-context option helps hear an earlier dispatch but is explicitly not
treated as citation evidence.

## Names and identifiers

Private-use output preserves names explicitly spoken in cited radio evidence.
Prompts and validators forbid guessing, correcting, or normalizing identities.
Phone numbers, dates of birth, emails, and long numeric identifiers remain
masked in derived/display text. The original audio and transcript stay local and
unchanged. See [evidence and privacy](../decisions/evidence-and-privacy.md).

## Versioning and reanalysis

Incident-window, day, week, and area results carry prompt versions and source
fingerprints. A policy change hides older claims until local reanalysis but
reuses downloads, combined audio, transcription, speaker labels, and embeddings
when their own identities remain valid.
