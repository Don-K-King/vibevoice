# VibeVoice im Evido-Stack (lokale Docker-Installation)

Diese Anleitung beschreibt die lokale Bereitstellung von VibeVoice als **separaten** Container im internen Evido-Netzwerk. Sie folgt der vorgegebenen Architektur (STT-Routing mit `pipeline_scope=diarization`) und fokussiert auf sichere, konfliktfreie Integration.

## 1) Architektur- und Sicherheitscheck (vorab lesen)

* **Netzwerk-Isolation:** VibeVoice ist **nur intern** erreichbar. Der Compose-Stack bindet den Host-Port auf `127.0.0.1`, damit kein externer Zugriff möglich ist. Externen Zugriff ausschließlich über TLS/mTLS oder einen Auth-Proxy anbieten.
* **Pipeline-Trennung:** Das STT-Mapping **muss** `pipeline_scope=diarization` verwenden, um Konflikte mit Standard-/Live-Translation-Pipelines zu vermeiden.
* **Secrets:** API-Keys **nur** serverseitig über `guardrails.headers` (nie im Frontend).
* **Datenminimierung:** Keine Audio-Payloads in Logs, nur Metadaten.

> **Wichtiger Abgleich:** Evido erwartet standardmäßig `POST /v1/transcribe`, VibeVoice stellt in dieser Repo-Version `POST /transcribe` bereit.  
> **Optionen:**  
> 1) **STT-Mapping** im Backend auf `/transcribe` anpassen, **oder**  
> 2) **Reverse-Proxy** vor VibeVoice (Path-Rewrite `/v1/transcribe` → `/transcribe`).  
> Ohne Abgleich entsteht ein **Architekturkonflikt** (Routing-Fehler / 404).

---

## 2) Voraussetzungen

* Docker 20.10+  
* NVIDIA Container Toolkit (GPU-Zugriff)  
* Ausreichend GPU- und RAM-Ressourcen (siehe README)  

Optional: Model-Cache vorab befüllen (schnellerer Erststart).

---

## 3) Evido-Compose-Konfiguration anlegen

Nutze die bereitgestellte Datei `docker-compose.evido.yml` im Repo-Root.

### Konfigurierbare Parameter (Env)

| Variable | Default | Zweck |
|----------|---------|------|
| `VIBEVOICE_IMAGE` | `vibevoice:latest` | Container-Image |
> **Hinweis:** Das Image `vibevoice:latest` muss entweder lokal gebaut oder über eine Registry verfügbar sein.

---

## 4) Schritt-für-Schritt Einrichtung

### Schritt 1: Evido-Netzwerk bereitstellen

Der Container wird in ein **externes** Netzwerk eingebunden (Name: `evido-live-translate`):

```bash
docker network create evido-live-translate
```

Falls das Netzwerk bereits existiert, kannst du den Schritt überspringen.

### Schritt 2: Image bereitstellen

**Option A (lokal bauen):**
```bash
docker build -t vibevoice:latest .
```

**Option B (Registry-Image nutzen):**
```bash
export VIBEVOICE_IMAGE=registry.example.com/vibevoice:latest
```

### Schritt 3: Container starten

```bash
docker compose -f docker-compose.evido.yml up -d
```

### Schritt 4: Health-Check prüfen

```bash
curl -fsS http://localhost:8001/health
```

Erwartete Antwort (Beispiel):
```json
{"status":"healthy","model_loaded":true,"device":"cuda"}
```

---

## 5) Backend → VibeVoice (STT-Mapping)

Empfohlene interne Ziel-URL im Backend:
```
http://vibevoice:8000/transcribe
```

**Wichtig:** Falls Evido zwingend `/v1/transcribe` erwartet, nutze einen Reverse-Proxy oder passe das Mapping an (siehe Architekturhinweis oben).

### Multipart-Upload (Backend → VibeVoice)

**Felder:**
* `file` (Audio-File/Chunk, konfigurierbar: `guardrails.multipart.file_field`)
* `metadata` (JSON, konfigurierbar: `guardrails.multipart.metadata_field`)

**Metadata-Felder:**
* `mime_type`, `language`, `model`, `device`, `compute_type`, `provider_id`

**Guardrails (Beispiele):**
* `headers` (API-Key, mTLS-Header etc.)
* `query` (optionale Query-Parameter)
* `multipart.chunk_size`
* `timeout_s`
* optional `client_cert`

---

## 6) Verarbeitung & erwartete Antwort

**Ablauf:**
1. UI lädt Audio-Segmente via `/upload_dual` ins Backend.
2. STT-Routing wählt `pipeline_scope=diarization`.
3. Backend sendet Multipart-Request an VibeVoice (`/transcribe`).
4. VibeVoice liefert strukturierte STT-Response inkl. Speaker-Metadaten.
5. Backend persistiert Ergebnisse in `transcript_structured.json`.

**Antwortanforderung:**
* Transkribierter Text
* Sprecher-Segmente / Speaker-Labels
* Optional: Zeitstempel/Segmente

---

## 7) Monitoring & Betrieb

* **Health:** `GET /health` (intern)
* **Metriken:** Falls `/metrics` verfügbar, nur intern scrapen
* **KPIs:** Latenz p50/p95/p99, Fehlerquoten, verarbeitete Audiominuten, Segment-Counts

---

## 8) Architekturkonflikte & Alternativen (proaktiv)

* **Konflikt:** `POST /v1/transcribe` vs. `POST /transcribe`  
  **Lösung:** STT-Mapping anpassen oder Reverse-Proxy mit Path-Rewrite.
* **Konflikt:** Direkte Exponierung des Ports auf `0.0.0.0`  
  **Lösung:** Host-Bind auf `127.0.0.1` (bereits im Compose), oder Zugriff ausschließlich über internen Proxy.
* **Konflikt:** Pipeline-Mix (Standard/Live vs. Diarization)  
  **Lösung:** `pipeline_scope=diarization` strikt verwenden, keine Fallbacks.
