# Red Fox historical data operations

The historical pipeline runs independently of market capture, board publication, official qualification, and the Favorite ledger. It never writes to those files. The raw capture archive remains the authoritative record of accepted observations.

## Production paths

- data/snapshots.csv holds the existing hot capture window. The existing runner rotates older rows into data/snapshot_archive when the hot file reaches its threshold. Keep all legacy raw files, including checked source copies outside that directory.
- /var/lib/redfox-history/private/history.sqlite3 indexes observations, fixture identities, captured qualification and supported-side evidence, and separate historical outcomes. Exact duplicate quotes index once. Conflicting values at the same market-side timestamp are retained as variants. The paired query excludes conflicting and post-start captures.
- /var/lib/redfox-history/private/research holds attributable imports and source manifests. Keep verified results separate from official records. Do not treat an unresolved or unavailable result as a loss.
- /var/lib/redfox-history/public/data/anomaly_event_details serves retired details as gzip through the existing authenticated URL. Nginx checks for a current uncompressed Game Detail first. The retirement job waits until detail files are inactive, compresses and verifies identical bytes, and takes the existing publication lock nonblocking before removing the redundant rendering. If publication is in progress, the job retries the first file during a bounded idle wait. A conflict keeps both copies for review.
- The history observer captures changes from the published board. The historical index also imports the retained decision ledger and compact publication transitions. Expiring rendered source HTML continues to serve diagnostics. Historically missing states are never reconstructed and presented as originally published.

## Scheduled jobs and monitoring

Install the history observer, maintenance, detail and health units in deploy. Enable the detail timer after testing the protected Nginx fallback with both gzip and identity clients. Check journal entries for the four units after rollout. The health job writes a daily /var/lib/redfox-history/private/health.json, rotates across stored SHA-256 checksums, and raises an early warning below 15 GiB available. Existing disk and freshness monitors remain in effect. The maintenance job indexes the hot tape and finalized shards, then reconciles pending results. It independently checks exact fixture identities and leaves ambiguous rows unresolved.

Below the 15 GiB reserve, the rebuildable quote index defers new imports while normal capture and compression continue. During initial backfill and seasonal changes, compare the index byte count and timestamped archive inventories before extrapolating. Use compressed date-partitioned quote indexes or compact bounded decision-window materializations for older dates before the SQLite index reaches the reserve. Retain the complete compressed tape as the reconstruction source. A cold partition should replace an older index only after parity checks on observations and an off-host restore drill.

## Backup and recovery

Enable redfox-history-backup.timer only after an existing private remote restic repository has been approved and tested. Install restic using the host package manager and store repository credentials in a root-only /etc/redfox-history-backup.env or its referenced credential files. Supply RESTIC_REPOSITORY and RESTIC_PASSWORD_FILE. Keep a recovery copy of the repository credentials outside production and separate from the repository.

The backup wrapper takes the manual SQLite database through SQLite's online backup API and exports private decision/outcome tables using one read transaction. It backs up canonical inputs and irreplaceable records with restic's versioning and chunk deduplication. Preserve closed gzip shards, legacy copies, research imports, qualification records and mappings. Check the repository on a recurring schedule with the wrapper's check command. Establish a retention policy for backup generations with the owner before running any restic forget or prune operation. Preserve permanent historical input data in the retained snapshots.

For a recovery drill restore a tagged backup into an isolated directory or instance. Check the repository integrity and restored SQLite databases, reconstruct the archive on a scratch path, and compare a pregame game trajectory and exact entry price against its verified result. Include a non-Favorite and an official or manual Favorite. Verify that archive timestamps precede the result, and that conflicting or unresolved fixtures stay ungraded. Measure the backup age and recovery time. A working same-host copy or a passing checksum alone leaves host-loss recovery unverified.

## Future comparisons

Query paired_as_of with the decision clock and scenario window. Treat the source clock, publication clock, scheduled start as known at the time, and eventual verified actual start as separate values. The historical analog layer will select one canonical game/matchup per cohort using provider identity, resolve calendar discrepancies before reporting results, and use only prior captures. UFC requires verified bout start evidence to protect against card-time and early-start ambiguity. Store final-score corrections as additional evidence. Settlements involving shortened MLB games or ambiguous contract terms need explicit gradeability.

Refresh future comparison results outside the live runner and use a small per-game payload in the existing popup after design approval. Do not join any of these research grades into the official Favorite records.
