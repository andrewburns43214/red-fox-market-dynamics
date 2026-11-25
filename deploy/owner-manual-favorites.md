# Owner Favorite controls

The HTTPS route /api/owner/favorites forwards only to the dedicated loopback
service on port 5054. The service verifies the Supabase session and the confirmed
owner email. The board determines button visibility from that same endpoint.
Requests to save require the same HTTPS origin and JSON. A revision check under
a SQLite write transaction prevents a second tab from overwriting a newer
decision. Changed odds, lines or source captures require a fresh confirmation.

Deploy the service unit and the Nginx location fragment from this directory.
Install the fragment under /etc/nginx/snippets and include it in the Red Fox
HTTPS server block. Verify the Nginx configuration before reloading. Create
the root-owned enable marker /etc/redfox-manual-favorites.enabled after
verifying the service and routes. The state directory is created by systemd.

To disable manual editing, remove the enable marker. Existing overrides
continue to publish and settle from the audit database. Normal automatic
qualification always continues calculating in the background. The owner can
restore it for an event from either pregame entry point while editing is
enabled. Active overrides should settle or be restored before reverting the
application revision that reads them.

Retain /var/lib/redfox-manual-favorites/decisions.sqlite3 and any SQLite recovery
files during a rollback. Back up the database with SQLite's online backup API,
or copy it while the service is stopped. Preserve the production release
backup and revert the release as a forward Git commit so later unrelated
production changes remain. Restore the prior Nginx configuration and remove
the separate route if the app itself is rolled back.

Manual entries use the newest complete paired sportsbook capture and a freshly
verified event. Independent provider unavailability, stale prices, changed
fixture identity and unclear bout status can prevent a save. The existing
one-minute score worker freezes the latest verified owner selection and only
grades a matching provider result. The scheduled board refresh publishes the
same choice for other viewers.
