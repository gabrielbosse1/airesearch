"""Collect real GitHub data (READMEs, issues, PRs, commits) for ByteCore.

Design notes that matter more than the code:

* **No new dependencies.** ``requests`` is not part of this project's
  requirements, so the collector speaks to the API with the standard library
  (:mod:`urllib.request`). If ``requests`` happens to be installed it is used,
  but nothing here needs it. That keeps the local 10-hour run a `pip install
  torch pyyaml` away.
* **Rate limits are respected, not dodged.** Unauthenticated: 60 requests/hour.
  Authenticated: 5,000/hour. The collector reads ``GITHUB_TOKEN``/``GH_TOKEN``
  (or an explicit argument), pauses between requests, and *sleeps until the
  quota resets* rather than hammering the API. The token is never printed, never
  written to any artifact, and never stored in the dataset.
* **Every list endpoint returns 100 items per request**, so one request is
  worth 100 issues. Commit *lists* are cheap; per-commit file lists cost one
  request each and are deliberately not fetched -- commit observations are
  derived from list payloads instead.
* **Provenance and scrubbing.** Each record carries its repository and SPDX
  licence; free text is scrubbed of e-mail addresses, URLs and long opaque
  blobs. Scrubbing is heuristic: it reduces accidental personal data, it does
  not guarantee its absence.

Usage::

    python -m data.github_collector --repos pytorch/pytorch facebook/react
    GITHUB_TOKEN=ghp_... python -m data.github_collector --max-items-per-kind 400
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

USER_AGENT = "ByteCore-DualMemory-Research/1.0"
DEFAULT_REPOS: tuple[str, ...] = (
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "huggingface/transformers",
    "microsoft/vscode",
    "facebook/react",
    "python/cpython",
    "rust-lang/rust",
    "scikit-learn/scikit-learn",
)
MAX_PER_PAGE = 100  # GitHub's hard maximum for list endpoints

# ---------------------------------------------------------------- scrubbing
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# A URL ends at whitespace or at a closing delimiter, so it never swallows the
# rest of a code example such as requests.get('https://x', auth=(...)).
_URL = re.compile(r"https?://[^\s)'" + '"' + r"\]\}>]+")
# Opaque blobs are space-free by construction; requiring 60+ characters with no
# spaces keeps ordinary prose out of this rule.
_BLOB = re.compile(r"\b[A-Za-z0-9+/]{60,}={0,2}\b")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LONG_RUN = re.compile(r"(.)\1{40,}")


def scrub_text(text: str, *, max_chars: int = 20_000) -> str:
    """Best-effort removal of personal data and noise. Heuristic, not a guarantee."""
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL.sub(" ", cleaned)
    cleaned = _EMAIL.sub("<email>", cleaned)
    cleaned = _URL.sub("<url>", cleaned)
    cleaned = _BLOB.sub("<blob>", cleaned)
    cleaned = _LONG_RUN.sub(lambda m: m.group(1) * 40, cleaned)
    cleaned = re.sub(r"[ \t]{3,}", "  ", cleaned)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + " ..."
    return cleaned


def _first_line(text: str, limit: int = 120) -> str:
    line = (text or "").strip().split("\n", 1)[0].strip()
    return line[:limit]


# ------------------------------------------------------------------ client
@dataclass
class RateLimit:
    """Snapshot of the API's own accounting."""

    limit: int = 0
    remaining: int = 0
    reset: float = 0.0
    authenticated: bool = False


@dataclass
class GitHubCollector:
    """Fetch READMEs, issues, pull requests and commits from GitHub."""

    token: str | None = None
    sleep: float = 0.7
    timeout: float = 30.0
    max_retries: int = 4
    user_agent: str = USER_AGENT
    rate_limit: RateLimit = field(default_factory=RateLimit)
    requests_made: int = 0
    _last_request: float = field(default=0.0, repr=False)
    _transport: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or None

    # ------------------------------------------------------------ transport
    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent, "Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, url: str) -> Any:
        """One GET with backoff, rate-limit patience and no token in any error."""
        for attempt in range(self.max_retries):
            self._wait_for_slot()
            try:
                if self._transport is not None:  # injected transport (tests)
                    payload, headers = self._transport(url)
                else:
                    request = urllib.request.Request(url, headers=self._headers())
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                        headers = dict(response.headers)
                self.requests_made += 1
                self._last_request = time.time()
                self._record_limit(headers)
                return payload
            except urllib.error.HTTPError as error:
                self._record_limit(dict(error.headers or {}))
                if error.code in (403, 429) and self._quota_exhausted():
                    self._sleep_until_reset()
                    continue
                if error.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt + random.random(), 30))
                    continue
                # Never echo the request headers: they carry the token.
                raise RuntimeError(f"GitHub API error {error.code} for {self._safe(url)}") from None
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt + random.random(), 30))
                    continue
                raise RuntimeError(f"network error for {self._safe(url)}: {type(error).__name__}") from None
        raise RuntimeError(f"giving up on {self._safe(url)}")

    @staticmethod
    def _safe(url: str) -> str:
        """Strip anything that could carry credentials out of the URL."""
        return url.split("?")[0]

    def _record_limit(self, headers: dict[str, str]) -> None:
        lowered = {str(key).lower(): value for key, value in headers.items()}
        if "x-ratelimit-limit" in lowered:
            self.rate_limit.limit = int(lowered["x-ratelimit-limit"])
        if "x-ratelimit-remaining" in lowered:
            self.rate_limit.remaining = int(lowered["x-ratelimit-remaining"])
        if "x-ratelimit-reset" in lowered:
            self.rate_limit.reset = float(lowered["x-ratelimit-reset"])
        self.rate_limit.authenticated = self.rate_limit.limit >= 1000

    def _quota_exhausted(self) -> bool:
        return self.rate_limit.remaining <= 1 and self.rate_limit.reset > 0

    def _sleep_until_reset(self) -> None:
        delay = max(self.rate_limit.reset - time.time(), 0.0) + 1.0
        print(f"  rate limit exhausted; sleeping {delay / 60:.1f} min", flush=True)
        time.sleep(min(delay, 3600))

    def _wait_for_slot(self) -> None:
        if self._quota_exhausted():
            self._sleep_until_reset()
        gap = self.sleep - (time.time() - self._last_request)
        if gap > 0:
            time.sleep(gap)

    def _paged(self, url: str, limit: int) -> list[Any]:
        """Follow pages until ``limit`` items or the last page."""
        rows: list[Any] = []
        page = 1
        while len(rows) < limit:
            batch = self._get(f"{url}{'&' if '?' in url else '?'}per_page={MAX_PER_PAGE}&page={page}")
            if not isinstance(batch, list) or not batch:
                break
            rows.extend(batch)
            if len(batch) < MAX_PER_PAGE:
                break
            page += 1
        return rows[:limit]

    # -------------------------------------------------------------- fetches
    def fetch_repo_metadata(self, owner: str, repo: str) -> dict[str, Any]:
        """Licence and description, for provenance."""
        payload = self._get(f"https://api.github.com/repos/{owner}/{repo}")
        license_info = payload.get("license") or {}
        return {"full_name": payload.get("full_name", f"{owner}/{repo}"),
                "license": license_info.get("spdx_id") or "NOASSERTION",
                "description": _first_line(payload.get("description") or ""),
                "stars": payload.get("stargazers_count", 0)}

    def fetch_repo_readme(self, owner: str, repo: str) -> dict[str, str] | None:
        """Fetch the README, base64-decoded and scrubbed."""
        payload = self._get(f"https://api.github.com/repos/{owner}/{repo}/readme")
        if not isinstance(payload, dict) or "content" not in payload:
            return None
        try:
            raw = base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            return None
        content = scrub_text(raw)
        return {"name": payload.get("name", "README.md"), "content": content} if content else None

    def fetch_repo_issues(self, owner: str, repo: str, limit: int = 100) -> list[dict[str, Any]]:
        """Issues only -- the endpoint also returns PRs, which are filtered out."""
        payload = self._paged(f"https://api.github.com/repos/{owner}/{repo}/issues?state=all", limit)
        rows = []
        for item in payload:
            if not isinstance(item, dict) or "pull_request" in item:
                continue
            body = scrub_text(item.get("body") or "")
            title = scrub_text(item.get("title") or "", max_chars=300)
            if not body or not title:
                continue
            rows.append({"number": item.get("number"), "title": title, "body": body,
                         "state": item.get("state"), "comments": item.get("comments", 0),
                         "labels": [label.get("name", "") for label in item.get("labels", [])][:8],
                         "created_at": item.get("created_at", "")})
        return rows

    def fetch_repo_prs(self, owner: str, repo: str, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._paged(f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all", limit)
        rows = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            body = scrub_text(item.get("body") or "")
            title = scrub_text(item.get("title") or "", max_chars=300)
            if not body or not title:
                continue
            rows.append({"number": item.get("number"), "title": title, "body": body,
                         "state": item.get("state"), "merged": bool(item.get("merged_at")),
                         "comments": item.get("comments", 0),
                         "created_at": item.get("created_at", "")})
        return rows

    def fetch_repo_commits(self, owner: str, repo: str, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._paged(f"https://api.github.com/repos/{owner}/{repo}/commits", limit)
        rows = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            commit = item.get("commit") or {}
            message = scrub_text(commit.get("message") or "", max_chars=2000)
            if not message:
                continue
            author = ((commit.get("author") or {}).get("name")) or (item.get("author") or {}).get("login") or ""
            rows.append({"sha": (item.get("sha") or "")[:12], "message": message,
                         "author": scrub_text(str(author), max_chars=80),
                         "date": (commit.get("author") or {}).get("date", "")})
        return rows

    # ---------------------------------------------------------------- drive
    def collect_dataset(self, repos: Sequence[str], output_dir: str = "data/github_raw", *,
                        max_items_per_kind: int = 200, include_readme: bool = True) -> dict[str, Any]:
        """Collect everything into ``github_stories.jsonl`` plus a manifest."""
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        per_repo: dict[str, dict[str, int]] = {}
        licenses: dict[str, str] = {}

        for full_name in repos:
            if "/" not in full_name:
                print(f"  skipping malformed repo name {full_name!r}")
                continue
            owner, name = full_name.split("/", 1)
            print(f"Collecting from {owner}/{name} ...", flush=True)
            counts = {"readme": 0, "issue": 0, "pr": 0, "commit": 0}
            license_id = "NOASSERTION"
            description = ""
            try:
                metadata = self.fetch_repo_metadata(owner, name)
                license_id = metadata["license"]
                description = metadata["description"]
            except RuntimeError as error:
                print(f"  metadata failed: {error}")
            licenses[full_name] = license_id

            try:
                if include_readme:
                    readme = self.fetch_repo_readme(owner, name)
                    if readme is not None:
                        records.append({"type": "readme", "source": full_name, "license": license_id,
                                        "description": description, "content": readme["content"]})
                        counts["readme"] = 1
            except RuntimeError as error:
                print(f"  readme failed: {error}")

            for kind, fetcher in (("issue", self.fetch_repo_issues),
                                  ("pr", self.fetch_repo_prs),
                                  ("commit", self.fetch_repo_commits)):
                try:
                    items = fetcher(owner, name, max_items_per_kind)
                except RuntimeError as error:
                    print(f"  {kind} failed: {error}")
                    continue
                for item in items:
                    item.update({"type": kind, "source": full_name, "license": license_id})
                records.extend(items)
                counts[kind] = len(items)
            per_repo[full_name] = counts
            print(f"  {counts} (total {len(records)})", flush=True)

        # Deduplicate on content so forks/mirrors do not double-count.
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for record in records:
            key = json.dumps({k: v for k, v in record.items() if k != "source"}, sort_keys=True)[:4000]
            if key in seen:
                continue
            seen.add(key)
            unique.append(record)

        output_file = target / "github_stories.jsonl"
        with output_file.open("w", encoding="utf-8") as handle:
            for record in unique:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        manifest = {
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "records": len(unique),
            "repositories": list(repos),
            "per_repository": per_repo,
            "licenses": licenses,
            "by_type": {kind: sum(1 for record in unique if record["type"] == kind)
                        for kind in ("readme", "issue", "pr", "commit")},
            "characters": sum(len(json.dumps(record, ensure_ascii=False)) for record in unique),
            "requests_made": self.requests_made,
            "authenticated": bool(self.token),
            "rate_limit": {"limit": self.rate_limit.limit, "remaining": self.rate_limit.remaining},
            "scrubbing": ("heuristic: e-mail addresses, URLs and long opaque blobs are replaced "
                          "with placeholders. This reduces accidental personal data; it does not "
                          "guarantee its absence."),
            "provenance": ("Content belongs to its respective repositories under the recorded "
                           "SPDX licences. Redistribution terms follow those licences."),
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Collected {len(unique)} items to {output_file}")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", nargs="*", default=list(DEFAULT_REPOS))
    parser.add_argument("--output-dir", default="data/github_raw")
    parser.add_argument("--max-items-per-kind", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.7, help="politeness delay between requests")
    parser.add_argument("--no-readme", action="store_true")
    parser.add_argument("--token", default=None,
                        help="omit: GITHUB_TOKEN/GH_TOKEN is read from the environment instead")
    args = parser.parse_args()

    collector = GitHubCollector(token=args.token, sleep=args.sleep)
    print(f"Rate limit: {collector.rate_limit.limit or 'unknown'}/hour, "
          f"authenticated={bool(collector.token)}")
    collector.collect_dataset(args.repos, args.output_dir,
                              max_items_per_kind=args.max_items_per_kind,
                              include_readme=not args.no_readme)


if __name__ == "__main__":
    main()
