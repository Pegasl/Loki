import hashlib
import json
import os
import secrets
from datetime import datetime
from pathlib import Path


def wiki_init(root: str | Path = ".") -> bool:
    """Create the small directory structure needed by the Wiki."""
    root = Path(root).expanduser().resolve()
    raw = root / "raw"
    wiki = root / "wiki"
    log = wiki / "log.md"
    aliases = wiki / "aliases.json"
    sources = wiki / "sources.json"

    if root.exists() and not root.is_dir():
        return False
    if raw.exists() and not raw.is_dir():
        return False
    if wiki.exists() and not wiki.is_dir():
        return False
    if log.exists() and not log.is_file():
        return False
    if aliases.exists() and not aliases.is_file():
        return False
    if sources.exists() and not sources.is_file():
        return False

    try:
        root.mkdir(parents=True, exist_ok=True)
        raw.mkdir(exist_ok=True)
        wiki.mkdir(exist_ok=True)

        if not log.exists():
            log.write_text("", encoding="utf-8")

        if not aliases.exists():
            aliases.write_text("", encoding="utf-8")

        if not sources.exists():
            sources.write_text("", encoding="utf-8")

    except OSError:
        return False

    return True


def wiki_register(root: str | Path = ".") -> bool:
    """Register every file below raw/ and create a Summary draft for new files."""
    root = Path(root).expanduser().resolve()
    raw = root / "raw"
    wiki = root / "wiki"
    sources_file = wiki / "sources.json"
    log_file = wiki / "log.md"

    errors = []
    log_messages = []

    try:
        root.mkdir(parents=True, exist_ok=True)
        raw.mkdir(exist_ok=True)
        wiki.mkdir(exist_ok=True)

        if sources_file.exists():
            text = sources_file.read_text(encoding="utf-8")
            sources = json.loads(text) if text.strip() else []
        else:
            sources = []

        if not isinstance(sources, list):
            raise ValueError("sources.json must contain a list")

        all_files = sorted(path for path in raw.rglob("*") if path.is_file())
        current_filenames = {
            path.relative_to(raw).as_posix() for path in all_files
        }

        active_sources = []
        for source in sources:
            filename = source.get("filename")
            if filename in current_filenames:
                active_sources.append(source)
                continue

            log_messages.append(
                f"## [{datetime.now().astimezone().date().isoformat()}] "
                f"register removed | {filename}\n\n"
                f"- File no longer exists under `raw/`; old registration was removed.\n"
                f"- SrcID: `{source.get('srcid')}`"
            )

        sources = active_sources
        files = [
            path
            for path in all_files
            if path.suffix.lower() in {".md", ".txt"}
        ]

        used_srcids = set()
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("Each source registration must be an object")
            if isinstance(source.get("srcid"), str):
                used_srcids.add(source["srcid"])

        for file in files:
            filename = file.relative_to(raw).as_posix()

            try:
                digest = hashlib.sha256(file.read_bytes()).hexdigest()
            except OSError as error:
                errors.append(f"{filename}: unable to read file: {error}")
                continue

            registered = None
            for source in sources:
                if source.get("filename") == filename:
                    registered = source
                    break

            if registered is not None:
                if registered.get("sha256") == digest:
                    continue

                errors.append(
                    f"{filename}: SHA256 changed; the file was not registered "
                    f"(old: {registered.get('sha256')}, new: {digest})"
                )
                log_messages.append(
                    f"## [{datetime.now().astimezone().date().isoformat()}] "
                    f"register skipped | {filename}\n\n"
                    f"- Error: SHA256 changed; file was not registered.\n"
                    f"- Old SHA256: `{registered.get('sha256')}`\n"
                    f"- New SHA256: `{digest}`"
                )
                continue

            srcid = "srcid-" + secrets.token_hex(3)
            while srcid in used_srcids:
                srcid = "srcid-" + secrets.token_hex(3)
            used_srcids.add(srcid)

            document_name = file.stem.strip() or "source"
            document = wiki / document_name
            if document.exists():
                document = wiki / f"{document_name}-{srcid}"

            document.mkdir(parents=True, exist_ok=False)
            (document / "concepts").mkdir()
            (document / "entities").mkdir()

            token = srcid.removeprefix("srcid-")
            summary = document / f"source-summary-{token}.md"
            source_path = os.path.relpath(file, summary.parent).replace(os.sep, "/")
            title = file.stem.strip() or file.name
            title = json.dumps(title, ensure_ascii=False)
            source_path = json.dumps(source_path, ensure_ascii=False)

            summary.write_text(
                "\n".join(
                    [
                        "---",
                        'type: "Source Summary"',
                        f"title: {title}",
                        'description: ""',
                        f"source: {source_path}",
                        f'srcid: "{srcid}"',
                        f'source_hash: "sha256:{digest}"',
                        "tags: []",
                        'timestamp: ""',
                        "---",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            sources.append(
                {
                    "filename": filename,
                    "srcid": srcid,
                    "sha256": digest,
                    "ingest": False,
                }
            )
            log_messages.append(
                f"## [{datetime.now().astimezone().date().isoformat()}] "
                f"register | {filename}\n\n"
                f"- SrcID: `{srcid}`\n"
                f"- SHA256: `{digest}`\n"
                f"- Ingest: `false`\n"
                f"- Summary: `{summary.relative_to(root).as_posix()}`"
            )

        sources_file.write_text(
            json.dumps(sources, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if not log_file.exists():
            log_file.write_text("", encoding="utf-8")
        old_log = log_file.read_text(encoding="utf-8").rstrip()

        if errors:
            log_messages.append(
                f"## [{datetime.now().astimezone().date().isoformat()}] "
                "register failed\n\n"
                + "\n".join(f"- Error: {error}" for error in errors)
            )
        else:
            log_messages.append(
                f"## [{datetime.now().astimezone().date().isoformat()}] "
                "register complete"
            )

        if log_messages:
            log_file.write_text(
                old_log + "\n\n" + "\n\n".join(log_messages) + "\n",
                encoding="utf-8",
            )

        return not errors

    except Exception as error:
        try:
            wiki.mkdir(parents=True, exist_ok=True)
            old_log = log_file.read_text(encoding="utf-8").rstrip() if log_file.exists() else ""
            message = (
                f"## [{datetime.now().astimezone().date().isoformat()}] "
                "register failed\n\n"
                f"- Error: {error}\n"
            )
            log_file.write_text(old_log + "\n\n" + message, encoding="utf-8")
        except Exception:
            pass
        return False
