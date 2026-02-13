#!/usr/bin/env python3
"""
Lightweight internal Jira-like tracker for roadmap execution.

Features:
- Projects
- Epics
- Issues
- Comments
- Dependencies
- Board view
- Markdown export
- CC CAPE roadmap seed data
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path
import secrets
import sqlite3
import sys
from typing import Sequence

DEFAULT_DB_PATH = "data/internal_jira.db"

STATUSES = ("backlog", "todo", "in_progress", "blocked", "in_review", "done")
PRIORITIES = ("p0", "p1", "p2", "p3")
ISSUE_TYPES = ("story", "task", "bug", "spike", "chore")
ROLES = ("admin", "editor", "viewer")

STATUS_LABELS = {
    "backlog": "Backlog",
    "todo": "To Do",
    "in_progress": "In Progress",
    "blocked": "Blocked",
    "in_review": "In Review",
    "done": "Done",
}

PRIORITY_LABELS = {
    "p0": "P0",
    "p1": "P1",
    "p2": "P2",
    "p3": "P3",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS epics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    owner TEXT,
    target_date TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    epic_id INTEGER,
    key TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    assignee TEXT,
    story_points INTEGER,
    sprint TEXT,
    due_date TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (epic_id) REFERENCES epics(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS issue_dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id INTEGER NOT NULL,
    depends_on_issue_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (issue_id, depends_on_issue_id),
    CHECK (issue_id != depends_on_issue_id),
    FOREIGN KEY (issue_id) REFERENCES issues(id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_issue_id) REFERENCES issues(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS issue_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id INTEGER NOT NULL,
    author TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (issue_id) REFERENCES issues(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS issue_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    actor TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (issue_id) REFERENCES issues(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
"""


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_choice(value: str, allowed: Sequence[str], field_name: str) -> str:
    normalized = value.strip().lower()
    if normalized not in allowed:
        options = ", ".join(allowed)
        raise ValueError(f"Invalid {field_name}: '{value}'. Allowed values: {options}")
    return normalized


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    if not password:
        raise ValueError("Password cannot be empty.")
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    salt = bytes.fromhex(salt_hex)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200_000,
    ).hex()
    return salt_hex, password_hash


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    _, candidate_hash = hash_password(password, salt_hex=salt_hex)
    return secrets.compare_digest(candidate_hash, expected_hash)


def connect(db_path: str) -> sqlite3.Connection:
    ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def get_user_by_username(conn: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,),
    ).fetchone()


def create_user(conn: sqlite3.Connection, username: str, password: str, role: str) -> sqlite3.Row:
    role_normalized = normalize_choice(role, ROLES, "role")
    username_clean = username.strip()
    if not username_clean:
        raise ValueError("Username cannot be empty.")
    if get_user_by_username(conn, username_clean):
        raise ValueError(f"User '{username_clean}' already exists.")
    salt_hex, password_hash = hash_password(password)
    ts = now_utc()
    conn.execute(
        """
        INSERT INTO users (username, password_salt, password_hash, role, active, created_at)
        VALUES (?, ?, ?, ?, 1, ?)
        """,
        (username_clean, salt_hex, password_hash, role_normalized, ts),
    )
    return conn.execute("SELECT * FROM users WHERE username = ?", (username_clean,)).fetchone()


def ensure_default_admin(conn: sqlite3.Connection, username: str = "admin", password: str = "admin123") -> None:
    if conn.execute("SELECT 1 FROM users LIMIT 1").fetchone():
        return
    create_user(conn, username=username, password=password, role="admin")
    conn.commit()


def get_project(conn: sqlite3.Connection, project_key: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM projects WHERE key = ?",
        (project_key.upper(),),
    ).fetchone()


def get_epic(conn: sqlite3.Connection, epic_key: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM epics WHERE key = ?",
        (epic_key.upper(),),
    ).fetchone()


def get_issue(conn: sqlite3.Connection, issue_key: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT i.*, p.key AS project_key, e.key AS epic_key, e.title AS epic_title
        FROM issues i
        JOIN projects p ON p.id = i.project_id
        LEFT JOIN epics e ON e.id = i.epic_id
        WHERE i.key = ?
        """,
        (issue_key.upper(),),
    ).fetchone()


def next_epic_key(conn: sqlite3.Connection, project: sqlite3.Row) -> str:
    prefix = f"{project['key']}-EP"
    rows = conn.execute(
        "SELECT key FROM epics WHERE project_id = ?",
        (project["id"],),
    ).fetchall()
    max_num = 0
    for row in rows:
        key = row["key"]
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix.isdigit():
            max_num = max(max_num, int(suffix))
    return f"{prefix}{max_num + 1}"


def next_issue_key(conn: sqlite3.Connection, project: sqlite3.Row) -> str:
    prefix = f"{project['key']}-"
    rows = conn.execute(
        "SELECT key FROM issues WHERE project_id = ?",
        (project["id"],),
    ).fetchall()
    max_num = 0
    for row in rows:
        key = row["key"]
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix.isdigit():
            max_num = max(max_num, int(suffix))
    return f"{prefix}{max_num + 1}"


def status_rank_sql(column: str) -> str:
    return (
        f"CASE {column} "
        "WHEN 'backlog' THEN 1 "
        "WHEN 'todo' THEN 2 "
        "WHEN 'in_progress' THEN 3 "
        "WHEN 'blocked' THEN 4 "
        "WHEN 'in_review' THEN 5 "
        "WHEN 'done' THEN 6 "
        "ELSE 99 END"
    )


def priority_rank_sql(column: str) -> str:
    return (
        f"CASE {column} "
        "WHEN 'p0' THEN 1 "
        "WHEN 'p1' THEN 2 "
        "WHEN 'p2' THEN 3 "
        "WHEN 'p3' THEN 4 "
        "ELSE 99 END"
    )


def label_status(status: str) -> str:
    return STATUS_LABELS.get(status, status)


def label_priority(priority: str) -> str:
    return PRIORITY_LABELS.get(priority, priority.upper())


def truncate(value: str | None, max_len: int) -> str:
    if value is None:
        return "-"
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not rows:
        print("(none)")
        return
    rendered_rows = []
    widths = [len(h) for h in headers]
    for row in rows:
        rendered = [str(col) if col is not None else "-" for col in row]
        rendered_rows.append(rendered)
        for idx, cell in enumerate(rendered):
            widths[idx] = max(widths[idx], len(cell))
    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    separator = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(separator)
    for row in rendered_rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def cmd_init(args: argparse.Namespace) -> int:
    with connect(args.db) as conn:
        init_db(conn)
        ensure_default_admin(conn, username=args.admin_user, password=args.admin_password)
    print(f"Initialized tracker database at {args.db}")
    print(f"Default admin ensured: {args.admin_user}")
    return 0


def cmd_user_create(args: argparse.Namespace) -> int:
    role = normalize_choice(args.role, ROLES, "role")
    with connect(args.db) as conn:
        init_db(conn)
        try:
            create_user(conn, username=args.username, password=args.password, role=role)
        except ValueError as error:
            print(str(error))
            return 1
        conn.commit()
    print(f"Created user {args.username} ({role})")
    return 0


def cmd_user_list(args: argparse.Namespace) -> int:
    with connect(args.db) as conn:
        init_db(conn)
        rows = conn.execute(
            """
            SELECT username, role, active, created_at
            FROM users
            ORDER BY username
            """
        ).fetchall()
    table_rows = [
        (
            row["username"],
            row["role"],
            "yes" if row["active"] else "no",
            row["created_at"],
        )
        for row in rows
    ]
    print_table(("Username", "Role", "Active", "Created"), table_rows)
    return 0


def cmd_user_update(args: argparse.Namespace) -> int:
    updates: list[str] = []
    params: list[object] = []

    if args.role is not None:
        role = normalize_choice(args.role, ROLES, "role")
        updates.append("role = ?")
        params.append(role)
    if args.password is not None:
        salt_hex, password_hash = hash_password(args.password)
        updates.append("password_salt = ?")
        params.append(salt_hex)
        updates.append("password_hash = ?")
        params.append(password_hash)
    if args.active is not None:
        active_val = 1 if args.active.lower() == "yes" else 0
        updates.append("active = ?")
        params.append(active_val)

    if not updates:
        print("No updates provided.")
        return 1

    with connect(args.db) as conn:
        init_db(conn)
        user = get_user_by_username(conn, args.username)
        if not user:
            print(f"User '{args.username}' not found.")
            return 1
        params.append(args.username)
        conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE username = ?",
            params,
        )
        conn.commit()

    print(f"Updated user {args.username}")
    return 0


def cmd_project_create(args: argparse.Namespace) -> int:
    project_key = args.key.upper()
    with connect(args.db) as conn:
        init_db(conn)
        existing = get_project(conn, project_key)
        if existing:
            print(f"Project {project_key} already exists.")
            return 1
        conn.execute(
            """
            INSERT INTO projects (key, name, description, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (project_key, args.name, args.description, now_utc()),
        )
        conn.commit()
    print(f"Created project {project_key}")
    return 0


def cmd_project_list(args: argparse.Namespace) -> int:
    with connect(args.db) as conn:
        init_db(conn)
        rows = conn.execute(
            """
            SELECT p.key, p.name,
                   (SELECT COUNT(*) FROM epics e WHERE e.project_id = p.id) AS epic_count,
                   (SELECT COUNT(*) FROM issues i WHERE i.project_id = p.id) AS issue_count,
                   p.created_at
            FROM projects p
            ORDER BY p.key
            """
        ).fetchall()
    table_rows = [(r["key"], r["name"], r["epic_count"], r["issue_count"], r["created_at"]) for r in rows]
    print_table(("Key", "Name", "Epics", "Issues", "Created"), table_rows)
    return 0


def cmd_epic_create(args: argparse.Namespace) -> int:
    status = normalize_choice(args.status, STATUSES, "status")
    priority = normalize_choice(args.priority, PRIORITIES, "priority")
    project_key = args.project.upper()
    with connect(args.db) as conn:
        init_db(conn)
        project = get_project(conn, project_key)
        if not project:
            print(f"Project {project_key} not found.")
            return 1
        epic_key = next_epic_key(conn, project)
        ts = now_utc()
        conn.execute(
            """
            INSERT INTO epics
            (project_id, key, title, description, status, priority, owner, target_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project["id"],
                epic_key,
                args.title,
                args.description,
                status,
                priority,
                args.owner,
                args.target_date,
                ts,
                ts,
            ),
        )
        conn.commit()
    print(f"Created epic {epic_key}")
    return 0


def cmd_epic_list(args: argparse.Namespace) -> int:
    project_key = args.project.upper()
    with connect(args.db) as conn:
        init_db(conn)
        project = get_project(conn, project_key)
        if not project:
            print(f"Project {project_key} not found.")
            return 1
        query = f"""
            SELECT key, title, status, priority, owner, target_date, updated_at
            FROM epics
            WHERE project_id = ?
            ORDER BY {status_rank_sql('status')}, {priority_rank_sql('priority')}, key
        """
        params: list[object] = [project["id"]]
        if args.status:
            status = normalize_choice(args.status, STATUSES, "status")
            query = f"""
                SELECT key, title, status, priority, owner, target_date, updated_at
                FROM epics
                WHERE project_id = ? AND status = ?
                ORDER BY {priority_rank_sql('priority')}, key
            """
            params.append(status)
        rows = conn.execute(query, params).fetchall()
    table_rows = [
        (
            r["key"],
            truncate(r["title"], 50),
            label_status(r["status"]),
            label_priority(r["priority"]),
            r["owner"] or "-",
            r["target_date"] or "-",
            r["updated_at"],
        )
        for r in rows
    ]
    print_table(("Key", "Title", "Status", "Priority", "Owner", "Target", "Updated"), table_rows)
    return 0


def cmd_issue_create(args: argparse.Namespace) -> int:
    issue_type = normalize_choice(args.type, ISSUE_TYPES, "issue type")
    status = normalize_choice(args.status, STATUSES, "status")
    priority = normalize_choice(args.priority, PRIORITIES, "priority")
    project_key = args.project.upper()
    epic_key = args.epic.upper() if args.epic else None
    with connect(args.db) as conn:
        init_db(conn)
        project = get_project(conn, project_key)
        if not project:
            print(f"Project {project_key} not found.")
            return 1
        epic_id = None
        if epic_key:
            epic = get_epic(conn, epic_key)
            if not epic:
                print(f"Epic {epic_key} not found.")
                return 1
            if epic["project_id"] != project["id"]:
                print(f"Epic {epic_key} does not belong to project {project_key}.")
                return 1
            epic_id = epic["id"]
        issue_key = next_issue_key(conn, project)
        ts = now_utc()
        conn.execute(
            """
            INSERT INTO issues
            (project_id, epic_id, key, type, title, description, status, priority, assignee,
             story_points, sprint, due_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project["id"],
                epic_id,
                issue_key,
                issue_type,
                args.title,
                args.description,
                status,
                priority,
                args.assignee,
                args.story_points,
                args.sprint,
                args.due_date,
                ts,
                ts,
            ),
        )
        issue_row = conn.execute("SELECT id FROM issues WHERE key = ?", (issue_key,)).fetchone()
        conn.execute(
            """
            INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
            VALUES (?, 'created', NULL, ?, ?, ?)
            """,
            (issue_row["id"], status, args.assignee or "system", ts),
        )
        conn.commit()
    print(f"Created issue {issue_key}")
    return 0


def cmd_issue_list(args: argparse.Namespace) -> int:
    project_key = args.project.upper()
    with connect(args.db) as conn:
        init_db(conn)
        project = get_project(conn, project_key)
        if not project:
            print(f"Project {project_key} not found.")
            return 1

        query = f"""
            SELECT i.key, i.type, i.title, i.status, i.priority, i.assignee, i.story_points,
                   i.sprint, i.due_date, e.key AS epic_key
            FROM issues i
            LEFT JOIN epics e ON e.id = i.epic_id
            WHERE i.project_id = ?
        """
        params: list[object] = [project["id"]]
        if args.status:
            status = normalize_choice(args.status, STATUSES, "status")
            query += " AND i.status = ?"
            params.append(status)
        if args.assignee:
            query += " AND i.assignee = ?"
            params.append(args.assignee)
        if args.type:
            issue_type = normalize_choice(args.type, ISSUE_TYPES, "issue type")
            query += " AND i.type = ?"
            params.append(issue_type)
        if args.epic:
            epic = get_epic(conn, args.epic.upper())
            if not epic:
                print(f"Epic {args.epic.upper()} not found.")
                return 1
            query += " AND i.epic_id = ?"
            params.append(epic["id"])

        query += f" ORDER BY {status_rank_sql('i.status')}, {priority_rank_sql('i.priority')}, i.key"
        rows = conn.execute(query, params).fetchall()

    table_rows = [
        (
            r["key"],
            r["type"],
            truncate(r["title"], 46),
            label_status(r["status"]),
            label_priority(r["priority"]),
            r["assignee"] or "-",
            r["story_points"] if r["story_points"] is not None else "-",
            r["sprint"] or "-",
            r["due_date"] or "-",
            r["epic_key"] or "-",
        )
        for r in rows
    ]
    print_table(("Key", "Type", "Title", "Status", "Prio", "Assignee", "Pts", "Sprint", "Due", "Epic"), table_rows)
    return 0


def cmd_issue_show(args: argparse.Namespace) -> int:
    issue_key = args.key.upper()
    with connect(args.db) as conn:
        init_db(conn)
        issue = get_issue(conn, issue_key)
        if not issue:
            print(f"Issue {issue_key} not found.")
            return 1
        deps = conn.execute(
            """
            SELECT i.key, i.title, i.status
            FROM issue_dependencies d
            JOIN issues i ON i.id = d.depends_on_issue_id
            WHERE d.issue_id = ?
            ORDER BY i.key
            """,
            (issue["id"],),
        ).fetchall()
        blocked_by = conn.execute(
            """
            SELECT i.key, i.title, i.status
            FROM issue_dependencies d
            JOIN issues i ON i.id = d.issue_id
            WHERE d.depends_on_issue_id = ?
            ORDER BY i.key
            """,
            (issue["id"],),
        ).fetchall()
        comments = conn.execute(
            """
            SELECT author, body, created_at
            FROM issue_comments
            WHERE issue_id = ?
            ORDER BY id
            """,
            (issue["id"],),
        ).fetchall()
        events = conn.execute(
            """
            SELECT event_type, old_value, new_value, actor, created_at
            FROM issue_events
            WHERE issue_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (issue["id"],),
        ).fetchall()

    print(f"{issue['key']} - {issue['title']}")
    print(f"Type: {issue['type']}")
    print(f"Project: {issue['project_key']}")
    print(f"Epic: {issue['epic_key'] or '-'}")
    print(f"Status: {label_status(issue['status'])}")
    print(f"Priority: {label_priority(issue['priority'])}")
    print(f"Assignee: {issue['assignee'] or '-'}")
    print(f"Story Points: {issue['story_points'] if issue['story_points'] is not None else '-'}")
    print(f"Sprint: {issue['sprint'] or '-'}")
    print(f"Due Date: {issue['due_date'] or '-'}")
    print(f"Created: {issue['created_at']}")
    print(f"Updated: {issue['updated_at']}")
    print("Description:")
    print(issue["description"] or "-")

    print("\nDepends on:")
    if deps:
        for dep in deps:
            print(f"- {dep['key']} [{label_status(dep['status'])}] {dep['title']}")
    else:
        print("- none")

    print("\nBlocks:")
    if blocked_by:
        for item in blocked_by:
            print(f"- {item['key']} [{label_status(item['status'])}] {item['title']}")
    else:
        print("- none")

    print("\nRecent Events:")
    if events:
        for event in events:
            old = event["old_value"] or "-"
            new = event["new_value"] or "-"
            print(f"- {event['created_at']} {event['event_type']} {old} -> {new} by {event['actor'] or '-'}")
    else:
        print("- none")

    print("\nComments:")
    if comments:
        for comment in comments:
            print(f"- {comment['created_at']} {comment['author']}: {comment['body']}")
    else:
        print("- none")
    return 0


def cmd_issue_move(args: argparse.Namespace) -> int:
    issue_key = args.key.upper()
    new_status = normalize_choice(args.status, STATUSES, "status")
    actor = args.actor or "system"
    with connect(args.db) as conn:
        init_db(conn)
        issue = get_issue(conn, issue_key)
        if not issue:
            print(f"Issue {issue_key} not found.")
            return 1
        if issue["status"] == new_status:
            print(f"Issue {issue_key} is already {label_status(new_status)}.")
            return 0
        ts = now_utc()
        conn.execute(
            "UPDATE issues SET status = ?, updated_at = ? WHERE id = ?",
            (new_status, ts, issue["id"]),
        )
        conn.execute(
            """
            INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
            VALUES (?, 'status_changed', ?, ?, ?, ?)
            """,
            (issue["id"], issue["status"], new_status, actor, ts),
        )
        conn.commit()
    print(f"Moved {issue_key}: {label_status(issue['status'])} -> {label_status(new_status)}")
    return 0


def cmd_issue_update(args: argparse.Namespace) -> int:
    issue_key = args.key.upper()
    with connect(args.db) as conn:
        init_db(conn)
        issue = get_issue(conn, issue_key)
        if not issue:
            print(f"Issue {issue_key} not found.")
            return 1

        set_clauses: list[str] = []
        params: list[object] = []
        ts = now_utc()
        actor = args.actor or "system"

        if args.title is not None:
            set_clauses.append("title = ?")
            params.append(args.title)
        if args.description is not None:
            set_clauses.append("description = ?")
            params.append(args.description)
        if args.assignee is not None:
            set_clauses.append("assignee = ?")
            params.append(args.assignee)
        if args.sprint is not None:
            set_clauses.append("sprint = ?")
            params.append(args.sprint)
        if args.due_date is not None:
            set_clauses.append("due_date = ?")
            params.append(args.due_date)
        if args.story_points is not None:
            set_clauses.append("story_points = ?")
            params.append(args.story_points)
        if args.priority is not None:
            priority = normalize_choice(args.priority, PRIORITIES, "priority")
            set_clauses.append("priority = ?")
            params.append(priority)
        if args.status is not None:
            status = normalize_choice(args.status, STATUSES, "status")
            set_clauses.append("status = ?")
            params.append(status)
            conn.execute(
                """
                INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
                VALUES (?, 'status_changed', ?, ?, ?, ?)
                """,
                (issue["id"], issue["status"], status, actor, ts),
            )

        if not set_clauses:
            print("No fields provided. Nothing to update.")
            return 1

        set_clauses.append("updated_at = ?")
        params.append(ts)
        params.append(issue["id"])

        conn.execute(
            f"UPDATE issues SET {', '.join(set_clauses)} WHERE id = ?",
            params,
        )
        conn.commit()
    print(f"Updated {issue_key}")
    return 0


def cmd_comment_add(args: argparse.Namespace) -> int:
    issue_key = args.key.upper()
    with connect(args.db) as conn:
        init_db(conn)
        issue = get_issue(conn, issue_key)
        if not issue:
            print(f"Issue {issue_key} not found.")
            return 1
        ts = now_utc()
        conn.execute(
            """
            INSERT INTO issue_comments (issue_id, author, body, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (issue["id"], args.author, args.body, ts),
        )
        conn.execute(
            """
            INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
            VALUES (?, 'comment_added', NULL, NULL, ?, ?)
            """,
            (issue["id"], args.author, ts),
        )
        conn.commit()
    print(f"Added comment to {issue_key}")
    return 0


def cmd_dep_add(args: argparse.Namespace) -> int:
    issue_key = args.key.upper()
    depends_on_key = args.depends_on.upper()
    with connect(args.db) as conn:
        init_db(conn)
        issue = get_issue(conn, issue_key)
        dependency = get_issue(conn, depends_on_key)
        if not issue:
            print(f"Issue {issue_key} not found.")
            return 1
        if not dependency:
            print(f"Issue {depends_on_key} not found.")
            return 1
        if issue["project_id"] != dependency["project_id"]:
            print("Dependencies across projects are not allowed.")
            return 1
        ts = now_utc()
        try:
            conn.execute(
                """
                INSERT INTO issue_dependencies (issue_id, depends_on_issue_id, created_at)
                VALUES (?, ?, ?)
                """,
                (issue["id"], dependency["id"], ts),
            )
        except sqlite3.IntegrityError:
            print(f"Dependency already exists: {issue_key} depends on {depends_on_key}")
            return 1
        conn.execute(
            """
            INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
            VALUES (?, 'dependency_added', NULL, ?, ?, ?)
            """,
            (issue["id"], depends_on_key, args.actor or "system", ts),
        )
        conn.commit()
    print(f"Added dependency: {issue_key} depends on {depends_on_key}")
    return 0


def cmd_board(args: argparse.Namespace) -> int:
    project_key = args.project.upper()
    with connect(args.db) as conn:
        init_db(conn)
        project = get_project(conn, project_key)
        if not project:
            print(f"Project {project_key} not found.")
            return 1
        rows = conn.execute(
            f"""
            SELECT i.key, i.title, i.status, i.priority, i.assignee, i.due_date, e.key AS epic_key
            FROM issues i
            LEFT JOIN epics e ON e.id = i.epic_id
            WHERE i.project_id = ?
            ORDER BY {priority_rank_sql('i.priority')}, i.key
            """,
            (project["id"],),
        ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {status: [] for status in STATUSES}
    for row in rows:
        grouped[row["status"]].append(row)
    print(f"Board: {project_key} ({project['name']})")
    for status in STATUSES:
        items = grouped.get(status, [])
        print(f"\n{STATUS_LABELS[status]} ({len(items)})")
        if not items:
            print("- none")
            continue
        for item in items:
            assignee = item["assignee"] or "unassigned"
            due = item["due_date"] or "-"
            epic_key = item["epic_key"] or "-"
            print(
                f"- {item['key']} [{label_priority(item['priority'])}] "
                f"{truncate(item['title'], 72)} (assignee: {assignee}, due: {due}, epic: {epic_key})"
            )
    return 0


def cmd_export_markdown(args: argparse.Namespace) -> int:
    project_key = args.project.upper()
    output_path = Path(args.out).expanduser()
    with connect(args.db) as conn:
        init_db(conn)
        project = get_project(conn, project_key)
        if not project:
            print(f"Project {project_key} not found.")
            return 1
        epics = conn.execute(
            f"""
            SELECT key, title, status, priority, owner, target_date
            FROM epics
            WHERE project_id = ?
            ORDER BY {status_rank_sql('status')}, {priority_rank_sql('priority')}, key
            """,
            (project["id"],),
        ).fetchall()
        issues = conn.execute(
            f"""
            SELECT i.key, i.type, i.title, i.status, i.priority, i.assignee, i.sprint, i.due_date, e.key AS epic_key
            FROM issues i
            LEFT JOIN epics e ON e.id = i.epic_id
            WHERE i.project_id = ?
            ORDER BY {status_rank_sql('i.status')}, {priority_rank_sql('i.priority')}, i.key
            """,
            (project["id"],),
        ).fetchall()

    grouped: dict[str, list[sqlite3.Row]] = {status: [] for status in STATUSES}
    for row in issues:
        grouped[row["status"]].append(row)

    lines: list[str] = []
    lines.append(f"# Internal Tracker Snapshot: {project_key}")
    lines.append("")
    lines.append(f"Generated: {now_utc()}")
    lines.append("")
    lines.append("## Epics")
    lines.append("")
    if epics:
        lines.append("| Key | Title | Status | Priority | Owner | Target Date |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for epic in epics:
            lines.append(
                f"| {epic['key']} | {epic['title']} | {label_status(epic['status'])} | "
                f"{label_priority(epic['priority'])} | {epic['owner'] or '-'} | {epic['target_date'] or '-'} |"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Board")
    lines.append("")
    for status in STATUSES:
        lines.append(f"### {label_status(status)}")
        lines.append("")
        items = grouped[status]
        if not items:
            lines.append("- none")
            lines.append("")
            continue
        for item in items:
            lines.append(
                f"- **{item['key']}** [{label_priority(item['priority'])}] "
                f"{item['title']} (`{item['type']}`, epic: {item['epic_key'] or '-'}, "
                f"assignee: {item['assignee'] or '-'}, sprint: {item['sprint'] or '-'}, due: {item['due_date'] or '-'})"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported tracker markdown to {output_path}")
    return 0


def get_or_create_epic(
    conn: sqlite3.Connection,
    project: sqlite3.Row,
    title: str,
    description: str,
    owner: str,
    target_date: str,
    priority: str = "p1",
    status: str = "backlog",
) -> sqlite3.Row:
    existing = conn.execute(
        """
        SELECT *
        FROM epics
        WHERE project_id = ? AND title = ?
        """,
        (project["id"], title),
    ).fetchone()
    if existing:
        return existing

    epic_key = next_epic_key(conn, project)
    ts = now_utc()
    conn.execute(
        """
        INSERT INTO epics
        (project_id, key, title, description, status, priority, owner, target_date, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project["id"],
            epic_key,
            title,
            description,
            status,
            priority,
            owner,
            target_date,
            ts,
            ts,
        ),
    )
    return conn.execute("SELECT * FROM epics WHERE key = ?", (epic_key,)).fetchone()


def get_or_create_issue(
    conn: sqlite3.Connection,
    project: sqlite3.Row,
    epic: sqlite3.Row,
    issue_type: str,
    title: str,
    description: str,
    priority: str,
    due_date: str,
    story_points: int,
    sprint: str,
) -> sqlite3.Row:
    existing = conn.execute(
        """
        SELECT *
        FROM issues
        WHERE project_id = ? AND epic_id = ? AND title = ?
        """,
        (project["id"], epic["id"], title),
    ).fetchone()
    if existing:
        return existing

    issue_key = next_issue_key(conn, project)
    ts = now_utc()
    conn.execute(
        """
        INSERT INTO issues
        (project_id, epic_id, key, type, title, description, status, priority, assignee,
         story_points, sprint, due_date, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'backlog', ?, NULL, ?, ?, ?, ?, ?)
        """,
        (
            project["id"],
            epic["id"],
            issue_key,
            issue_type,
            title,
            description,
            priority,
            story_points,
            sprint,
            due_date,
            ts,
            ts,
        ),
    )
    created = conn.execute("SELECT * FROM issues WHERE key = ?", (issue_key,)).fetchone()
    conn.execute(
        """
        INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
        VALUES (?, 'created', NULL, 'backlog', 'system', ?)
        """,
        (created["id"], ts),
    )
    return created


def cmd_seed_cc_cape(args: argparse.Namespace) -> int:
    roadmap = [
        {
            "epic": {
                "title": "Phase 0: Discovery and Specification",
                "description": "Lock methodology assumptions, tolerance bands, and data contracts.",
                "owner": "Product + Quant Lead",
                "target_date": "2026-02-27",
                "priority": "p1",
                "sprint": "Phase-0",
            },
            "issues": [
                ("story", "Finalize CC CAPE methodology specification", "Define formulas, edge-case policy, and acceptance tolerances.", "p1", 8),
                ("task", "Confirm data vendors and legal usage rights", "Validate licensing for constituents, fundamentals, CPI, and benchmark series.", "p1", 5),
                ("task", "Publish technical architecture design", "Capture ETL, compute engine, API, dashboard, and observability design.", "p2", 3),
            ],
        },
        {
            "epic": {
                "title": "Phase 1: Data Foundation",
                "description": "Build source ingestion with data quality gates and lineage.",
                "owner": "Data Engineering",
                "target_date": "2026-03-20",
                "priority": "p1",
                "sprint": "Phase-1",
            },
            "issues": [
                ("story", "Implement constituents and market data ingestion", "Ingest daily close, shares outstanding, and market cap for S&P 500 constituents.", "p1", 8),
                ("story", "Implement fundamentals, CPI, and Shiller CAPE ingestion", "Ingest quarterly earnings, CPI series, and benchmark CAPE history.", "p1", 8),
                ("task", "Add data quality checks and alerting", "Missing, stale, and schema validation checks with alerts.", "p1", 5),
                ("task", "Create raw and curated schemas with lineage metadata", "Store source version IDs and run lineage for auditable recomputes.", "p2", 5),
            ],
        },
        {
            "epic": {
                "title": "Phase 2: Calculation Engine",
                "description": "Ship deterministic CAPE calculations and decomposition outputs.",
                "owner": "Quant Engineering",
                "target_date": "2026-04-10",
                "priority": "p0",
                "sprint": "Phase-2",
            },
            "issues": [
                ("story", "Implement company-level CAPE and CC CAPE calculation", "Compute 10-year real earnings denominator and weighted aggregate CC CAPE.", "p0", 13),
                ("story", "Implement CAPE Spread and percentile analytics", "Compute spread vs Shiller CAPE and rolling percentile/z-score views.", "p1", 8),
                ("task", "Add constituent and sector decomposition outputs", "Produce contribution tables for spread drivers by name and sector.", "p1", 5),
                ("task", "Backfill 10-year history and reconcile sample dates", "Run historical backfill and compare output against known benchmarks.", "p1", 8),
            ],
        },
        {
            "epic": {
                "title": "Phase 3: API and Dashboard MVP",
                "description": "Expose CC CAPE outputs through API and internal dashboard.",
                "owner": "Backend + Frontend Engineering",
                "target_date": "2026-05-01",
                "priority": "p1",
                "sprint": "Phase-3",
            },
            "issues": [
                ("story", "Build read-only API for latest metrics and history", "Ship endpoints for latest values, time series, and decomposition snapshot.", "p1", 8),
                ("story", "Build overview and time-series dashboard pages", "Display latest values, percentile context, and CC CAPE vs Shiller CAPE trends.", "p1", 8),
                ("task", "Build contributors view and CSV export", "Expose top positive/negative spread contributors and downloadable outputs.", "p2", 5),
                ("task", "Instrument usage telemetry for API and dashboard", "Capture weekly active usage and query-level metrics.", "p2", 3),
            ],
        },
        {
            "epic": {
                "title": "Phase 4: Hardening and Pilot",
                "description": "Raise reliability and run internal pilot feedback loop.",
                "owner": "Platform",
                "target_date": "2026-05-22",
                "priority": "p1",
                "sprint": "Phase-4",
            },
            "issues": [
                ("story", "Write operational runbook and failure recovery procedures", "Document incident paths, recovery steps, and escalation owners.", "p1", 8),
                ("task", "Implement role-based access and audit logs", "Protect admin operations and audit changes to data and configuration.", "p1", 5),
                ("task", "Run pilot and triage feedback backlog", "Collect pilot issues, classify severity, and close MVP blockers.", "p1", 5),
            ],
        },
        {
            "epic": {
                "title": "Phase 5: Launch",
                "description": "Transition from pilot to stable production usage.",
                "owner": "Product + Platform",
                "target_date": "2026-06-05",
                "priority": "p1",
                "sprint": "Phase-5",
            },
            "issues": [
                ("story", "Run production release checklist", "Validate release criteria, rollback path, and sign-offs.", "p1", 8),
                ("task", "Publish onboarding docs and quickstart", "Create user guide for analysts and PM workflows.", "p2", 3),
                ("task", "Track and publish MVP KPI baseline", "Report freshness SLA, reliability, and adoption baseline.", "p1", 5),
            ],
        },
    ]

    with connect(args.db) as conn:
        init_db(conn)

        project = get_project(conn, "CAPE")
        if not project:
            conn.execute(
                """
                INSERT INTO projects (key, name, description, created_at)
                VALUES ('CAPE', 'Current Constituents CAPE Tracker',
                        'Internal roadmap execution tracker for CC CAPE product delivery.', ?)
                """,
                (now_utc(),),
            )
            project = get_project(conn, "CAPE")

        created_epics = 0
        created_issues = 0

        for phase in roadmap:
            epic_spec = phase["epic"]
            epic_existing = conn.execute(
                "SELECT * FROM epics WHERE project_id = ? AND title = ?",
                (project["id"], epic_spec["title"]),
            ).fetchone()
            epic = get_or_create_epic(
                conn,
                project,
                title=epic_spec["title"],
                description=epic_spec["description"],
                owner=epic_spec["owner"],
                target_date=epic_spec["target_date"],
                priority=epic_spec["priority"],
                status="backlog",
            )
            if not epic_existing:
                created_epics += 1

            for issue_type, title, description, priority, points in phase["issues"]:
                existing_issue = conn.execute(
                    """
                    SELECT id
                    FROM issues
                    WHERE project_id = ? AND epic_id = ? AND title = ?
                    """,
                    (project["id"], epic["id"], title),
                ).fetchone()
                get_or_create_issue(
                    conn,
                    project=project,
                    epic=epic,
                    issue_type=issue_type,
                    title=title,
                    description=description,
                    priority=priority,
                    due_date=epic_spec["target_date"],
                    story_points=points,
                    sprint=epic_spec["sprint"],
                )
                if not existing_issue:
                    created_issues += 1

        conn.commit()

    print("Seed complete for project CAPE")
    print(f"Created epics: {created_epics}")
    print(f"Created issues: {created_issues}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal Jira-like tracker")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"SQLite DB path (default: {DEFAULT_DB_PATH})")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize the tracker database")
    init_parser.add_argument("--admin-user", default="admin", help="Default admin username")
    init_parser.add_argument("--admin-password", default="admin123", help="Default admin password")
    init_parser.set_defaults(func=cmd_init)

    user_create = subparsers.add_parser("user-create", help="Create a user")
    user_create.add_argument("--username", required=True, help="Username")
    user_create.add_argument("--password", required=True, help="Password")
    user_create.add_argument("--role", required=True, help=f"Role ({', '.join(ROLES)})")
    user_create.set_defaults(func=cmd_user_create)

    user_list = subparsers.add_parser("user-list", help="List users")
    user_list.set_defaults(func=cmd_user_list)

    user_update = subparsers.add_parser("user-update", help="Update role/password/active state")
    user_update.add_argument("--username", required=True, help="Username")
    user_update.add_argument("--role", help=f"Role ({', '.join(ROLES)})")
    user_update.add_argument("--password", help="New password")
    user_update.add_argument("--active", choices=("yes", "no"), help="Whether user is active")
    user_update.set_defaults(func=cmd_user_update)

    project_create = subparsers.add_parser("project-create", help="Create a project")
    project_create.add_argument("--key", required=True, help="Project key, e.g. CAPE")
    project_create.add_argument("--name", required=True, help="Project display name")
    project_create.add_argument("--description", default="", help="Project description")
    project_create.set_defaults(func=cmd_project_create)

    project_list = subparsers.add_parser("project-list", help="List projects")
    project_list.set_defaults(func=cmd_project_list)

    epic_create = subparsers.add_parser("epic-create", help="Create an epic")
    epic_create.add_argument("--project", required=True, help="Project key")
    epic_create.add_argument("--title", required=True, help="Epic title")
    epic_create.add_argument("--description", default="", help="Epic description")
    epic_create.add_argument("--owner", default="", help="Epic owner")
    epic_create.add_argument("--status", default="backlog", help=f"Epic status ({', '.join(STATUSES)})")
    epic_create.add_argument("--priority", default="p2", help=f"Epic priority ({', '.join(PRIORITIES)})")
    epic_create.add_argument("--target-date", default="", help="Target date YYYY-MM-DD")
    epic_create.set_defaults(func=cmd_epic_create)

    epic_list = subparsers.add_parser("epic-list", help="List epics in a project")
    epic_list.add_argument("--project", required=True, help="Project key")
    epic_list.add_argument("--status", help=f"Filter status ({', '.join(STATUSES)})")
    epic_list.set_defaults(func=cmd_epic_list)

    issue_create = subparsers.add_parser("issue-create", help="Create an issue")
    issue_create.add_argument("--project", required=True, help="Project key")
    issue_create.add_argument("--epic", help="Epic key, e.g. CAPE-EP1")
    issue_create.add_argument("--type", default="task", help=f"Issue type ({', '.join(ISSUE_TYPES)})")
    issue_create.add_argument("--title", required=True, help="Issue title")
    issue_create.add_argument("--description", default="", help="Issue description")
    issue_create.add_argument("--status", default="backlog", help=f"Issue status ({', '.join(STATUSES)})")
    issue_create.add_argument("--priority", default="p2", help=f"Issue priority ({', '.join(PRIORITIES)})")
    issue_create.add_argument("--assignee", default="", help="Assignee")
    issue_create.add_argument("--story-points", type=int, help="Story points")
    issue_create.add_argument("--sprint", default="", help="Sprint label")
    issue_create.add_argument("--due-date", default="", help="Due date YYYY-MM-DD")
    issue_create.set_defaults(func=cmd_issue_create)

    issue_list = subparsers.add_parser("issue-list", help="List issues in a project")
    issue_list.add_argument("--project", required=True, help="Project key")
    issue_list.add_argument("--status", help=f"Filter status ({', '.join(STATUSES)})")
    issue_list.add_argument("--assignee", help="Filter assignee")
    issue_list.add_argument("--type", help=f"Filter issue type ({', '.join(ISSUE_TYPES)})")
    issue_list.add_argument("--epic", help="Filter by epic key")
    issue_list.set_defaults(func=cmd_issue_list)

    issue_show = subparsers.add_parser("issue-show", help="Show issue details")
    issue_show.add_argument("--key", required=True, help="Issue key, e.g. CAPE-1")
    issue_show.set_defaults(func=cmd_issue_show)

    issue_move = subparsers.add_parser("issue-move", help="Move issue status")
    issue_move.add_argument("--key", required=True, help="Issue key")
    issue_move.add_argument("--status", required=True, help=f"New status ({', '.join(STATUSES)})")
    issue_move.add_argument("--actor", default="system", help="Actor name")
    issue_move.set_defaults(func=cmd_issue_move)

    issue_update = subparsers.add_parser("issue-update", help="Update issue fields")
    issue_update.add_argument("--key", required=True, help="Issue key")
    issue_update.add_argument("--title", help="New title")
    issue_update.add_argument("--description", help="New description")
    issue_update.add_argument("--assignee", help="New assignee")
    issue_update.add_argument("--priority", help=f"New priority ({', '.join(PRIORITIES)})")
    issue_update.add_argument("--status", help=f"New status ({', '.join(STATUSES)})")
    issue_update.add_argument("--story-points", type=int, help="Story points")
    issue_update.add_argument("--sprint", help="Sprint label")
    issue_update.add_argument("--due-date", help="Due date YYYY-MM-DD")
    issue_update.add_argument("--actor", default="system", help="Actor name")
    issue_update.set_defaults(func=cmd_issue_update)

    comment_add = subparsers.add_parser("comment-add", help="Add comment to issue")
    comment_add.add_argument("--key", required=True, help="Issue key")
    comment_add.add_argument("--author", required=True, help="Comment author")
    comment_add.add_argument("--body", required=True, help="Comment text")
    comment_add.set_defaults(func=cmd_comment_add)

    dep_add = subparsers.add_parser("dep-add", help="Add issue dependency")
    dep_add.add_argument("--key", required=True, help="Issue key")
    dep_add.add_argument("--depends-on", required=True, help="Blocking issue key")
    dep_add.add_argument("--actor", default="system", help="Actor name")
    dep_add.set_defaults(func=cmd_dep_add)

    board = subparsers.add_parser("board", help="Show project board grouped by status")
    board.add_argument("--project", required=True, help="Project key")
    board.set_defaults(func=cmd_board)

    export_md = subparsers.add_parser("export-markdown", help="Export project snapshot as markdown")
    export_md.add_argument("--project", required=True, help="Project key")
    export_md.add_argument("--out", required=True, help="Output path")
    export_md.set_defaults(func=cmd_export_markdown)

    seed = subparsers.add_parser("seed-cc-cape", help="Seed CAPE roadmap epics and issues")
    seed.set_defaults(func=cmd_seed_cc_cape)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as error:
        print(str(error))
        return 1
    except sqlite3.Error as error:
        print(f"Database error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
