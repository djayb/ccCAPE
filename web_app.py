#!/usr/bin/env python3
"""Minimal internal web UI for the CC CAPE tracker."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from internal_jira import (
    PRIORITY_LABELS,
    ROLES,
    STATUSES,
    STATUS_LABELS,
    connect,
    create_user,
    get_issue,
    get_project,
    get_user_by_username,
    init_db,
    ensure_default_admin,
    now_utc,
    normalize_choice,
    priority_rank_sql,
    status_rank_sql,
    verify_password,
)

DB_PATH = os.environ.get("TRACKER_DB", "data/internal_jira.db")
SESSION_SECRET = os.environ.get("TRACKER_SESSION_SECRET", "replace-this-secret")
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="CC CAPE Internal Tracker")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _open_conn():
    conn = connect(DB_PATH)
    init_db(conn)
    ensure_default_admin(conn)
    return conn


def _current_user(request: Request, conn):
    username = request.session.get("username")
    if not username:
        return None
    user = get_user_by_username(conn, username)
    if not user:
        request.session.clear()
        return None
    if not user["active"]:
        request.session.clear()
        return None
    return user


def _require_login(request: Request, conn):
    user = _current_user(request, conn)
    if not user:
        return None, RedirectResponse("/login", status_code=303)
    return user, None


def _require_api_login(request: Request, conn):
    user = _current_user(request, conn)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def _can_edit(user) -> bool:
    return user["role"] in {"admin", "editor"}


@app.on_event("startup")
def startup() -> None:
    with _open_conn() as conn:
        init_db(conn)
        ensure_default_admin(conn)


@app.get("/", response_class=HTMLResponse)
def root() -> RedirectResponse:
    return RedirectResponse("/board", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": request.query_params.get("error", ""),
            "message": request.query_params.get("msg", ""),
        },
    )


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    with _open_conn() as conn:
        user = get_user_by_username(conn, username)
        if not user or not user["active"]:
            return RedirectResponse("/login?error=Invalid%20credentials", status_code=303)
        if not verify_password(password, user["password_salt"], user["password_hash"]):
            return RedirectResponse("/login?error=Invalid%20credentials", status_code=303)

    request.session["username"] = user["username"]
    request.session["role"] = user["role"]
    return RedirectResponse("/board", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login?msg=Signed%20out", status_code=303)


@app.get("/board", response_class=HTMLResponse)
def board(request: Request, project: str | None = None):
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect

        projects = conn.execute("SELECT key, name FROM projects ORDER BY key").fetchall()
        selected_project = (project or (projects[0]["key"] if projects else "")).upper()
        project_row = get_project(conn, selected_project) if selected_project else None

        grouped: dict[str, list] = {status: [] for status in STATUSES}
        if project_row:
            issues = conn.execute(
                f"""
                SELECT i.key, i.title, i.status, i.priority, i.assignee, i.due_date, e.key AS epic_key
                FROM issues i
                LEFT JOIN epics e ON e.id = i.epic_id
                WHERE i.project_id = ?
                ORDER BY {status_rank_sql('i.status')}, {priority_rank_sql('i.priority')}, i.key
                """,
                (project_row["id"],),
            ).fetchall()
            for issue in issues:
                grouped[issue["status"]].append(issue)

    return templates.TemplateResponse(
        "board.html",
        {
            "request": request,
            "user": user,
            "projects": projects,
            "selected_project": selected_project,
            "project_row": project_row,
            "grouped": grouped,
            "statuses": STATUSES,
            "status_labels": STATUS_LABELS,
            "priority_labels": PRIORITY_LABELS,
            "can_edit": _can_edit(user),
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


def _issue_summary_payload(issue) -> dict:
    return {
        "key": issue["key"],
        "title": issue["title"],
        "status": issue["status"],
        "status_label": STATUS_LABELS.get(issue["status"], issue["status"]),
        "priority": issue["priority"],
        "priority_label": PRIORITY_LABELS.get(issue["priority"], issue["priority"]),
        "assignee": issue["assignee"],
        "due_date": issue["due_date"],
        "epic_key": issue["epic_key"],
    }


@app.get("/api/board")
def api_board(request: Request, project: str | None = None):
    with _open_conn() as conn:
        user = _require_api_login(request, conn)
        projects = conn.execute("SELECT key, name FROM projects ORDER BY key").fetchall()
        if not projects:
            return {
                "project": None,
                "statuses": [],
                "generated_at": now_utc(),
                "viewer_role": user["role"],
            }

        selected_project = (project or projects[0]["key"]).upper()
        project_row = get_project(conn, selected_project)
        if not project_row:
            raise HTTPException(status_code=404, detail=f"Project '{selected_project}' not found.")

        issues = conn.execute(
            f"""
            SELECT i.key, i.title, i.status, i.priority, i.assignee, i.due_date, e.key AS epic_key
            FROM issues i
            LEFT JOIN epics e ON e.id = i.epic_id
            WHERE i.project_id = ?
            ORDER BY {status_rank_sql('i.status')}, {priority_rank_sql('i.priority')}, i.key
            """,
            (project_row["id"],),
        ).fetchall()

    grouped: dict[str, list] = {status: [] for status in STATUSES}
    for issue in issues:
        grouped[issue["status"]].append(_issue_summary_payload(issue))

    return {
        "project": {"key": project_row["key"], "name": project_row["name"]},
        "statuses": [
            {
                "key": status,
                "label": STATUS_LABELS.get(status, status),
                "count": len(grouped[status]),
                "issues": grouped[status],
            }
            for status in STATUSES
        ],
        "generated_at": now_utc(),
        "viewer_role": user["role"],
    }


@app.get("/api/issues")
def api_issues(
    request: Request,
    project: str | None = None,
    status: str | None = None,
    assignee: str | None = None,
    epic: str | None = None,
    issue_type: str | None = None,
):
    with _open_conn() as conn:
        user = _require_api_login(request, conn)

        projects = conn.execute("SELECT key FROM projects ORDER BY key").fetchall()
        if not projects:
            return {"project": None, "issues": [], "count": 0, "generated_at": now_utc(), "viewer_role": user["role"]}

        selected_project = (project or projects[0]["key"]).upper()
        project_row = get_project(conn, selected_project)
        if not project_row:
            raise HTTPException(status_code=404, detail=f"Project '{selected_project}' not found.")

        query = f"""
            SELECT i.key, i.type, i.title, i.status, i.priority, i.assignee, i.story_points,
                   i.sprint, i.due_date, i.created_at, i.updated_at, e.key AS epic_key
            FROM issues i
            LEFT JOIN epics e ON e.id = i.epic_id
            WHERE i.project_id = ?
        """
        params: list[object] = [project_row["id"]]

        if status:
            try:
                status_normalized = normalize_choice(status, STATUSES, "status")
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            query += " AND i.status = ?"
            params.append(status_normalized)

        if assignee:
            query += " AND i.assignee = ?"
            params.append(assignee)

        if epic:
            epic_key = epic.upper()
            epic_row = conn.execute(
                "SELECT id FROM epics WHERE project_id = ? AND key = ?",
                (project_row["id"], epic_key),
            ).fetchone()
            if not epic_row:
                raise HTTPException(status_code=404, detail=f"Epic '{epic_key}' not found in project '{selected_project}'.")
            query += " AND i.epic_id = ?"
            params.append(epic_row["id"])

        if issue_type:
            query += " AND i.type = ?"
            params.append(issue_type.lower())

        query += f" ORDER BY {status_rank_sql('i.status')}, {priority_rank_sql('i.priority')}, i.key"
        rows = conn.execute(query, params).fetchall()

    issues = []
    for row in rows:
        item = _issue_summary_payload(row)
        item.update(
            {
                "type": row["type"],
                "story_points": row["story_points"],
                "sprint": row["sprint"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
        issues.append(item)

    return {
        "project": {"key": selected_project},
        "count": len(issues),
        "issues": issues,
        "generated_at": now_utc(),
        "viewer_role": user["role"],
    }


@app.get("/api/issues/{issue_key}")
def api_issue_detail(request: Request, issue_key: str):
    issue_key = issue_key.upper()
    with _open_conn() as conn:
        user = _require_api_login(request, conn)
        issue = get_issue(conn, issue_key)
        if not issue:
            raise HTTPException(status_code=404, detail=f"Issue '{issue_key}' not found.")

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
            ORDER BY id DESC
            """,
            (issue["id"],),
        ).fetchall()
        events = conn.execute(
            """
            SELECT event_type, old_value, new_value, actor, created_at
            FROM issue_events
            WHERE issue_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (issue["id"],),
        ).fetchall()
        project_row = conn.execute("SELECT key, name FROM projects WHERE id = ?", (issue["project_id"],)).fetchone()

    return {
        "issue": {
            "key": issue["key"],
            "title": issue["title"],
            "description": issue["description"],
            "type": issue["type"],
            "status": issue["status"],
            "status_label": STATUS_LABELS.get(issue["status"], issue["status"]),
            "priority": issue["priority"],
            "priority_label": PRIORITY_LABELS.get(issue["priority"], issue["priority"]),
            "assignee": issue["assignee"],
            "story_points": issue["story_points"],
            "sprint": issue["sprint"],
            "due_date": issue["due_date"],
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "project_key": project_row["key"] if project_row else None,
            "project_name": project_row["name"] if project_row else None,
            "epic_key": issue["epic_key"],
        },
        "depends_on": [
            {"key": dep["key"], "title": dep["title"], "status": dep["status"], "status_label": STATUS_LABELS.get(dep["status"], dep["status"])}
            for dep in deps
        ],
        "blocks": [
            {
                "key": item["key"],
                "title": item["title"],
                "status": item["status"],
                "status_label": STATUS_LABELS.get(item["status"], item["status"]),
            }
            for item in blocked_by
        ],
        "comments": [{"author": row["author"], "body": row["body"], "created_at": row["created_at"]} for row in comments],
        "events": [
            {
                "event_type": row["event_type"],
                "old_value": row["old_value"],
                "new_value": row["new_value"],
                "actor": row["actor"],
                "created_at": row["created_at"],
            }
            for row in events
        ],
        "generated_at": now_utc(),
        "viewer_role": user["role"],
    }


@app.get("/issue/{issue_key}", response_class=HTMLResponse)
def issue_detail(request: Request, issue_key: str):
    issue_key = issue_key.upper()
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect

        issue = get_issue(conn, issue_key)
        if not issue:
            return RedirectResponse("/board?error=Issue%20not%20found", status_code=303)

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
            ORDER BY id DESC
            """,
            (issue["id"],),
        ).fetchall()

        events = conn.execute(
            """
            SELECT event_type, old_value, new_value, actor, created_at
            FROM issue_events
            WHERE issue_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (issue["id"],),
        ).fetchall()

        project_row = conn.execute("SELECT key, name FROM projects WHERE id = ?", (issue["project_id"],)).fetchone()

    return templates.TemplateResponse(
        "issue.html",
        {
            "request": request,
            "user": user,
            "issue": issue,
            "project_row": project_row,
            "deps": deps,
            "blocked_by": blocked_by,
            "comments": comments,
            "events": events,
            "statuses": STATUSES,
            "status_labels": STATUS_LABELS,
            "priority_labels": PRIORITY_LABELS,
            "can_edit": _can_edit(user),
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@app.post("/issue/{issue_key}/status")
def update_issue_status(request: Request, issue_key: str, status: str = Form(...)):
    issue_key = issue_key.upper()
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if not _can_edit(user):
            return RedirectResponse(f"/issue/{issue_key}?error=Insufficient%20role", status_code=303)

        issue = get_issue(conn, issue_key)
        if not issue:
            return RedirectResponse("/board?error=Issue%20not%20found", status_code=303)

        new_status = normalize_choice(status, STATUSES, "status")
        if issue["status"] != new_status:
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
                (issue["id"], issue["status"], new_status, user["username"], ts),
            )
            conn.commit()

    return RedirectResponse(f"/issue/{issue_key}?msg=Status%20updated", status_code=303)


@app.post("/issue/{issue_key}/comment")
def add_issue_comment(request: Request, issue_key: str, body: str = Form(...)):
    issue_key = issue_key.upper()
    body = body.strip()
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if not _can_edit(user):
            return RedirectResponse(f"/issue/{issue_key}?error=Insufficient%20role", status_code=303)
        if not body:
            return RedirectResponse(f"/issue/{issue_key}?error=Comment%20cannot%20be%20empty", status_code=303)

        issue = get_issue(conn, issue_key)
        if not issue:
            return RedirectResponse("/board?error=Issue%20not%20found", status_code=303)

        ts = now_utc()
        conn.execute(
            "INSERT INTO issue_comments (issue_id, author, body, created_at) VALUES (?, ?, ?, ?)",
            (issue["id"], user["username"], body, ts),
        )
        conn.execute(
            """
            INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
            VALUES (?, 'comment_added', NULL, NULL, ?, ?)
            """,
            (issue["id"], user["username"], ts),
        )
        conn.execute(
            "UPDATE issues SET updated_at = ? WHERE id = ?",
            (ts, issue["id"]),
        )
        conn.commit()

    return RedirectResponse(f"/issue/{issue_key}?msg=Comment%20added", status_code=303)


@app.post("/issue/{issue_key}/dependency")
def add_issue_dependency(request: Request, issue_key: str, depends_on: str = Form(...)):
    issue_key = issue_key.upper()
    depends_on = depends_on.upper().strip()

    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if not _can_edit(user):
            return RedirectResponse(f"/issue/{issue_key}?error=Insufficient%20role", status_code=303)

        issue = get_issue(conn, issue_key)
        dep = get_issue(conn, depends_on)
        if not issue or not dep:
            return RedirectResponse(f"/issue/{issue_key}?error=Issue%20or%20dependency%20not%20found", status_code=303)
        if issue["project_id"] != dep["project_id"]:
            return RedirectResponse(f"/issue/{issue_key}?error=Dependency%20must%20be%20in%20same%20project", status_code=303)

        ts = now_utc()
        try:
            conn.execute(
                "INSERT INTO issue_dependencies (issue_id, depends_on_issue_id, created_at) VALUES (?, ?, ?)",
                (issue["id"], dep["id"], ts),
            )
        except Exception:
            return RedirectResponse(f"/issue/{issue_key}?error=Dependency%20already%20exists", status_code=303)

        conn.execute(
            """
            INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
            VALUES (?, 'dependency_added', NULL, ?, ?, ?)
            """,
            (issue["id"], depends_on, user["username"], ts),
        )
        conn.execute(
            "UPDATE issues SET updated_at = ? WHERE id = ?",
            (ts, issue["id"]),
        )
        conn.commit()

    return RedirectResponse(f"/issue/{issue_key}?msg=Dependency%20added", status_code=303)


@app.post("/issue/{issue_key}/assignee")
def update_issue_assignee(request: Request, issue_key: str, assignee: str = Form("")):
    issue_key = issue_key.upper()
    assignee = assignee.strip()

    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if not _can_edit(user):
            return RedirectResponse(f"/issue/{issue_key}?error=Insufficient%20role", status_code=303)

        issue = get_issue(conn, issue_key)
        if not issue:
            return RedirectResponse("/board?error=Issue%20not%20found", status_code=303)

        ts = now_utc()
        conn.execute(
            "UPDATE issues SET assignee = ?, updated_at = ? WHERE id = ?",
            (assignee if assignee else None, ts, issue["id"]),
        )
        conn.execute(
            """
            INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
            VALUES (?, 'assignee_changed', ?, ?, ?, ?)
            """,
            (issue["id"], issue["assignee"], assignee if assignee else None, user["username"], ts),
        )
        conn.commit()

    return RedirectResponse(f"/issue/{issue_key}?msg=Assignee%20updated", status_code=303)


@app.get("/admin/users", response_class=HTMLResponse)
def users_page(request: Request):
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if user["role"] != "admin":
            return RedirectResponse("/board?error=Admin%20role%20required", status_code=303)

        users = conn.execute(
            "SELECT username, role, active, created_at FROM users ORDER BY username"
        ).fetchall()

    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
            "user": user,
            "users": users,
            "roles": ROLES,
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@app.post("/admin/users")
def users_create(request: Request, username: str = Form(...), password: str = Form(...), role: str = Form(...)):
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if user["role"] != "admin":
            return RedirectResponse("/board?error=Admin%20role%20required", status_code=303)

        try:
            create_user(conn, username=username, password=password, role=role)
            conn.commit()
        except Exception as error:
            msg = str(error).replace(" ", "%20")
            return RedirectResponse(f"/admin/users?error={msg}", status_code=303)

    return RedirectResponse("/admin/users?msg=User%20created", status_code=303)


@app.post("/admin/users/{username}/active")
def users_toggle_active(request: Request, username: str, active: str = Form(...)):
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if user["role"] != "admin":
            return RedirectResponse("/board?error=Admin%20role%20required", status_code=303)

        value = 1 if active.lower() == "yes" else 0
        conn.execute("UPDATE users SET active = ? WHERE username = ?", (value, username))
        conn.commit()

    return RedirectResponse("/admin/users?msg=User%20updated", status_code=303)
