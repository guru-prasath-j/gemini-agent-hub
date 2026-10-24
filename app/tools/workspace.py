"""Enterprise "workspace" connectors: calendar, ticketing and directory.

Agents talk to a ``WorkspaceBackend`` protocol, never to an API directly.
``DemoBackend`` is an in-memory implementation seeded with sample data so
the project runs anywhere; ``GoogleCalendarBackend`` shows the same calendar
contract implemented against the Google Calendar API using Application
Default Credentials (the Cloud Run service account / domain-wide delegation).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol


@dataclass
class Event:
    id: str
    title: str
    start: str  # ISO-8601
    end: str
    attendees: list[str] = field(default_factory=list)


@dataclass
class Ticket:
    id: str
    title: str
    description: str
    priority: str
    status: str = "open"


class WorkspaceBackend(Protocol):
    def list_events(self, day: str) -> list[Event]: ...
    def create_event(self, title: str, start: str, end: str, attendees: list[str]) -> Event: ...
    def create_ticket(self, title: str, description: str, priority: str) -> Ticket: ...
    def lookup_employee(self, name: str) -> list[dict[str, str]]: ...


class DemoBackend:
    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self.events: list[Event] = []
        self.tickets: list[Ticket] = []
        self.directory = [
            {"name": "Asha Rao", "team": "Platform SRE", "title": "On-call lead", "location": "Bengaluru"},
            {"name": "Vikram Iyer", "team": "Payments", "title": "Engineering Manager", "location": "Hyderabad"},
            {"name": "Meera Nair", "team": "People Ops", "title": "HR Business Partner", "location": "Bengaluru"},
            {"name": "Rahul Menon", "team": "Security", "title": "Security Engineer", "location": "Pune"},
        ]

    def seed_day(self, day: str) -> None:
        self.events = [
            Event("e-standup", "Platform stand-up", f"{day}T09:30:00", f"{day}T10:00:00", ["Asha Rao"]),
            Event("e-review", "Architecture review", f"{day}T14:00:00", f"{day}T15:00:00", ["Vikram Iyer"]),
        ]

    def list_events(self, day: str) -> list[Event]:
        return sorted((e for e in self.events if e.start.startswith(day)), key=lambda e: e.start)

    def create_event(self, title: str, start: str, end: str, attendees: list[str]) -> Event:
        ev = Event(f"e-{next(self._ids)}", title, start, end, attendees)
        self.events.append(ev)
        return ev

    def create_ticket(self, title: str, description: str, priority: str) -> Ticket:
        t = Ticket(f"TCK-{1000 + next(self._ids)}", title, description, priority)
        self.tickets.append(t)
        return t

    def lookup_employee(self, name: str) -> list[dict[str, str]]:
        q = name.lower()
        return [p for p in self.directory if q in p["name"].lower() or q in p["team"].lower()]


class GoogleCalendarBackend(DemoBackend):
    """Calendar ops on Google Calendar; ticketing/directory stay on the demo store."""

    def __init__(self, calendar_id: str = "primary") -> None:
        super().__init__()
        import google.auth
        from googleapiclient.discovery import build

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/calendar"])
        self._svc = build("calendar", "v3", credentials=creds, cache_discovery=False)
        self._cal = calendar_id

    def list_events(self, day: str) -> list[Event]:
        items = (
            self._svc.events()
            .list(calendarId=self._cal, timeMin=f"{day}T00:00:00Z", timeMax=f"{day}T23:59:59Z",
                  singleEvents=True, orderBy="startTime")
            .execute()
            .get("items", [])
        )
        return [
            Event(i["id"], i.get("summary", ""), i["start"].get("dateTime", day), i["end"].get("dateTime", day),
                  [a.get("email", "") for a in i.get("attendees", [])])
            for i in items
        ]

    def create_event(self, title: str, start: str, end: str, attendees: list[str]) -> Event:
        body = {"summary": title, "start": {"dateTime": start}, "end": {"dateTime": end},
                "attendees": [{"email": a} for a in attendees if "@" in a]}
        i = self._svc.events().insert(calendarId=self._cal, body=body).execute()
        return Event(i["id"], title, start, end, attendees)


_BACKEND: WorkspaceBackend = DemoBackend()


def set_backend(backend: WorkspaceBackend) -> None:
    global _BACKEND
    _BACKEND = backend


def get_backend() -> WorkspaceBackend:
    return _BACKEND


# --------------------------------------------------------------------------
# Tool functions. Docstrings are the tool descriptions Gemini sees.
# --------------------------------------------------------------------------

def list_calendar_events(day: str) -> dict[str, Any]:
    """List the user's calendar events on a given day.

    Args:
        day: Date in YYYY-MM-DD format.
    """
    return {"status": "ok", "events": [asdict(e) for e in _BACKEND.list_events(day)]}


def find_free_slots(day: str, duration_minutes: int = 30, work_start: str = "09:00", work_end: str = "18:00") -> dict[str, Any]:
    """Find free time slots on a day, inside working hours.

    Args:
        day: Date in YYYY-MM-DD format.
        duration_minutes: Length of the slot needed.
        work_start: Start of the working day, HH:MM.
        work_end: End of the working day, HH:MM.
    """
    fmt = "%Y-%m-%dT%H:%M:%S"
    cursor = datetime.strptime(f"{day}T{work_start}:00", fmt)
    end_of_day = datetime.strptime(f"{day}T{work_end}:00", fmt)
    need = timedelta(minutes=int(duration_minutes))
    slots = []
    for ev in _BACKEND.list_events(day):
        s, e = datetime.fromisoformat(ev.start[:19]), datetime.fromisoformat(ev.end[:19])
        if s - cursor >= need:
            slots.append({"start": cursor.strftime(fmt), "end": s.strftime(fmt)})
        cursor = max(cursor, e)
    if end_of_day - cursor >= need:
        slots.append({"start": cursor.strftime(fmt), "end": end_of_day.strftime(fmt)})
    return {"status": "ok", "slots": slots}


def create_calendar_event(title: str, start: str, end: str, attendees: list[str]) -> dict[str, Any]:
    """Create a calendar event. Confirm details with the user before calling.

    Args:
        title: Event title.
        start: ISO-8601 start, e.g. 2026-10-26T15:00:00.
        end: ISO-8601 end.
        attendees: Names or emails of attendees.
    """
    return {"status": "ok", "event": asdict(_BACKEND.create_event(title, start, end, attendees))}


def create_ticket(title: str, description: str, priority: str = "P3") -> dict[str, Any]:
    """Open an IT / ops support ticket.

    Args:
        title: One-line summary.
        description: What happened, impact, and steps already tried.
        priority: One of P1 (outage), P2, P3, P4.
    """
    priority = priority.upper() if priority.upper() in {"P1", "P2", "P3", "P4"} else "P3"
    return {"status": "ok", "ticket": asdict(_BACKEND.create_ticket(title, description, priority))}


def lookup_employee(name: str) -> dict[str, Any]:
    """Look up people in the company directory by name or team.

    Args:
        name: Full or partial person name, or a team name.
    """
    return {"status": "ok", "matches": _BACKEND.lookup_employee(name)}
