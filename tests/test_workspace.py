from app.tools.workspace import create_ticket, find_free_slots, list_calendar_events, lookup_employee


def test_free_slots_skip_busy_blocks(demo_backend):
    slots = find_free_slots("2026-10-26", 30)["slots"]
    assert slots[0] == {"start": "2026-10-26T09:00:00", "end": "2026-10-26T09:30:00"}
    assert {"start": "2026-10-26T10:00:00", "end": "2026-10-26T14:00:00"} in slots
    assert slots[-1]["end"] == "2026-10-26T18:00:00"


def test_ticket_priority_is_normalised(demo_backend):
    t = create_ticket("Laptop stolen", "Lost at airport", "p1")["ticket"]
    assert t["priority"] == "P1" and t["id"].startswith("TCK-")
    assert create_ticket("x", "y", "urgent!!")["ticket"]["priority"] == "P3"


def test_directory_and_calendar(demo_backend):
    assert lookup_employee("sre")["matches"][0]["name"] == "Asha Rao"
    assert len(list_calendar_events("2026-10-26")["events"]) == 2
