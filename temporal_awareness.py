from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
import pytz
from dateutil.parser import parse as parse_datetime
import re

logger = logging.getLogger(__name__)

class TimeUnit(Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"

@dataclass
class ScheduledEvent:
    id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    description: str = ""
    priority: int = 2  # 1-5, 5 being highest
    category: str = "general"
    recurrence: Optional[str] = None  # e.g., "daily", "weekly", "monthly"
    timezone: str = "UTC"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "description": self.description,
            "priority": self.priority,
            "category": self.category,
            "recurrence": self.recurrence,
            "timezone": self.timezone,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScheduledEvent':
        return cls(
            id=data["id"],
            name=data["name"],
            start_time=parse_datetime(data["start_time"]),
            end_time=parse_datetime(data["end_time"]) if data["end_time"] else None,
            description=data.get("description", ""),
            priority=data.get("priority", 2),
            category=data.get("category", "general"),
            recurrence=data.get("recurrence"),
            timezone=data.get("timezone", "UTC"),
            metadata=data.get("metadata", {})
        )

class TemporalAwareness:
    def __init__(self, timezone_str: str = "UTC", storage_path: str = "temporal_data.json"):
        self.timezone = pytz.timezone(timezone_str)
        self.storage_path = storage_path
        self.events: Dict[str, ScheduledEvent] = {}
        self.load_events()
    
    def add_event(self, event: ScheduledEvent) -> bool:
        """Add or update an event"""
        # Ensure times are timezone-aware
        if event.start_time.tzinfo is None:
            event.start_time = self.timezone.localize(event.start_time)
        if event.end_time and event.end_time.tzinfo is None:
            event.end_time = self.timezone.localize(event.end_time)
        
        self.events[event.id] = event
        self.save_events()
        return True
    
    def remove_event(self, event_id: str) -> bool:
        """Remove an event by ID"""
        if event_id in self.events:
            del self.events[event_id]
            self.save_events()
            return True
        return False
    
    def get_upcoming_events(self, hours: int = 24) -> List[ScheduledEvent]:
        """Get events occurring in the next N hours"""
        now = datetime.now(self.timezone)
        future = now + timedelta(hours=hours)
        
        upcoming = []
        for event in self.events.values():
            if event.start_time <= future and (event.end_time is None or event.end_time >= now):
                upcoming.append(event)
        
        return sorted(upcoming, key=lambda e: e.start_time)
    
    def get_events_in_range(self, start: datetime, end: datetime) -> List[ScheduledEvent]:
        """Get events within a specific time range"""
        # Ensure timezone awareness
        if start.tzinfo is None:
            start = self.timezone.localize(start)
        if end.tzinfo is None:
            end = self.timezone.localize(end)
        
        in_range = []
        for event in self.events.values():
            event_end = event.end_time or event.start_time + timedelta(hours=1)
            if (event.start_time <= end and event_end >= start):
                in_range.append(event)
        
        return sorted(in_range, key=lambda e: e.start_time)
    
    def parse_relative_time(self, text: str, reference_time: Optional[datetime] = None) -> Optional[datetime]:
        """Parse relative time expressions like 'in 2 hours' or 'tomorrow at 3pm'"""
        if reference_time is None:
            reference_time = datetime.now(self.timezone)
        
        text = text.lower().strip()
        now = reference_time
        
        # Handle simple cases
        if text == "now":
            return now
        elif text == "tomorrow":
            return now.replace(hour=9, minute=0, second=0) + timedelta(days=1)
        
        # Handle "in X minutes/hours/days"
        time_match = re.match(r'in (\d+) (second|minute|hour|day|week|month|year)s?', text)
        if time_match:
            amount = int(time_match.group(1))
            unit = time_match.group(2)
            
            if unit.startswith('second'):
                return now + timedelta(seconds=amount)
            elif unit.startswith('minute'):
                return now + timedelta(minutes=amount)
            elif unit.startswith('hour'):
                return now + timedelta(hours=amount)
            elif unit.startswith('day'):
                return now + timedelta(days=amount)
            elif unit.startswith('week'):
                return now + timedelta(weeks=amount)
            elif unit.startswith('month'):
                # Approximate month as 30 days
                return now + timedelta(days=30 * amount)
            elif unit.startswith('year'):
                # Approximate year as 365 days
                return now + timedelta(days=365 * amount)
        
        # Handle "at [time]" or "tomorrow at [time]"
        time_match = re.match(r'(?:tomorrow )?at (\d{1,2})(?::(\d{2}))?\s?(am|pm)?', text, re.IGNORECASE)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            period = time_match.group(3)
            
            # Convert to 24-hour format
            if period:
                if period.lower() == 'pm' and hour < 12:
                    hour += 12
                elif period.lower() == 'am' and hour == 12:
                    hour = 0
            
            # Create the time
            result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if 'tomorrow' in text or (result < now and 'tomorrow' not in text):
                result += timedelta(days=1)
            
            return result
        
        return None
    
    def schedule_reminder(self, text: str, reminder_time: Optional[datetime] = None) -> Optional[ScheduledEvent]:
        """Schedule a reminder based on natural language"""
        if reminder_time is None:
            # Try to extract time from text
            reminder_time = self.parse_relative_time(text)
            if reminder_time is None:
                return None
        
        # Create a reminder event
        event = ScheduledEvent(
            id=f"reminder_{int(reminder_time.timestamp())}",
            name=f"Reminder: {text}",
            start_time=reminder_time,
            end_time=reminder_time + timedelta(minutes=5),
            category="reminder",
            priority=4
        )
        
        self.add_event(event)
        return event
    
    def save_events(self):
        """Save events to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({"events": [e.to_dict() for e in self.events.values()]}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving temporal events: {e}")
    
    def load_events(self):
        """Load events from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.events = {e["id"]: ScheduledEvent.from_dict(e) for e in data.get("events", [])}
        except FileNotFoundError:
            logger.info("No existing temporal data found, starting with empty event store")
            self.events = {}
        except Exception as e:
            logger.error(f"Error loading temporal events: {e}")
            self.events = {}
    
    def get_time_until_next_event(self) -> Optional[Tuple[timedelta, ScheduledEvent]]:
        """Get time until the next scheduled event"""
        now = datetime.now(self.timezone)
        upcoming = []
        
        for event in self.events.values():
            if event.start_time > now:
                time_until = event.start_time - now
                upcoming.append((time_until, event))
        
        if not upcoming:
            return None
            
        return min(upcoming, key=lambda x: x[0])
    
    def get_time_since_last_event(self, category: Optional[str] = None) -> Optional[timedelta]:
        """Get time since the last event of a given category"""
        now = datetime.now(self.timezone)
        past_events = []
        
        for event in self.events.values():
            if event.start_time < now and (category is None or event.category == category):
                past_events.append(event)
        
        if not past_events:
            return None
            
        most_recent = max(past_events, key=lambda e: e.start_time)
        return now - most_recent.start_time
