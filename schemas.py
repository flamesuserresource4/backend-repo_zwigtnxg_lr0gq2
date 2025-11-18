"""
Database Schemas for EvenTheField

Each Pydantic model represents a collection in MongoDB.
Collection name is the lowercase of the class name.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

# Core domain
class Match(BaseModel):
    competition: str = Field(..., description="Competition name, e.g., Premier League")
    season: str = Field(..., description="Season identifier, e.g., 2024/25")
    home_team: str
    away_team: str
    kickoff: datetime
    match_id: Optional[str] = Field(None, description="External/provider match id")
    status: Literal["scheduled", "in_play", "finished"] = "scheduled"

class HistoricalMatch(BaseModel):
    competition: str
    season: str
    date: datetime
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    match_id: Optional[str] = Field(None, description="If known, link to the scheduled match id")
    # Optional richer context
    home_lineup_rating: Optional[float] = Field(None, description="0-100 rating summarizing home lineup strength")
    away_lineup_rating: Optional[float] = Field(None, description="0-100 rating summarizing away lineup strength")

class ModelConfig(BaseModel):
    name: str = Field(..., description="Model name, e.g., soccer_v1")
    created_at: Optional[datetime] = None
    parameters: dict = Field(default_factory=dict)
    notes: Optional[str] = None

# Bet-along domain
class Tipster(BaseModel):
    display_name: str
    bio: Optional[str] = None
    twitter: Optional[str] = None

class Follower(BaseModel):
    email: str
    name: Optional[str] = None

class Subscription(BaseModel):
    tipster_id: str
    follower_id: str
    started_at: datetime
    status: Literal["active", "canceled"] = "active"
    plan_name: str = "monthly"
    price_cents: int = 0

class Bet(BaseModel):
    tipster_id: str
    match_id: str
    market: Literal["outright", "goals"]
    selection: str  # e.g., "home", "draw", "away" or "over_2.5"
    odds: float
    stake: float
    placed_at: datetime

class Payment(BaseModel):
    subscription_id: str
    amount_cents: int
    processed_at: datetime
    tipster_share_cents: int
    platform_share_cents: int

class Notification(BaseModel):
    follower_id: str
    tipster_id: str
    message: str
    created_at: datetime
    read: bool = False

# The previous example schemas are not used in this app and were removed to avoid confusion.
