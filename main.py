import os
from datetime import datetime, timezone
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents

app = FastAPI(title="EvenTheField API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Pydantic DTOs
# -----------------------
class MatchIn(BaseModel):
    competition: str
    season: str
    home_team: str
    away_team: str
    kickoff: datetime
    match_id: Optional[str] = None

class MatchOut(MatchIn):
    id: Optional[str] = None
    status: Literal["scheduled", "in_play", "finished"] = "scheduled"

class HistoricalMatchIn(BaseModel):
    competition: str
    season: str
    date: datetime
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_lineup_rating: Optional[float] = None
    away_lineup_rating: Optional[float] = None
    match_id: Optional[str] = None

class TrainRequest(BaseModel):
    name: str = Field("soccer_v1", description="Model name")
    lookback_seasons: Optional[int] = 3
    include_lineups: bool = True

class PredictRequest(BaseModel):
    match_id: str

class TipsterIn(BaseModel):
    display_name: str
    bio: Optional[str] = None
    twitter: Optional[str] = None

class TipsterOut(TipsterIn):
    id: Optional[str] = None

class FollowerIn(BaseModel):
    email: str
    name: Optional[str] = None

class FollowerOut(FollowerIn):
    id: Optional[str] = None

class SubscriptionIn(BaseModel):
    tipster_id: str
    follower_id: str
    plan_name: str = "monthly"
    price_cents: int = 0

class SubscriptionOut(SubscriptionIn):
    id: Optional[str] = None
    status: Literal["active", "canceled"] = "active"
    started_at: datetime

class BetIn(BaseModel):
    tipster_id: str
    match_id: str
    market: Literal["outright", "goals"]
    selection: str
    odds: float
    stake: float

class BetOut(BetIn):
    id: Optional[str] = None
    placed_at: datetime
    status: Literal["open", "settled", "void"] = "open"
    result: Optional[Literal["win", "lose", "void"]] = None
    profit: Optional[float] = None

class SettleBetRequest(BaseModel):
    bet_id: str
    result: Literal["win", "lose", "void"]

class PaymentIn(BaseModel):
    subscription_id: str
    amount_cents: int

class PaymentOut(PaymentIn):
    id: Optional[str] = None
    processed_at: datetime
    tipster_share_cents: int
    platform_share_cents: int


# -----------------------
# Health & utility
# -----------------------
@app.get("/")
def read_root():
    return {"message": "EvenTheField API running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:120]}"
    return response

# -----------------------
# Matches & historical data
# -----------------------
@app.post("/api/matches", response_model=Dict[str, Any])
def create_match(match: MatchIn):
    payload = match.model_dump()
    payload["status"] = "scheduled"
    inserted_id = create_document("match", payload)
    return {"id": inserted_id}

@app.get("/api/matches", response_model=List[Dict[str, Any]])
def list_matches():
    docs = get_documents("match", {"status": "scheduled"})
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs

@app.post("/api/historical", response_model=Dict[str, Any])
def add_historical(item: HistoricalMatchIn):
    inserted_id = create_document("historicalmatch", item)
    return {"id": inserted_id}

# -----------------------
# Simple training & prediction (baseline model)
# -----------------------

def _team_stats(competition: Optional[str] = None, seasons_back: Optional[int] = 3):
    """Compute simple attack/defense ratings from historical matches."""
    filt: Dict[str, Any] = {}
    if competition:
        filt["competition"] = competition
    docs = get_documents("historicalmatch", filt)
    team = {}
    for d in docs:
        ht = d.get("home_team")
        at = d.get("away_team")
        hg = d.get("home_goals", 0)
        ag = d.get("away_goals", 0)
        team.setdefault(ht, {"gf": 0, "ga": 0, "n": 0})
        team.setdefault(at, {"gf": 0, "ga": 0, "n": 0})
        team[ht]["gf"] += hg
        team[ht]["ga"] += ag
        team[ht]["n"] += 1
        team[at]["gf"] += ag
        team[at]["ga"] += hg
        team[at]["n"] += 1
    ratings = {}
    for t, s in team.items():
        n = max(s["n"], 1)
        att = (s["gf"] / n)
        deff = (s["ga"] / n)
        ratings[t] = {
            "attack": att,
            "defense": deff,
            "net": att - deff
        }
    return ratings

@app.post("/api/model/train", response_model=Dict[str, Any])
def train_model(req: TrainRequest):
    ratings = _team_stats()
    cfg = {
        "name": req.name,
        "created_at": datetime.now(timezone.utc),
        "parameters": {
            "lookback_seasons": req.lookback_seasons,
            "include_lineups": req.include_lineups,
            "home_advantage": 0.25,
        },
        "ratings": ratings,
        "notes": "Baseline model using per-team average goals for/against as ratings",
    }
    model_id = create_document("modelconfig", cfg)
    return {"model_id": model_id, "teams": len(ratings)}

@app.post("/api/model/predict", response_model=Dict[str, Any])
def predict(req: PredictRequest):
    # find the match
    m = db["match"].find_one({"match_id": req.match_id}) or db["match"].find_one({"_id": req.match_id})
    if not m:
        raise HTTPException(status_code=404, detail="Match not found")

    home = m["home_team"]
    away = m["away_team"]

    # get most recent model
    model = db["modelconfig"].find_one(sort=[("created_at", -1)])
    if not model:
        # fallback compute quickly
        ratings = _team_stats()
        home_adv = 0.25
    else:
        ratings = model.get("ratings", {})
        home_adv = model.get("parameters", {}).get("home_advantage", 0.25)

    rh = ratings.get(home, {"attack": 1.2, "defense": 1.2, "net": 0})
    ra = ratings.get(away, {"attack": 1.2, "defense": 1.2, "net": 0})

    # Expected goals as simple function of attacks vs opponent defense
    home_xg = max(0.2, (rh["attack"] + (2.0 - ra["defense"]) + home_adv))
    away_xg = max(0.2, (ra["attack"] + (2.0 - rh["defense"])) )

    total_goals_line = round(home_xg + away_xg, 1)

    # Convert net strengths to probabilities (softmax-like)
    h_score = rh["net"] + home_adv
    d_score = 0.0  # neutral baseline for draw
    a_score = ra["net"]
    import math
    exps = [math.exp(h_score), math.exp(d_score), math.exp(a_score)]
    s = sum(exps)
    probs = [e / s for e in exps]
    labels = ["home", "draw", "away"]
    outright_probs = {k: round(v, 3) for k, v in zip(labels, probs)}
    outright_odds = {k: round(1.0 / max(v, 1e-6), 2) for k, v in outright_probs.items()}

    return {
        "match": {
            "home_team": home,
            "away_team": away,
            "kickoff": m.get("kickoff"),
        },
        "expected_goals": {"home": round(home_xg, 2), "away": round(away_xg, 2)},
        "goal_line": total_goals_line,
        "outright": {"probs": outright_probs, "odds_decimal": outright_odds},
    }

# -----------------------
# Tipsters, followers, subscriptions, bets, payments
# -----------------------
@app.post("/api/tipsters", response_model=Dict[str, Any])
def create_tipster(t: TipsterIn):
    _id = create_document("tipster", t)
    return {"id": _id}

@app.get("/api/tipsters", response_model=List[Dict[str, Any]])
def list_tipsters():
    docs = get_documents("tipster")
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs

@app.post("/api/followers", response_model=Dict[str, Any])
def create_follower(f: FollowerIn):
    _id = create_document("follower", f)
    return {"id": _id}

@app.get("/api/followers", response_model=List[Dict[str, Any]])
def list_followers():
    docs = get_documents("follower")
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs

@app.post("/api/subscribe", response_model=Dict[str, Any])
def subscribe(s: SubscriptionIn):
    sub_doc = {
        **s.model_dump(),
        "status": "active",
        "started_at": datetime.now(timezone.utc),
    }
    _id = create_document("subscription", sub_doc)
    return {"id": _id}

@app.get("/api/tipsters/{tipster_id}/followers", response_model=List[Dict[str, Any]])
def tipster_followers(tipster_id: str):
    docs = get_documents("subscription", {"tipster_id": tipster_id, "status": "active"})
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs

@app.post("/api/bets", response_model=Dict[str, Any])
def place_bet(b: BetIn):
    # Validate match & cutoff
    m = db["match"].find_one({"match_id": b.match_id}) or db["match"].find_one({"_id": b.match_id})
    if not m:
        raise HTTPException(status_code=404, detail="Related match not found")
    kickoff = m.get("kickoff")
    if isinstance(kickoff, str):
        kickoff_dt = datetime.fromisoformat(kickoff.replace('Z','+00:00'))
    else:
        kickoff_dt = kickoff
    now = datetime.now(timezone.utc)
    if kickoff_dt and now >= kickoff_dt.replace(tzinfo=timezone.utc):
        raise HTTPException(status_code=400, detail="Bet must be posted before game start time")

    bet_doc = {
        **b.model_dump(),
        "placed_at": now,
        "status": "open",
    }
    _id = create_document("bet", bet_doc)

    # Create notifications for followers
    active_subs = get_documents("subscription", {"tipster_id": b.tipster_id, "status": "active"})
    for s in active_subs:
        note = {
            "follower_id": s.get("follower_id"),
            "tipster_id": b.tipster_id,
            "message": f"New bet posted: {b.market} - {b.selection} @ {b.odds}",
            "created_at": now,
            "read": False,
        }
        create_document("notification", note)

    return {"id": _id}

@app.get("/api/bets", response_model=List[Dict[str, Any]])
def list_bets(tipster_id: Optional[str] = Query(None)):
    filt = {"status": {"$in": ["open", "settled"]}}
    if tipster_id:
        filt["tipster_id"] = tipster_id
    docs = get_documents("bet", filt)
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs

@app.post("/api/bets/settle", response_model=Dict[str, Any])
def settle_bet(payload: SettleBetRequest):
    bet = db["bet"].find_one({"_id": payload.bet_id}) or db["bet"].find_one({"_id": payload.bet_id})
    # Allow querying by string id
    if not bet:
        try:
            from bson import ObjectId
            bet = db["bet"].find_one({"_id": ObjectId(payload.bet_id)})
        except Exception:
            bet = None
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")

    profit = 0.0
    if payload.result == "win":
        profit = bet.get("stake", 0) * (bet.get("odds", 0) - 1)
    elif payload.result == "lose":
        profit = -bet.get("stake", 0)
    else:
        profit = 0.0

    db["bet"].update_one({"_id": bet["_id"]}, {"$set": {"status": "settled", "result": payload.result, "profit": profit}})
    return {"status": "ok", "profit": profit}

@app.get("/api/tipsters/{tipster_id}/roi", response_model=Dict[str, Any])
def tipster_roi(tipster_id: str):
    bets = get_documents("bet", {"tipster_id": tipster_id, "status": "settled"})
    total_stake = sum(b.get("stake", 0.0) for b in bets)
    total_profit = sum(b.get("profit", 0.0) for b in bets)
    roi = (total_profit / total_stake) if total_stake > 0 else 0.0
    return {"bets": len(bets), "total_stake": total_stake, "total_profit": total_profit, "roi": round(roi, 4)}

@app.get("/api/followers/{follower_id}/notifications", response_model=List[Dict[str, Any]])
def get_notifications(follower_id: str):
    notes = get_documents("notification", {"follower_id": follower_id})
    for n in notes:
        n["id"] = str(n.pop("_id"))
    return notes

@app.post("/api/payments", response_model=Dict[str, Any])
def create_payment(p: PaymentIn):
    tipster_share = int(round(p.amount_cents * 0.90))
    platform_share = p.amount_cents - tipster_share
    doc = {
        **p.model_dump(),
        "processed_at": datetime.now(timezone.utc),
        "tipster_share_cents": tipster_share,
        "platform_share_cents": platform_share,
    }
    _id = create_document("payment", doc)
    return {"id": _id, "tipster_share_cents": tipster_share, "platform_share_cents": platform_share}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
