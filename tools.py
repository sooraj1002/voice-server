"""
Tool definitions and implementations for the expense tracker voice assistant.
Tools call the backend REST API on behalf of the authenticated user.
"""
import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8082")

# ---------------------------------------------------------------------------
# Ollama-compatible tool schemas
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "add_expense",
            "description": (
                "Add a new expense to the tracker. "
                "Use this when the user wants to record a purchase or payment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Amount spent (positive number)",
                    },
                    "description": {
                        "type": "string",
                        "description": "What the expense was for (e.g. 'lunch', 'Uber ride')",
                    },
                    "category_name": {
                        "type": "string",
                        "description": "Category name. Will be fuzzy-matched to available categories.",
                    },
                    "account_name": {
                        "type": "string",
                        "description": "Account name. Will be fuzzy-matched to available accounts.",
                    },
                    "date": {
                        "type": "string",
                        "description": "ISO date string (YYYY-MM-DD). Defaults to today if omitted.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags (e.g. ['food', 'work'])",
                    },
                },
                "required": ["amount", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_categories",
            "description": "Get all available spending categories for this user.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_accounts",
            "description": "Get all user accounts with their current balances.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_summary",
            "description": "Get total spending for a time period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["today", "week", "month", "year"],
                        "description": "Time period to summarize",
                    }
                },
                "required": ["period"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


async def execute_tool(
    name: str,
    args: dict,
    jwt: str,
    session_cache: dict,
) -> tuple[str, dict | None]:
    """
    Execute a tool and return (human_readable_result, action_event_or_None).
    action_event is sent to the frontend for cache invalidation.
    """
    headers = {"Authorization": f"Bearer {jwt}"}
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=15.0) as client:
        match name:
            case "list_categories":
                return await _list_categories(client, headers, session_cache)
            case "list_accounts":
                return await _list_accounts(client, headers, session_cache)
            case "get_spending_summary":
                return await _get_spending_summary(client, headers, args)
            case "add_expense":
                return await _add_expense(client, headers, args, session_cache)
            case _:
                return f"Unknown tool: {name}", None


async def _list_categories(client, headers, cache) -> tuple[str, None]:
    resp = await client.get("/api/categories", headers=headers)
    resp.raise_for_status()
    data = resp.json()
    cats = data.get("data", [])
    cache["categories"] = cats
    names = ", ".join(c["name"] for c in cats) if cats else "none"
    return f"Available categories: {names}", None


async def _list_accounts(client, headers, cache) -> tuple[str, None]:
    resp = await client.get("/api/accounts", headers=headers)
    resp.raise_for_status()
    data = resp.json()
    accounts = data.get("data", [])
    cache["accounts"] = accounts
    parts = []
    for a in accounts:
        bal = a.get("currentBalance") or a.get("initialBalance") or 0
        parts.append(f"{a['name']} (balance: {bal:,.0f})")
    return "Accounts: " + (", ".join(parts) if parts else "none"), None


async def _get_spending_summary(client, headers, args) -> tuple[str, None]:
    period = args.get("period", "month")
    resp = await client.get(f"/api/expenses?period={period}&limit=1", headers=headers)
    resp.raise_for_status()
    data = resp.json()
    total = data.get("pagination", {}).get("totalAmount") or 0
    return f"You spent {total:,.2f} this {period}.", None


async def _add_expense(client, headers, args, cache) -> tuple[str, dict]:
    amount = float(args["amount"])
    description = args.get("description", "")
    category_name = args.get("category_name", "")
    account_name = args.get("account_name", "")
    date_str = args.get("date", "")
    tags = args.get("tags", [])

    # Ensure we have fresh category/account data in cache
    if not cache.get("categories"):
        resp = await client.get("/api/categories", headers=headers)
        resp.raise_for_status()
        cache["categories"] = resp.json().get("data", [])

    if not cache.get("accounts"):
        resp = await client.get("/api/accounts", headers=headers)
        resp.raise_for_status()
        cache["accounts"] = resp.json().get("data", [])

    # Fuzzy-match category
    category = _fuzzy_match(cache["categories"], category_name)
    if not category and cache["categories"]:
        category = cache["categories"][0]

    # Fuzzy-match account
    account = _fuzzy_match(cache["accounts"], account_name)
    if not account and cache["accounts"]:
        account = cache["accounts"][0]

    if not category:
        return "No categories available. Please create a category first.", None
    if not account:
        return "No accounts available. Please create an account first.", None

    # Resolve date
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT12:00:00.000Z")
    elif len(date_str) == 10:  # YYYY-MM-DD
        date_str = f"{date_str}T12:00:00.000Z"

    payload = {
        "amount": amount,
        "categoryId": category["id"],
        "accountId": account["id"],
        "date": date_str,
        "description": description,
        "tags": tags if tags else ["misc"],
    }

    resp = await client.post("/api/expenses", headers=headers, json=payload)
    resp.raise_for_status()

    confirmation = (
        f"Added {amount:,.0f} for {description} "
        f"under {category['name']} from {account['name']}."
    )
    action = {"tool": "add_expense", "success": True}
    return confirmation, action


def _fuzzy_match(items: list[dict], name: str) -> dict | None:
    if not name or not items:
        return None
    name_lower = name.lower().strip()
    # Exact match first
    for item in items:
        if item["name"].lower() == name_lower:
            return item
    # Substring match
    for item in items:
        if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
            return item
    return None


async def prefetch_session_data(jwt: str) -> dict:
    """Fetch categories and accounts at WebSocket connect time."""
    cache: dict = {}
    headers = {"Authorization": f"Bearer {jwt}"}
    try:
        async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=10.0) as client:
            cats_resp = await client.get("/api/categories", headers=headers)
            if cats_resp.is_success:
                cache["categories"] = cats_resp.json().get("data", [])

            accs_resp = await client.get("/api/accounts", headers=headers)
            if accs_resp.is_success:
                cache["accounts"] = accs_resp.json().get("data", [])
    except Exception as e:
        logger.warning("Failed to prefetch session data: %s", e)
    return cache
