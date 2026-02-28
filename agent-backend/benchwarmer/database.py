import json
import logging
import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client

logger = logging.getLogger(__name__)

_supabase: Optional[Client] = None

def _get_client() -> Client:
    global _supabase
    if _supabase is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        _supabase = create_client(url, key)
    return _supabase

def init_db():
    # Tables are created in Supabase dashboard / SQL editor.
    # This is a no-op kept for backward compatibility with server.py startup.
    _get_client()  # validate credentials early

def save_message(session_id: str, role: str, content: str, tools: Optional[List[Dict]] = None, plot_image: Optional[str] = None):
    try:
        sb = _get_client()
        row = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "tools": json.dumps(tools) if tools else None,
            "plot_image": plot_image,
        }
        sb.table("messages").insert(row).execute()
    except Exception as e:
        logger.error(f"Failed to save message for session {session_id}: {e}")
        raise

def get_messages(session_id: str) -> List[Dict[str, Any]]:
    try:
        sb = _get_client()
        result = sb.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("id") \
            .execute()

        messages = []
        for row in result.data:
            try:
                tools = json.loads(row["tools"]) if row.get("tools") else []
            except (json.JSONDecodeError, TypeError):
                tools = []
            msg = {
                "id": str(row["id"]),
                "role": row["role"],
                "content": row["content"],
                "tools": tools,
                "plotImage": row.get("plot_image"),
            }
            messages.append(msg)
        return messages
    except Exception as e:
        logger.error(f"Failed to get messages for session {session_id}: {e}")
        return []

def get_all_sessions() -> List[Dict[str, Any]]:
    try:
        sb = _get_client()

        msgs = sb.table("messages") \
            .select("session_id, content, timestamp") \
            .order("id") \
            .execute()

        titles_result = sb.table("session_titles") \
            .select("session_id, title") \
            .execute()
        titles_map = {r["session_id"]: r["title"] for r in titles_result.data}

        session_info: Dict[str, Dict[str, Any]] = {}
        for row in msgs.data:
            sid = row["session_id"]
            if sid not in session_info:
                session_info[sid] = {
                    "first_message": row["content"],
                    "created_at": row["timestamp"],
                }

        sessions = []
        for sid, info in session_info.items():
            custom_title = titles_map.get(sid)
            if custom_title:
                title = custom_title
            elif info["first_message"]:
                fm = info["first_message"]
                title = (fm[:30] + "...") if len(fm) > 30 else fm
            else:
                title = "New Chat"
            sessions.append({
                "id": sid,
                "created_at": info["created_at"],
                "title": title,
            })

        sessions.sort(key=lambda s: s["created_at"] or "", reverse=True)
        return sessions
    except Exception as e:
        logger.error(f"Failed to get all sessions: {e}")
        return []

def delete_session(session_id: str):
    try:
        sb = _get_client()
        sb.table("messages").delete().eq("session_id", session_id).execute()
        sb.table("session_titles").delete().eq("session_id", session_id).execute()
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise

def rename_session(session_id: str, title: str):
    try:
        sb = _get_client()
        sb.table("session_titles").upsert({
            "session_id": session_id,
            "title": title,
        }).execute()
    except Exception as e:
        logger.error(f"Failed to rename session {session_id}: {e}")
        raise

def get_session_title(session_id: str) -> Optional[str]:
    try:
        sb = _get_client()
        result = sb.table("session_titles") \
            .select("title") \
            .eq("session_id", session_id) \
            .execute()
        return result.data[0]["title"] if result.data else None
    except Exception as e:
        logger.error(f"Failed to get session title for {session_id}: {e}")
        return None

def save_algorithm(name: str, code: str):
    try:
        sb = _get_client()
        sb.table("algorithms").upsert({
            "name": name,
            "code": code,
        }, on_conflict="name").execute()
    except Exception as e:
        logger.error(f"Failed to save algorithm {name}: {e}")
        raise

def get_all_algorithms() -> List[Dict[str, Any]]:
    try:
        sb = _get_client()
        result = sb.table("algorithms") \
            .select("name, created_at") \
            .order("created_at", desc=True) \
            .execute()
        return [{"name": r["name"], "created_at": r["created_at"]} for r in result.data]
    except Exception as e:
        logger.error(f"Failed to get all algorithms: {e}")
        return []

def get_algorithm_code(name: str) -> Optional[str]:
    try:
        sb = _get_client()
        result = sb.table("algorithms") \
            .select("code") \
            .eq("name", name) \
            .execute()
        return result.data[0]["code"] if result.data else None
    except Exception as e:
        logger.error(f"Failed to get algorithm code for {name}: {e}")
        return None
