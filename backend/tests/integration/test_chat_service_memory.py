from database.db import SessionLocal
from database.tables import ChatRecord
from database.tables import ChatSession
from memory.store import chat_memory
from services.chat_service import ChatService
from utils.common import generate_uuid


def test_load_agent_history_restores_recent_records_from_mysql() -> None:
    db_session = SessionLocal()
    session_id = generate_uuid()
    user_id = 77
    chat_memory.clear(session_id)
    try:
        db_session.add(
            ChatSession(
                session_id=session_id,
                user_id=user_id,
                session_name="restore history test",
                model_name="local-rule-model",
            )
        )
        db_session.flush()
        db_session.add(
            ChatRecord(
                session_id=session_id,
                role="user",
                content="first question",
            )
        )
        db_session.add(
            ChatRecord(
                session_id=session_id,
                role="assistant",
                content="first answer",
            )
        )
        db_session.commit()

        service = ChatService.__new__(ChatService)
        service._db_session = db_session

        history = service._load_agent_history(session_id, user_id)

        assert history == [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        assert chat_memory.get_recent(session_id) == history
    finally:
        chat_memory.clear(session_id)
        db_session.query(ChatRecord).filter_by(session_id=session_id).delete()
        db_session.query(ChatSession).filter_by(session_id=session_id).delete()
        db_session.commit()
        db_session.close()
