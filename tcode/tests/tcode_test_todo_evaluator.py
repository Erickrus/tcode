import asyncio
from tcode.todo_evaluator import evaluate_todo
from tcode.storage import Storage
from tcode.session import SessionManager
from tcode.event import EventBus


def test_evaluate_todo_schema_and_attachment():
    async def run():
        storage = Storage()
        events = EventBus()
        sessions = SessionManager(storage=storage, events=events)
        session_id = await sessions.create_session({})
        message_id = await sessions.create_message(session_id, 'assistant')

        # create a part with schema metadata and JSON output
        part = {
            'id': 'p-test-json',
            'session_id': session_id,
            'message_id': message_id,
            'type': 'tool',
            'state': {'status': 'completed', 'output': '{"name": "test", "value": 42}'},
            'metadata': {'schema': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'value': {'type': 'number'}}}}
        }
        await storage.append_part(part)

        res = await evaluate_todo(session_id, message_id, part, sessions, None)
        assert res['status'] == 'verified'

        # create a part with attachment reference
        import base64
        data = b'hello'
        from tcode.attachments import AttachmentStore
        store = AttachmentStore()
        url = store.store(data, 'hello.txt', 'text/plain')

        part2 = {
            'id': 'p-test-attach',
            'session_id': session_id,
            'message_id': message_id,
            'type': 'tool',
            'state': {'status': 'completed', 'output': ''},
            'metadata': {'attachment': url}
        }
        await storage.append_part(part2)
        res2 = await evaluate_todo(session_id, message_id, part2, sessions, None)
        assert res2['status'] == 'verified'

    asyncio.run(run())
