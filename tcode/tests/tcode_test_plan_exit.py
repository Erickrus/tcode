import asyncio
from tcode.plan import PlanManager
from tcode.session import SessionManager
from tcode.event import EventBus
from tcode.storage import Storage
from tcode.providers.litellm_adapter import LitellmAdapter
from tcode.tools import ToolContext
from tcode.builtin_tools import structured_execute


def test_plan_exit_marks_plan_ready():
    async def run():
        storage = Storage()
        events = EventBus()
        sessions = SessionManager(storage=storage, events=events)

        session_id = await sessions.create_session({})

        adapter = LitellmAdapter(api_url='http://localhost:4000/v1/completions', api_key='sk-123456', model='gpt-5-mini')
        class ProvidersWrapper:
            def get_adapter(self, name, opts=None):
                return adapter

        providers = ProvidersWrapper()

        pm = PlanManager(sessions, providers, events)
        instructions = 'Make a plan to add a new feature that prints Hello World to console and add tests.'
        res = await pm.create_plan(session_id, title='Hello Plan', instructions=instructions, provider='litellm', model='gpt-5-mini')

        # call builtin plan_exit via direct tool invocation
        from tcode.builtin_tools import plan_exit_execute
        ctx = ToolContext(session_id=session_id, message_id='m', call_id='c', extra={'sessions': sessions})
        result = await plan_exit_execute({}, ctx)
        assert 'Plan exit' in result.output

        sess = await sessions.get_session(session_id)
        meta = sess.get('metadata', {})
        assert meta.get('plan_ready') == True

    asyncio.run(run())
