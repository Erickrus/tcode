from __future__ import annotations
from typing import Dict, Any, Optional
from .session import SessionManager
from .providers.factory import ProviderFactory
from .event import Event, EventBus
import asyncio
from .util import next_id
import time

class PlanManager:
    def __init__(self, sessions: SessionManager, providers: ProviderFactory, events: EventBus, tools: Optional[object] = None, mcp: Optional[object] = None, attachments: Optional[AttachmentStore] = None):
        self.sessions = sessions
        self.providers = providers
        self.events = events
        self.tools = tools
        self.mcp = mcp
        self.attachments = attachments

    async def _run_subagent(self, session_id: str, instructions: str, subagent_type: str, provider: str, model: str) -> Dict[str, Any]:
        """Run a small scoped agent for exploration or design. Returns a dict with 'text' and optional 'facts'."""
        adapter = self.providers.get_adapter(provider, {'type': provider})
        prompt = f"You are an explorer agent (type={subagent_type}). Follow these instructions:\n{instructions}\n\nProvide a concise findings summary and a few facts as bullet points."
        try:
            resp = await adapter.chat([{"role": "user", "content": prompt}], model, {})
            text = ''
            if isinstance(resp, dict):
                choices = resp.get('choices')
                if choices and isinstance(choices, list):
                    c0 = choices[0]
                    text = c0.get('text') or (c0.get('message') and c0.get('message').get('content')) or ''
                else:
                    text = resp.get('text') or ''
            else:
                text = str(resp)
        except Exception as e:
            text = f"Subagent failed: {e}"
        # parse facts
        facts = []
        for ln in text.split('\n'):
            s = ln.strip()
            if s.startswith('-') or s.startswith('*'):
                facts.append(s.lstrip('-* ').strip())
        return {'text': text, 'facts': facts}

    async def create_plan(self, session_id: str, title: Optional[str], instructions: str, provider: str = 'litellm', model: str = 'gpt-5-mini', explore_count: int = 2) -> Dict[str, Any]:
        # compose prompt
        prompt = f"Create a succinct development plan based on the following instructions:\n\n{instructions}\n\nRespond with a numbered list of steps and brief descriptions."
        adapter = self.providers.get_adapter(provider, {'type': provider})
        try:
            resp = await adapter.chat([{"role": "user", "content": prompt}], model, {})
            text = ''
            if isinstance(resp, dict):
                choices = resp.get('choices')
                if choices and isinstance(choices, list):
                    c0 = choices[0]
                    text = c0.get('text') or (c0.get('message') and c0.get('message').get('content')) or ''
                else:
                    text = resp.get('text') or ''
            else:
                text = str(resp)
        except Exception as e:
            text = f'Plan generation failed: {e}'

        # persist plan in session metadata and as a plan message part
        meta = {'title': title, 'instructions': instructions, 'plan_text': text, 'timestamp': int(time.time()), 'plan_file': f'plan_{session_id}.md'}
        await self.sessions.set_summary(session_id, {'plan': meta})
        # also persist top-level 'plan' key for compatibility
        try:
            sess = await self.sessions.get_session(session_id)
            meta2 = sess.get('metadata', {}) or {}
            meta2['plan'] = meta
            meta2['plan_active'] = True
            meta2['plan_file'] = meta['plan_file']
            await self.sessions.storage.update_session(session_id, meta2)
            # set session-level permissions: deny write_file by default, allow only for the plan file
            try:
                import os
                plan_path = meta2.get('plan_file') or ''
                abs_plan = os.path.abspath(plan_path) if plan_path else plan_path
                # rules: default deny, specific allow for plan file (specific allow should come last so it wins)
                rules = [
                    {'permission': 'builtin_write_file', 'action': 'deny'},
                    {'permission': 'builtin_write_file', 'action': 'allow', 'pattern': abs_plan},
                ]
                await self.sessions.set_permission(session_id, rules)
            except Exception:
                pass
        except Exception:
            pass

        # insert a plan message/part
        message_id = await self.sessions.create_message(session_id, 'assistant')
        part = {
            'id': next_id('part'),
            'session_id': session_id,
            'message_id': message_id,
            'type': 'plan',
            'title': title,
            'text': text,
            'metadata': {'timestamp': int(time.time())}
        }
        await self.sessions.storage.append_part(part)
        seq = await self.sessions.storage.next_sequence(session_id)
        self.events.publish_fire_and_forget(Event.create('session.plan.created', {'sessionID': session_id, 'plan': meta}, session_id=session_id, sequence=seq))

        # Phase 1: run explore subagents in parallel (up to 3)
        # Phase 1: launch subagents using SubagentRunner
        try:
            from .subagent import SubagentRunner
            subrunner = SubagentRunner(self.providers, None, None, self.sessions, self.events, None)
            # create configurable number of parallel explores
            tasks = [subrunner.run_explore(provider, model, session_id, instructions + '\nFocus on code locations and tests.') for _ in range(min(max(1, explore_count), 3))]
            explore_results = await asyncio.gather(*tasks)
        except Exception:
            # fallback to lightweight provider-based subagents
            explore_tasks = []
            for i in range(min(max(1, explore_count), 3)):
                explore_tasks.append(self._run_subagent(session_id, instructions + '\nFocus on code locations and tests.', 'explore', provider, model))
            explore_results = await asyncio.gather(*explore_tasks)

        # attach explore results as subtask parts
        for er in explore_results:
            subpart = {
                'id': next_id('part'),
                'session_id': session_id,
                'message_id': message_id,
                'type': 'subtask',
                'text': er.get('text'),
                'facts': er.get('facts'),
                'metadata': {'phase': 'explore', 'timestamp': int(time.time())}
            }
            await self.sessions.storage.append_part(subpart)
            seq = await self.sessions.storage.next_sequence(session_id)
            self.events.publish_fire_and_forget(Event.create('message.part.updated', {'part': subpart}, session_id=session_id, sequence=seq))

        # Phase 2: Design - aggregate explores and draft candidate designs
        design_input = '\n\n'.join([er.get('text','') for er in explore_results])
        design_prompt = f"Based on the exploration findings below, draft 2-3 candidate approaches to accomplish the instructions. For each approach, list pros/cons and a recommended approach.\n\nFindings:\n{design_input}\n\nInstructions:\n{instructions}"
        try:
            adapter = self.providers.get_adapter(provider, {'type': provider})
            resp = await adapter.chat([{"role": "user", "content": design_prompt}], model, {})
            design_text = ''
            if isinstance(resp, dict):
                choices = resp.get('choices')
                if choices and isinstance(choices, list):
                    c0 = choices[0]
                    design_text = c0.get('text') or (c0.get('message') and c0.get('message').get('content')) or ''
                else:
                    design_text = resp.get('text') or ''
            else:
                design_text = str(resp)
        except Exception as e:
            design_text = f'Design generation failed: {e}'

        design_part = {
            'id': next_id('part'),
            'session_id': session_id,
            'message_id': message_id,
            'type': 'design',
            'text': design_text,
            'metadata': {'phase': 'design', 'timestamp': int(time.time())}
        }
        await self.sessions.storage.append_part(design_part)
        seq = await self.sessions.storage.next_sequence(session_id)
        self.events.publish_fire_and_forget(Event.create('message.part.updated', {'part': design_part}, session_id=session_id, sequence=seq))

        # Phase 3: Review - ask model to review the draft plan and list concerns/questions
        review_prompt = f"Review the following draft design and list any missing concerns, ambiguities, or checklist items needed to implement the plan.\n\nDraft:\n{design_text}"
        try:
            resp = await adapter.chat([{"role": "user", "content": review_prompt}], model, {})
            review_text = ''
            if isinstance(resp, dict):
                choices = resp.get('choices')
                if choices and isinstance(choices, list):
                    c0 = choices[0]
                    review_text = c0.get('text') or (c0.get('message') and c0.get('message').get('content')) or ''
                else:
                    review_text = resp.get('text') or ''
            else:
                review_text = str(resp)
        except Exception as e:
            review_text = f'Review failed: {e}'

        review_part = {
            'id': next_id('part'),
            'session_id': session_id,
            'message_id': message_id,
            'type': 'review',
            'text': review_text,
            'metadata': {'phase': 'review', 'timestamp': int(time.time())}
        }
        await self.sessions.storage.append_part(review_part)
        seq = await self.sessions.storage.next_sequence(session_id)
        self.events.publish_fire_and_forget(Event.create('message.part.updated', {'part': review_part}, session_id=session_id, sequence=seq))

        # Phase 4: Final Plan - instruct model to produce final plan and write to plan file
        # If a schema is provided via meta, ask model to output JSON using structured_output semantics
        schema = meta.get('schema')
        if schema:
            # instruct model to return JSON only
            final_prompt = f"Produce the final plan as JSON conforming to this schema: {schema}. Provide only the JSON object.\n\nDesign:\n{design_text}\n\nReview:\n{review_text}\n\nInstructions:\n{instructions}"
            try:
                # use structured_output tool via provider directly
                resp = await adapter.chat([{"role": "user", "content": final_prompt}], model, {})
                final_text = ''
                if isinstance(resp, dict):
                    choices = resp.get('choices')
                    if choices and isinstance(choices, list):
                        c0 = choices[0]
                        final_text = c0.get('text') or (c0.get('message') and c0.get('message').get('content')) or ''
                    else:
                        final_text = resp.get('text') or ''
                else:
                    final_text = str(resp)
                # try to parse JSON
                import json as _json
                parsed = None
                try:
                    parsed = _json.loads(final_text)
                    # validate via simple validator
                    from .jsonschema_util import validate_simple_schema
                    ok, reason = validate_simple_schema(parsed, schema)
                    if not ok:
                        final_text = f'Structured output validation failed: {reason} -- raw: {final_text}'
                    else:
                        # store structured final_text as JSON string for plan_text
                        final_text = _json.dumps(parsed, ensure_ascii=False, indent=2)
                except Exception:
                    final_text = f'Final plan parse failed: {final_text}'
            except Exception as e:
                final_text = f'Final plan generation failed: {e}'
        else:
            final_prompt = f"Produce the final plan text based on the chosen approach and review. Include numbered steps, verification steps, and file edits.\n\nDesign:\n{design_text}\n\nReview:\n{review_text}\n\nInstructions:\n{instructions}"
            try:
                resp = await adapter.chat([{"role": "user", "content": final_prompt}], model, {})
                final_text = ''
                if isinstance(resp, dict):
                    choices = resp.get('choices')
                    if choices and isinstance(choices, list):
                        c0 = choices[0]
                        final_text = c0.get('text') or (c0.get('message') and c0.get('message').get('content')) or ''
                    else:
                        final_text = resp.get('text') or ''
                else:
                    final_text = str(resp)
            except Exception as e:
                final_text = f'Final plan generation failed: {e}'

        # write plan file using AttachmentStore to keep workspace safe
        plan_path = meta.get('plan_file')
        plan_url = None
        try:
            # store as attachment
            from .attachments import AttachmentStore
            store = AttachmentStore()
            # create file content as bytes
            b = final_text.encode('utf-8')
            fname = plan_path if plan_path else f'plan_{session_id}.md'
            plan_url = store.store(b, fname, 'text/markdown')
            # also persist file path as local filename if needed
            meta['plan_file'] = fname
        except Exception:
            plan_url = None

        final_part = {
            'id': next_id('part'),
            'session_id': session_id,
            'message_id': message_id,
            'type': 'plan_final',
            'text': final_text,
            'metadata': {'phase': 'final', 'timestamp': int(time.time()), 'file': plan_path, 'attachment': plan_url}
        }
        await self.sessions.storage.append_part(final_part)
        seq = await self.sessions.storage.next_sequence(session_id)
        self.events.publish_fire_and_forget(Event.create('message.part.updated', {'part': final_part}, session_id=session_id, sequence=seq))

        # Phase 5: finalize state and emit completed event
        try:
            sess = await self.sessions.get_session(session_id)
            meta2 = sess.get('metadata', {}) or {}
            meta2['plan_active'] = False
            meta2['plan_ready'] = True
            meta2['plan_text'] = final_text
            meta2['plan_file'] = meta.get('plan_file')
            # store attachment url if available
            if plan_url:
                meta2.setdefault('attachments', {})
                meta2['attachments']['plan'] = plan_url
            await self.sessions.storage.update_session(session_id, meta2)
        except Exception:
            pass

        seq = await self.sessions.storage.next_sequence(session_id)
        self.events.publish_fire_and_forget(Event.create('session.plan.completed', {'sessionID': session_id, 'plan': meta}, session_id=session_id, sequence=seq))

        return meta
