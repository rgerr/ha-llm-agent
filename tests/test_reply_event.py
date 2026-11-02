"""The bus event at the end of a turn, and the phone approval in Home Assistant's language."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

from homeassistant.core import Event, HomeAssistant
from pytest_homeassistant_custom_component.common import async_mock_service

from custom_components.llm_agent.agent import Agent, ChatRun
from custom_components.llm_agent.const import EVENT_REPLY
from custom_components.llm_agent.providers.base import ProviderError

from .conftest import entry_options
from .test_agent import ScriptedProvider, reply_text, reply_tool


async def _turn(
    hass: HomeAssistant, replies: list[Any], *, member: bool = False, user_id: str | None = None
) -> tuple[list[Event], dict[str, Any]]:
    events: list[Event] = []
    hass.bus.async_listen(EVENT_REPLY, events.append)
    agent = Agent(hass, entry_options())
    await agent.store.async_load()
    conv = await agent.store.create("p1", "gpt-5.4-mini", "", user_id)
    run = ChatRun(agent, conv, user_id, member=member)
    with patch(
        "custom_components.llm_agent.agent.make_provider", return_value=ScriptedProvider(replies)
    ):
        await run.run("what is on?")
    await hass.async_block_till_done()
    return events, conv


async def test_a_finished_turn_fires_its_reply(hass: HomeAssistant) -> None:
    hass.states.async_set("light.porch", "on")
    events, conv = await _turn(
        hass,
        [
            reply_tool("c1", "get_entities", {"entity_ids": ["light.porch"]}),
            reply_text("The porch light is on."),
        ],
        user_id="u1",
    )
    assert len(events) == 1
    assert events[0].data == {
        "conversation_id": conv["id"],
        "user_id": "u1",
        "unattended": False,
        "watched": False,
        "stopped": None,
        "failed": False,
        "title": "what is on?",
        "reply": "The porch light is on.",
    }


async def test_a_long_reply_is_cut_and_a_member_chat_keeps_its_text(hass: HomeAssistant) -> None:
    events, _ = await _turn(hass, [reply_text("x" * 600)])
    assert len(events[0].data["reply"]) == 500 and events[0].data["reply"].endswith("…")
    events, _ = await _turn(hass, [reply_text("private")], member=True)
    assert "reply" not in events[-1].data and "title" not in events[-1].data


async def test_a_failed_turn_says_so_without_an_older_reply(hass: HomeAssistant) -> None:
    class Failing(ScriptedProvider):
        async def stream(self, **kwargs: Any):
            raise ProviderError("HTTP 503")
            yield  # pragma: no cover - makes this an async generator

    events: list[Event] = []
    hass.bus.async_listen(EVENT_REPLY, events.append)
    agent = Agent(hass, entry_options())
    await agent.store.async_load()
    conv = await agent.store.create("p1", "gpt-5.4-mini", "")
    conv["messages"] += [
        {"role": "user", "content": "earlier", "display": "earlier"},
        {"role": "assistant", "content": "an earlier answer", "tool_calls": []},
    ]
    with patch("custom_components.llm_agent.agent.make_provider", return_value=Failing([])):
        await ChatRun(agent, conv, None).run("again")
    await hass.async_block_till_done()
    assert events[0].data["failed"] is True
    assert events[0].data["reply"] == ""


async def test_the_phone_asks_in_dutch_and_names_who_ran_the_script(
    hass: HomeAssistant, hass_admin_user
) -> None:
    hass.config.language = "nl"
    notified = async_mock_service(hass, "notify", "mobile_app_pixel")
    async_mock_service(hass, "lock", "unlock")
    hass.states.async_set("lock.front", "locked")
    agent = Agent(hass, entry_options(approval_phone="notify.mobile_app_pixel"))
    await agent.store.async_load()
    conv = await agent.store.create("p1", "gpt-5.4-mini", "", hass_admin_user.id)
    run = ChatRun(agent, conv, hass_admin_user.id, unattended=True)
    args = {"domain": "lock", "service": "unlock", "target": {"entity_id": "lock.front"}}
    provider = ScriptedProvider([reply_tool("c1", "call_service", args), reply_text("Ok.")])
    with patch("custom_components.llm_agent.agent.make_provider", return_value=provider):
        task = hass.async_create_task(run.run("unlock"))
        for _ in range(100):
            if notified:
                break
            await asyncio.sleep(0.01)
        assert notified
        data = notified[0].data
        run.answer("c1", False, "test over")
        await task
    agent.phone.close()
    assert data["title"] == "Deze actie goedkeuren?"
    assert [a["title"] for a in data["data"]["actions"]] == ["Goedkeuren", "Weigeren"]
    assert data["message"].startswith(f"{hass_admin_user.name} vraagt via een automatisering: ")
