"""Regression coverage for Bot.set_config's narrowed validation (this
pass's remediation section 11C): it used to silently accept ANY
keyword, a leftover meant to keep a retired GUI from crashing while it
was being replaced -- which meant typos/obsolete fields were accepted
forever with no signal. Bot.py's own default self.config dict is the
one canonical set of known keys; anything else must now raise."""

from __future__ import annotations

import pytest

from bot.Bot import Bot


def _bare_bot() -> Bot:
    bot = Bot.__new__(Bot)
    bot.config = {
        "show_frames": False,
        "selected_mobs": [],
        "selected_map_name": None,
        "native_monster_local_radius_cells": 50,
        "eva_hotkey": "F1",
    }
    bot._native_map_overlay = "sentinel"
    bot._native_map_overlay_name = "sentinel"
    bot._reload_mob_templates_calls = 0
    bot._reload_mob_templates = lambda: setattr(
        bot, "_reload_mob_templates_calls", bot._reload_mob_templates_calls + 1
    )
    return bot


def test_set_config_accepts_a_known_key() -> None:
    bot = _bare_bot()
    bot.set_config(show_frames=True)
    assert bot.config["show_frames"] is True


def test_set_config_rejects_an_unknown_key() -> None:
    bot = _bare_bot()
    with pytest.raises(ValueError, match="Unknown bot config key"):
        bot.set_config(mob_still_alive_match_threshold=0.7)
    assert "mob_still_alive_match_threshold" not in bot.config


def test_set_config_rejects_unknown_keys_even_mixed_with_known_ones() -> None:
    bot = _bare_bot()
    with pytest.raises(ValueError, match="Unknown bot config key"):
        bot.set_config(show_frames=True, mobs_kill_goal=None)
    # Nothing from a rejected batch is applied, including the valid key.
    assert bot.config["show_frames"] is False


def test_set_config_reloads_mob_templates_when_selected_mobs_changes() -> None:
    bot = _bare_bot()
    bot.set_config(selected_mobs=[{"species_id": 1}])
    assert bot._reload_mob_templates_calls == 1


def test_set_config_resets_native_map_overlay_on_map_change() -> None:
    bot = _bare_bot()
    bot.set_config(selected_map_name="Tower AoE")
    assert bot._native_map_overlay is None
    assert bot._native_map_overlay_name is None


def test_set_config_resets_native_map_overlay_on_radius_change() -> None:
    bot = _bare_bot()
    bot.set_config(native_monster_local_radius_cells=75)
    assert bot._native_map_overlay is None
    assert bot._native_map_overlay_name is None
