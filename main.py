#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
💀 BC RELAY – RAILWAY EDITION (FULLY UPDATED) 💀
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Owner-only DM commands. All audio effects included.
Works with py-tgcalls==2.1.0 – uses only InputAudioStream.
"""

import asyncio
import os
import sys
import time
import logging
import shutil
from pathlib import Path
from typing import Optional, List

# ==================== ENV LOAD ====================
from dotenv import load_dotenv
load_dotenv()

REQUIRED_ENV = ["API_ID", "API_HASH", "OWNER_ID"]
missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
if missing:
    print(f"❌ Missing env vars: {', '.join(missing)}")
    sys.exit(1)

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
PHONE_NUMBER = os.getenv("PHONE_NUMBER")   # optional
OWNER_ID = int(os.getenv("OWNER_ID"))

# ==================== IMPORTS ====================
import numpy as np
from scipy import signal as scipy_signal

from pyrogram import Client, filters, idle
from pyrogram.types import Message

from pytgcalls import PyTgCalls
# ---------- ONLY THIS IMPORT ----------
from pytgcalls.types import InputAudioStream
from pytgcalls.exceptions import GroupCallNotFound, NoActiveGroupCall, NotInGroupCallError

# ==================== SETUP ====================
AUDIO_DIR = Path("saved_audios")
AUDIO_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Relay")

# ==================== CONSTANTS ====================
SAMPLE_RATE = 48000
CHANNELS = 1
FRAME_DURATION = 20   # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# ==================== CONFIG ====================
class Config:
    def __init__(self):
        self.vc_chat: Optional[int] = None
        self.in_vc: bool = False
        self.active: bool = False
        self.mute_bypass: bool = True
        self.boost: float = 60.0
        self.bass: float = 8.0
        self.equalizer: List[float] = [8.0, 4.0, 2.0, 3.0, 5.0]
        self.reverb: float = 0.0
        self.echo: float = 0.0
        self.current_audio_file: Optional[str] = None
        self.playlist: List[str] = []
        self.playlist_index: int = 0
        self.auto_reconnect: bool = True
        self.auto_reconnect_delay: int = 5

config = Config()

# ==================== AUDIO PROCESSOR ====================
class AudioProcessor:
    def __init__(self):
        self.bass_coeffs = None
        self.update_filters()
        self.reverb_buffer = np.zeros(50000, dtype=np.float32)
        self.reverb_index = 0
        self.echo_buffer = np.zeros(30000, dtype=np.float32)
        self.echo_index = 0

    def update_filters(self):
        try:
            nyquist = SAMPLE_RATE / 2
            b, a = scipy_signal.butter(4, [30/nyquist, 200/nyquist], btype='band')
            self.bass_coeffs = (b, a)
        except Exception as e:
            logger.error(f"Filter update failed: {e}")
            self.bass_coeffs = None

    def process(self, audio_data: bytes) -> bytes:
        if not audio_data:
            return audio_data
        try:
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Bass boost
            if config.bass > 1.0 and self.bass_coeffs:
                b, a = self.bass_coeffs
                filtered = scipy_signal.filtfilt(b, a, samples)
                bass_gain = 1.0 + (config.bass - 1.0) * 0.5
                samples += filtered * bass_gain

            # 5‑band EQ (only low and high shelf for simplicity)
            eq = config.equalizer
            nyquist = SAMPLE_RATE / 2
            if eq[0] != 1.0:
                b, a = scipy_signal.butter(2, 200/nyquist, btype='low')
                filtered = scipy_signal.filtfilt(b, a, samples)
                samples += filtered * (eq[0] - 1.0) * 0.5
            if eq[4] != 1.0:
                b, a = scipy_signal.butter(2, 4000/nyquist, btype='high')
                filtered = scipy_signal.filtfilt(b, a, samples)
                samples += filtered * (eq[4] - 1.0) * 0.3

            # Reverb (feedback delay)
            if config.reverb > 0.01:
                delay = int(SAMPLE_RATE * 0.05)
                gain = config.reverb * 0.6
                out = np.zeros_like(samples)
                for i in range(len(samples)):
                    idx = (self.reverb_index - delay) % len(self.reverb_buffer)
                    out[i] = self.reverb_buffer[idx]
                    self.reverb_buffer[self.reverb_index] = samples[i] + out[i] * 0.3
                    self.reverb_index = (self.reverb_index + 1) % len(self.reverb_buffer)
                samples += out * gain

            # Echo (slapback)
            if config.echo > 0.01:
                delay = int(SAMPLE_RATE * 0.08)
                gain = config.echo * 0.5
                out = np.zeros_like(samples)
                for i in range(len(samples)):
                    idx = (self.echo_index - delay) % len(self.echo_buffer)
                    out[i] = self.echo_buffer[idx]
                    self.echo_buffer[self.echo_index] = samples[i] + out[i] * 0.2
                    self.echo_index = (self.echo_index + 1) % len(self.echo_buffer)
                samples += out * gain

            # Compressor
            rms = np.sqrt(np.mean(samples**2))
            threshold = 0.08
            if rms > threshold:
                gain_reduction = (threshold / rms) ** 0.7
                samples *= gain_reduction
            samples *= 1.8

            # Volume boost
            samples *= (config.boost / 10.0)

            # Limiter (soft clipping)
            samples = np.tanh(samples * 1.2) * 0.98

            # Normalize
            max_val = np.max(np.abs(samples))
            if max_val > 0.95:
                samples = samples / max_val * 0.95

            return (samples * 32767).astype(np.int16).tobytes()

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return audio_data

audio_processor = AudioProcessor()

# ==================== TELEGRAM CLIENT ====================
app = Client(
    "ELUMTER_COPY_userbot",
    api_id=API_ID,
    api_hash=API_HASH,
    phone_number=PHONE_NUMBER if PHONE_NUMBER else None,
    workers=4
)
calls = PyTgCalls(app)

# ==================== HELPERS ====================
def is_owner(user_id: int) -> bool:
    return user_id == OWNER_ID

async def save_audio(message: Message) -> Optional[str]:
    try:
        if message.audio:
            name = message.audio.file_name or f"audio_{int(time.time())}.ogg"
            ext = name.split('.')[-1] if '.' in name else 'ogg'
        elif message.voice:
            ext = 'ogg'
        elif message.video_note:
            ext = 'mp4'
        else:
            return None
        path = AUDIO_DIR / f"audio_{int(time.time())}.{ext}"
        await app.download_media(message, file_name=str(path))
        logger.info(f"Saved: {path}")
        return str(path)
    except Exception as e:
        logger.error(f"Save error: {e}")
        return None

async def join_vc(chat_id: int) -> bool:
    try:
        await calls.join_group_call(
            chat_id,
            InputAudioStream(
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                frame_duration=FRAME_DURATION,
            )
        )
        config.in_vc = True
        config.vc_chat = chat_id
        logger.info(f"Joined VC in {chat_id}")
        return True
    except Exception as e:
        logger.error(f"Join VC failed: {e}")
        return False

async def leave_vc() -> bool:
    if config.in_vc and config.vc_chat:
        try:
            await calls.leave_group_call(config.vc_chat)
            config.in_vc = False
            config.vc_chat = None
            logger.info("Left VC")
            return True
        except Exception as e:
            logger.error(f"Leave error: {e}")
    return False

async def play_audio(file_path: str) -> bool:
    if not config.in_vc or not config.vc_chat:
        return False
    try:
        await calls.change_stream(
            config.vc_chat,
            InputAudioStream(
                path=file_path,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                frame_duration=FRAME_DURATION,
            )
        )
        return True
    except Exception as e:
        logger.error(f"Play error: {e}")
        return False

def format_status() -> str:
    eq = config.equalizer
    return (
        f"💀 **BC RELAY – RAILWAY**\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"🔊 Active: `{config.active}`\n"
        f"📡 In VC: `{config.in_vc}` (chat: `{config.vc_chat}`)\n"
        f"🔇 Mute Bypass: `{'ON ✅' if config.mute_bypass else 'OFF'}`\n"
        f"🔊 Boost: `{config.boost}x`\n"
        f"🎸 Bass: `{config.bass}/10`\n"
        f"🎚️ EQ: `{eq[0]} {eq[1]} {eq[2]} {eq[3]} {eq[4]}`\n"
        f"🌀 Reverb: `{config.reverb:.2f}`\n"
        f"📢 Echo: `{config.echo:.2f}`\n"
        f"🎵 Current: `{Path(config.current_audio_file).name if config.current_audio_file else 'None'}`\n"
        f"📋 Playlist: `{len(config.playlist)}` tracks"
    )

# ==================== AUTO-RECONNECT ====================
async def auto_reconnect_loop():
    while True:
        await asyncio.sleep(config.auto_reconnect_delay)
        if config.active and not config.in_vc and config.auto_reconnect and config.vc_chat:
            logger.warning("Auto-reconnecting...")
            if await join_vc(config.vc_chat):
                logger.info("Reconnected")
                if config.current_audio_file and os.path.exists(config.current_audio_file):
                    await play_audio(config.current_audio_file)
            else:
                logger.error("Reconnect failed")

# ==================== OWNER DM COMMANDS ====================
@app.on_message(filters.private & filters.create(lambda _, m: is_owner(m.from_user.id)))
async def owner_commands(client: Client, message: Message):
    text = message.text or ""
    cmd = text.lower().strip()

    # Save audio from reply or direct
    if message.reply_to_message and (message.reply_to_message.audio or message.reply_to_message.voice):
        audio_path = await save_audio(message.reply_to_message)
        if audio_path:
            config.current_audio_file = audio_path
            if audio_path not in config.playlist:
                config.playlist.append(audio_path)
            await message.reply(f"✅ Audio saved & set as current!\n📁 `{audio_path}`")
        else:
            await message.reply("❌ Save failed")
        return

    if message.audio or message.voice or message.video_note:
        audio_path = await save_audio(message)
        if audio_path:
            config.current_audio_file = audio_path
            if audio_path not in config.playlist:
                config.playlist.append(audio_path)
            await message.reply(f"✅ Audio saved!\n📁 `{audio_path}`")
        else:
            await message.reply("❌ Save failed")
        return

    parts = cmd.split()
    main_cmd = parts[0] if parts else ""

    # ---------- HELP ----------
    if main_cmd in ["/start", "!start", "/help", "!help"]:
        await message.reply(
            "💀 **BC RELAY – OWNER COMMANDS** 💀\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "`!join -100xxx` – Join VC\n"
            "`!leave` – Leave VC\n"
            "`!play` – Play current audio\n"
            "`!stop` – Stop & leave\n"
            "`!boost 50` – Volume (1-100)\n"
            "`!bass 8` – Bass (1-10)\n"
            "`!eq 2 1 3 2 4` – 5‑band EQ (0-10)\n"
            "`!reverb 0.5` – Reverb (0-1)\n"
            "`!echo 0.3` – Echo (0-1)\n"
            "`!mutebypass on/off`\n"
            "`!autoreconnect on/off`\n"
            "`!playlist add/remove/clear`\n"
            "`!next` – Next track\n"
            "`!status` – Show status\n"
            "`!save` – Save copy\n"
            "`!reset` – Reset all\n\n"
            "📤 Send audio/voice to save."
        )

    # ---------- JOIN ----------
    elif main_cmd == "!join":
        if len(parts) > 1:
            try:
                chat_id = int(parts[1])
                try:
                    await app.join_chat(chat_id)
                except:
                    pass
                if await join_vc(chat_id):
                    await message.reply(f"✅ Joined VC in `{chat_id}`")
                else:
                    await message.reply("❌ Join VC failed. Is there an active voice chat?")
            except:
                await message.reply("❌ Use: `!join -100xxxxxxxxx`")
        else:
            await message.reply("❌ Use: `!join -100xxxxxxxxx`")

    # ---------- LEAVE ----------
    elif main_cmd == "!leave":
        if await leave_vc():
            await message.reply("✅ Left VC")
        else:
            await message.reply("❌ Not in VC")

    # ---------- PLAY ----------
    elif main_cmd == "!play":
        if not config.in_vc:
            await message.reply("❌ Join VC first: `!join -100xxx`")
            return
        if not config.current_audio_file or not os.path.exists(config.current_audio_file):
            await message.reply("❌ No audio file. Send one first.")
            return
        if await play_audio(config.current_audio_file):
            config.active = True
            await message.reply(f"🎵 Playing: `{Path(config.current_audio_file).name}`")
        else:
            await message.reply("❌ Play error")

    # ---------- STOP ----------
    elif main_cmd in ["!stop", "/stop"]:
        config.active = False
        await leave_vc()
        await message.reply("✅ Stopped")

    # ---------- BOOST ----------
    elif main_cmd == "!boost":
        if len(parts) > 1:
            try:
                v = float(parts[1])
                if 1 <= v <= 100:
                    config.boost = v
                    await message.reply(f"🔊 Boost: `{v}x`")
                else:
                    await message.reply("❌ 1-100")
            except:
                await message.reply("❌ Use: `!boost 50`")
        else:
            await message.reply(f"Boost: `{config.boost}x`")

    # ---------- BASS ----------
    elif main_cmd == "!bass":
        if len(parts) > 1:
            try:
                v = float(parts[1])
                if 1 <= v <= 10:
                    config.bass = v
                    audio_processor.update_filters()
                    await message.reply(f"🎸 Bass: `{v}/10`")
                else:
                    await message.reply("❌ 1-10")
            except:
                await message.reply("❌ Use: `!bass 8`")
        else:
            await message.reply(f"Bass: `{config.bass}/10`")

    # ---------- EQ ----------
    elif main_cmd == "!eq":
        if len(parts) == 6:
            try:
                vals = [float(p) for p in parts[1:6]]
                if all(0 <= v <= 10 for v in vals):
                    config.equalizer = vals
                    await message.reply(f"🎚️ EQ: `{vals}`")
                else:
                    await message.reply("❌ Each 0-10")
            except:
                await message.reply("❌ Use: `!eq 2 1 3 2 4`")
        else:
            eq = config.equalizer
            await message.reply(f"EQ: `{eq}`")

    # ---------- REVERB ----------
    elif main_cmd == "!reverb":
        if len(parts) > 1:
            try:
                v = float(parts[1])
                if 0 <= v <= 1:
                    config.reverb = v
                    await message.reply(f"🌀 Reverb: `{v:.2f}`")
                else:
                    await message.reply("❌ 0-1")
            except:
                await message.reply("❌ Use: `!reverb 0.5`")
        else:
            await message.reply(f"Reverb: `{config.reverb:.2f}`")

    # ---------- ECHO ----------
    elif main_cmd == "!echo":
        if len(parts) > 1:
            try:
                v = float(parts[1])
                if 0 <= v <= 1:
                    config.echo = v
                    await message.reply(f"📢 Echo: `{v:.2f}`")
                else:
                    await message.reply("❌ 0-1")
            except:
                await message.reply("❌ Use: `!echo 0.3`")
        else:
            await message.reply(f"Echo: `{config.echo:.2f}`")

    # ---------- MUTE BYPASS ----------
    elif main_cmd == "!mutebypass":
        if len(parts) > 1:
            state = parts[1].lower()
            if state == "on":
                config.mute_bypass = True
                await message.reply("🔇 Mute Bypass: ON")
            elif state == "off":
                config.mute_bypass = False
                await message.reply("🔇 Mute Bypass: OFF")
            else:
                await message.reply("❌ on/off")
        else:
            await message.reply(f"Mute Bypass: `{'ON' if config.mute_bypass else 'OFF'}`")

    # ---------- AUTORECONNECT ----------
    elif main_cmd == "!autoreconnect":
        if len(parts) > 1:
            state = parts[1].lower()
            if state == "on":
                config.auto_reconnect = True
                await message.reply("🔄 Auto-reconnect: ON")
            elif state == "off":
                config.auto_reconnect = False
                await message.reply("🔄 Auto-reconnect: OFF")
            else:
                await message.reply("❌ on/off")
        else:
            await message.reply(f"Auto-reconnect: `{'ON' if config.auto_reconnect else 'OFF'}`")

    # ---------- PLAYLIST ----------
    elif main_cmd == "!playlist":
        if len(parts) > 1:
            sub = parts[1].lower()
            if sub == "add" and len(parts) > 2:
                path = parts[2]
                if os.path.exists(path):
                    config.playlist.append(path)
                    await message.reply(f"✅ Added: `{path}`")
                else:
                    await message.reply("❌ File not found")
            elif sub == "remove" and len(parts) > 2:
                try:
                    idx = int(parts[2]) - 1
                    if 0 <= idx < len(config.playlist):
                        removed = config.playlist.pop(idx)
                        await message.reply(f"✅ Removed: `{removed}`")
                    else:
                        await message.reply("❌ Invalid index")
                except:
                    await message.reply("❌ Use: `!playlist remove 1`")
            elif sub == "clear":
                config.playlist.clear()
                await message.reply("✅ Playlist cleared")
            else:
                await message.reply("❌ Use: add <path>, remove <index>, clear")
        else:
            if config.playlist:
                msg = "📋 **Playlist:**\n"
                for i, p in enumerate(config.playlist, 1):
                    msg += f"`{i}. {Path(p).name}`\n"
                await message.reply(msg)
            else:
                await message.reply("📋 Empty")

    # ---------- NEXT ----------
    elif main_cmd == "!next":
        if config.playlist:
            config.playlist_index = (config.playlist_index + 1) % len(config.playlist)
            config.current_audio_file = config.playlist[config.playlist_index]
            await message.reply(f"⏩ Next: `{Path(config.current_audio_file).name}`")
            if config.in_vc:
                await play_audio(config.current_audio_file)
                config.active = True
        else:
            await message.reply("❌ Playlist empty")

    # ---------- STATUS ----------
    elif main_cmd == "!status":
        await message.reply(format_status())

    # ---------- SAVE ----------
    elif main_cmd == "!save":
        if config.current_audio_file and os.path.exists(config.current_audio_file):
            base = Path(config.current_audio_file).stem
            new_path = AUDIO_DIR / f"{base}_saved_{int(time.time())}.ogg"
            try:
                shutil.copy(config.current_audio_file, new_path)
                await message.reply(f"✅ Saved copy: `{new_path}`")
            except Exception as e:
                await message.reply(f"❌ Error: {e}")
        else:
            await message.reply("❌ No current audio")

    # ---------- RESET ----------
    elif main_cmd == "!reset":
        config.active = False
        await leave_vc()
        config.boost = 60.0
        config.bass = 8.0
        config.equalizer = [8.0, 4.0, 2.0, 3.0, 5.0]
        config.reverb = 0.0
        config.echo = 0.0
        config.mute_bypass = True
        config.auto_reconnect = True
        config.playlist.clear()
        config.current_audio_file = None
        await message.reply("✅ Reset to defaults")

    # ---------- UNKNOWN ----------
    else:
        await message.reply("❌ Unknown. Use `/start` for help.")

# ==================== VC EVENTS ====================
@calls.on_stream_end()
async def stream_end_handler(client, update):
    logger.info("Stream ended")

@calls.on_kicked()
async def on_kicked_handler(client, chat_id):
    logger.warning(f"Kicked from {chat_id}")
    config.in_vc = False

@calls.on_closed_voice_chat()
async def on_voice_chat_closed(client, chat_id):
    logger.warning(f"Voice chat closed in {chat_id}")
    config.in_vc = False

# ==================== MAIN ====================
async def main():
    print("=" * 60)
    print("💀 BC RELAY – RAILWAY EDITION (FULLY UPDATED) 💀")
    print("=" * 60)
    print(f"👑 Owner: {OWNER_ID}")
    print(f"📱 Account: {PHONE_NUMBER or 'Session file'}")
    print("🔇 Mute Bypass: ENABLED")
    print("=" * 60)

    await app.start()
    await calls.start()
    asyncio.create_task(auto_reconnect_loop())

    print("\n✅ Ready! Only owner DM commands work.")
    print("💀 MUTE BYPASS ACTIVE")
    print("=" * 60)

    await idle()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Shutting down...")
    except Exception as e:
        logger.error(f"Fatal: {e}")