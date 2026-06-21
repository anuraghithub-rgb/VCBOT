#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
💀 NUCLEAR VOICE RELAY – RENDER EDITION 💀
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Only owner DM commands
• Audio processing: bass, EQ, reverb, echo, compressor, volume boost
• Auto-reconnect, playlist, mute bypass
• Deploy on Render with env vars
"""

import asyncio
import os
import sys
import time
import logging
import shutil
from pathlib import Path
from typing import Optional, List

import numpy as np
from scipy import signal as scipy_signal
from dotenv import load_dotenv

from pyrogram import Client, filters, idle
from pyrogram.types import Message
from pyrogram.errors import FloodWait, PeerIdInvalid, ChannelInvalid
from pytgcalls import PyTgCalls
from pytgcalls.types.input_stream import AudioStream, InputAudioStream
from pytgcalls.exceptions import GroupCallNotFound, NoActiveGroupCall, NotInGroupCallError

# ============ LOAD ENV ============
load_dotenv()

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
PHONE_NUMBER = os.getenv("PHONE_NUMBER")  # optional, if session exists
OWNER_ID = int(os.getenv("OWNER_ID"))

if not all([API_ID, API_HASH, OWNER_ID]):
    print("❌ Missing environment variables: API_ID, API_HASH, OWNER_ID are required.")
    sys.exit(1)

AUDIO_DIR = Path("saved_audios")
AUDIO_DIR.mkdir(exist_ok=True)

# ============ LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ELUMTER_COPY_userbot")

# ============ CONSTANTS ============
SAMPLE_RATE = 48000
CHANNELS = 1
FRAME_DURATION = 20  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# ============ CONFIG ============
class Config:
    def __init__(self):
        self.vc_chat: Optional[int] = None          # group where bot will play
        self.in_vc: bool = False
        self.active: bool = False                   # whether playing is active
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

# ============ AUDIO PROCESSOR ============
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

            # Bass Boost
            if config.bass > 1.0 and self.bass_coeffs:
                b, a = self.bass_coeffs
                filtered = scipy_signal.filtfilt(b, a, samples)
                bass_gain = 1.0 + (config.bass - 1.0) * 0.5
                samples = samples + (filtered * bass_gain)

            # EQ
            eq = config.equalizer
            nyquist = SAMPLE_RATE / 2
            if eq[0] != 1.0:
                b, a = scipy_signal.butter(2, 200/nyquist, btype='low')
                filtered = scipy_signal.filtfilt(b, a, samples)
                samples = samples + (filtered * (eq[0] - 1.0) * 0.5)
            if eq[4] != 1.0:
                b, a = scipy_signal.butter(2, 4000/nyquist, btype='high')
                filtered = scipy_signal.filtfilt(b, a, samples)
                samples = samples + (filtered * (eq[4] - 1.0) * 0.3)

            # Reverb
            if config.reverb > 0.01:
                delay_samples = int(SAMPLE_RATE * 0.05)
                reverb_gain = config.reverb * 0.6
                reverb_out = np.zeros_like(samples)
                for i in range(len(samples)):
                    idx = (self.reverb_index - delay_samples) % len(self.reverb_buffer)
                    reverb_out[i] = self.reverb_buffer[idx]
                    self.reverb_buffer[self.reverb_index] = samples[i] + reverb_out[i] * 0.3
                    self.reverb_index = (self.reverb_index + 1) % len(self.reverb_buffer)
                samples = samples + reverb_out * reverb_gain

            # Echo
            if config.echo > 0.01:
                delay_samples = int(SAMPLE_RATE * 0.08)
                echo_gain = config.echo * 0.5
                echo_out = np.zeros_like(samples)
                for i in range(len(samples)):
                    idx = (self.echo_index - delay_samples) % len(self.echo_buffer)
                    echo_out[i] = self.echo_buffer[idx]
                    self.echo_buffer[self.echo_index] = samples[i] + echo_out[i] * 0.2
                    self.echo_index = (self.echo_index + 1) % len(self.echo_buffer)
                samples = samples + echo_out * echo_gain

            # Compressor
            rms = np.sqrt(np.mean(samples**2))
            threshold = 0.08
            if rms > threshold:
                gain_reduction = threshold / rms
                gain_reduction = gain_reduction ** 0.7
                samples = samples * gain_reduction
            samples = samples * 1.8

            # Volume Boost
            boost_factor = config.boost / 10.0
            samples = samples * boost_factor

            # Limiter
            samples = np.tanh(samples * 1.2) * 0.98

            # Normalize
            max_val = np.max(np.abs(samples))
            if max_val > 0.95:
                samples = samples / max_val * 0.95

            samples = (samples * 32767).astype(np.int16)
            return samples.tobytes()

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return audio_data

audio_processor = AudioProcessor()

# ============ TELEGRAM CLIENT ============
app = Client(
    "ELUMTER_COPY_userbot",
    api_id=API_ID,
    api_hash=API_HASH,
    phone_number=PHONE_NUMBER if PHONE_NUMBER else None,
    workers=4
)
calls = PyTgCalls(app)

# ============ HELPERS ============
def is_owner(user_id: int) -> bool:
    return user_id == OWNER_ID

async def save_audio(message: Message) -> Optional[str]:
    try:
        if message.audio:
            file_name = message.audio.file_name or f"audio_{int(time.time())}.ogg"
            ext = file_name.split('.')[-1] if '.' in file_name else 'ogg'
        elif message.voice:
            ext = 'ogg'
        elif message.video_note:
            ext = 'mp4'
        else:
            return None
        timestamp = int(time.time())
        path = AUDIO_DIR / f"audio_{timestamp}.{ext}"
        await app.download_media(message, file_name=str(path))
        logger.info(f"Audio saved: {path}")
        return str(path)
    except Exception as e:
        logger.error(f"Save audio error: {e}")
        return None

async def join_vc(chat_id: int) -> bool:
    try:
        await calls.join_group_call(
            chat_id,
            AudioStream(
                InputAudioStream(
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    frame_duration=FRAME_DURATION,
                )
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
            logger.error(f"Leave VC error: {e}")
    return False

async def play_audio(file_path: str):
    if not config.in_vc or not config.vc_chat:
        logger.error("Not in VC to play")
        return False
    try:
        await calls.change_stream(
            config.vc_chat,
            AudioStream(
                InputAudioStream(
                    path=file_path,
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    frame_duration=FRAME_DURATION,
                )
            )
        )
        return True
    except Exception as e:
        logger.error(f"Play error: {e}")
        return False

def format_status() -> str:
    eq = config.equalizer
    return (
        f"💀 **NUCLEAR RELAY – RENDER**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔊 **Active:** `{config.active}`\n"
        f"📡 **In VC:** `{config.in_vc}` (chat: `{config.vc_chat}`)\n"
        f"🔇 **Mute Bypass:** `{'ON ✅' if config.mute_bypass else 'OFF'}`\n"
        f"🔊 **Boost:** `{config.boost}x`\n"
        f"🎸 **Bass:** `{config.bass}/10`\n"
        f"🎚️ **EQ:** `{eq[0]} {eq[1]} {eq[2]} {eq[3]} {eq[4]}`\n"
        f"🌀 **Reverb:** `{config.reverb:.2f}`\n"
        f"📢 **Echo:** `{config.echo:.2f}`\n"
        f"🎵 **Current Audio:** `{config.current_audio_file or 'None'}`\n"
        f"📋 **Playlist:** `{len(config.playlist)}` tracks"
    )

# ============ AUTO-RECONNECT ============
async def auto_reconnect_loop():
    while True:
        await asyncio.sleep(config.auto_reconnect_delay)
        if config.active and not config.in_vc and config.auto_reconnect and config.vc_chat:
            logger.warning("Auto-reconnecting...")
            if await join_vc(config.vc_chat):
                logger.info("Reconnected successfully")
                # Resume playing if we have a file
                if config.current_audio_file and os.path.exists(config.current_audio_file):
                    await play_audio(config.current_audio_file)
            else:
                logger.error("Auto-reconnect failed")

# ============ DM COMMANDS (OWNER ONLY) ============
@app.on_message(filters.private & filters.create(lambda _, m: is_owner(m.from_user.id)))
async def owner_commands(client: Client, message: Message):
    text = message.text or ""
    cmd = text.lower().strip()

    # Save audio from reply
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

    # Direct audio file
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

    # Parse command
    parts = cmd.split()
    main_cmd = parts[0] if parts else ""

    # ---------- HELP ----------
    if main_cmd in ["/start", "!start", "/help", "!help"]:
        await message.reply(
            "💀 **NUCLEAR VOICE RELAY – RENDER EDITION** 💀\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "👑 **Owner Only**\n\n"
            "**📝 COMMANDS:**\n"
            "`!join -100xxx` – Join voice chat in that group\n"
            "`!leave` – Leave current VC\n"
            "`!play` – Play current audio (if in VC)\n"
            "`!stop` – Stop playing & leave VC\n"
            "`!boost 50` – Volume (1-100)\n"
            "`!bass 8` – Bass (1-10)\n"
            "`!eq 2 1 3 2 4` – 5‑band EQ (0-10 each)\n"
            "`!reverb 0.5` – Reverb (0-1)\n"
            "`!echo 0.3` – Echo (0-1)\n"
            "`!mutebypass on/off` – Toggle mute bypass\n"
            "`!autoreconnect on/off` – Toggle auto-reconnect\n"
            "`!playlist add/remove/clear` – Manage playlist\n"
            "`!next` – Play next track in playlist\n"
            "`!status` – Show full status\n"
            "`!save` – Save current audio to a new file\n"
            "`!reset` – Reset all settings\n\n"
            "💬 **Send audio/voice/video_note** to save as current audio."
        )

    # ---------- JOIN ----------
    elif main_cmd == "!join":
        if len(parts) > 1:
            try:
                chat_id = int(parts[1])
                # First join the chat (as member)
                try:
                    await app.join_chat(chat_id)
                except Exception as e:
                    # Might already be member
                    logger.warning(f"Join chat warning: {e}")
                if await join_vc(chat_id):
                    await message.reply(f"✅ Joined VC in `{chat_id}`")
                else:
                    await message.reply("❌ Failed to join VC. Is there an active voice chat?")
            except:
                await message.reply("❌ Use: `!join -100xxxxxxxxx`")
        else:
            await message.reply("❌ Use: `!join -100xxxxxxxxx`")

    # ---------- LEAVE ----------
    elif main_cmd == "!leave":
        if await leave_vc():
            await message.reply("✅ Left voice chat!")
        else:
            await message.reply("❌ Not in VC or error.")

    # ---------- PLAY ----------
    elif main_cmd == "!play":
        if not config.in_vc:
            await message.reply("❌ Pehle `!join -100xxx` karo!")
            return
        if not config.current_audio_file or not os.path.exists(config.current_audio_file):
            await message.reply("❌ No audio file selected. Send one first.")
            return
        if await play_audio(config.current_audio_file):
            config.active = True
            await message.reply(f"🎵 Now playing: `{Path(config.current_audio_file).name}`")
        else:
            await message.reply("❌ Play failed.")

    # ---------- STOP ----------
    elif main_cmd in ["!stop", "/stop"]:
        config.active = False
        await leave_vc()
        await message.reply("✅ Stopped and left VC.")

    # ---------- BOOST ----------
    elif main_cmd == "!boost":
        if len(parts) > 1:
            try:
                val = float(parts[1])
                if 1 <= val <= 100:
                    config.boost = val
                    await message.reply(f"🔊 Boost set to `{val}x`")
                else:
                    await message.reply("❌ Range: 1-100")
            except:
                await message.reply("❌ Use: `!boost 50`")
        else:
            await message.reply(f"Current boost: `{config.boost}x`")

    # ---------- BASS ----------
    elif main_cmd == "!bass":
        if len(parts) > 1:
            try:
                val = float(parts[1])
                if 1 <= val <= 10:
                    config.bass = val
                    audio_processor.update_filters()
                    await message.reply(f"🎸 Bass set to `{val}/10`")
                else:
                    await message.reply("❌ Range: 1-10")
            except:
                await message.reply("❌ Use: `!bass 8`")
        else:
            await message.reply(f"Current bass: `{config.bass}/10`")

    # ---------- EQ ----------
    elif main_cmd == "!eq":
        if len(parts) == 6:
            try:
                vals = [float(p) for p in parts[1:6]]
                if all(0 <= v <= 10 for v in vals):
                    config.equalizer = vals
                    await message.reply(f"🎚️ EQ set to: `{vals[0]} {vals[1]} {vals[2]} {vals[3]} {vals[4]}`")
                else:
                    await message.reply("❌ Each band must be 0-10")
            except:
                await message.reply("❌ Use: `!eq 2 1 3 2 4`")
        else:
            eq = config.equalizer
            await message.reply(f"Current EQ: `{eq[0]} {eq[1]} {eq[2]} {eq[3]} {eq[4]}`")

    # ---------- REVERB ----------
    elif main_cmd == "!reverb":
        if len(parts) > 1:
            try:
                val = float(parts[1])
                if 0 <= val <= 1:
                    config.reverb = val
                    await message.reply(f"🌀 Reverb set to `{val:.2f}`")
                else:
                    await message.reply("❌ Range: 0-1")
            except:
                await message.reply("❌ Use: `!reverb 0.5`")
        else:
            await message.reply(f"Current reverb: `{config.reverb:.2f}`")

    # ---------- ECHO ----------
    elif main_cmd == "!echo":
        if len(parts) > 1:
            try:
                val = float(parts[1])
                if 0 <= val <= 1:
                    config.echo = val
                    await message.reply(f"📢 Echo set to `{val:.2f}`")
                else:
                    await message.reply("❌ Range: 0-1")
            except:
                await message.reply("❌ Use: `!echo 0.3`")
        else:
            await message.reply(f"Current echo: `{config.echo:.2f}`")

    # ---------- MUTE BYPASS ----------
    elif main_cmd == "!mutebypass":
        if len(parts) > 1:
            state = parts[1].lower()
            if state == "on":
                config.mute_bypass = True
                await message.reply("🔇 **Mute Bypass: ON**")
            elif state == "off":
                config.mute_bypass = False
                await message.reply("🔇 **Mute Bypass: OFF**")
            else:
                await message.reply("❌ Use: `!mutebypass on/off`")
        else:
            await message.reply(f"Mute Bypass is `{'ON' if config.mute_bypass else 'OFF'}`")

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
                await message.reply("❌ Use: `!autoreconnect on/off`")
        else:
            await message.reply(f"Auto-reconnect is `{'ON' if config.auto_reconnect else 'OFF'}`")

    # ---------- PLAYLIST ----------
    elif main_cmd == "!playlist":
        if len(parts) > 1:
            sub = parts[1].lower()
            if sub == "add" and len(parts) > 2:
                path = parts[2]
                if os.path.exists(path):
                    config.playlist.append(path)
                    await message.reply(f"✅ Added to playlist: `{path}`")
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
                await message.reply("❌ Use: `!playlist add <path>`, `remove <index>`, `clear`")
        else:
            if config.playlist:
                msg = "📋 **Playlist:**\n"
                for i, p in enumerate(config.playlist, 1):
                    msg += f"`{i}. {Path(p).name}`\n"
                await message.reply(msg)
            else:
                await message.reply("📋 Playlist is empty")

    # ---------- NEXT ----------
    elif main_cmd == "!next":
        if config.playlist:
            config.playlist_index = (config.playlist_index + 1) % len(config.playlist)
            config.current_audio_file = config.playlist[config.playlist_index]
            await message.reply(f"⏩ Next track: `{Path(config.current_audio_file).name}`")
            if config.in_vc:
                if await play_audio(config.current_audio_file):
                    config.active = True
        else:
            await message.reply("❌ Playlist empty")

    # ---------- STATUS ----------
    elif main_cmd == "!status":
        await message.reply(format_status())

    # ---------- SAVE CURRENT ----------
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
            await message.reply("❌ No current audio file")

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
        await message.reply("✅ All settings reset to default!")

    # ---------- UNKNOWN ----------
    else:
        await message.reply("❌ Unknown command. Use `/start` for help.")

# ============ VC EVENTS ============
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

# ============ MAIN ============
async def main():
    print("=" * 60)
    print("💀 NUCLEAR VOICE RELAY – RENDER EDITION 💀")
    print("=" * 60)
    print(f"👑 Owner ID: {OWNER_ID}")
    print(f"📱 Account: {PHONE_NUMBER or 'Using session file'}")
    print("🔇 Mute Bypass: ENABLED")
    print("=" * 60)

    await app.start()
    await calls.start()

    # Start auto-reconnect loop
    asyncio.create_task(auto_reconnect_loop())

    print("\n✅ System Ready!")
    print("📝 Only owner's DM commands will work.")
    print("💀 MUTE BYPASS ACTIVE")
    print("=" * 60)

    await idle()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ System band ho raha hai...")
    except Exception as e:
        logger.error(f"Fatal Error: {e}")
