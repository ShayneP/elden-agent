import asyncio
import logging
import os
from livekit import rtc
from collections import deque
import time

from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, silero, cartesia, openai
from livekit.agents.llm import ChatMessage, ChatImage

# Configure Logging
logger = logging.getLogger("elden-ring")
logger.setLevel(logging.INFO)

async def entrypoint(ctx: JobContext):
    room = rtc.Room()

    # Initialize Chat Context with System Prompt
    initial_chat_ctx = llm.ChatContext()
    initial_chat_ctx.messages.append(
        llm.ChatMessage(
            content=(
                "You are a Finger Maiden, a servant of the gods. Your job is to help the Tarnished. "
                "The Tarnished will ask you for help, and you must provide them with the information they need. "
                "Stay in character. Respond with one or two sentences. Everything that you say will be said aloud, so it's "
                "very important that you don't use lists, bolding, or any other markdown. Don't include 'action' words like "
                "*speaks softly* or *laughs*. "
                "Just use the words that you want to be spoken."
            ),
            role="system",
        )
    )

    async def _will_synthesize_assistant_reply(
            assistant: VoiceAssistant, chat_ctx: llm.ChatContext
        ):
        latest_image = await get_latest_image()
        if latest_image:
            image = [ChatImage(image=latest_image)]
            chat_ctx.messages.append(ChatMessage(role="user", content=image))
            logger.debug("Assistant reply augmented with latest image.")

    # Initialize Voice Assistant
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(voice="71a7ad14-091c-4e8e-a314-022ece01c121"),
        chat_ctx=initial_chat_ctx,
        will_synthesize_assistant_reply=_will_synthesize_assistant_reply,
        interrupt_min_words=10
    )

    async def get_video_track(room: rtc.Room):
        """
        Retrieves the first available remote video track from the room.
        Raises:
            ValueError: If no remote video track is found.
        """
        for participant_id, participant in room.remote_participants.items():
            for track_id, track_publication in participant.track_publications.items():
                if track_publication.track and isinstance(track_publication.track, rtc.RemoteVideoTrack):
                    logger.info(f"Using video track {track_publication.track.sid} from participant {participant_id}.")
                    return track_publication.track
        raise ValueError("No remote video track found in the room.")

    async def get_latest_image():
        video_stream = None
        try:
            video_track = await get_video_track(ctx.room)
            video_stream = rtc.VideoStream(video_track)
            async for event in video_stream:
                logger.debug("Latest image frame retrieved.")
                return event.frame
        except Exception as e:
            logger.error(f"Failed to get latest image: {e}")
            return None
        finally:
            if video_stream:
                await video_stream.aclose()

    await ctx.connect()
    chat = rtc.ChatManager(ctx.room) 
    logger.info("Connected to the room and ChatManager initialized.")

    assistant.start(ctx.room)
    logger.info("VoiceAssistant started.")

    await asyncio.sleep(2)
    await assistant.say("I am here.")
    logger.info("Assistant greeted with 'I am here.'")


if __name__ == "__main__":
    # Configure the worker options with the revised entrypoint
    worker_options = WorkerOptions(entrypoint_fnc=entrypoint)
    cli.run_app(worker_options)
