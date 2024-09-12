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

logger = logging.getLogger("elden-ring")
logger.setLevel(logging.INFO) 

async def entrypoint(ctx: JobContext):
    room = rtc.Room()

    initial_chat_ctx = llm.ChatContext()
    initial_chat_ctx.messages.append(
        llm.ChatMessage(
            content=(
                "You are a Finger Maiden, a servant of the gods. Your job is to help the Tarnished. \
                The Tarnished will ask you for help, and you must provide them with the information they need. \
                Stay in character. Respond with one or two sentences. Everything that you say will be said aloud, so it's \
                very important that you don't use lists, bolding, or any other markdown. Don't include 'action' words like *speaks softly*' or *laughs*. \
                Just use the words that you want to be spoken."
            ),
            role="system",
        )
    )

    image_buffer = deque(maxlen=3)
    last_capture_time = 0

    async def update_image_buffer():
        nonlocal last_capture_time
        current_time = time.time()
        if current_time - last_capture_time >= 1:
            latest_image = await get_latest_image()
            if latest_image:
                image_buffer.append(latest_image)
                last_capture_time = current_time

    async def _will_synthesize_assistant_reply(
            assistant: VoiceAssistant, chat_ctx: llm.ChatContext
        ):
        latest_image = await get_latest_image()
        if latest_image:
            image = [ChatImage(image=latest_image)]
            chat_ctx.messages.append(ChatMessage(role="user", content=image))

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(voice="71a7ad14-091c-4e8e-a314-022ece01c121"),
        chat_ctx=initial_chat_ctx,
        will_synthesize_assistant_reply=_will_synthesize_assistant_reply,
        interrupt_min_words=10
    )

    async def get_video_track(room: ctx.room):
        video_track = asyncio.Future[rtc.RemoteVideoTrack]()
        for _, participant in room.remote_participants.items():
            for _, track_publication in participant.track_publications.items():
                if track_publication.track is not None and isinstance(
                    track_publication.track, rtc.RemoteVideoTrack
                ):
                    video_track.set_result(track_publication.track)
                    print(f"Using video track {track_publication.track.sid}")
                    break
        return await video_track

    async def get_latest_image():
        video_track = await get_video_track(ctx.room)
        async for event in rtc.VideoStream(video_track):
            return event.frame

    await ctx.connect()
    chat = rtc.ChatManager(ctx.room) 

    assistant.start(ctx.room)

    async def update_image_buffer_periodically():
        while True:
            await update_image_buffer()
            await asyncio.sleep(1)


    asyncio.create_task(update_image_buffer_periodically())

    await asyncio.sleep(2)
    await assistant.say("I am here.")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))