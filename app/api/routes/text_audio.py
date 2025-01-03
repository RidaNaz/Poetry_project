import os
import wave
import logging
import contextlib
from IPython.display import Audio

from google.genai import Client
from typing import Literal, Union, Dict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()  
router = APIRouter()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

client = Client(
  http_options= {'api_version': 'v1alpha'}
)

MODEL: str = "gemini-2.0-flash-exp"
FILE_NAME = 'audio.wav'
config={"configurable": {"thread_id": "1"}, "generation_config": {"response_modalities": ["AUDIO"]}}

logger = logging.getLogger('Live')
logger.setLevel('INFO')

memory: MemorySaver = MemorySaver()

class State(MessagesState):
    summary: str

@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
        
# Summarization

async def summarize_conversation(state: State) -> Dict[str, object]:
    """
    Summarizes the conversation if the number of messages exceeds 6 messages.
    
    Args:
        state (State): The current conversation state.

    Returns:
        Dict[str, object]: A dictionary containing updated messages and the summary.
    """
    # Get any existing summary
    summary = state.get("summary", [])
    
    # Create the summarization prompt based on the presence of an existing summary
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add the summarization prompt to the conversation history
    messages = [SystemMessage(content=summary_message)] + state["messages"]
    
    history_str = "\n".join([f"{m.type}: {m.content}" for m in messages])
    input_text = f"{history_str}\nUser: {summary_message}"

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send(input=input_text, end_of_turn=True)
        turn = session.receive()
        response_text = ""

        async for chunk in turn:
            if chunk.text is not None:
                response_text += chunk.text

    # Update and return the new summary
    summary = response_text
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=getattr(m, "id", None)) for m in state["messages"][:-2]]

    return {"summary": response_text, "messages": delete_messages}
        
def select_next_node(state: State) -> Union[Literal["summarize"], str]:

    messages = state["messages"]
    last_message = messages[-1]

    # If there are more than six messages, route to "summarize_conversation"
    if len(messages) > 6:
        return "summarize"
    
    # Otherwise, route to "final" or end
    return END
        
async def call_modle(state: State):

  # Initialize messages from the state
  messages = state["messages"]

  # Prepend a summary if it exists
  summary = state.get("summary", "")
  if summary:
    system_message = f"Summary of conversation earlier: {summary}"
    messages = [SystemMessage(content=system_message)] + messages
        
    # Extract the content from HumanMessage objects
  message_texts = [message.content for message in messages]

  async with client.aio.live.connect(model=MODEL, config=config) as session:
    
    with wave_file(FILE_NAME) as wav:
      await session.send(input=" ".join(message_texts), end_of_turn=True)

      receive = session.receive()

      # Track if this is the first response to display metadata
      first = True
      async for response in receive:
        if response.data is not None:
          model_turn = response.server_content.model_turn
          if first:
            print(model_turn.parts[0].inline_data.mime_type)
            first = False
          wav.writeframes(response.data)

      response_text = ""
      async for chunk in receive:
          if chunk.text is not None:
              response_text += chunk.text
              
  # Append the response to messages
  response_message = AIMessage(content=response_text)
  messages.append(response_message)

  # Display the audio file in a Jupyter/Colab notebook
  return Audio(FILE_NAME, autoplay=True), {"messages": messages}
  

builder:StateGraph = StateGraph(State)

builder.add_node("agent", call_modle)
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "agent")
builder.add_edge("summarize", END)

builder.add_conditional_edges(
    "agent",
    select_next_node,
    {"summarize": "summarize", END: END},
)

graph = builder.compile(checkpointer=memory)

@router.post("/")
async def run(message: str):
    system_prompt = SystemMessage(content=(
        "You are a professional healthcare assistant. Provide clear, accurate, and evidence-based information on: "
        "general health, symptoms, and preventive care; medication uses and precautions. "
        "Be empathetic, concise, and avoid diagnostics or treatment advice. Always encourage consulting a licensed healthcare provider."
    ))

    initial_messages = [system_prompt, HumanMessage(content=message)]

    # Invoke the graph to process user input and generate audio
    audio, response_message = await call_modle(State(messages=initial_messages))

    # Return the audio file as a response
    return FileResponse(
        FILE_NAME,
        media_type="audio/wav",
        filename="output_audio.wav"
    )