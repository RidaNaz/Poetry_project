import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from google.genai import Client
from typing import Literal, Union, Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
import fitz  # PyMuPDF for PDF text extraction
from PIL import Image
import pytesseract  # For OCR text extraction from images

router = APIRouter()

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

client = Client(
  http_options={'api_version': 'v1alpha'}
)

MODEL: str = "gemini-2.0-flash-exp"

config = {
  "configurable": {"thread_id": "1"},
  "generation_config": {"response_modalities": ["TEXT"]}
}

memory: MemorySaver = MemorySaver()

# State

class State(MessagesState):
    summary: str

# Summarization

async def summarize_conversation(state: State) -> Dict[str, object]:
    """
    Summarizes the conversation if the number of messages exceeds 6 messages.
    """
    summary = state.get("summary", [])
    
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

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

    summary = response_text
    delete_messages = [RemoveMessage(id=getattr(m, "id", None)) for m in state["messages"][:-2]]

    return {"summary": response_text, "messages": delete_messages}

# Conditional Function

def select_next_node(state: State) -> Union[Literal["summarize"], str]:
    messages = state["messages"]

    if len(messages) > 6:
        return "summarize"
    return END

async def call_model(state: State):
    """
    Handles the model invocation using the Gemini 2.0 Live API.
    """
    if "messages" not in state:
        raise ValueError("State must contain a 'messages' key.")

    messages = state["messages"]
    message_texts = [message.content for message in messages]

    summary = state.get("summary", "")
    if summary:
        message_texts.insert(0, f"Summary of conversation earlier: {summary}")

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        try:
            await session.send(input=" ".join(message_texts), end_of_turn=True)
            turn = session.receive()
            response_text = ""

            async for chunk in turn:
                if chunk.text is not None:
                    response_text += chunk.text

        except Exception as e:
            raise RuntimeError(f"Error invoking the model: {e}")

    response_message = AIMessage(content=response_text)
    messages.append(response_message)

    return {"messages": messages[-1]}

# Helper functions to extract text from PDF and images

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Build Graph

builder = StateGraph(State)

builder.add_node("agent", call_model)
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
async def run(message: str = None, file: UploadFile = File(None)):
    # Ensure the temp_files directory exists
    os.makedirs("./temp_files", exist_ok=True)

    # Validate input: ensure at least one of message or file is provided
    if not message and not file:
        raise HTTPException(status_code=400, detail="Either 'message' or 'file' must be provided.")
    
    # Initialize conversation messages
    initial_messages = []

    # Check if a file is provided
    if file:
        file_extension = file.filename.split('.')[-1].lower()

        # Save the file temporarily
        file_path = os.path.join("temp_files", file.filename)  # Path where the file will be saved
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract text from PDF or image
        extracted_text = ""
        if file_extension == "pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif file_extension in ["png", "jpg", "jpeg"]:
            extracted_text = extract_text_from_image(file_path)
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Only PDF, PNG, JPG, and JPEG are accepted."
            )

        # Combine the extracted text with the message if both are provided
        if message:
            combined_message = f"{message}\n\nExtracted from file:\n{extracted_text}"
        else:
            combined_message = extracted_text

        # Add the combined message to the conversation
        initial_messages.append(HumanMessage(content=combined_message))

        # Clean up the temporary file
        os.remove(file_path)

    # Process the text message, if provided
    if message and not file:
        initial_messages = message
    
    # If only a message is provided, initial_messages already has it

    # Invoke the state graph with the initial messages
    try:
        event = await graph.ainvoke({"messages": initial_messages}, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph invocation failed: {e}")

    return event
