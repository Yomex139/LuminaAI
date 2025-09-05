#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pypdf import PdfReader
from openai import OpenAI
import json
from dotenv import load_dotenv
from pydantic import BaseModel
import requests
import gradio as gr
import logging
from supabase import create_client
from urllib3 import connection_from_url

record_user_details_json = {
    "name": "record_user_details",
    "description": "record the users details in Json formats",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "store the email of the users"
            },
            "name": {
                "type": "string",
                "description": "store the name of users"
            },
            "note": {
                "type": "string",
                "description": "store the extra note that the user input"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "record the unknown question asked by the user",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "store the unknown question asked by the user"
            },
            "email": {
                "type": "string",
                "description": "store the email of the user"
            }
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

class ChatBot:
    # Create a Pydantic model for evaluation
    class Evaluation(BaseModel):
        is_acceptable: bool
        feedback: str
    def __init__(self) -> None:
        # load envrioment variable
        load_dotenv(override=True)

        self.openai = OpenAI()
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.supabase_url=os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase = create_client(supabase_url=self.supabase_url, supabase_key=self.supabase_key)
        self.pushover_url = "https://api.pushover.net/1/messages.json"
        self.linkedin = ""
        self.summary = ""
        self.name = "Oluwayomi Daniel"
        self.tools = [{"type": "function", "function": record_user_details_json},
                      {"type": "function", "function": record_unknown_question_json}]

        self.system_prompt = f"""
                You are acting as {self.name}, a professional representative on {self.name}'s personal website. Your primary role is to answer questions related to {self.name}'s career, background, skills, and experience. You are engaging with potential clients, collaborators, or employers, so maintain a professional, helpful, and personable tone.
                You are provided with a summary of {self.name}'s background and LinkedIn profile to guide your responses. Use this context to stay in character and represent {self.name} faithfully.
                If a user asks a question that is unrelated to {self.name}'s career but is simple and answerable (e.g., general knowledge, travel info, tech tips), you may answer it helpfully. After responding, gently steer the conversation back toward {self.name}'s professional interests or invite the user to get in touch via email.
                If a question is too obscure, highly technical, or outside your scope, use the `record_unknown_question` tool to log it.
                If the user expresses interest in connecting, ask for their email and name and record it using the `record_user_details` tool.
                Always stay in character as {self.name}, and aim to be both informative and engaging.
                ## Summary: {self.summary}
                ## LinkedIn Profile: {self.linkedin}
                With this context, please chat with the user, always staying in character as {self.name}.
                """

    # function to send message
    def send_message_notification(self, message):
        payload = {"user":self.pushover_user, "token":self.pushover_token, "message":message}
        result = requests.post(self.pushover_url, data=payload)
        if result:
            logging.info("Message Pushed")

    def record_user_details(self, email="not provided yet", name="name not provided", note="not provided"):
        self.send_message_notification(f"Recording interest from {name} with email {email} and notes:\n {note}")
        data = {"name": name, "email": email, "note": note}
        self.supabase.table("user_details").insert(data).execute()
        return {"recorded": "ok"}

    def record_unknown_question(self, question):
        self.send_message_notification(f"Recording \n{question}\n asked that I couldn't answer")
        data = {"question": question}
        self.supabase.table("user_questions").insert(data).execute()
        return {"recorded": "ok"}

    def hand_tool_call(self, tool_calls):
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            logging.info(f"Tool called: {tool_name} with arguments: {arguments}")
            tool = getattr(self, tool_name, None)
            if tool is None:
                raise ValueError(f"Tool {tool_name} not found in ChatBot instance.")
            result = tool(**arguments)
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def build_system_prompt(self):
        try:
            reader = PdfReader("me/LinkedIn_Profile_Updated.pdf")
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    self.linkedin += text
        except FileNotFoundError:
            logging.warning("LinkedIn profile PDF not found.")

        try:
            with open("me/summary.txt", "r", encoding="utf-8") as f:
                my_summary = f.read()
                if my_summary:
                    self.summary += my_summary
        except FileNotFoundError:
            logging.warning("Summary file not found.")
        self.system_prompt += f"## Summary:{self.summary}## LinkedIn Profile:{self.linkedin}"
        self.system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return self.system_prompt
    # evaluator
    def evaluator(self, reply, message, history) -> Evaluation:
        # evaluator_system_prompt
        evaluator_system_prompt = f"""You are an evaluator that decides whether a response to a question is acceptable.
        You are provided with a conversation between a User and an Agent. 
        Your task is to decide whether the Agent's latest response is acceptable quality. 

        The Agent is playing the role of {self.name} and is representing {self.name} on their website. 
        The Agent has been instructed to be professional and engaging, 
        as if talking to a potential client or future employer who came across the website. 

        The Agent has been provided with context on {self.name} in the form of their summary and LinkedIn details. 

        ## Summary:
        {self.summary}

        ## LinkedIn Profile:
        {self.linkedin}

        With this context, please evaluate the latest response, replying with whether it is acceptable and your feedback.
        """

        # evaluate user prompt
        user_prompt = f"""Here's the conversation between the User and Agent:

        Conversation history:
        {history}

        Latest response from Agent:
        {reply}

        Latest message from User:
        {message}

        Please evaluate the response, replying with whether it is acceptable and your feedback.
        """

        # build the message
        messages = [{"role": "system", "content": evaluator_system_prompt},
                    {"role": "user", "content": user_prompt}]
        # define evaluator
        gemini = OpenAI(api_key=self.gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

        response = gemini.beta.chat.completions.parse(model="gemini-2.0-flash",
                                                      messages=messages,
                                                      response_format="json")
        parsed = ChatBot.Evaluation(**response.choices[0].message.content)
        return parsed
    #rerun if response is not validated by validator
    def rerun(self, reply, message, history, feedback):
        updated_system_prompt = self.system_prompt + "## Previous answer rejected\nYou just tried to reply, but \
        the quality control rejected your reply"
        updated_system_prompt += f"## Your attempted answer: {reply}"
        updated_system_prompt += f"## Reason for rejection: {feedback}"
        messages = [{"role":"system", "content":updated_system_prompt}] + history + [{"role":"user", "content":message}]
        response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content

    def stream_response(self, message, history):
        """
        Stream assistant responses token-by-token for Gradio.
        If a tool_call is detected, resolve it afterward and trigger notifications.
        """
        # Build chat history
        messages = [{"role": "system", "content": self.system_prompt}]
        for user_msg, bot_msg in history or []:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})

        # Stream from OpenAI
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tools,
            stream=True
        )

        partial, tool_call_detected = "", False

        try:
            # 1) Stream text as it arrives
            for chunk in response:
                delta = chunk.choices[0].delta

                # Detect tool call fragments
                if getattr(delta, "tool_calls", None) or (
                    isinstance(delta, dict) and delta.get("tool_calls")
                ):
                    tool_call_detected = True
                    continue  # skip tool fragments in stream

                # Extract text
                content = getattr(delta, "content", None) or (
                    delta.get("content") if isinstance(delta, dict) else None
                )
                if content:
                    partial += content
                    yield partial  # push accumulated text to Gradio

            # 2) If a tool call was detected, resolve it with a non-streaming call
            if tool_call_detected:
                final_resp = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=self.tools,
                )
                tool_calls = final_resp.choices[0].message.tool_calls
                if tool_calls:
                    self.hand_tool_call(tool_calls)  # triggers notification
                    yield partial + "📩 I’ve recorded your details and sent a notification. Do you have any other questions?"
                else:
                    yield partial + " m.⚠️ Tool call suspected but missing in final response."

            # 3) If nothing was streamed at all
            if not partial and not tool_call_detected:
                yield "I couldn't generate a text reply for that request."

        except Exception as e:
            logging.exception("Streaming error")
            yield f"⚠️ Streaming error: {type(e).__name__}: {e}"






# ----------------- Gradio Integration -----------------

import gradio as gr

# Instantiate your bot
bot = ChatBot()
bot.build_system_prompt()


def respond(message, history):
    """
    Gradio expects a generator that yields strings.
    DO NOT return the generator object — re-yield its values.
    """
    for partial in bot.stream_response(message, history):
        yield partial

demo=gr.ChatInterface(
    fn=respond,
    title="🤖 Oluwayomi Daniel's AI Assistant",
    description="Ask me anything about Oluwayomi Daniel's career, skills, background or any question. I'm here to help!",
    examples=[
        "What is Oluwayomi's experience with ML and AI",
        "Can I get in touch with Oluwayomi?",
        "Tell me about his recent projects."
    ]
)
if __name__ == "__main__":
    demo.launch()

