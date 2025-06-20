import os
from typing import List

from dotenv import load_dotenv
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description, Tool
from langchain_openai import ChatOpenAI

@tool
def get_text_length(text: str) -> int:
    """ Returns the length of the text by characters """
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool not found with name {tool_name}")


if __name__ == "__main__":
    load_dotenv()
    print("Welcome to ReAct LangChain!")

    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = (PromptTemplate.from_template(template=template)
              .partial(
                    tools=render_text_description(tools),
                    tool_names=", ".join(t.name for t in tools)
              )
    )

    intermediate_steps = []

    llm = ChatOpenAI(temperature=0, model=os.environ["MODEL"], stop_sequences=["\nObservation"])
    agent = {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
            } | prompt | llm | ReActSingleInputOutputParser()

    agent_step: AgentAction | AgentFinish = agent.invoke(
        {
            "input": "Text Length for DOG?",
            "agent_scratchpad": intermediate_steps
        }
    )

    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")

        intermediate_steps.append((agent_step, str(observation)))

    agent_step: AgentAction | AgentFinish = agent.invoke(
        {
            "input": "Text Length for DOG?",
            "agent_scratchpad": intermediate_steps
        }
    )

    print(agent_step)

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values )