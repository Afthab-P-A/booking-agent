from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from datetime import datetime
import dateparser
import pandas as pd
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, List
import json
from langchain.agents import tool

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages:Annotated[list,add_messages]
    
import os
from dotenv import load_dotenv
print(load_dotenv())

from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

llm=ChatGroq(model="llama3-8b-8192")

def chatbot(state:State):
    return {"messages":[llm.invoke(state["messages"])]}

import pandas as pd
df = pd.read_excel("schedule.xlsx")

from langchain.tools import tool

@tool
def check_availability(employee: str = "all", date: str = "today") -> str:
    """
    Returns scheduled tasks for a given employee or all employees on a given date (e.g., 'today', 'tomorrow').
    If employee='all', it checks for all employees.
    """

    # Parse natural language date
    parsed_date = dateparser.parse(date, settings={'PREFER_DATES_FROM': 'future', 'TIMEZONE': 'Asia/Kolkata'})
    if not parsed_date:
        return "Couldn't understand the date. Try 'today', 'tomorrow', or a specific date like 'July 30, 2025'."

    date_str = parsed_date.strftime("%Y-%m-%d")

    result = []

    if employee.lower() == "all":
        all_emps = df["employee_name"].unique()
        for emp in all_emps:
            tasks = df[(df["employee_name"] == emp) & (df["date"] == date_str)]
            if not tasks.empty:
                for _, row in tasks.iterrows():
                    result.append(f"{row['employee_name']} has {row['task']} at {row['time']}")
            else:
                result.append(f"{emp} has no tasks on {date_str}.")
    else:
        emp = employee.strip().lower()
        tasks = df[(df["employee_name"].str.lower() == emp) & (df["date"] == date_str)]
        if tasks.empty:
            return f"{employee.title()} has no tasks on {date_str}."
        for _, row in tasks.iterrows():
            result.append(f"{row['employee_name']} has {row['task']} at {row['time']}")

    return " | ".join(result)


from datetime import datetime, timedelta

@tool
def assign_task(employee: str, task: str, date: str = "tomorrow", time: str = "11:00 AM") -> str:
    """
    Assign a task to an employee on a given date and time.
    - If employee is already booked at or 1 hour before, suggest alternative staff.
    - If free, assign and update schedule.xlsx file.
    """

    # Parse date
    parsed_date = dateparser.parse(date, settings={'PREFER_DATES_FROM': 'future', 'TIMEZONE': 'Asia/Kolkata'})
    if not parsed_date:
        return "❌ Couldn't understand the date. Try 'tomorrow' or 'August 1 2025'."

    date_str = parsed_date.strftime("%Y-%m-%d")

    # Parse time input
    try:
        task_time = datetime.strptime(time.strip(), "%I:%M %p")
    except:
        return "❌ Invalid time format. Use something like '11:00 AM'."

    # Load Excel file each time
    df = pd.read_excel("schedule.xlsx")

    # Clean employee name
    employee = employee.strip().lower()

    # Convert existing time strings for comparison
    df["parsed_time"] = pd.to_datetime(df["time"], format="%I:%M %p", errors="coerce").dt.time

    # Filter employee's tasks on same date
    emp_tasks = df[(df["employee_name"].str.lower() == employee) & (df["date"] == date_str)]

    for _, row in emp_tasks.iterrows():
        if pd.isna(row["parsed_time"]):
            continue

        scheduled_time = datetime.combine(parsed_date.date(), row["parsed_time"])
        task_dt = datetime.combine(parsed_date.date(), task_time.time())
        time_diff = abs((task_dt - scheduled_time).total_seconds()) / 60  # in minutes

        if time_diff <= 60:
            # Conflict found
            all_emps = df["employee_name"].str.title().unique().tolist()
            booked = df[(df["date"] == date_str) & (df["time"].str.strip().str.lower() == time.strip().lower())]
            booked_emps = booked["employee_name"].str.title().tolist()
            available_emps = [e for e in all_emps if e not in booked_emps]

            conflict_msg = f"⚠️ {employee.title()} is already booked for '{row['task']}' at {row['time']} on {date_str}."
            if available_emps:
                return f"{conflict_msg} Available employees at {time} are: {', '.join(available_emps)}"
            else:
                return f"{conflict_msg} No other employees are free at {time} on {date_str}."

    # No conflict — assign task
    new_row = {
        "employee_name": employee.title(),
        "date": date_str,
        "time": time.strip(),
        "task": task.strip()
    }

    new_df = pd.concat([df.drop(columns=["parsed_time"]), pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_excel("schedule.xlsx", index=False)

    return f"✅ Assigned '{task}' to {employee.title()} at {time} on {date_str}."


@tool
def remove_task(employee: str, time: str = None, date: str = "today") -> str:
    """
    Removes a task assigned to an employee at a specific date and time (or all tasks on that date).
    Updates the 'schedule.xlsx' file accordingly.
    """

    # Load current data
    try:
        df = pd.read_excel("schedule.xlsx")
    except Exception as e:
        return f"Error reading schedule file: {str(e)}"

    employee = employee.strip().lower()

    # Parse date
    parsed_date = dateparser.parse(date, settings={'PREFER_DATES_FROM': 'future', 'TIMEZONE': 'Asia/Kolkata'})
    if not parsed_date:
        return "Couldn't understand the date. Try 'today', 'tomorrow', or 'August 1, 2025'."
    date_str = parsed_date.strftime("%Y-%m-%d")

    # Filter tasks
    df["employee_name_lower"] = df["employee_name"].str.lower()
    mask = (df["employee_name_lower"] == employee) & (df["date"] == date_str)

    if time:
        try:
            formatted_time = datetime.strptime(time.strip(), "%I:%M %p").strftime("%I:%M %p")
            mask &= (df["time"].str.strip().str.lower() == formatted_time.strip().lower())
        except:
            return "Invalid time format. Use like '12:30 PM'."

    removed = df[mask]

    if removed.empty:
        return f"No matching task found for {employee.title()} on {date_str} {'at ' + time if time else ''}."

    # Drop and save
    df = df[~mask].drop(columns=["employee_name_lower"])
    df.to_excel("schedule.xlsx", index=False)

    if time:
        return f"✅ Task for {employee.title()} at {time} on {date_str} has been removed."
    else:
        return f"✅ All tasks for {employee.title()} on {date_str} have been removed."
    
    
tools=[check_availability,assign_task,remove_task]

llm_with_tools=llm.bind_tools(tools)

from langgraph.prebuilt import ToolNode,tools_condition

##NODE DEFINITION
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

#Graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

#Add Edges
builder.add_edge(START,"tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition
)
builder.add_edge("tools","tool_calling_llm")

#compile graph
graph= builder.compile()
from langchain_core.messages import HumanMessage



user_input = ""  # example input

result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
for m in result['messages']:
    m.pretty_print()
# print(result)