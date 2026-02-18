import os
import logging
import google.cloud.logging


from callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv


from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.models import Gemini
from google.genai import types
from google.adk.tools import exit_loop


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()


load_dotenv()


model_name = os.getenv("MODEL", "gemini-1.5-flash-001")


RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)


# Tools




def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """Append new output to an existing state key.


    Args:
        field (str): a field name to append to
        response (str): a string to append to the field


    Returns:
        dict[str, str]: {"status": "success"}
    """
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}




def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:
    """Writes content to a file in a specified directory.

    Args:
        directory (str): The directory to write the file in.
        filename (str): The name of the file.
        content (str): The content to write to the file.

    Returns:
        dict[str, str]: {"status": "success"}
    """
    target_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding='utf-8') as f:
        f.write(content)
    logging.info(f"File '{target_path}' written successfully.")
    return {"status": "success"}




# Agents

# Step 2: The Investigation (Parallel)
admirer = Agent(
    name="The_Admirer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="ค้นหาและรวบรวมข้อมูลเกี่ยวกับความสำเร็จและด้านบวกของหัวข้อที่กำหนด",
    instruction="""
    คุณคือ 'ผู้ชื่นชม' มีหน้าที่ค้นหาข้อมูลด้านบวกและความสำเร็จของ '{PROMPT?}' จาก Wikipedia
    ใช้เครื่องมือ Wikipedia เพื่อค้นหาหัวข้อ เช่น '{PROMPT} achievements', '{PROMPT} success', '{PROMPT} legacy'
    จากนั้นใช้เครื่องมือ 'append_to_state' เพื่อบันทึกข้อมูลที่ค้นพบลงใน state key ชื่อ 'pos_data'
    สรุปข้อมูลที่ค้นพบสั้นๆ เป็นภาษาไทย
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)

critic = Agent(
    name="The_Critic",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="ค้นหาและรวบรวมข้อมูลเกี่ยวกับข้อผิดพลาด, ด้านลบ, และข้อโต้แย้งของหัวข้อที่กำหนด",
    instruction="""
    คุณคือ 'ผู้วิจารณ์' มีหน้าที่ค้นหาข้อมูลด้านลบ ข้อผิดพลาด และข้อโต้แย้งของ '{PROMPT?}' จาก Wikipedia
    ใช้เครื่องมือ Wikipedia เพื่อค้นหาหัวข้อ เช่น '{PROMPT} controversy', '{PROMPT} failures', '{PROMPT} criticism'
    จากนั้นใช้เครื่องมือ 'append_to_state' เพื่อบันทึกข้อมูลที่ค้นพบลงใน state key ชื่อ 'neg_data'
    สรุปข้อมูลที่ค้นพบสั้นๆ เป็นภาษาไทย
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)

investigation_team = ParallelAgent(
    name="Investigation_Team",
    sub_agents=[
        admirer,
        critic
    ]
)

# Step 3: The Trial & Review (Loop)
judge = Agent(
    name="The_Judge",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="ตรวจสอบข้อมูลจากทั้งสองฝ่ายและตัดสินใจว่าจะค้นหาข้อมูลเพิ่มเติมหรือยุติการพิจารณา",
    instruction="""
    คุณคือ 'ผู้พิพากษา' มีหน้าที่ตรวจสอบข้อมูลที่รวบรวมมา
    ข้อมูลด้านบวก: {pos_data?}
    ข้อมูลด้านลบ: {neg_data?}

    พิจารณาว่าข้อมูลจากทั้งสองฝ่ายมีความสมดุลและเพียงพอหรือไม่
    - หากข้อมูลฝั่งใดฝั่งหนึ่งน้อยเกินไปหรือไม่สมดุล ให้สั่งให้ทีมกลับไปค้นหาข้อมูลเพิ่มเติมโดยระบุสิ่งที่ต้องการค้นหาเพิ่ม
    - หากข้อมูลสมดุลและเพียงพอแล้ว ให้ใช้เครื่องมือ 'exit_loop' เพื่อยุติการพิจารณาคดี
    อธิบายการตัดสินใจของคุณเป็นภาษาไทย
    """,
    tools=[exit_loop]
)

trial_and_review = LoopAgent(
    name="Trial_and_Review",
    description="วนลูปการสืบสวนและพิจารณาจนกว่าจะได้ข้อมูลที่สมดุล",
    sub_agents=[
        investigation_team,
        judge
    ],
    max_iterations=3,
)

# Step 4: The Verdict (Output)
verdict_writer = Agent(
    name="Verdict_Writer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="สรุปรายงานเปรียบเทียบข้อเท็จจริงและบันทึกเป็นไฟล์",
    instruction="""
    คุณมีหน้าที่สรุปผลการพิจารณาคดีเกี่ยวกับ '{PROMPT?}'
    สร้างรายงานเปรียบเทียบข้อเท็จจริงจากข้อมูลทั้งสองด้านที่ได้รับมาอย่างเป็นกลางและสรุปเป็นภาษาไทย

    ข้อมูลด้านบวก:
    {pos_data?}

    ข้อมูลด้านลบ:
    {neg_data?}

    จากนั้นใช้เครื่องมือ 'write_file' เพื่อบันทึกรายงานสรุปผลลงในไฟล์ .txt
    - directory: 'historical_verdicts'
    - filename: ใช้ชื่อหัวข้อ '{PROMPT?}' ตามด้วย '.txt' (เช่น 'Genghis Khan.txt')
    - content: เนื้อหารายงานสรุปผลทั้งหมด
    """,
    tools=[write_file],
)

# Step 1: The Inquiry (Sequential Root Agent)
historical_court_root = SequentialAgent(
    name="Historical_Court",
    description="ระบบศาลจำลองประวัติศาสตร์",
    sub_agents=[
        trial_and_review,
        verdict_writer
    ],
)


root_agent = Agent(
    name="Inquiry_Clerk",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="รับเรื่องราวและเริ่มกระบวนการศาลจำลองประวัติศาสตร์",
    instruction="""
    คุณคือเสมียนศาลประวัติศาสตร์
    - ทักทายผู้ใช้และแจ้งว่าคุณสามารถช่วยวิเคราะห์บุคคลหรือเหตุการณ์ในประวัติศาสตร์ได้
    - สอบถามผู้ใช้ว่าต้องการให้วิเคราะห์เรื่องอะไร
    - เมื่อผู้ใช้ตอบกลับ ให้ใช้เครื่องมือ 'append_to_state' เพื่อบันทึกคำตอบของผู้ใช้ลงใน state key ชื่อ 'PROMPT'
    - จากนั้นส่งต่อให้ 'Historical_Court' Agent เพื่อเริ่มกระบวนการพิจารณาคดี
    """,
    tools=[append_to_state],
    sub_agents=[historical_court_root],
)