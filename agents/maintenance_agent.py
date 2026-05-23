from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(
    temperature=0
)

telemetry_summary = '''
Motor vibration is increasing.
Battery temperature is high.
State of Health is degrading.
Failure probability is 0.84.
'''

prompt = f'''
Analyze this telemetry report.
Explain probable root cause.
Suggest maintenance actions.

{telemetry_summary}
'''

response = llm([
    HumanMessage(content=prompt)
])

print(response.content)
