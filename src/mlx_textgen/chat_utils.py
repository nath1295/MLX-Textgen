from mlx_lm.tokenizer_utils import TokenizerWrapper
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase
import json
from typing import List, Dict, Optional, Any, Literal, Union

def get_chat_prompt(msgs: List[Dict[str, str]], tokenizer: TokenizerWrapper) -> str:
    last_role = msgs[-1]['role']
    if last_role != 'assistant':
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    else:
        last_content = msgs[-1]['content']
        lc_strip = last_content.strip()
        wa = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        woa = tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
        diff = wa.removeprefix(woa)
        last_chunk = lc_strip
        if last_content in diff:
            last_chunk = last_content
        pref = ''.join(diff.split(last_chunk)[:-1]) + last_content.lstrip()
        return woa + pref
    
def convert_tool_to_json_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
    params_json = tool.get('function', dict()).get('parameters')
    if params_json is not None:
        return params_json
    else:
        return tool
    
def get_tool_name(tool: Dict[str, Any]) -> str:
    # OpenAI format
    name = tool.get('function', dict()).get('name')
    if name is None:
        name = tool.get('title')
    return name
    
TOOL_LIST = [{'properties': {'example_arg': {'title': 'Example Arg', 'type': 'string'}}, 'required': ['example_arg'], 'title': 'ExampleTool', 'type': 'function'}]
TOOL_CALL_LIST = [dict(function=dict(name='ExampleTool', arguments=dict(example_arg='example')))]
MSG_WITH_TOOL = [
    dict(role='user', content='Hello'),
    dict(role='assistant', tool_calls=TOOL_CALL_LIST)
]
MSG_SINGLE_ASSISTANT = [dict(role='assistant', content='')]
DEFAULT_TOOL_SYSTEM = '''

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
$$tool_list$$
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>'''

class ToolCallContent(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCall(BaseModel):
    function: ToolCallContent

class ChatMessage(BaseModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: Optional[Any] = None
    tool_call: Optional[List[ToolCall]] = None


class ChatTemplate:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    @property
    def support_tool_call(self) -> bool:
        if not hasattr(self, '_support_tool_call'):
            try:
                p_with_tool = self.tokenizer.apply_chat_template(MSG_WITH_TOOL[:1], tools=TOOL_LIST, add_generation_prompt=True, tokenize=False)
                p_wo_tool = self.tokenizer.apply_chat_template(MSG_WITH_TOOL[:1], add_generation_prompt=True, tokenize=False)
                self._support_tool_call = p_with_tool != p_wo_tool
            except:
                self._support_tool_call = False
        return self._support_tool_call
    
    @property
    def allow_multiple_assistant(self) -> bool:
        if not hasattr(self, '_allow_multiple_assistant'):
            try:
                self.tokenizer.apply_chat_template(MSG_SINGLE_ASSISTANT * 2, tokenize=False, add_generation_prompt=False, continue_final_message=True)
                self._allow_multiple_assistant = True
            except:
                self._allow_multiple_assistant = False
        return self._allow_multiple_assistant

    @property
    def tool_start(self) -> str:
        if not hasattr(self, '_tool_start'):
            if self.support_tool_call:
                p_with_tool = self.tokenizer.apply_chat_template(MSG_WITH_TOOL, tools=TOOL_LIST, add_generation_prompt=True, tokenize=False)
                p_wo_tool = self.tokenizer.apply_chat_template(MSG_WITH_TOOL[:1], tools=TOOL_LIST, add_generation_prompt=False, tokenize=False)
                diff_str = p_with_tool.removeprefix(p_wo_tool)
                tool_first_index = diff_str.find('{')
                self._tool_start = diff_str[:tool_first_index]
            else:
                self._tool_start = '<tool_call>\n'
        return self._tool_start
    
    def _validate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for msg in messages:
            ChatMessage(**msg)
        if len(messages) == 0:
            raise Exception(f'Must have at least one message.')
        return messages
    
    def _validate_message_seq(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.allow_multiple_assistant and self.support_tool_call:
            return messages
        elif self.allow_multiple_assistant and (not self.support_tool_call):
            new_messages = []
            for msg in messages:
                if (msg['role'] == 'assistant') and msg.get('tool_call'):
                    tool_call = '\n'.join(['<tool_call>\n' + json.dumps(tc) + '\n</tool_call>' for tc in msg['tool_call']])
                    new_messages.append(dict(role='assistant', content=msg.get('content', '') + tool_call))
                elif (msg['role'] == 'tool'):
                    content = msg.get('content', '')
                    content = content if isinstance(content, str) else json.dumps(content)
                    content = '<tool_response>\n' + content + '\n</tool_response>'
                    new_messages.append(dict(role='user', content=content))
                else:
                    new_messages.append(msg)
            return new_messages
        elif self.support_tool_call:
            new_messages = []
            last_role = None
            for msg in messages:
                if (msg['role'] == 'assistant') and last_role != 'user':
                    new_messages.extend([dict(role='user', content=''), msg])
                else:
                    new_messages.append(msg)
                last_role = msg['role']
            return new_messages
        else:
            new_messages = []
            last_role = None
            for msg in messages:
                if (msg['role'] == 'assistant') and msg.get('tool_call'):
                    tool_call = '\n'.join(['<tool_call>\n' + json.dumps(tc) + '\n</tool_call>' for tc in msg['tool_call']])
                    to_append = dict(role='assistant', content=msg.get('content', '') + tool_call)
                elif (msg['role'] == 'tool'):
                    content = msg.get('content', '')
                    content = content if isinstance(content, str) else json.dumps(content)
                    content = '<tool_response>\n' + content + '\n</tool_response>'
                    to_append = dict(role='user', content=content)
                else:
                    to_append = msg
                if (to_append['role'] == 'assistant') and last_role != 'user':
                    new_messages.extend([dict(role='user', content=''), to_append])
                else:
                    new_messages.append(to_append)
                last_role = to_append['role']
            return new_messages
                    
    def apply_chat_template(self, 
            messages: List[Dict[str, Any]], 
            tools: Optional[List[Dict[str, Any]]] = None, 
            tool_choice: Union[Literal['none', 'auto', 'required'], Dict[str, Union[str, Dict[str, str]]]] = 'auto',
            add_generation_prompt: bool = True
            ) -> str:
        messages = self._validate_messages(messages=messages)
        messages = self._validate_message_seq(messages=messages)
        tools = tools if tool_choice != 'none' else None
        continue_final_message = False
        if messages[-1]['role'] =='assistant':
            continue_final_message = True
            add_generation_prompt = False
        if tools and (not self.support_tool_call):
            tool_json = '\n'.join([json.dumps(tool) for tool in tools])
            if messages[0]['role'] == 'system':
                content = messages[0].get('content', '') + DEFAULT_TOOL_SYSTEM.replace('$$tool_list$$', tool_json)
                messages[0]['content'] = content.strip()
            else:
                messages = [dict(role='system', content=DEFAULT_TOOL_SYSTEM.replace('$$tool_list$$', tool_json).strip())] + messages
        prompt = self.tokenizer.apply_chat_template(conversation=messages, 
            tools=tools if self.support_tool_call else None, tokenize=False, 
            add_generation_prompt=add_generation_prompt, 
            continue_final_message=continue_final_message)
        if tool_choice == 'required':
            prompt += self.tool_start + '{"function": {' + '"name": "'
        elif isinstance(tool_choice, dict):
            tool_name = tool_choice['function']['name']
            prompt += self.tool_start + '{"function": {' + f'"name": "{tool_name}", arguments": '
        return prompt


