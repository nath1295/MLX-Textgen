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
        if 'title' not in params_json.keys():
            params_json['title'] = tool['function']['name']
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
    arguments: str

class ToolCall(BaseModel):
    function: ToolCallContent

class ChatMessage(BaseModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: Optional[Any] = None
    tool_calls: Optional[List[ToolCall]] = None


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
    def support_system(self) -> bool:
        if not hasattr(self, '_support_system'):
            try:
                system = 'Test system message, see if exist.'
                messages = [dict(role='system', content=system), dict(role='user', content='Hi there')]
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                self._support_system = system in prompt
            except:
                self._support_system = False
        return self._support_system

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
        def format_tool_call(tc):
            call = tc['function']
            try:
                args = json.loads(call['arguments'])
            except:
                args = call['arguments']
            call['arguments'] = args
            return json.dumps(call)
        if self.allow_multiple_assistant and self.support_tool_call:
            new_messages = []
            for msg in messages:
                if msg.get('tool_calls') is not None:
                    msg['tool_calls'] = [format_tool_call(tc) for tc in msg['tool_calls']]
                    new_messages.append(msg)
                else:
                    new_messages.append(msg)
        elif self.allow_multiple_assistant and (not self.support_tool_call):
            new_messages = []
            for msg in messages:
                if (msg['role'] == 'assistant') and msg.get('tool_calls'):
                    tool_call = '\n'.join(['<tool_call>\n' + format_tool_call(tc) + '\n</tool_call>' for tc in msg['tool_calls']])
                    content = '' if msg.get('content') is None else msg.get('content')
                    new_messages.append(dict(role='assistant', content=content + tool_call))
                elif (msg['role'] == 'tool'):
                    content = '' if msg.get('content') is None else msg.get('content')
                    content = content if isinstance(content, str) else json.dumps(content)
                    content = '<tool_response>\n' + content + '\n</tool_response>'
                    new_messages.append(dict(role='user', content=content))
                else:
                    new_messages.append(msg)
        elif self.support_tool_call:
            new_messages = []
            last_role = None
            for msg in messages:
                if (msg['role'] == 'assistant') and last_role != 'user':
                    if msg.get('tool_calls') is not None:
                        msg['tool_calls'] = [format_tool_call(tc) for tc in msg['tool_calls']]
                    new_messages.extend([dict(role='user', content=''), msg])
                else:
                    new_messages.append(msg)
                last_role = msg['role']
        else:
            new_messages = []
            last_role = None
            for msg in messages:
                if (msg['role'] == 'assistant') and msg.get('tool_calls'):
                    tool_call = '\n'.join(['<tool_call>\n' + format_tool_call(tc) + '\n</tool_call>' for tc in msg['tool_calls']])
                    content = '' if msg.get('content') is None else msg.get('content')
                    to_append = dict(role='assistant', content=content + tool_call)
                elif (msg['role'] == 'tool'):
                    content = '' if msg.get('content') is None else msg.get('content')
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
        if not self.support_system and (new_messages[0]['role'] == 'system'):
            system = new_messages[0]['content']
            new_messages = new_messages[1:]
            if (len(new_messages) > 0) and (new_messages[0]['role'] == 'user'):
                first_message = new_messages[0]['content']
                first_message = '<system>\n' + system + '\n</system>\n\n' + first_message
                new_messages[0]['content'] = first_message
            else:
                raise Exception(f'First message after the system message is not a user message.')
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
            prompt += self.tool_start + '{"name": "'
        elif isinstance(tool_choice, dict):
            tool_name = tool_choice['function']['name']
            prompt += self.tool_start + '{' + f'"name": "{tool_name}", arguments": '
        return prompt


