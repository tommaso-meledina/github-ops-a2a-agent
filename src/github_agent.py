import os
from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, SecretStr

from src.util.github_util import build_github_client
from src.util.parse_env import AgentEnvironment, parse_env

memory = MemorySaver()

environment: AgentEnvironment = parse_env()

@tool
def read_github_issue(
        repo_owner: str,
        repo_name: str,
        issue_number: int
) -> dict:
    """
    Read details of a GitHub issue given its numeric ID, the name of the repository hosting it, and the name of the owner of the repo.
    Returns a JSON-like dict with key fields.

    Args:
        repo_owner: owner of the GitHub repository hosting the issue (required)
        repo_name: name of the GitHub repository hosting the issue (required)
        issue_number: GitHub Issue's numeric ID (required)
    """
    gh = build_github_client()
    retrieved_repo = gh.get_repo(f"{repo_owner}/{repo_name}")
    issue = retrieved_repo.get_issue(number=issue_number)
    return {
        "url": issue.html_url,
        "title": issue.title,
        "body": issue.body,
        "state": issue.state,
        "created_at": issue.created_at.isoformat(),
        "updated_at": issue.updated_at.isoformat(),
    }


@tool
def open_github_pr(
        repo_owner: str,
        repo_name: str,
        head_branch: str,
        base_branch: str,
        title: str | None = None,
        body: str | None = None,
        related_issue: int | None = None
) -> dict:
    """
    Create a pull request on GitHub. Either the `related_issue` parameter or the `title` and `body` parameters are required.

    Args:
      repo_owner: owner of the GitHub repository where the PR should be created
      repo_name: name of the GitHub repository where the PR should be created
      head_branch: The name of the branch containing the commits
      base_branch: The branch you want to merge into
      title: Pull request title; not honored if "related_issue" is provided (optional)
      body: Pull request body text; not honored if "related_issue" is provided (optional)
      related_issue: ID of the issue that the PR relates to (optional)

    Returns:
      A dict with details of the created PR.
    """
    gh = build_github_client()
    retrieved_repo = gh.get_repo(f"{repo_owner}/{repo_name}")

    retrieved_issue = retrieved_repo.get_issue(number=related_issue) if related_issue is not None else None

    pr = retrieved_repo.create_pull(
        title=title,
        body=body,
        head=head_branch,
        base=base_branch
    ) if related_issue is None else retrieved_repo.create_pull(
        head=head_branch,
        base=base_branch,
        issue=retrieved_issue
    )

    return {
        "url": pr.html_url,
        "title": pr.title,
        "state": pr.state,
        "created_at": pr.created_at.isoformat(),
        "head_ref": pr.head.ref,
        "base_ref": pr.base.ref,
    }


class AgentResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class GitHubAgent:
    """GitHubAgent - a specialized assistant for reading issues and writing PRs on GitHub."""

    SYSTEM_INSTRUCTION = (
        'You are a specialized assistant for GitHub operations. '
        "Your sole purpose is to either use the 'read_github_issue' tool to read the contents of a GitHub issue, "
        "or use the 'open_github_pr' to open a Pull Request on GitHub. "
        'If the user asks about anything other than these two operations, '
        'politely state that you cannot help with that topic and can only assist with the aforementioned operations. '
        'Do not attempt to answer unrelated questions or use tools for other purposes. '
        'Make sure that you are explicitly given all the necessary input parameters for the tools you can use; if one or more parameters are missing, ask the user to provide them. '
        'Never use your tools unless the user explicitly gave you all the necessary input parameters for those tools. '
        'When answering, set response status to input_required if the user needs to provide more information to complete the request. '
        'set response status to error if there is an error while processing the request. '
        'Set response status to completed if the request is complete.'
    )

    def __init__(self):
        model_source = environment.llm_source
        if model_source == 'google':
            self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        else:
            self.model = ChatOpenAI(
                model=os.getenv('TOOL_LLM_NAME'),
                api_key=SecretStr(environment.llm_api_key),
                base_url=os.getenv('TOOL_LLM_URL'),
                temperature=0,
            )
        self.tools = [
            read_github_issue,
            open_github_pr
        ]

        self.graph = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            system_prompt=self.SYSTEM_INSTRUCTION,
            response_format=AgentResponseFormat
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = Command(
            update={
                "messages": [HumanMessage(content=query)]
            }
        )
        config: RunnableConfig = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Invoking GitHub API...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing GitHub API response...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, AgentResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }