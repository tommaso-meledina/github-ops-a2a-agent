from a2a.types import AgentSkill, AgentCard, AgentCapabilities, SecurityScheme
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

import uvicorn

from src.auth_middleware import BearerAuthMiddleware
from src.github_agent_executor import GitHubAgentExecutor


def main():
    read_issue_skill = AgentSkill(
        id="read_issue",
        name="Read GitHub Issue",
        description="Read a GitHub issue given its URL",
        tags=["github", "issue"],
        examples=[
            '{"skill_id": "read_issue", "issue_url": "https://github.com/org/repo/issues/1"}'
        ]
    )

    open_pr_skill = AgentSkill(
        id="open_pr",
        name="Open Pull Request",
        description="Open a pull request on a GitHub repository",
        tags=["github", "pull-request"],
        examples=[
            '{"skill_id": "open_pr", "repo_url": "https://github.com/org/repo", "branch": "feature", "title": "Title", "body": "Description"}'
        ]
    )

    agent_card = AgentCard(
        name="GitHub A2A Agent",
        description="An A2A agent exposing GitHub issue and PR skills",
        url="http://localhost:9999/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[read_issue_skill, open_pr_skill],
        default_input_modes=["application/json", "application/text", "text/event-stream"],
        default_output_modes=["application/json", "application/text", "text/event-stream"],
        security=[{ "bearerAuth": [] }]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=GitHubAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    ).build()

    server.add_middleware(BearerAuthMiddleware)

    uvicorn.run(server, host='0.0.0.0', port=9999)


if __name__ == "__main__":
    main()
