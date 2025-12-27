import os
from dataclasses import dataclass
from dotenv import load_dotenv

DEFAULT_GITHUB_HOST = 'github.com'
DEFAULT_LLM_SOURCE = 'google'

@dataclass(slots=True)
class AgentEnvironment:
    github_token: str
    github_repo: str
    github_owner: str
    github_host: str
    llm_source: str
    llm_api_key: str

def parse_env() -> AgentEnvironment:
    load_dotenv()
    # TODO: can do better
    for required_env in ["GITHUB_TOKEN"]:
        if not os.environ.get(required_env):
            raise EnvironmentError(f"Missing required environment variable: {required_env}")
    # noinspection PyTypeChecker
    return AgentEnvironment(
        github_token=os.environ.get("GITHUB_TOKEN"),
        github_repo=os.environ.get("GITHUB_REPO"),
        github_owner=os.environ.get("GITHUB_OWNER"),
        github_host=os.environ.get("GITHUB_HOST") or DEFAULT_GITHUB_HOST,
        llm_source=os.environ.get("LLM_SOURCE") or DEFAULT_LLM_SOURCE,
        llm_api_key=os.environ.get("LLM_API_KEY")
    )