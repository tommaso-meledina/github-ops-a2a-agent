import os

from github import Github
from urllib.parse import urlparse


def build_github_client() -> Github:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN env var must be set")
    return Github(token)


def parse_github_repo_url(url: str):
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Not a valid GitHub repo URL: {url}")
    return parts[0], parts[1]
