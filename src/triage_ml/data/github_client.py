"""
Robust GitHub REST API Client.
Fetches issues, extracts metadata, handles pagination, rate limits, and authentication securely.
"""
import time
from typing import Any, Dict, List, Optional
import requests

from config.settings import settings
from src.triage_ml.utils.logger import get_logger

logger = get_logger("github_client")


class GitHubClient:
    """Production client for the GitHub REST API v3."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or settings.GITHUB_TOKEN
        self.session = requests.Session()
        self.session.headers.update(settings.get_github_headers())

    def get_rate_limit(self) -> Dict[str, Any]:
        """
        Checks remaining GitHub API rate limits.

        Returns:
            Dict[str, Any]: Rate limit status information.
        """
        url = f"{settings.GITHUB_API_BASE}/rate_limit"
        try:
            res = self.session.get(url, timeout=settings.API_TIMEOUT)
            if res.status_code == 200:
                data = res.json().get("resources", {}).get("core", {})
                return {
                    "limit": data.get("limit", 60),
                    "remaining": data.get("remaining", 0),
                    "reset_epoch": data.get("reset", 0),
                    "authenticated": bool(self.token and "your_github" not in self.token)
                }
            return {"error": f"HTTP {res.status_code}", "remaining": 0}
        except Exception as e:
            logger.warning(f"Unable to query rate limit: {e}")
            return {"error": str(e), "remaining": 0}

    def fetch_issues(
        self,
        owner: str,
        repo: str,
        max_pages: int = 5,
        state: str = "all",
        per_page: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetches GitHub issues for a specific repository.
        Skips pull requests automatically and handles rate limiting.

        Args:
            owner: Repository owner (e.g., 'pallets').
            repo: Repository name (e.g., 'flask').
            max_pages: Maximum number of pages to fetch.
            state: Issue state ('all', 'open', 'closed').
            per_page: Issues per page (max 100).

        Returns:
            List[Dict[str, Any]]: List of parsed issue records.
        """
        url = f"{settings.GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
        issues: List[Dict[str, Any]] = []

        logger.info(f"Fetching issues for {owner}/{repo} (up to {max_pages} pages)...")

        for page in range(1, max_pages + 1):
            params = {
                "state": state,
                "per_page": min(per_page, 100),
                "page": page
            }
            try:
                res = self.session.get(url, params=params, timeout=settings.API_TIMEOUT)
                if res.status_code == 403:
                    logger.warning("GitHub API rate limit exceeded or access forbidden.")
                    break
                if res.status_code != 200:
                    logger.warning(f"Error fetching page {page} from {owner}/{repo}: HTTP {res.status_code}")
                    break

                page_data = res.json()
                if not page_data or not isinstance(page_data, list):
                    break

                for item in page_data:
                    # Skip pull requests (GitHub API /issues endpoint returns PRs unless filtered)
                    if "pull_request" in item:
                        continue

                    label_names = [lbl["name"] for lbl in item.get("labels", []) if isinstance(lbl, dict) and "name" in lbl]
                    assignee_names = [a["login"] for a in item.get("assignees", []) if isinstance(a, dict) and "login" in a]

                    issues.append({
                        "id": item.get("id"),
                        "number": item.get("number"),
                        "repository": f"{owner}/{repo}",
                        "title": item.get("title", ""),
                        "body": item.get("body", "") or "",
                        "labels": label_names,
                        "assignees": assignee_names,
                        "state": item.get("state", "open"),
                        "comments_count": item.get("comments", 0),
                        "created_at": item.get("created_at", "")
                    })

            except Exception as e:
                logger.error(f"Failed to fetch page {page} for {owner}/{repo}: {e}")
                break

        logger.info(f"Successfully collected {len(issues)} pure issues from {owner}/{repo}")
        return issues
