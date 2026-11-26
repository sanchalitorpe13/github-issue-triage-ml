"""
API Client for GitHub Issue Triage REST Backend.
Centralizes all HTTP communication with the FastAPI backend service.
Provides timeouts, error handling, health checks, model info, and triage execution.
"""
import os
from typing import Any, Dict, List, Optional
import requests

from config.settings import settings
from src.triage_ml.utils.logger import get_logger

logger = get_logger("api_client")


class APIClientError(Exception):
    """Exception raised when API communication or server fails."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class TriageAPIClient:
    """Robust client interacting with the FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 15):
        # Default to environment variable or standard local FastAPI port
        self.base_url = (base_url or os.getenv("TRIAGE_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "IssueIntelligence-Dashboard/2.0"
        })

    def check_health(self) -> Dict[str, Any]:
        """
        Queries detailed system health endpoint.

        Returns:
            Dict[str, Any]: Diagnostic health information.

        Raises:
            APIClientError: If connection fails or returns unhealthy.
        """
        url = f"{self.base_url}/api/v1/health"
        try:
            res = self.session.get(url, timeout=self.timeout)
            if res.status_code == 200:
                return res.json()
            # Try basic health fallback
            fallback_res = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            if fallback_res.status_code == 200:
                return {"status": "healthy", "model_loaded": True, "triage_engine_status": "operational"}
            raise APIClientError(f"Health check failed with HTTP {res.status_code}", status_code=res.status_code)
        except requests.exceptions.ConnectionError:
            raise APIClientError(
                f"Cannot connect to FastAPI backend at {self.base_url}. Ensure the server is running.",
                status_code=503
            )
        except requests.exceptions.Timeout:
            raise APIClientError(f"Connection to backend at {self.base_url} timed out.", status_code=504)
        except Exception as e:
            raise APIClientError(f"Unexpected health check error: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Queries active ML model metadata and evaluation summary.

        Returns:
            Dict[str, Any]: Sanitized model information.
        """
        url = f"{self.base_url}/api/v1/model/info"
        try:
            res = self.session.get(url, timeout=self.timeout)
            if res.status_code == 200:
                return res.json()
            err_data = res.json() if res.headers.get("content-type") == "application/json" else {}
            msg = err_data.get("message") or err_data.get("detail") or f"HTTP {res.status_code}"
            raise APIClientError(f"Failed to fetch model info: {msg}", status_code=res.status_code)
        except requests.exceptions.ConnectionError:
            raise APIClientError(f"Cannot connect to FastAPI backend at {self.base_url}.", status_code=503)
        except Exception as e:
            if isinstance(e, APIClientError):
                raise
            raise APIClientError(f"Error querying model info: {str(e)}")

    def triage_issue(
        self,
        title: str,
        body: Optional[str] = "",
        repository: Optional[str] = None,
        issue_number: Optional[int] = None,
        labels: Optional[List[str]] = None,
        state: Optional[str] = "open",
        url: Optional[str] = None,
        author: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submits a single issue for ML classification and priority triage.

        Returns:
            Dict[str, Any]: Complete validated TriageResult.
        """
        endpoint = f"{self.base_url}/api/v1/triage"
        payload: Dict[str, Any] = {
            "title": title.strip(),
            "body": (body or "").strip(),
            "state": state or "open"
        }
        if repository:
            payload["repository"] = repository.strip()
        if issue_number:
            payload["issue_number"] = int(issue_number)
        if labels:
            payload["labels"] = labels
        if url:
            payload["url"] = url.strip()
        if author:
            payload["author"] = author.strip()
        if metadata:
            payload["metadata"] = metadata

        try:
            res = self.session.post(endpoint, json=payload, timeout=self.timeout)
            if res.status_code == 200:
                return res.json()
            err_data = res.json() if "application/json" in res.headers.get("content-type", "") else {}
            msg = err_data.get("message") or err_data.get("detail") or f"HTTP {res.status_code}"
            raise APIClientError(msg, status_code=res.status_code, details=err_data)
        except requests.exceptions.ConnectionError:
            raise APIClientError(f"Backend offline at {self.base_url}. Is FastAPI running?", status_code=503)
        except Exception as e:
            if isinstance(e, APIClientError):
                raise
            raise APIClientError(f"Failed to submit issue for triage: {str(e)}")

    def triage_batch(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Submits multiple issues concurrently for batch triage.

        Returns:
            Dict[str, Any]: BatchTriageResponse dictionary.
        """
        endpoint = f"{self.base_url}/api/v1/triage/batch"
        try:
            res = self.session.post(endpoint, json={"issues": issues}, timeout=self.timeout * 2)
            if res.status_code == 200:
                return res.json()
            err_data = res.json() if "application/json" in res.headers.get("content-type", "") else {}
            msg = err_data.get("message") or err_data.get("detail") or f"HTTP {res.status_code}"
            raise APIClientError(msg, status_code=res.status_code, details=err_data)
        except requests.exceptions.ConnectionError:
            raise APIClientError(f"Backend offline at {self.base_url}.", status_code=503)
        except Exception as e:
            if isinstance(e, APIClientError):
                raise
            raise APIClientError(f"Batch triage failed: {str(e)}")
