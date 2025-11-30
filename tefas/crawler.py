"""Tefas Crawler

Crawls public invenstment fund information from Turkey Electronic Fund Trading Platform.
"""

import ssl
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
import time

from tefas.schema import BreakdownSchema, InfoSchema

logger = logging.getLogger(__name__)

# User-Agent pool for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class Crawler:
    """Fetch public fund information from ``https://www.tefas.gov.tr``.

    Improved with rate limiting, robot detection, and better error handling.

    Examples:

    >>> tefas = Crawler()
    >>> data = tefas.fetch(start="2020-11-20")
    >>> data.head(1)
           price  number_of_shares code  ... precious_metals  stock  private_sector_bond
    0  41.302235         1898223.0  AAK  ...             0.0  31.14                 3.28
    >>> data = tefas.fetch(name="YAC",
    >>>                    start="2020-11-15",
    >>>                    end="2020-11-20",
    >>>                    columns=["date", "code", "price"])
    >>> data.head()
             date code     price
    0  2020-11-20  YAC  1.844274
    1  2020-11-19  YAC  1.838618
    2  2020-11-18  YAC  1.833198
    3  2020-11-17  YAC  1.838440
    4  2020-11-16  YAC  1.827832
    """

    root_url = "https://www.tefas.gov.tr"
    detail_endpoint = "/api/DB/BindHistoryAllocation"
    info_endpoint = "/api/DB/BindHistoryInfo"
    
    def __init__(
        self,
        request_delay: float = 1.0,
        max_retries: int = 5,
        timeout: int = 30,
        base_backoff: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize the crawler with configurable rate limiting.
        
        Args:
            request_delay: Minimum delay between requests in seconds (default: 1.0)
            max_retries: Maximum number of retry attempts (default: 5)
            timeout: Request timeout in seconds (default: 30)
            base_backoff: Base multiplier for exponential backoff (default: 2.0)
            jitter: Whether to add random jitter to backoff delays (default: True)
        """
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_backoff = base_backoff
        self.jitter = jitter
        self.last_request_time = 0.0
        
        # Rotate User-Agent
        self.user_agent = random.choice(USER_AGENTS)
        self.headers = {
            "Connection": "keep-alive",
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": self.user_agent,
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Origin": "https://www.tefas.gov.tr",
            "Referer": "https://www.tefas.gov.tr/TarihselVeriler.aspx",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        
        self.session = _get_session()
        # Initial request to get cookies
        _ = self.session.get(self.root_url, timeout=self.timeout)
        self.cookies = self.session.cookies.get_dict()
        
        logger.info(f"Crawler initialized with User-Agent: {self.user_agent[:50]}...")

    def _enforce_rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            sleep_time = self.request_delay - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter.
        
        Args:
            attempt: Current retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_backoff ** attempt
        
        if self.jitter:
            # Add random jitter (±25%) to prevent thundering herd
            jitter_amount = delay * 0.25 * (2 * random.random() - 1)
            delay = max(0.1, delay + jitter_amount)
        
        return delay

    def fetch(
        self,
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        name: Optional[str] = None,
        columns: Optional[List[str]] = None,
        kind: Optional[str] = "YAT",
    ) -> pd.DataFrame:
        """Main entry point of the public API. Get fund information.

        Args:
            start: The date that fund information is crawled for.
            end: End of the period that fund information is crawled for. (optional)
            name: Name of the fund. If not given, all funds will be returned. (optional)
            columns: List of columns to be returned. (optional)
            kind: Type of the fund. One of `YAT`, `EMK`, or `BYF`. Defaults to `YAT`. (optional)
                - `YAT`: Securities Mutual Funds
                - `EMK`: Pension Funds
                - `BYF`: Exchange Traded Funds

        Returns:
            A pandas DataFrame where each row is the information for a fund.

        Raises:
            ValueError if date format is wrong.
        """  # noqa
        assert kind in [
            "YAT",
            "EMK",
            "BYF",
        ], "`kind` should be one of `YAT`, `EMK`, or `BYF`"
        start_date = _parse_date(start)
        end_date = _parse_date(end or start)
        data = {
            "fontip": kind,
            "bastarih": start_date,
            "bittarih": end_date,
            "fonkod": name.upper() if name else "",
        }

        # General info pane
        info_schema = InfoSchema(many=True)
        info = self._do_post(self.info_endpoint, data)
        info = info_schema.load(info)
        info = pd.DataFrame(info, columns=info_schema.fields.keys())

        # Portfolio breakdown pane
        detail_schema = BreakdownSchema(many=True)
        detail = self._do_post(self.detail_endpoint, data)
        detail = detail_schema.load(detail)
        detail = pd.DataFrame(detail, columns=detail_schema.fields.keys())

        # Merge two panes
        merged = pd.merge(info, detail, on=["code", "date"])

        # Return only desired columns
        merged = merged[columns] if columns else merged

        return merged

    def _do_post(
        self, 
        endpoint: str, 
        data: Dict[str, str], 
        attempt: int = 0
    ) -> Dict[str, str]:
        """Make POST request with rate limiting, retry logic, and error handling.
        
        Args:
            endpoint: API endpoint path
            data: POST data payload
            attempt: Current retry attempt (0-indexed)
            
        Returns:
            Response data dictionary
            
        Raises:
            requests.exceptions.RequestException: On network errors
            ValueError: On invalid response format
            Exception: On rate limiting or robot check failures
        """
        # Enforce rate limit before request
        self._enforce_rate_limit()
        
        try:
            response = self.session.post(
                url=f"{self.root_url}{endpoint}",
                data=data,
                cookies=self.cookies,
                headers=self.headers,
                timeout=self.timeout,
            )
            
            # Check HTTP status code
            if response.status_code == 429:
                # Too Many Requests - rate limited
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(
                    f"Rate limited (429). Retry-After: {retry_after}s. "
                    f"Attempt {attempt + 1}/{self.max_retries + 1}"
                )
                if attempt < self.max_retries:
                    time.sleep(retry_after)
                    return self._do_post(endpoint, data, attempt + 1)
                else:
                    raise Exception(
                        f"Rate limited. Max attempts ({self.max_retries + 1}) reached. "
                        f"Please wait {retry_after} seconds before trying again."
                    )
            
            elif response.status_code == 403:
                # Forbidden - possible robot check
                logger.warning(
                    f"Forbidden (403) - possible robot check. "
                    f"Attempt {attempt + 1}/{self.max_retries + 1}"
                )
                if attempt < self.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    logger.info(f"Waiting {backoff:.2f} seconds before retry...")
                    time.sleep(backoff)
                    # Rotate User-Agent on retry
                    self.user_agent = random.choice(USER_AGENTS)
                    self.headers["User-Agent"] = self.user_agent
                    # Refresh session and cookies
                    self.session = _get_session()
                    _ = self.session.get(self.root_url, timeout=self.timeout)
                    self.cookies = self.session.cookies.get_dict()
                    return self._do_post(endpoint, data, attempt + 1)
                else:
                    raise Exception(
                        "Forbidden (403) - possible robot check. "
                        "Max attempts reached. Please try again later."
                    )
            
            elif response.status_code == 503:
                # Service Unavailable
                logger.warning(
                    f"Service unavailable (503). Attempt {attempt + 1}/{self.max_retries + 1}"
                )
                if attempt < self.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    logger.info(f"Waiting {backoff:.2f} seconds before retry...")
                    time.sleep(backoff)
                    return self._do_post(endpoint, data, attempt + 1)
                else:
                    raise Exception("Service unavailable. Max attempts reached.")
            
            elif not response.ok:
                # Other HTTP errors
                response.raise_for_status()
            
            # Check if response is HTML (robot check page) or contains rate limiting message
            response_text = response.text.lower()
            is_html_response = response.headers.get('Content-Type', '').startswith('text/html')
            is_rate_limit_message = any(keyword in response_text for keyword in [
                'rate limit', 'rate limiting', 'robot check', 'too many requests',
                'stuck at rate limiting', 'blocked', 'captcha', 'access denied'
            ])
            
            if is_html_response or is_rate_limit_message:
                logger.warning(
                    f"Received HTML response or rate limiting message (possible robot check). "
                    f"Attempt {attempt + 1}/{self.max_retries + 1}"
                )
                if attempt < self.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    logger.info(f"Waiting {backoff:.2f} seconds before retry...")
                    time.sleep(backoff)
                    # Rotate User-Agent and refresh session
                    self.user_agent = random.choice(USER_AGENTS)
                    self.headers["User-Agent"] = self.user_agent
                    self.session = _get_session()
                    _ = self.session.get(self.root_url, timeout=self.timeout)
                    self.cookies = self.session.cookies.get_dict()
                    return self._do_post(endpoint, data, attempt + 1)
                else:
                    raise Exception(
                        "Rate limiting or robot check detected. "
                        "Max attempts reached. Please try again later."
                    )
            
            # Parse JSON response
            try:
                json_response = response.json()
            except ValueError as e:
                logger.error(f"Invalid JSON response: {response.text[:200]}")
                if attempt < self.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    logger.info(f"Retrying after {backoff:.2f} seconds...")
                    time.sleep(backoff)
                    return self._do_post(endpoint, data, attempt + 1)
                else:
                    raise ValueError(f"Invalid JSON response after {self.max_retries + 1} attempts: {e}")
            
            # Check if response contains data
            data_dict = json_response.get("data", {})
            if not data_dict:
                logger.warning("Empty data in response")
                # Empty data might be valid (no funds found), but log it
                if isinstance(data_dict, dict) and len(data_dict) == 0:
                    logger.info("Response data is empty dict (might be valid)")
            
            return data_dict
            
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout. Attempt {attempt + 1}/{self.max_retries + 1}")
            if attempt < self.max_retries:
                backoff = self._calculate_backoff(attempt)
                logger.info(f"Retrying after {backoff:.2f} seconds...")
                time.sleep(backoff)
                return self._do_post(endpoint, data, attempt + 1)
            else:
                raise Exception(f"Request timeout after {self.max_retries + 1} attempts")
        
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error: {e}. Attempt {attempt + 1}/{self.max_retries + 1}")
            if attempt < self.max_retries:
                backoff = self._calculate_backoff(attempt)
                logger.info(f"Retrying after {backoff:.2f} seconds...")
                time.sleep(backoff)
                return self._do_post(endpoint, data, attempt + 1)
            else:
                raise Exception(f"Connection error after {self.max_retries + 1} attempts: {e}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}. Attempt {attempt + 1}/{self.max_retries + 1}")
            if attempt < self.max_retries:
                backoff = self._calculate_backoff(attempt)
                logger.info(f"Retrying after {backoff:.2f} seconds...")
                time.sleep(backoff)
                return self._do_post(endpoint, data, attempt + 1)
            else:
                raise

def _parse_date(date: Union[str, datetime]) -> str:
    if isinstance(date, datetime):
        formatted = datetime.strftime(date, "%d.%m.%Y")
    elif isinstance(date, str):
        try:
            parsed = datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                "Date string format is incorrect. It should be `YYYY-MM-DD`"
            ) from exc
        else:
            formatted = datetime.strftime(parsed, "%d.%m.%Y")
    else:
        raise ValueError(
            "`date` should be a string like 'YYYY-MM-DD' "
            "or a `datetime.datetime` object."
        )
    return formatted


def _get_session() -> requests.Session:
    """
    Create and return a custom requests session with optimized settings.

    This function configures a custom SSL context to use the `OP_LEGACY_SERVER_CONNECT`
    option, which allows for legacy server connections, addressing specific issues
    with OpenSSL 3.0.0.

    The custom session uses a custom HTTP adapter that incorporates this modified
    SSL context for the session's connections, along with optimized connection pooling.

    This approach is based on solutions found at:
    - https://stackoverflow.com/questions/71603314/ssl-error-unsafe-legacy-renegotiation-disabled/
    - https://github.com/urllib3/urllib3/issues/2653
    """

    class CustomHttpAdapter(HTTPAdapter):
        def __init__(self, ssl_context=None, **kwargs):
            self.ssl_context = ssl_context
            super().__init__(**kwargs)

        def init_poolmanager(
            self, connections, maxsize, block=False
        ):  # pylint: disable=arguments-differ
            self.poolmanager = PoolManager(
                num_pools=connections,
                maxsize=maxsize,
                block=block,
                ssl_context=self.ssl_context,
            )
    
    # Configure retry strategy (we handle retries manually, but this is for urllib3)
    retry_strategy = Retry(
        total=0,  # We handle retries manually
        backoff_factor=0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    
    session = requests.Session()
    
    # Mount custom SSL adapter
    session.mount("https://", CustomHttpAdapter(ctx))
    
    # Configure connection pooling with retry strategy
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=retry_strategy,
    )
    session.mount("https://", adapter)
    
    return session
