"""`SearchBackend` protocol and its adapters (spec C-10).

`web_search` holds no vendor knowledge. Serper is the only adapter in v1 and it is
registered only when `SERPER_API_KEY` is set; `StaticSearchBackend` covers tests and
offline use.
"""

__all__: list[str] = []
