"""WordPress publish: REST API + Application Password auth.

Default status is "draft" so a human can review before going public.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import markdown as md
import requests
from langsmith import traceable
from requests.auth import HTTPBasicAuth


@dataclass
class PublishResult:
    id: int
    url: str
    edit_url: str
    status: str


def render_html(content_md: str) -> str:
    return md.markdown(content_md, extensions=["fenced_code", "tables"])


def _base_and_auth(
    site_url: str | None,
    username: str | None,
    app_password: str | None,
) -> tuple[str, HTTPBasicAuth]:
    site = (site_url or os.environ["WORDPRESS_URL"]).rstrip("/")
    user = username or os.environ["WORDPRESS_USERNAME"]
    pw = app_password or os.environ["WORDPRESS_APP_PASSWORD"]
    return f"{site}/wp-json/wp/v2", HTTPBasicAuth(user, pw)


def _resolve_tag_ids(base: str, auth: HTTPBasicAuth, tags: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for name in tags:
        name = name.strip()
        if not name:
            continue
        r = requests.get(
            f"{base}/tags",
            params={"search": name, "per_page": 20},
            auth=auth,
            timeout=15,
        )
        r.raise_for_status()
        match = next(
            (t for t in r.json() if t["name"].lower() == name.lower()),
            None,
        )
        if match:
            ids.append(match["id"])
            continue
        r = requests.post(f"{base}/tags", json={"name": name}, auth=auth, timeout=15)
        r.raise_for_status()
        ids.append(r.json()["id"])
    return ids


@traceable(name="publish.wp", run_type="tool")
def publish_post(
    *,
    title: str,
    content_md: str,
    excerpt: str = "",
    slug: str = "",
    tags: list[str] | None = None,
    status: str = "draft",
    site_url: str | None = None,
    username: str | None = None,
    app_password: str | None = None,
) -> PublishResult:
    base, auth = _base_and_auth(site_url, username, app_password)
    site_root = base.rsplit("/wp-json", 1)[0]

    payload: dict = {
        "title": title,
        "content": render_html(content_md),
        "status": status,
    }
    if excerpt:
        payload["excerpt"] = excerpt
    if slug:
        payload["slug"] = slug
    if tags:
        payload["tags"] = _resolve_tag_ids(base, auth, tags)

    r = requests.post(f"{base}/posts", json=payload, auth=auth, timeout=30)
    r.raise_for_status()
    data = r.json()

    return PublishResult(
        id=data["id"],
        url=data.get("link", ""),
        edit_url=f"{site_root}/wp-admin/post.php?post={data['id']}&action=edit",
        status=data.get("status", status),
    )
