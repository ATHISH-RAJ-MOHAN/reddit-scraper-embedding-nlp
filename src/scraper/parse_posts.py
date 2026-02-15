from bs4 import BeautifulSoup

def parse_reddit_html(html):
    """Parses a listing page (e.g. /r/Cooking/top) for post summaries."""
    soup = BeautifulSoup(html, "lxml")
    posts = []

    post_elements = soup.select("div.thing")

    for post in post_elements:
        try:
            if "promoted" in post.get("class", []):
                continue

            # Title
            title_tag = post.select_one("a.title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            
            # Permalink (Critical for fetching full body)
            # We prefer the data-permalink attribute if available, else the link href
            permalink = post.get("data-permalink")
            if not permalink and title_tag:
                href = title_tag.get("href")
                # Only use if it's a relative reddit link (self post)
                if href and href.startswith("/r/"):
                    permalink = href

            # Author
            author_tag = post.select_one("a.author")
            author = author_tag.get_text(strip=True) if author_tag else "unknown"

            # Timestamp
            time_tag = post.select_one("time")
            timestamp = time_tag["datetime"] if time_tag else ""

            if title:
                posts.append({
                    "title": title,
                    "author": author,
                    "timestamp": timestamp,
                    "permalink": permalink, # Store this to visit later
                    "body": "", # Will be filled by the second request
                })

        except Exception as e:
            print(f"Error parsing post: {e}")
            continue

    return posts

def parse_single_post(html):
    """Parses a specific post page to get the full body text."""
    soup = BeautifulSoup(html, "lxml")
    
    # On a post detail page, the main content is in div.usertext-body within the main entry
    # We look for the distinct 'link' class which wraps the OP
    submission_text = soup.select_one("div.link div.usertext-body div.md")
    
    if submission_text:
        return submission_text.get_text("\n", strip=True)
    return ""

def parse_top_comment(html):
    """Parses a specific post page to get the top visible comment text."""
    soup = BeautifulSoup(html, "lxml")
    comment = soup.select_one("div.comment div.md")
    if comment:
        return comment.get_text("\n", strip=True)
    return ""
