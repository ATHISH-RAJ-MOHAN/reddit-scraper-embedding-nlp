from bs4 import BeautifulSoup

def parse_reddit_html(html):
    soup = BeautifulSoup(html, "lxml")
    posts = []

    post_elements = soup.select("div.thing")

    for post in post_elements:
        try:
            if "promoted" in post.get("class", []):
                continue

            title_tag = post.select_one("a.title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            
            permalink = post.get("data-permalink")
            if not permalink and title_tag:
                href = title_tag.get("href")
                if href and href.startswith("/r/"):
                    permalink = href

            author_tag = post.select_one("a.author")
            author = author_tag.get_text(strip=True) if author_tag else "unknown"

            time_tag = post.select_one("time")
            timestamp = time_tag["datetime"] if time_tag else ""

            if title:
                posts.append({
                    "title": title,
                    "author": author,
                    "timestamp": timestamp,
                    "permalink": permalink,
                    "body": "",
                })

        except Exception as e:
            print(f"Error parsing post: {e}")
            continue

    return posts

def parse_single_post(html):
    soup = BeautifulSoup(html, "lxml")
    
    submission_text = soup.select_one("div.link div.usertext-body div.md")
    body = submission_text.get_text("\n", strip=True) if submission_text else ""
    
    top_comment = ""
    comments = soup.select("div.commentarea > div.sitetable > div.comment")
    
    for comment in comments:
        comment_body = comment.select_one("div.usertext-body div.md")
        if comment_body:
            text = comment_body.get_text("\n", strip=True)
            if text and text != "[deleted]" and text != "[removed]":
                top_comment = text
                break

    return {
        "body": body,
        "top_comment": top_comment
    }