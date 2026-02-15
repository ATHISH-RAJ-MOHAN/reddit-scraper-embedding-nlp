import argparse
import json
import os
import time
from bs4 import BeautifulSoup
from scraper.fetch_html import fetch_html
from scraper.parse_posts import parse_reddit_html, parse_single_post, parse_top_comment

DEFAULT_SUBREDDITS = ["Cooking"]

def scrape_subreddit(subreddit, limit=10):
    base_url = f"https://old.reddit.com/r/{subreddit}/?limit=100"
    after = None
    all_posts = []

    # --- Phase 1: Collect Post Links ---
    print(f"--- Collecting headers from r/{subreddit} ---")
    while len(all_posts) < limit:
        url = f"{base_url}&after={after}" if after else base_url
        print(f"Fetching Listing: {url}")

        html = fetch_html(url)
        if not html:
            break

        posts = parse_reddit_html(html)
        if not posts:
            break
            
        all_posts.extend(posts)

        # Pagination
        soup = BeautifulSoup(html, "lxml")
        next_button = soup.select_one("span.next-button a")
        if next_button and "after=" in next_button["href"]:
            after = next_button["href"].split("after=")[1].split("&")[0]
        else:
            break
    
    # Trim to limit before fetching details to save time
    all_posts = all_posts[:limit]

    # --- Phase 2: Fetch Full Body for Each Post ---
    print(f"\n--- Fetching full body text for {len(all_posts)} posts ---")
    for index, post in enumerate(all_posts):
        if post.get('permalink'):
            # Construct full URL. Permalinks usually start with /r/...
            post_url = f"https://old.reddit.com{post['permalink']}"
            
            print(f"[{index+1}/{len(all_posts)}] Fetching details: {post['title'][:30]}...")
            
            detail_html = fetch_html(post_url)
            if detail_html:
                full_body = parse_single_post(detail_html)
                post['body'] = full_body
                post['top_comment'] = parse_top_comment(detail_html)
            else:
                print("  Failed to load detail page.")
        else:
            print(f"[{index+1}/{len(all_posts)}] Skipping (External Link): {post['title'][:30]}...")

    return all_posts

def scrape_all(subreddits=None, limit=100):
    os.makedirs("data/parsed_json", exist_ok=True)
    final_data = {}

    if subreddits is None:
        subreddits = DEFAULT_SUBREDDITS

    for sub in subreddits:
        posts = scrape_subreddit(sub, limit=limit) # Set to 10 for testing speed
        final_data[sub] = posts

        with open(f"data/parsed_json/{sub}.json", "w", encoding="utf-8") as f:
            json.dump(posts, f, indent=4)

    print("\nScraping complete.")
    return final_data

def _parse_args():
    parser = argparse.ArgumentParser(description="Scrape Reddit posts and save as JSON.")
    parser.add_argument(
        "--subreddits",
        type=str,
        default=",".join(DEFAULT_SUBREDDITS),
        help="Comma-separated list of subreddits to scrape.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of posts to fetch per subreddit.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    subs = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    scrape_all(subreddits=subs, limit=args.limit)
