import json
import os
import time
from bs4 import BeautifulSoup
from scraper.fetch_html import fetch_html
from scraper.parse_posts import parse_reddit_html, parse_single_post

SUBREDDITS = ["Cooking", "Recipes", "AskCulinary", "Baking", "FoodScience"]

def scrape_subreddit(subreddit, limit=100):
    base_url = f"https://old.reddit.com/r/{subreddit}/?limit=100"
    after = None
    all_posts = []

    # Phase 1: Collect Post Links
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

        soup = BeautifulSoup(html, "lxml")
        next_button = soup.select_one("span.next-button a")
        if next_button and "after=" in next_button["href"]:
            after = next_button["href"].split("after=")[1].split("&")[0]
        else:
            break
    
    all_posts = all_posts[:limit]

    # Phase 2: Fetch Full Body for Each Post
    print(f"\n--- Fetching full body text and top comments for {len(all_posts)} posts ---")
    for index, post in enumerate(all_posts):
        if post.get('permalink'):
            post_url = f"https://old.reddit.com{post['permalink']}"
            
            print(f"[{index+1}/{len(all_posts)}] Fetching details: {post['title'][:30]}...")
            
            detail_html = fetch_html(post_url)
            if detail_html:
                details = parse_single_post(detail_html)
                post['body'] = details['body']
                post['top_comment'] = details['top_comment']
            else:
                print("  Failed to load detail page.")
                post['body'] = ""
                post['top_comment'] = ""
        else:
            print(f"[{index+1}/{len(all_posts)}] Skipping (External Link): {post['title'][:30]}...")
            post['body'] = ""
            post['top_comment'] = ""

    return all_posts

def scrape_all():
    os.makedirs("data/parsed_json", exist_ok=True)
    final_data = {}

    for sub in SUBREDDITS:
        posts = scrape_subreddit(sub, limit=1000)
        final_data[sub] = posts

        with open(f"data/parsed_json/{sub}.json", "w", encoding="utf-8") as f:
            json.dump(posts, f, indent=4)

    print("\nScraping complete.")
    return final_data