from scraper.scrape_reddit import scrape_all

data = scrape_all()
print("Scraped subreddits:", list(data.keys()))
