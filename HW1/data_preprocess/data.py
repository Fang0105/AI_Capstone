from googleapiclient.discovery import build
import pandas as pd
import time

# 🔹 Replace with your YouTube API Key
API_KEY = "AIzaSyBJ66nS9e7K0avLGDfh2KEtFJcDwxwdpuA"

# 🔹 Replace with the YouTube video ID (from the URL after 'v=')
VIDEO_ID = "FGptJQ73a28"  # Example: Rick Astley's "Never Gonna Give You Up"

# 🔹 Initialize YouTube API
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_all_video_comments(video_id):
    comments = []
    next_page_token = None

    while True:
        # API request to get comments
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,  # Maximum allowed per request
            textFormat="plainText",
            pageToken=next_page_token  # Handle pagination
        )
        response = request.execute()

        # Extract comments
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": comment["authorDisplayName"],
                "text": comment["textDisplay"],
                "likes": comment["likeCount"],
                "published_at": comment["publishedAt"]
            })

        # Check if more comments exist
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break  # Exit loop if no more pages

        time.sleep(1)  # Prevent API rate limit

    return comments

# 🔹 Fetch comments
comments_data = get_all_video_comments(VIDEO_ID)

# 🔹 Convert to Pandas DataFrame
df = pd.DataFrame(comments_data)

# 🔹 Save to CSV
df.to_csv("./../data/newjeans_court_reaction.csv", index=False)

# 🔹 Show first few comments
print(df.head())
