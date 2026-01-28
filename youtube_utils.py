import re
from youtube_transcript_api import YouTubeTranscriptApi 

def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def getTranscript(url):
    video_id = get_video_id(url)

    ytt_api = YouTubeTranscriptApi()

    transcripts = ytt_api.list(video_id)

    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop

    return transcript.to_raw_data()
        
def process_transcript(transcript):
    print(transcript)
    txt = ""
    # 'i' is now a dictionary like {'text': 'Hello', 'start': 0.5}
    for i in transcript:
        try:
            txt += f"Text: {i['text']} Start: {i['start']}\n"
        except (KeyError, TypeError):
            pass
    return txt