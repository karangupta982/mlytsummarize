from flask import Flask, request, jsonify
import yt_dlp
import assemblyai as aai
import os
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from flask_cors import CORS
import time
import traceback
import re
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
import json
from dotenv import load_dotenv

load_dotenv()


nltk.download('punkt')

app = Flask(__name__)
# Configure CORS properly to allow requests from your React frontend
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",
    "https://mlytsummarize-kdso.vercel.app",
    "https://mlytsummarize-omjx.vercel.app"
]}})

aai.settings.api_key = "85dfd1af7ea047f1abf886314afcbd7d"

# Initialize Groq client
# groq_client = Groq(
#     api_key=os.getenv("GROQ_API_KEY")
# )

groq_client = Groq(
    api_key=""
    # api_key=os.getenv("GROQ_API_KEY")
)

# Enhanced yt_dlp options for cloud environments
ydl_opts = {
    'format': 'worstaudio[ext=m4a]/worstaudio/worst',  # Use worst quality for faster processing
    'outtmpl': '/tmp/sameAudio.%(ext)s',  # Use /tmp directory
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
    }],
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'referer': 'https://www.youtube.com/',
    'sleep_interval': 1,
    'max_sleep_interval': 3,
    'retries': 2,
    'fragment_retries': 2,
    'socket_timeout': 30,
    'nocheckcertificate': True,
}

def download_audio(video_url):
    """Enhanced download_audio with better error handling"""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([video_url])
            if error_code != 0:
                raise Exception(f"yt_dlp failed with exit code: {error_code}")
        
        # Check both possible locations
        possible_files = ["/tmp/sameAudio.mp3", "sameAudio.mp3"]
        for file_path in possible_files:
            if os.path.exists(file_path):
                return file_path
        
        raise Exception("Audio file not found after download")
    except Exception as e:
        raise Exception(f"Audio download failed: {str(e)}")

def get_transcript_from_youtube_api(videolink, language='en'):
    """Get transcript directly from YouTube's captions (no audio download needed)"""
    try:
        # Extract video ID from various YouTube URL formats
        video_id_patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'youtu\.be\/([0-9A-Za-z_-]{11})',
            r'youtube\.com\/embed\/([0-9A-Za-z_-]{11})'
        ]
        
        video_id = None
        for pattern in video_id_patterns:
            match = re.search(pattern, videolink)
            if match:
                video_id = match.group(1)
                break
        
        if not video_id:
            raise Exception("Could not extract video ID from URL")
        
        print(f"Extracted video ID: {video_id}")
        
        # Try different language codes based on the requested language
        if language == 'hi':
            lang_codes = ['hi', 'hi-IN']
        else:
            lang_codes = ['en', 'en-US', 'en-GB', 'en-CA']
        
        transcript_list = None
        used_lang = None
        
        # Try specific language codes first
        for lang_code in lang_codes:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
                used_lang = lang_code
                print(f"Successfully got transcript using language code: {lang_code}")
                break
            except Exception as e:
                print(f"Failed to get transcript for {lang_code}: {str(e)}")
                continue
        
        # If specific languages fail, try auto-generated
        if not transcript_list:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                used_lang = "auto"
                print("Successfully got auto-generated transcript")
            except Exception as e:
                raise Exception(f"Could not retrieve any transcript: {str(e)}")
        
        # Convert transcript list to text
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        
        if not transcript_text or len(transcript_text.strip()) == 0:
            raise Exception("Transcript is empty")
        
        print(f"Retrieved transcript with {len(transcript_text)} characters using {used_lang}")
        return transcript_text
        
    except Exception as e:
        raise Exception(f"YouTube transcript API failed: {str(e)}")

def groq_summarization(transcript, language='en'):
    """
    Summarizes the input transcript using Groq API with improved error handling.
    """
    try:
        # If the transcript is empty, return a default message
        if not transcript or len(transcript.strip()) == 0:
            raise Exception("Empty transcript provided")
        
        print(f"🚀 Starting Groq summarization for {len(transcript)} characters in {language}")
        
        # Check if Groq client is properly initialized
        if not groq_client:
            raise Exception("Groq client is not initialized")
        
        # More aggressive transcript truncation for Groq limits
        # Llama-3.1-8b-instant has roughly 8K token context limit
        # Rough estimate: 4 characters per token
        max_chars = 20000  # Conservative limit
        original_length = len(transcript)
        
        if len(transcript) > max_chars:
            # Try to truncate at sentence boundary
            truncated = transcript[:max_chars]
            
            # Look for sentence endings in the last 2000 characters
            search_area = truncated[-2000:]
            sentence_endings = ['.', '!', '?', '।']  # Include Hindi purna viram
            
            best_cut = -1
            for ending in sentence_endings:
                last_occurrence = search_area.rfind(ending)
                if last_occurrence > best_cut:
                    best_cut = last_occurrence
            
            if best_cut > 0:
                # Cut at the sentence boundary
                transcript = transcript[:max_chars - 2000 + best_cut + 1]
            else:
                transcript = truncated
                
            print(f"✂️ Truncated transcript from {original_length} to {len(transcript)} characters")
        
        # Prepare the prompt based on language
        if language == 'hi':
            prompt = f"""कृपया निम्नलिखित हिंदी पाठ का एक संक्षिप्त सारांश प्रदान करें। 
            सारांश में सभी मुख्य बिंदु होने चाहिए और यह 50-70 शब्दों में होना चाहिए।
            
            पाठ:
            {transcript}
            
            कृपया केवल हिंदी में सारांश दें।"""
        else:
            prompt = f"""Please provide a concise and comprehensive summary of the following text. 
            The summary should:
            - Capture all key points and main ideas
            - Be between 150-200 words
            - Be well-structured and coherent
            - Maintain the original context and meaning
            
            Text to summarize:
            {transcript}
            
            Summary:"""
        
        print(f"📤 Sending request to Groq API with prompt length: {len(prompt)}")
        
        # Make the API call to Groq with better parameters
        completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that creates accurate and concise summaries. Always respond in the same language as the input text."
                },
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,  # Lower temperature for more consistent summaries
            max_tokens=500,   # Increased token limit for better summaries
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        print("📥 Received response from Groq API")
        
        # Extract the summary
        summary = completion.choices[0].message.content.strip()
        
        # Validate the summary
        if not summary or len(summary.strip()) == 0:
            raise Exception("Groq returned empty summary")
        
        # Check if the summary is actually a summary (not just echoing the input)
        if len(summary) > len(transcript) * 0.8:  # If summary is more than 80% of original
            raise Exception("Groq returned content that's too similar to original (possible echo)")
        
        print(f"✅ Successfully generated Groq summary: {len(summary)} characters")
        
        # Return the summary with a marker
        return {"summary": summary, "method": "groq", "success": True}
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ GROQ API ERROR: {error_msg}")
        
        # Log specific error types for debugging
        if "rate_limit" in error_msg.lower():
            print("🚫 ERROR TYPE: Rate limit exceeded")
        elif "token" in error_msg.lower():
            print("🚫 ERROR TYPE: Token limit exceeded")
        elif "api" in error_msg.lower() or "auth" in error_msg.lower():
            print("🚫 ERROR TYPE: API authentication or connection issue")
        else:
            print(f"🚫 ERROR TYPE: Unknown - {error_msg}")
        
        # Return error info instead of raising exception
        return {"summary": None, "method": "groq", "success": False, "error": error_msg}

def extractive_summarization(transcript):
    """
    Fallback summarization using the original Extractive Summarization technique.
    """
    try:
        print("🔄 Using extractive summarization fallback")
        
        # If the transcript is empty, return a default message
        if not transcript or len(transcript.strip()) == 0:
            return "The transcript appears to be empty. Please try a different video."
        
        # Use the NLTK punkt tokenizer to split into sentences
        sentences = sent_tokenize(transcript)
        
        print(f"📝 Number of sentences detected: {len(sentences)}")
        
        if len(sentences) <= 5:  # If transcript is very short, return it as is
            return transcript
        
        # Vectorize sentences
        vectorizer = text.CountVectorizer(stop_words='english')
        
        try:
            X = vectorizer.fit_transform(sentences)
            
            # Check if we can perform SVD
            if X.shape[0] < 2 or X.shape[1] < 2:
                return transcript
                
            # Perform Truncated SVD for dimensionality reduction
            n_components = min(1, X.shape[1] - 1)  # Ensure we don't exceed matrix dimensions
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd.fit(X)
            components = svd.transform(X)
            
            # Rank sentences based on the first singular vector
            ranked_sentences = [item[0] for item in sorted(enumerate(components), key=lambda item: -item[1])]
            
            # Select top sentences for summary (at least 3 sentences or 30% of the original)
            num_sentences = max(3, int(0.3 * len(sentences)))
            selected_sentences = sorted(ranked_sentences[:min(num_sentences, len(ranked_sentences))])
            
            # Compile the final summary
            summary = " ".join([sentences[idx] for idx in selected_sentences])
            return summary
        except Exception as e:
            print(f"Error during summarization process: {str(e)}")
            # Return first few complete sentences if summarization fails
            return " ".join(sentences[:min(10, len(sentences))])
    except Exception as e:
        print(f"Error in sentence tokenization: {str(e)}")
        # If tokenization fails, return complete sentences from the beginning
        if transcript and len(transcript) > 0:
            # Get the first 1000 characters but ensure we end with a complete sentence
            partial_text = transcript[:1000]
            # Find the last period in the partial text
            last_period_index = partial_text.rfind('.')
            if last_period_index > 0:
                return transcript[:last_period_index + 1]  # Include the period
            else:
                # If no period is found, return as is without ellipsis
                return partial_text
        else:
            return "Unable to generate summary. The video may not contain sufficient spoken content."

def smart_summarization(transcript, language='en'):
    """
    Try Groq first, with extractive summarization as explicit fallback
    """
    print(f"🎯 Starting smart summarization for {language} text")
    
    # First, try Groq
    groq_result = groq_summarization(transcript, language)
    
    if groq_result["success"]:
        print("✅ Groq summarization successful")
        return groq_result["summary"], "groq"
    else:
        print(f"❌ Groq failed: {groq_result['error']}")
        print("🔄 Falling back to extractive summarization")
        
        # Fallback to extractive summarization
        try:
            extractive_summary = extractive_summarization(transcript)
            return extractive_summary, "extractive"
        except Exception as extractive_error:
            print(f"❌ Extractive summarization also failed: {extractive_error}")
            return f"Unable to generate summary. Groq error: {groq_result['error']}. Extractive error: {extractive_error}", "failed"

def get_transcript_with_retry(transcriber, audio_file, max_retries=3):
    """Attempts to get a transcript with retries"""
    for attempt in range(max_retries):
        try:
            transcript = transcriber.transcribe(audio_file)
            
            # Check if transcript has text
            if hasattr(transcript, 'text') and transcript.text:
                text_content = transcript.text
                if len(text_content) > 0:
                    print(f"Successful transcription on attempt {attempt+1} with {len(text_content)} characters")
                    return text_content
                else:
                    print(f"Empty transcript.text on attempt {attempt+1}")
            else:
                print(f"No text attribute found on attempt {attempt+1}")
                
                # Try to find the text in other attributes
                for attr in ['transcript', 'result', 'content', 'output']:
                    if hasattr(transcript, attr):
                        content = getattr(transcript, attr)
                        if isinstance(content, str) and len(content) > 0:
                            return content
                
                # If all else fails, check if there's a summary attribute
                if hasattr(transcript, 'summary') and transcript.summary:
                    return f"Summary from AssemblyAI: {transcript.summary}"
                    
            # If we get here, we need to try again
            if attempt < max_retries - 1:
                print(f"Retry transcription in 2 seconds...")
                time.sleep(2)
                
        except Exception as e:
            print(f"Transcription error on attempt {attempt+1}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retry transcription in 2 seconds...")
                time.sleep(2)
    
    # If we get here, all retries failed
    raise Exception("Failed to get transcript after multiple attempts")


@app.route("/testGroq", methods=["GET"])
def test_groq():
    """Test endpoint to verify Groq API is working"""
    try:
        test_text = "This is a test message to verify that the Groq API is working correctly."
        
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": f"Please summarize this in one sentence: {test_text}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=100
        )
        
        response = completion.choices[0].message.content
        
        return jsonify({
            "status": "success",
            "groq_response": response,
            "message": "Groq API is working correctly"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "Groq API is not working"
        }), 500


@app.route("/summarizeText", methods=["POST"])
def summarize_text():
    """Endpoint to summarize any text using Groq API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')  # Default to English
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        print(f"Summarizing text of {len(text)} characters in {language}")
        
        # Generate summary using smart summarization
        summary, method = smart_summarization(text, language)
        
        return jsonify({
            "summary": summary,
            "method": method,
            "original_length": len(text),
            "summary_length": len(summary)
        })
        
    except Exception as e:
        print(f"Error in text summarization: {str(e)}")
        return jsonify({
            "error": f"Failed to summarize text: {str(e)}"
        }), 500

@app.route("/getEnglishTranscript", methods=["GET"])
def get_english_transcript():
    videolink = request.args.get('videolink')
    if not videolink:
        return jsonify({"error": "No video link provided"}), 400

    print(f"Processing English video: {videolink}")
    audio_file = None
    transcript_method = "unknown"

    try:
        # Method 1: Try YouTube transcript API first
        try:
            print("Attempting to get transcript using YouTube transcript API...")
            text = get_transcript_from_youtube_api(videolink, language='en')
            print(f"Successfully got transcript from YouTube API: {len(text)} characters")
            transcript_method = "youtube-api"
            
        except Exception as api_error:
            print(f"YouTube transcript API failed: {api_error}")
            print("Falling back to yt_dlp + AssemblyAI method...")
            
            # Method 2: Fallback to yt_dlp + AssemblyAI
            audio_file = download_audio(videolink)
            
            if not os.path.exists(audio_file):
                raise Exception(f"Audio file {audio_file} was not created successfully")
            
            audio_size = os.path.getsize(audio_file)
            print(f"🎵 Audio file size: {audio_size} bytes")
            
            if audio_size < 1000:
                raise Exception("Audio file is too small, possibly download failed")

            transcriber = aai.Transcriber()
            text = get_transcript_with_retry(transcriber, audio_file)
            transcript_method = "assemblyai"
        
        if not text:
            return jsonify({"summary": "The video does not appear to contain any recognizable speech. Please try another video."}), 200
        
        print(f"📝 Successfully got transcript: {len(text)} characters using {transcript_method}")
        
        # Generate summary using smart summarization
        summary, summarization_method = smart_summarization(text, language='en')
        
        return jsonify({
            "summary": summary, 
            "method": f"{transcript_method}-{summarization_method}",
            "transcript_length": len(text),
            "summary_length": len(summary)
        })
        
    except Exception as e:
        print(f"❌ Error in English transcript processing: {str(e)}")
        return jsonify({
            "error": f"Failed to process video: {str(e)}", 
            "suggestion": "This might be due to video restrictions or server limitations. Please try another video."
        }), 500
    finally:
        # Cleanup
        cleanup_files = ["/tmp/sameAudio.mp3", "sameAudio.mp3"]
        if audio_file:
            cleanup_files.append(audio_file)
        
        for file_path in cleanup_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🗑️ Cleaned up: {file_path}")
                except Exception as e:
                    print(f"⚠️ Error cleaning up {file_path}: {e}")

@app.route("/getHindiTranscript", methods=["GET"])
def get_hindi_transcript():
    videolink = request.args.get('videolink')
    if not videolink:
        return jsonify({"error": "No video link provided"}), 400

    print(f"🎬 Processing Hindi video: {videolink}")
    audio_file = None
    transcript_method = "unknown"

    try:
        # Method 1: Try YouTube transcript API first for Hindi
        try:
            print("📺 Attempting to get Hindi transcript using YouTube transcript API...")
            text = get_transcript_from_youtube_api(videolink, language='hi')
            print(f"✅ Successfully got Hindi transcript from YouTube API: {len(text)} characters")
            transcript_method = "youtube-api"
            
        except Exception as api_error:
            print(f"❌ YouTube Hindi transcript API failed: {api_error}")
            print("🔄 Falling back to yt_dlp + AssemblyAI method for Hindi...")
        
            # Method 2: Fallback to yt_dlp + AssemblyAI
            audio_file = download_audio(videolink)
            
            # Add error checking for audio download
            if not os.path.exists(audio_file):
                raise Exception(f"Audio file {audio_file} was not created successfully")
            
            # Check audio file size
            audio_size = os.path.getsize(audio_file)
            print(f"🎵 Audio file size: {audio_size} bytes")
            
            if audio_size < 1000:  # Less than 1KB, likely an empty file
                raise Exception("Audio file is too small, possibly download failed")
                
            config = aai.TranscriptionConfig(language_code="hi")
            transcriber = aai.Transcriber(config=config)
            
            # Try to get transcript with retries
            try:
                text = get_transcript_with_retry(transcriber, audio_file)
                transcript_method = "assemblyai"
            except Exception as e:
                print(f"❌ Error getting Hindi transcript: {str(e)}")
                return jsonify({"summary": "Failed to transcribe Hindi audio. The video may not contain Hindi speech."}), 200
        
        if not text:
            return jsonify({"summary": "The video does not appear to contain any recognizable Hindi speech. Please try another video."}), 200
        
        print(f"📝 Successfully got Hindi transcript: {len(text)} characters using {transcript_method}")
        
        # Generate summary using smart summarization
        summary, summarization_method = smart_summarization(text, language='hi')
        
        return jsonify({
            "summary": summary, 
            "method": f"{transcript_method}-{summarization_method}",
            "transcript_length": len(text),
            "summary_length": len(summary)
        })
        
    except Exception as e:
        print(f"❌ Error in Hindi transcript processing: {str(e)}")
        return jsonify({
            "error": f"Failed to process Hindi video: {str(e)}", 
            "suggestion": "This might be due to video restrictions or server limitations. Please try another video."
        }), 500
    finally:
        # Cleanup
        cleanup_files = ["/tmp/sameAudio.mp3", "sameAudio.mp3"]
        if audio_file:
            cleanup_files.append(audio_file)
        
        for file_path in cleanup_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🗑️ Cleaned up: {file_path}")
                except Exception as e:
                    print(f"⚠️ Error cleaning up {file_path}: {e}")

@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200

# for development
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")