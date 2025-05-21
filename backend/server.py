# from flask import Flask, request, jsonify
# import yt_dlp
# import assemblyai as aai
# import os
# from transformers import pipeline
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import TruncatedSVD
# from nltk.tokenize import sent_tokenize
# from langdetect import detect

# app = Flask(__name__)

# aai.settings.api_key = "85dfd1af7ea047f1abf886314afcbd7d"

# ydl_opts = {
#     'format': 'm4a/bestaudio/best',
#     'outtmpl': 'sameAudio.%(ext)s',
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#     }]
# }

# def download_audio(video_url):
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         error_code = ydl.download([video_url])
#     return "sameAudio.mp3"

# @app.route("/getEnglishTranscript", methods=["GET"])
# def get_english_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return "No video link provided", 400

#     print(f"Downloading and transcribing: {videolink}")

#     try:
#         download_audio(videolink)
#         transcriber = aai.Transcriber()
#         transcript = transcriber.transcribe("sameAudio.mp3")
#         summary = extractive_summarization(transcript.text)
#         # return jsonify([summary])
#         return jsonify({"summary": transcript.text})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         if os.path.exists("sameAudio.mp3"):
#             os.remove("sameAudio.mp3")

# @app.route("/getHindiTranscript", methods=["GET"])
# def get_hindi_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return "No video link provided", 400

#     print(f"Downloading and transcribing (Hindi): {videolink}")

#     try:
#         download_audio(videolink)
#         config = aai.TranscriptionConfig(language_code="hi")
#         transcriber = aai.Transcriber(config=config)
#         transcript = transcriber.transcribe("sameAudio.mp3")
#         #summary = extractive_summarization(transcript.text)
#         return jsonify([transcript.text])
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         if os.path.exists("sameAudio.mp3"):
#             os.remove("sameAudio.mp3")

# def extractive_summarization(transcript):
#     """
#     Summarizes the input transcript using the Extractive Summarization technique.
#     Latent Semantic Analysis (LSA) is used for dimensionality reduction and the sentences are ranked
#     based on their singular values. The top-ranked sentences are selected to form the summary.
    
#     Parameters:
#     - transcript (str): The transcript text to be summarized.
    
#     Returns:
#     - summary (str): The summarized text.
#     """
#     sentences = sent_tokenize(transcript)
    
#     # Vectorize sentences
#     vectorizer = CountVectorizer(stop_words='english')
#     X = vectorizer.fit_transform(sentences)
    
#     # Perform Truncated SVD for dimensionality reduction
#     svd = TruncatedSVD(n_components=1, random_state=42)
#     svd.fit(X)
#     components = svd.transform(X)
    
#     # Rank sentences based on the first singular vector
#     ranked_sentences = [item[0] for item in sorted(enumerate(components), key=lambda item: -item[1])]
    
#     # Select top sentences for summary
#     num_sentences = int(0.4 * len(sentences))  # 20% of the original sentences
#     selected_sentences = sorted(ranked_sentences[:num_sentences])
    
#     # Compile the final summary
#     summary = " ".join([sentences[idx] for idx in selected_sentences])
#     # return summary
#     return jsonify({"summary": summary})

# if __name__ == "__main__":
#     app.run(debug=True)


















# from flask import Flask, request, jsonify
# import yt_dlp
# import assemblyai as aai
# import os
# from transformers import pipeline
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import TruncatedSVD
# from nltk.tokenize import sent_tokenize
# from langdetect import detect
# from flask_cors import CORS  # Add CORS support

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# aai.settings.api_key = "85dfd1af7ea047f1abf886314afcbd7d"

# ydl_opts = {
#     'format': 'm4a/bestaudio/best',
#     'outtmpl': 'sameAudio.%(ext)s',
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#     }]
# }

# def download_audio(video_url):
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         error_code = ydl.download([video_url])
#     return "sameAudio.mp3"

# def extractive_summarization(transcript):
#     """
#     Summarizes the input transcript using the Extractive Summarization technique.
#     Latent Semantic Analysis (LSA) is used for dimensionality reduction and the sentences are ranked
#     based on their singular values. The top-ranked sentences are selected to form the summary.
    
#     Parameters:
#     - transcript (str): The transcript text to be summarized.
    
#     Returns:
#     - summary (str): The summarized text.
#     """
#     sentences = sent_tokenize(transcript)
    
#     if len(sentences) <= 5:  # If transcript is very short, return it as is
#         return transcript
    
#     # Vectorize sentences
#     vectorizer = CountVectorizer(stop_words='english')
    
#     try:
#         X = vectorizer.fit_transform(sentences)
        
#         # Perform Truncated SVD for dimensionality reduction
#         n_components = min(1, X.shape[1] - 1)  # Ensure we don't exceed matrix dimensions
#         svd = TruncatedSVD(n_components=n_components, random_state=42)
#         svd.fit(X)
#         components = svd.transform(X)
        
#         # Rank sentences based on the first singular vector
#         ranked_sentences = [item[0] for item in sorted(enumerate(components), key=lambda item: -item[1])]
        
#         # Select top sentences for summary (at least 3 sentences or 30% of the original)
#         num_sentences = max(3, int(0.3 * len(sentences)))
#         selected_sentences = sorted(ranked_sentences[:num_sentences])
        
#         # Compile the final summary
#         summary = " ".join([sentences[idx] for idx in selected_sentences])
#         return summary
#     except Exception as e:
#         print(f"Error in extractive summarization: {str(e)}")
#         # Return a shortened version of transcript if summarization fails
#         return " ".join(sentences[:min(10, len(sentences))])

# @app.route("/getEnglishTranscript", methods=["GET"])
# def get_english_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing: {videolink}")

#     try:
#         audio_file = download_audio(videolink)
#         transcriber = aai.Transcriber()
#         transcript = transcriber.transcribe(audio_file)
        
#         if not transcript or not transcript.text:
#             return jsonify({"error": "Failed to generate transcript"}), 500
            
#         # Generate summary from transcript
#         summary = extractive_summarization(transcript.text)
        
#         return jsonify({"summary": summary})
#     except Exception as e:
#         print(f"Error in English transcript: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#     finally:
#         if os.path.exists("sameAudio.mp3"):
#             try:
#                 os.remove("sameAudio.mp3")
#             except:
#                 pass

# @app.route("/getHindiTranscript", methods=["GET"])
# def get_hindi_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing (Hindi): {videolink}")

#     try:
#         audio_file = download_audio(videolink)
#         config = aai.TranscriptionConfig(language_code="hi")
#         transcriber = aai.Transcriber(config=config)
#         transcript = transcriber.transcribe(audio_file)
        
#         if not transcript or not transcript.text:
#             return jsonify({"error": "Failed to generate Hindi transcript"}), 500
            
#         # For Hindi, we'll use a simpler approach since our extractive summarization
#         # might not work well with Hindi. We'll return a portion of the transcript.
#         sentences = sent_tokenize(transcript.text)
#         summary = " ".join(sentences[:min(10, len(sentences))])
        
#         return jsonify({"summary": summary})
#     except Exception as e:
#         print(f"Error in Hindi transcript: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#     finally:
#         if os.path.exists("sameAudio.mp3"):
#             try:
#                 os.remove("sameAudio.mp3")
#             except:
#                 pass

# @app.route("/ping", methods=["GET"])
# def ping():
#     """Health check endpoint"""
#     return jsonify({"status": "ok"}), 200

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0")







from flask import Flask, request, jsonify
import yt_dlp
import assemblyai as aai
import os
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.tokenize import sent_tokenize
from langdetect import detect
from flask_cors import CORS  # Add CORS support

# Download NLTK data - this is crucial for the sent_tokenize function to work
nltk.download('punkt')
# Remove punkt_tab download as it's causing errors and is not needed
# nltk.download('punkt_tab')

app = Flask(__name__)
# Configure CORS properly to allow requests from your React frontend
CORS(app, resources={r"/*": {"origins": "*"}})

aai.settings.api_key = "85dfd1af7ea047f1abf886314afcbd7d"

ydl_opts = {
    'format': 'm4a/bestaudio/best',
    'outtmpl': 'sameAudio.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
    }]
}

def download_audio(video_url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([video_url])
    return "sameAudio.mp3"

def extractive_summarization(transcript):
    """
    Summarizes the input transcript using the Extractive Summarization technique.
    Latent Semantic Analysis (LSA) is used for dimensionality reduction and the sentences are ranked
    based on their singular values. The top-ranked sentences are selected to form the summary.
    
    Parameters:
    - transcript (str): The transcript text to be summarized.
    
    Returns:
    - summary (str): The summarized text.
    """
    try:
        sentences = sent_tokenize(transcript)
        
        if len(sentences) <= 5:  # If transcript is very short, return it as is
            return transcript
        
        # Vectorize sentences
        vectorizer = CountVectorizer(stop_words='english')
        
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
            # Return a shortened version of transcript if summarization fails
            return " ".join(sentences[:min(10, len(sentences))])
    except Exception as e:
        print(f"Error in sentence tokenization: {str(e)}")
        # If tokenization fails, return the first 500 characters as a fallback
        return transcript
        # return transcript[:500] + "... (Summary truncated due to processing error)"

@app.route("/getEnglishTranscript", methods=["GET"])
def get_english_transcript():
    videolink = request.args.get('videolink')
    if not videolink:
        return jsonify({"error": "No video link provided"}), 400

    print(f"Downloading and transcribing: {videolink}")

    try:
        audio_file = download_audio(videolink)
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file)

        print("transcript",transcript)
        
        if not transcript or not transcript.text:
            return jsonify({"error": "Failed to generate transcript"}), 500
            
        # Generate summary from transcript
        text = transcript.text
        print(f"Successfully transcribed {len(text)} characters of text")
        
        try:
            summary = extractive_summarization(text)
            print(f"Generated summary of {len(summary)} characters")
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            summary = text[:1000] + "... (Summary unavailable, showing transcript excerpt)"
        
        # Add better debug output
        print(f"Returning summary response with {len(summary)} characters")
        return jsonify({"summary": summary})
    except Exception as e:
        print(f"Error in English transcript: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists("sameAudio.mp3"):
                os.remove("sameAudio.mp3")
                print("Audio file deleted successfully")
        except Exception as e:
            print(f"Error deleting audio file: {str(e)}")

@app.route("/getHindiTranscript", methods=["GET"])
def get_hindi_transcript():
    videolink = request.args.get('videolink')
    if not videolink:
        return jsonify({"error": "No video link provided"}), 400

    print(f"Downloading and transcribing (Hindi): {videolink}")

    try:
        audio_file = download_audio(videolink)
        config = aai.TranscriptionConfig(language_code="hi")
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_file)
        
        if not transcript or not transcript.text:
            return jsonify({"error": "Failed to generate Hindi transcript"}), 500
        
        text = transcript.text
        print(f"Successfully transcribed {len(text)} characters of Hindi text")
        
        # For Hindi, we'll use a simpler approach since our extractive summarization
        # might not work well with Hindi. We'll return a portion of the transcript.
        try:
            # Try to tokenize Hindi text into sentences
            sentences = sent_tokenize(text)
            summary = " ".join(sentences[:min(10, len(sentences))])
        except Exception as e:
            print(f"Error in Hindi summarization: {str(e)}")
            summary = text[:1000] + "... (Summary unavailable, showing transcript excerpt)"
        
        # Add better debug output
        print(f"Returning Hindi summary response with {len(summary)} characters")
        return jsonify({"summary": summary})
    except Exception as e:
        print(f"Error in Hindi transcript: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists("sameAudio.mp3"):
                os.remove("sameAudio.mp3")
                print("Audio file deleted successfully")
        except Exception as e:
            print(f"Error deleting audio file: {str(e)}")

@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")