# from flask import Flask, request, jsonify
# import yt_dlp
# import assemblyai as aai
# import os
# from transformers import pipeline
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import TruncatedSVD
# import nltk
# from nltk.tokenize import sent_tokenize
# from langdetect import detect
# from flask_cors import CORS  # Add CORS support

# # Download NLTK data - this is crucial for the sent_tokenize function to work
# nltk.download('punkt')
# # Remove punkt_tab download as it's causing errors and is not needed
# # nltk.download('punkt_tab')

# app = Flask(__name__)
# # Configure CORS properly to allow requests from your React frontend
# CORS(app, resources={r"/*": {"origins": "*"}})

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
#     try:
#         sentences = sent_tokenize(transcript)
        
#         if len(sentences) <= 5:  # If transcript is very short, return it as is
#             return transcript
        
#         # Vectorize sentences
#         vectorizer = CountVectorizer(stop_words='english')
        
#         try:
#             X = vectorizer.fit_transform(sentences)
            
#             # Check if we can perform SVD
#             if X.shape[0] < 2 or X.shape[1] < 2:
#                 return transcript
                
#             # Perform Truncated SVD for dimensionality reduction
#             n_components = min(1, X.shape[1] - 1)  # Ensure we don't exceed matrix dimensions
#             svd = TruncatedSVD(n_components=n_components, random_state=42)
#             svd.fit(X)
#             components = svd.transform(X)
            
#             # Rank sentences based on the first singular vector
#             ranked_sentences = [item[0] for item in sorted(enumerate(components), key=lambda item: -item[1])]
            
#             # Select top sentences for summary (at least 3 sentences or 30% of the original)
#             num_sentences = max(3, int(0.3 * len(sentences)))
#             selected_sentences = sorted(ranked_sentences[:min(num_sentences, len(ranked_sentences))])
            
#             # Compile the final summary
#             summary = " ".join([sentences[idx] for idx in selected_sentences])
#             return summary
#         except Exception as e:
#             print(f"Error during summarization process: {str(e)}")
#             # Return a shortened version of transcript if summarization fails
#             return " ".join(sentences[:min(10, len(sentences))])
#     except Exception as e:
#         print(f"Error in sentence tokenization: {str(e)}")
#         # If tokenization fails, return the first 500 characters as a fallback
#         return transcript
#         # return transcript[:500] + "... (Summary truncated due to processing error)"

# @app.route("/getEnglishTranscript", methods=["GET"])
# def get_english_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing: {videolink}")

#     try:
#         audio_file = download_audio(videolink)
        
#         # Add more robust error checking for audio download
#         if not os.path.exists(audio_file):
#             raise Exception(f"Audio file {audio_file} was not created successfully")
        
#         # Check audio file size
#         audio_size = os.path.getsize(audio_file)
#         print(f"Audio file size: {audio_size} bytes")
        
#         if audio_size < 1000:  # Less than 1KB, likely an empty file
#             raise Exception("Audio file is too small, possibly download failed")

#         transcriber = aai.Transcriber()
        
#         # Add timeout and more error handling
#         try:
#             transcript = transcriber.transcribe(audio_file, timeout=300)  # 5-minute timeout
#         except Exception as transcribe_error:
#             print(f"Transcription error: {transcribe_error}")
#             raise

#         print("transcript", transcript)
        
#         if not transcript or not transcript.text:
#             return jsonify({"error": "Failed to generate transcript"}), 500
            
#         # Generate summary from transcript
#         text = transcript.text
#         print(f"Successfully transcribed {len(text)} characters of text")
        
#         try:
#             summary = extractive_summarization(text)
#             print(f"Generated summary of {len(summary)} characters")
#         except Exception as e:
#             print(f"Error in summarization: {str(e)}")
#             summary = text[:1000] + "... (Summary unavailable, showing transcript excerpt)"
        
#         # Add better debug output
#         print(f"Returning summary response with {len(summary)} characters")
#         return jsonify({"summary": summary})
#     except Exception as e:
#         # Log the full traceback
#         import traceback
#         traceback.print_exc()
        
#         print(f"Comprehensive error in English transcript: {str(e)}")
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
#     finally:
#         try:
#             if os.path.exists("sameAudio.mp3"):
#                 os.remove("sameAudio.mp3")
#                 print("Audio file deleted successfully")
#         except Exception as e:
#             print(f"Error deleting audio file: {str(e)}")

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
        
#         text = transcript.text
#         print(f"Successfully transcribed {len(text)} characters of Hindi text")
        
#         # For Hindi, we'll use a simpler approach since our extractive summarization
#         # might not work well with Hindi. We'll return a portion of the transcript.
#         try:
#             # Try to tokenize Hindi text into sentences
#             sentences = sent_tokenize(text)
#             summary = " ".join(sentences[:min(10, len(sentences))])
#         except Exception as e:
#             print(f"Error in Hindi summarization: {str(e)}")
#             summary = text[:1000] + "... (Summary unavailable, showing transcript excerpt)"
        
#         # Add better debug output
#         print(f"Returning Hindi summary response with {len(summary)} characters")
#         return jsonify({"summary": summary})
#     except Exception as e:
#         print(f"Error in Hindi transcript: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#     finally:
#         try:
#             if os.path.exists("sameAudio.mp3"):
#                 os.remove("sameAudio.mp3")
#                 print("Audio file deleted successfully")
#         except Exception as e:
#             print(f"Error deleting audio file: {str(e)}")

# @app.route("/ping", methods=["GET"])
# def ping():
#     """Health check endpoint"""
#     return jsonify({"status": "ok"}), 200

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0")


















# from flask import Flask, request, jsonify
# import yt_dlp
# import assemblyai as aai
# import os
# from transformers import pipeline
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import TruncatedSVD
# import nltk
# from nltk.tokenize import sent_tokenize
# from langdetect import detect
# from flask_cors import CORS  # Add CORS support

# # Download NLTK data - this is crucial for the sent_tokenize function to work
# nltk.download('punkt')

# app = Flask(__name__)
# # Configure CORS properly to allow requests from your React frontend
# CORS(app, resources={r"/*": {"origins": "*"}})

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
#     try:
#         sentences = sent_tokenize(transcript)
        
#         if len(sentences) <= 5:  # If transcript is very short, return it as is
#             return transcript
        
#         # Vectorize sentences
#         vectorizer = CountVectorizer(stop_words='english')
        
#         try:
#             X = vectorizer.fit_transform(sentences)
            
#             # Check if we can perform SVD
#             if X.shape[0] < 2 or X.shape[1] < 2:
#                 return transcript
                
#             # Perform Truncated SVD for dimensionality reduction
#             n_components = min(1, X.shape[1] - 1)  # Ensure we don't exceed matrix dimensions
#             svd = TruncatedSVD(n_components=n_components, random_state=42)
#             svd.fit(X)
#             components = svd.transform(X)
            
#             # Rank sentences based on the first singular vector
#             ranked_sentences = [item[0] for item in sorted(enumerate(components), key=lambda item: -item[1])]
            
#             # Select top sentences for summary (at least 3 sentences or 30% of the original)
#             num_sentences = max(3, int(0.3 * len(sentences)))
#             selected_sentences = sorted(ranked_sentences[:min(num_sentences, len(ranked_sentences))])
            
#             # Compile the final summary
#             summary = " ".join([sentences[idx] for idx in selected_sentences])
#             return summary
#         except Exception as e:
#             print(f"Error during summarization process: {str(e)}")
#             # Return a shortened version of transcript if summarization fails
#             return " ".join(sentences[:min(10, len(sentences))])
#     except Exception as e:
#         print(f"Error in sentence tokenization: {str(e)}")
#         # If tokenization fails, return the transcript as fallback
#         return transcript

# @app.route("/getEnglishTranscript", methods=["GET"])
# def get_english_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing: {videolink}")

#     try:
#         audio_file = download_audio(videolink)
        
#         # Add more robust error checking for audio download
#         if not os.path.exists(audio_file):
#             raise Exception(f"Audio file {audio_file} was not created successfully")
        
#         # Check audio file size
#         audio_size = os.path.getsize(audio_file)
#         print(f"Audio file size: {audio_size} bytes")
        
#         if audio_size < 1000:  # Less than 1KB, likely an empty file
#             raise Exception("Audio file is too small, possibly download failed")

#         transcriber = aai.Transcriber()
        
#         # Remove the timeout parameter which was causing the error
#         try:
#             transcript = transcriber.transcribe(audio_file)
#         except Exception as transcribe_error:
#             print(f"Transcription error: {transcribe_error}")
#             raise

#         print("transcript", transcript)
        
#         if not transcript or not transcript.text:
#             return jsonify({"error": "Failed to generate transcript"}), 500
            
#         # Generate summary from transcript
#         text = transcript.text
#         print(f"Successfully transcribed {len(text)} characters of text")
        
#         try:
#             summary = extractive_summarization(text)
#             print(f"Generated summary of {len(summary)} characters")
#         except Exception as e:
#             print(f"Error in summarization: {str(e)}")
#             summary = text[:1000] + "... (Summary unavailable, showing transcript excerpt)"
        
#         # Add better debug output
#         print(f"Returning summary response with {len(summary)} characters")
#         return jsonify({"summary": summary})
#     except Exception as e:
#         # Log the full traceback
#         import traceback
#         traceback.print_exc()
        
#         print(f"Comprehensive error in English transcript: {str(e)}")
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
#     finally:
#         try:
#             if os.path.exists("sameAudio.mp3"):
#                 os.remove("sameAudio.mp3")
#                 print("Audio file deleted successfully")
#         except Exception as e:
#             print(f"Error deleting audio file: {str(e)}")

# @app.route("/getHindiTranscript", methods=["GET"])
# def get_hindi_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing (Hindi): {videolink}")

#     try:
#         audio_file = download_audio(videolink)
        
#         # Add error checking for audio download
#         if not os.path.exists(audio_file):
#             raise Exception(f"Audio file {audio_file} was not created successfully")
        
#         # Check audio file size
#         audio_size = os.path.getsize(audio_file)
#         print(f"Audio file size: {audio_size} bytes")
        
#         if audio_size < 1000:  # Less than 1KB, likely an empty file
#             raise Exception("Audio file is too small, possibly download failed")
            
#         config = aai.TranscriptionConfig(language_code="hi")
#         transcriber = aai.Transcriber(config=config)
        
#         try:
#             transcript = transcriber.transcribe(audio_file)
#         except Exception as transcribe_error:
#             print(f"Hindi transcription error: {transcribe_error}")
#             raise
        
#         if not transcript or not transcript.text:
#             return jsonify({"error": "Failed to generate Hindi transcript"}), 500
        
#         text = transcript.text
#         print(f"Successfully transcribed {len(text)} characters of Hindi text")
        
#         # For Hindi, we'll use a simpler approach since our extractive summarization
#         # might not work well with Hindi. We'll return a portion of the transcript.
#         try:
#             # Try to tokenize Hindi text into sentences
#             sentences = sent_tokenize(text)
#             summary = " ".join(sentences[:min(10, len(sentences))])
#         except Exception as e:
#             print(f"Error in Hindi summarization: {str(e)}")
#             summary = text[:1000] + "... (Summary unavailable, showing transcript excerpt)"
        
#         # Add better debug output
#         print(f"Returning Hindi summary response with {len(summary)} characters")
#         return jsonify({"summary": summary})
#     except Exception as e:
#         # Log the full traceback
#         import traceback
#         traceback.print_exc()
        
#         print(f"Error in Hindi transcript: {str(e)}")
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
#     finally:
#         try:
#             if os.path.exists("sameAudio.mp3"):
#                 os.remove("sameAudio.mp3")
#                 print("Audio file deleted successfully")
#         except Exception as e:
#             print(f"Error deleting audio file: {str(e)}")

# @app.route("/ping", methods=["GET"])
# def ping():
#     """Health check endpoint"""
#     return jsonify({"status": "ok"}), 200

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0")











# somewhat correctone
# from flask import Flask, request, jsonify
# import yt_dlp
# import assemblyai as aai
# import os
# from transformers import pipeline
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import TruncatedSVD
# import nltk
# from nltk.tokenize import sent_tokenize
# from langdetect import detect
# from flask_cors import CORS  # Add CORS support

# # Download NLTK data - this is crucial for the sent_tokenize function to work
# nltk.download('punkt')

# app = Flask(__name__)
# # Configure CORS properly to allow requests from your React frontend
# CORS(app, resources={r"/*": {"origins": "*"}})

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
#     try:
#         sentences = sent_tokenize(transcript)
        
#         if len(sentences) <= 5:  # If transcript is very short, return it as is
#             return transcript
        
#         # Vectorize sentences
#         vectorizer = CountVectorizer(stop_words='english')
        
#         try:
#             X = vectorizer.fit_transform(sentences)
            
#             # Check if we can perform SVD
#             if X.shape[0] < 2 or X.shape[1] < 2:
#                 return transcript
                
#             # Perform Truncated SVD for dimensionality reduction
#             n_components = min(1, X.shape[1] - 1)  # Ensure we don't exceed matrix dimensions
#             svd = TruncatedSVD(n_components=n_components, random_state=42)
#             svd.fit(X)
#             components = svd.transform(X)
            
#             # Rank sentences based on the first singular vector
#             ranked_sentences = [item[0] for item in sorted(enumerate(components), key=lambda item: -item[1])]
            
#             # Select top sentences for summary (at least 3 sentences or 30% of the original)
#             num_sentences = max(3, int(0.3 * len(sentences)))
#             selected_sentences = sorted(ranked_sentences[:min(num_sentences, len(ranked_sentences))])
            
#             # Compile the final summary
#             summary = " ".join([sentences[idx] for idx in selected_sentences])
#             return summary
#         except Exception as e:
#             print(f"Error during summarization process: {str(e)}")
#             # Return a shortened version of transcript if summarization fails
#             return " ".join(sentences[:min(10, len(sentences))])
#     except Exception as e:
#         print(f"Error in sentence tokenization: {str(e)}")
#         # If tokenization fails, return the transcript as fallback
#         return transcript

# @app.route("/getEnglishTranscript", methods=["GET"])
# def get_english_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing: {videolink}")
    
#     # For debugging, create a temporary direct response for testing
#     try_direct_response = False  # Set to True to test without full processing
#     if try_direct_response:
#         return jsonify({"summary": "This is a test summary to verify the API works without processing."})

#     try:
#         audio_file = download_audio(videolink)
        
#         # Add more robust error checking for audio download
#         if not os.path.exists(audio_file):
#             raise Exception(f"Audio file {audio_file} was not created successfully")
        
#         # Check audio file size
#         audio_size = os.path.getsize(audio_file)
#         print(f"Audio file size: {audio_size} bytes")
        
#         if audio_size < 1000:  # Less than 1KB, likely an empty file
#             raise Exception("Audio file is too small, possibly download failed")

#         # Print AssemblyAI version to help with debugging
#         try:
#             print(f"AssemblyAI version: {aai.__version__}")
#         except AttributeError:
#             print("Could not determine AssemblyAI version")

#         transcriber = aai.Transcriber()
        
#         # Remove the timeout parameter which was causing the error
#         try:
#             print("Starting transcription...")
#             transcript = transcriber.transcribe(audio_file)
#             print("Transcription complete.")
#         except Exception as transcribe_error:
#             print(f"Transcription error: {transcribe_error}")
#             raise

#         print("transcript", transcript)
        
#         # Debug: Print more information about the transcript object
#         print(f"Transcript type: {type(transcript)}")
#         print(f"Transcript attributes: {dir(transcript)}")
        
#         # Check if transcript exists and has the expected attributes
#         if not transcript:
#             return jsonify({"error": "Failed to generate transcript - transcript is None"}), 500
            
#         # Check if transcript has text attribute
#         try:
#             text = transcript.text
#             print(f"Successfully transcribed {len(text)} characters of text")
#         except AttributeError as ae:
#             print(f"AttributeError accessing transcript.text: {ae}")
#             # Try to access the text using a different approach based on transcript object structure
#             try:
#                 # Check if it might be under a different attribute name
#                 possible_attrs = ['transcript', 'content', 'data', 'transcription', 'result']
#                 for attr in possible_attrs:
#                     if hasattr(transcript, attr):
#                         text = getattr(transcript, attr)
#                         if isinstance(text, str) and text:
#                             print(f"Found text under attribute '{attr}'")
#                             break
#                 else:
#                     # If we couldn't find the text in any of the expected attributes
#                     return jsonify({"error": f"Failed to access transcript text: {ae}"}), 500
#             except Exception as e2:
#                 print(f"Second attempt to access transcript text failed: {e2}")
#                 return jsonify({"error": f"Failed to access transcript text: {ae}, {e2}"}), 500
        
#         try:
#             summary = extractive_summarization(text)
#             print(f"Generated summary of {len(summary)} characters")
#         except Exception as e:
#             print(f"Error in summarization: {str(e)}")
#             summary = text[:1000] + "... (Summary unavailable, showing transcript excerpt)"
        
#         # Add better debug output
#         print(f"Returning summary response with {len(summary)} characters")
#         return jsonify({"summary": summary})
#     except Exception as e:
#         # Log the full traceback
#         import traceback
#         traceback.print_exc()
        
#         print(f"Comprehensive error in English transcript: {str(e)}")
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
#     finally:
#         try:
#             if os.path.exists("sameAudio.mp3"):
#                 os.remove("sameAudio.mp3")
#                 print("Audio file deleted successfully")
#         except Exception as e:
#             print(f"Error deleting audio file: {str(e)}")

# @app.route("/getHindiTranscript", methods=["GET"])
# def get_hindi_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing (Hindi): {videolink}")

#     try:
#         audio_file = download_audio(videolink)
        
#         # Add error checking for audio download
#         if not os.path.exists(audio_file):
#             raise Exception(f"Audio file {audio_file} was not created successfully")
        
#         # Check audio file size
#         audio_size = os.path.getsize(audio_file)
#         print(f"Audio file size: {audio_size} bytes")
        
#         if audio_size < 1000:  # Less than 1KB, likely an empty file
#             raise Exception("Audio file is too small, possibly download failed")
            
#         config = aai.TranscriptionConfig(language_code="hi")
#         transcriber = aai.Transcriber(config=config)
        
#         try:
#             transcript = transcriber.transcribe(audio_file)
#         except Exception as transcribe_error:
#             print(f"Hindi transcription error: {transcribe_error}")
#             raise
        
#         if not transcript or not transcript.text:
#             return jsonify({"error": "Failed to generate Hindi transcript"}), 500
        
#         text = transcript.text
#         print(f"Successfully transcribed {len(text)} characters of Hindi text")
        
#         # For Hindi, we'll use a simpler approach since our extractive summarization
#         # might not work well with Hindi. We'll return a portion of the transcript.
#         try:
#             # Try to tokenize Hindi text into sentences
#             sentences = sent_tokenize(text)
#             summary = " ".join(sentences[:min(10, len(sentences))])
#         except Exception as e:
#             print(f"Error in Hindi summarization: {str(e)}")
#             summary = text[:1000] + "... (Summary unavailable, showing transcript excerpt)"
        
#         # Add better debug output
#         print(f"Returning Hindi summary response with {len(summary)} characters")
#         return jsonify({"summary": summary})
#     except Exception as e:
#         # Log the full traceback
#         import traceback
#         traceback.print_exc()
        
#         print(f"Error in Hindi transcript: {str(e)}")
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
#     finally:
#         try:
#             if os.path.exists("sameAudio.mp3"):
#                 os.remove("sameAudio.mp3")
#                 print("Audio file deleted successfully")
#         except Exception as e:
#             print(f"Error deleting audio file: {str(e)}")

# @app.route("/ping", methods=["GET"])
# def ping():
#     """Health check endpoint"""
#     return jsonify({"status": "ok"}), 200

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0")






# --------
from flask import Flask, request, jsonify
import yt_dlp
import assemblyai as aai
import os
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from flask_cors import CORS  # Add CORS support
import time
import traceback

# Download ONLY the punkt package that's needed (not punkt_tab)
nltk.download('punkt')

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

# def extractive_summarization(transcript):
#     """
#     Summarizes the input transcript using the Extractive Summarization technique.
#     """
#     try:
#         # If the transcript is empty, return a default message
#         if not transcript or len(transcript.strip()) == 0:
#             return "The transcript appears to be empty. Please try a different video."
        
#         # Use the NLTK punkt tokenizer to split into sentences
#         sentences = sent_tokenize(transcript)
        
#         print(f"Number of sentences detected: {len(sentences)}")
        
#         if len(sentences) <= 5:  # If transcript is very short, return it as is
#             return transcript
        
#         # Vectorize sentences
#         vectorizer = CountVectorizer(stop_words='english')
        
#         try:
#             X = vectorizer.fit_transform(sentences)
            
#             # Check if we can perform SVD
#             if X.shape[0] < 2 or X.shape[1] < 2:
#                 return transcript
                
#             # Perform Truncated SVD for dimensionality reduction
#             n_components = min(1, X.shape[1] - 1)  # Ensure we don't exceed matrix dimensions
#             svd = TruncatedSVD(n_components=n_components, random_state=42)
#             svd.fit(X)
#             components = svd.transform(X)
            
#             # Rank sentences based on the first singular vector
#             ranked_sentences = [item[0] for item in sorted(enumerate(components), key=lambda item: -item[1])]
            
#             # Select top sentences for summary (at least 3 sentences or 30% of the original)
#             num_sentences = max(3, int(0.3 * len(sentences)))
#             selected_sentences = sorted(ranked_sentences[:min(num_sentences, len(ranked_sentences))])
            
#             # Compile the final summary
#             summary = " ".join([sentences[idx] for idx in selected_sentences])
#             return summary
#         except Exception as e:
#             print(f"Error during summarization process: {str(e)}")
#             # Return a shortened version of transcript if summarization fails
#             return " ".join(sentences[:min(10, len(sentences))])
#     except Exception as e:
#         print(f"Error in sentence tokenization: {str(e)}")
#         # If tokenization fails, return a truncated version of the transcript
#         if transcript and len(transcript) > 0:
#             return transcript[:1000] + "... (Summary unavailable due to processing error, showing partial transcript)"
            
#         else:
#             return "Unable to generate summary. The video may not contain sufficient spoken content."





def extractive_summarization(transcript):
    """
    Summarizes the input transcript using the Extractive Summarization technique.
    Ensures summaries end with complete sentences (ending with full stops).
    """
    try:
        # If the transcript is empty, return a default message
        if not transcript or len(transcript.strip()) == 0:
            return "The transcript appears to be empty. Please try a different video."
        
        # Use the NLTK punkt tokenizer to split into sentences
        sentences = sent_tokenize(transcript)
        
        print(f"Number of sentences detected: {len(sentences)}")
        
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
            # Return first few complete sentences if summarization fails
            # Instead of truncating with "...", return complete sentences
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

@app.route("/getEnglishTranscript", methods=["GET"])
def get_english_transcript():
    videolink = request.args.get('videolink')
    if not videolink:
        return jsonify({"error": "No video link provided"}), 400

    print(f"Downloading and transcribing: {videolink}")

    try:
        audio_file = download_audio(videolink)
        
        # Add more robust error checking for audio download
        if not os.path.exists(audio_file):
            raise Exception(f"Audio file {audio_file} was not created successfully")
        
        # Check audio file size
        audio_size = os.path.getsize(audio_file)
        print(f"Audio file size: {audio_size} bytes")
        
        if audio_size < 1000:  # Less than 1KB, likely an empty file
            raise Exception("Audio file is too small, possibly download failed")

        # Print AssemblyAI version to help with debugging
        try:
            print(f"AssemblyAI version: {aai.__version__}")
        except AttributeError:
            print("Could not determine AssemblyAI version")

        transcriber = aai.Transcriber()
        
        # Try to get transcript with retries
        text = get_transcript_with_retry(transcriber, audio_file)
        
        if not text:
            # If text is still empty, return a helpful error message
            return jsonify({"summary": "The video does not appear to contain any recognizable speech. Please try another video."}), 200
        
        print(f"Successfully transcribed {len(text)} characters of text")
        
        # Generate summary from transcript
        try:
            summary = extractive_summarization(text)
            print(f"Generated summary of {len(summary)} characters")
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            summary = text[:1000]
        
        # Return an empty response if summary is empty
        if not summary or len(summary.strip()) == 0:
            summary = "Unable to generate summary for this video. The content may not contain sufficient speech."
        
        # Add better debug output
        print(f"Returning summary response with {len(summary)} characters")
        return jsonify({"summary": summary})
    except Exception as e:
        # Log the full traceback
        traceback.print_exc()
        
        print(f"Comprehensive error in English transcript: {str(e)}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    finally:
        try:
            if os.path.exists("sameAudio.mp3"):
                os.remove("sameAudio.mp3")
                print("Audio file deleted successfully")
        except Exception as e:
            print(f"Error deleting audio file: {str(e)}")

# @app.route("/getHindiTranscript", methods=["GET"])
# def get_hindi_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing (Hindi): {videolink}")

#     try:
#         audio_file = download_audio(videolink)
        
#         # Add error checking for audio download
#         if not os.path.exists(audio_file):
#             raise Exception(f"Audio file {audio_file} was not created successfully")
        
#         # Check audio file size
#         audio_size = os.path.getsize(audio_file)
#         print(f"Audio file size: {audio_size} bytes")
        
#         if audio_size < 1000:  # Less than 1KB, likely an empty file
#             raise Exception("Audio file is too small, possibly download failed")
            
#         config = aai.TranscriptionConfig(language_code="hi")
#         transcriber = aai.Transcriber(config=config)
        
#         # Try to get transcript with retries
#         try:
#             text = get_transcript_with_retry(transcriber, audio_file)
#         except Exception as e:
#             print(f"Error getting Hindi transcript: {str(e)}")
#             return jsonify({"summary": "Failed to transcribe Hindi audio. The video may not contain Hindi speech."}), 200
        
#         if not text:
#             # If text is still empty, return a helpful error message
#             return jsonify({"summary": "The video does not appear to contain any recognizable Hindi speech. Please try another video."}), 200
        
#         print(f"Successfully transcribed {len(text)} characters of Hindi text")
        
#         # For Hindi, we'll use a simpler approach since our extractive summarization
#         # might not work well with Hindi. We'll return a portion of the transcript.
#         try:
#             # Try to tokenize Hindi text into sentences
#             sentences = sent_tokenize(text)
#             summary = " ".join(sentences[:min(10, len(sentences))])
#         except Exception as e:
#             print(f"Error in Hindi summarization: {str(e)}")
#             summary = text[:1000]
        
#         # Add better debug output
#         print(f"Returning Hindi summary response with {len(summary)} characters")
#         return jsonify({"summary": summary})
#     except Exception as e:
#         # Log the full traceback
#         traceback.print_exc()
        
#         print(f"Error in Hindi transcript: {str(e)}")
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
#     finally:
#         try:
#             if os.path.exists("sameAudio.mp3"):
#                 os.remove("sameAudio.mp3")
#                 print("Audio file deleted successfully")
#         except Exception as e:
#             print(f"Error deleting audio file: {str(e)}")


@app.route("/getHindiTranscript", methods=["GET"])
def get_hindi_transcript():
    videolink = request.args.get('videolink')
    if not videolink:
        return jsonify({"error": "No video link provided"}), 400

    print(f"Downloading and transcribing (Hindi): {videolink}")

    try:
        audio_file = download_audio(videolink)
        
        # Add error checking for audio download
        if not os.path.exists(audio_file):
            raise Exception(f"Audio file {audio_file} was not created successfully")
        
        # Check audio file size
        audio_size = os.path.getsize(audio_file)
        print(f"Audio file size: {audio_size} bytes")
        
        if audio_size < 1000:  # Less than 1KB, likely an empty file
            raise Exception("Audio file is too small, possibly download failed")
            
        config = aai.TranscriptionConfig(language_code="hi")
        transcriber = aai.Transcriber(config=config)
        
        # Try to get transcript with retries
        try:
            text = get_transcript_with_retry(transcriber, audio_file)
        except Exception as e:
            print(f"Error getting Hindi transcript: {str(e)}")
            return jsonify({"summary": "Failed to transcribe Hindi audio. The video may not contain Hindi speech."}), 200
        
        if not text:
            # If text is still empty, return a helpful error message
            return jsonify({"summary": "The video does not appear to contain any recognizable Hindi speech. Please try another video."}), 200
        
        print(f"Successfully transcribed {len(text)} characters of Hindi text")
        
        # For Hindi, we'll use a simpler approach since our extractive summarization
        # might not work well with Hindi. We'll return a portion of the transcript.
        try:
            # Try to tokenize Hindi text into sentences
            # Look for both Hindi purna viram (।) and Latin period (.) as sentence delimiters
            hindi_sentences = []
            current_sentence = ""
            
            for char in text:
                current_sentence += char
                if char in ['।', '.']:  # Both Hindi purna viram and Latin period
                    hindi_sentences.append(current_sentence.strip())
                    current_sentence = ""
            
            # Add any remaining text as a sentence if it exists
            if current_sentence.strip():
                hindi_sentences.append(current_sentence.strip())
            
            # If tokenization worked, use the first few sentences
            if hindi_sentences:
                summary = " ".join(hindi_sentences[:min(25, len(hindi_sentences))])
            else:
                # Fallback to NLTK's sent_tokenize with additional handling
                sentences = sent_tokenize(text)
                summary = " ".join(sentences[:min(25, len(sentences))])
        except Exception as e:
            print(f"Error in Hindi summarization: {str(e)}")
            # If all else fails, truncate the text but ensure it ends with a Hindi purna viram or period
            partial_text = text[:1000]
            # Find the last purna viram in the partial text
            last_purna_viram = partial_text.rfind('।')
            last_period = partial_text.rfind('.')
            
            # Use the later of the two ending marks
            last_mark = max(last_purna_viram, last_period)
            
            if last_mark > 0:
                summary = text[:last_mark + 1]  # Include the punctuation mark
            else:
                summary = partial_text  # If no sentence ending found, use as is
        
        # Add better debug output
        print(f"Returning Hindi summary response with {len(summary)} characters")
        return jsonify({"summary": summary})
    except Exception as e:
        # Log the full traceback
        traceback.print_exc()
        
        print(f"Error in Hindi transcript: {str(e)}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
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



# for development
# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0")


# for production
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")








# implementing alternate approach   

# from flask import Flask, request, jsonify
# import yt_dlp
# import os
# import requests
# import json
# import nltk
# from nltk.tokenize import sent_tokenize
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import TruncatedSVD
# from flask_cors import CORS

# # Download NLTK data
# nltk.download('punkt')

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # AssemblyAI API key
# ASSEMBLYAI_API_KEY = "85dfd1af7ea047f1abf886314afcbd7d"
# ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
# ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

# # YouTube download options
# ydl_opts = {
#     'format': 'm4a/bestaudio/best',
#     'outtmpl': 'sameAudio.%(ext)s',
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#     }]
# }

# def download_audio(video_url):
#     """Download audio from YouTube video"""
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         error_code = ydl.download([video_url])
#     return "sameAudio.mp3"

# def upload_to_assemblyai(audio_file_path):
#     """Upload audio file to AssemblyAI"""
#     print(f"Uploading file to AssemblyAI: {audio_file_path}")
    
#     headers = {
#         "authorization": ASSEMBLYAI_API_KEY,
#         "content-type": "application/json"
#     }
    
#     with open(audio_file_path, "rb") as f:
#         response = requests.post(
#             ASSEMBLYAI_UPLOAD_URL,
#             headers=headers,
#             data=f
#         )
    
#     if response.status_code != 200:
#         print(f"Upload failed with status code: {response.status_code}")
#         print(f"Response: {response.text}")
#         raise Exception(f"Upload failed: {response.text}")
        
#     upload_url = response.json()["upload_url"]
#     print(f"File uploaded. Upload URL: {upload_url}")
#     return upload_url

# def start_transcription(upload_url, language_code="en"):
#     """Start transcription job at AssemblyAI"""
#     print(f"Starting transcription job for audio at {upload_url}")
    
#     headers = {
#         "authorization": ASSEMBLYAI_API_KEY,
#         "content-type": "application/json"
#     }
    
#     data = {
#         "audio_url": upload_url,
#         "language_code": language_code
#     }
    
#     response = requests.post(
#         ASSEMBLYAI_TRANSCRIPT_URL,
#         json=data,
#         headers=headers
#     )
    
#     if response.status_code != 200:
#         print(f"Transcription request failed with status code: {response.status_code}")
#         print(f"Response: {response.text}")
#         raise Exception(f"Transcription request failed: {response.text}")
        
#     transcript_id = response.json()["id"]
#     print(f"Transcription job started with ID: {transcript_id}")
#     return transcript_id

# def get_transcript(transcript_id):
#     """Get transcript from AssemblyAI when it's ready"""
#     headers = {
#         "authorization": ASSEMBLYAI_API_KEY,
#         "content-type": "application/json"
#     }
    
#     polling_endpoint = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"
    
#     print(f"Polling for transcript status at: {polling_endpoint}")
    
#     while True:
#         response = requests.get(polling_endpoint, headers=headers)
#         response_json = response.json()
        
#         if response_json["status"] == "completed":
#             print("Transcription completed successfully")
#             return response_json["text"]
#         elif response_json["status"] == "error":
#             raise Exception(f"Transcription failed: {response_json.get('error', 'Unknown error')}")
#         else:
#             print(f"Transcription status: {response_json['status']}. Waiting...")
#             import time
#             time.sleep(5)

# def extractive_summarization(transcript):
#     """
#     Summarizes the transcript using Extractive Summarization with LSA.
#     """
#     try:
#         sentences = sent_tokenize(transcript)
        
#         if len(sentences) <= 5:  # If transcript is very short, return it as is
#             return transcript
        
#         # Vectorize sentences
#         vectorizer = CountVectorizer(stop_words='english')
        
#         try:
#             X = vectorizer.fit_transform(sentences)
            
#             # Check if we can perform SVD
#             if X.shape[0] < 2 or X.shape[1] < 2:
#                 return transcript
                
#             # Perform Truncated SVD for dimensionality reduction
#             n_components = min(3, X.shape[1] - 1)  # Ensure we don't exceed matrix dimensions
#             svd = TruncatedSVD(n_components=n_components, random_state=42)
#             svd.fit(X)
#             components = svd.transform(X)
            
#             # Rank sentences based on the singular values
#             ranked_sentences = [item[0] for item in sorted(enumerate(components[:, 0]), key=lambda item: -abs(item[1]))]
            
#             # Select top sentences for summary (at least 3 sentences or 30% of the original)
#             num_sentences = max(3, int(0.3 * len(sentences)))
#             selected_sentences = sorted(ranked_sentences[:min(num_sentences, len(ranked_sentences))])
            
#             # Compile the final summary
#             summary = " ".join([sentences[idx] for idx in selected_sentences])
#             return summary
#         except Exception as e:
#             print(f"Error during summarization process: {str(e)}")
#             # Return a shortened version of transcript if summarization fails
#             return " ".join(sentences[:min(10, len(sentences))])
#     except Exception as e:
#         print(f"Error in sentence tokenization: {str(e)}")
#         # Return first 1000 chars if everything fails
#         return transcript[:1000] + "... (Summary unavailable, showing transcript excerpt)"

# @app.route("/getEnglishTranscript", methods=["GET"])
# def get_english_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing: {videolink}")

#     try:
#         # Download audio
#         audio_file = download_audio(videolink)
#         if not os.path.exists(audio_file):
#             raise Exception(f"Audio file {audio_file} was not created successfully")
        
#         audio_size = os.path.getsize(audio_file)
#         print(f"Audio file size: {audio_size} bytes")
        
#         if audio_size < 1000:
#             raise Exception("Audio file is too small, possibly download failed")
        
#         # Process with AssemblyAI (direct API approach)
#         upload_url = upload_to_assemblyai(audio_file)
#         transcript_id = start_transcription(upload_url)
#         transcript_text = get_transcript(transcript_id)
        
#         print(f"Successfully transcribed {len(transcript_text)} characters of text")
        
#         # Generate summary
#         summary = extractive_summarization(transcript_text)
#         print(f"Generated summary of {len(summary)} characters")
        
#         return jsonify({"summary": summary})
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
        
#         print(f"Error in English transcript: {str(e)}")
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
#     finally:
#         try:
#             if os.path.exists("sameAudio.mp3"):
#                 os.remove("sameAudio.mp3")
#                 print("Audio file deleted successfully")
#         except Exception as e:
#             print(f"Error deleting audio file: {str(e)}")

# @app.route("/getHindiTranscript", methods=["GET"])
# def get_hindi_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing (Hindi): {videolink}")

#     try:
#         # Download audio
#         audio_file = download_audio(videolink)
#         if not os.path.exists(audio_file):
#             raise Exception(f"Audio file {audio_file} was not created successfully")
        
#         audio_size = os.path.getsize(audio_file)
#         print(f"Audio file size: {audio_size} bytes")
        
#         if audio_size < 1000:
#             raise Exception("Audio file is too small, possibly download failed")
        
#         # Process with AssemblyAI (direct API approach)
#         upload_url = upload_to_assemblyai(audio_file)
#         transcript_id = start_transcription(upload_url, language_code="hi")
#         transcript_text = get_transcript(transcript_id)
        
#         print(f"Successfully transcribed {len(transcript_text)} characters of Hindi text")
        
#         # For Hindi, use a simpler approach to summarization
#         try:
#             sentences = sent_tokenize(transcript_text)
#             summary = " ".join(sentences[:min(10, len(sentences))])
#         except Exception as e:
#             print(f"Error in Hindi summarization: {str(e)}")
#             summary = transcript_text[:1000] + "... (Summary unavailable, showing transcript excerpt)"
        
#         return jsonify({"summary": summary})
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
        
#         print(f"Error in Hindi transcript: {str(e)}")
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
#     finally:
#         try:
#             if os.path.exists("sameAudio.mp3"):
#                 os.remove("sameAudio.mp3")
#                 print("Audio file deleted successfully")
#         except Exception as e:
#             print(f"Error deleting audio file: {str(e)}")

# @app.route("/ping", methods=["GET"])
# def ping():
#     """Health check endpoint"""
#     return jsonify({"status": "ok"}), 200

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0")