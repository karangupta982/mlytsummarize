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
# from flask import Flask, request, jsonify
# import yt_dlp
# import assemblyai as aai
# import os
# import nltk
# from nltk.tokenize import sent_tokenize
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import TruncatedSVD
# from flask_cors import CORS  # Add CORS support
# import time
# import traceback

# # Download ONLY the punkt package that's needed (not punkt_tab)
# nltk.download('punkt')

# app = Flask(__name__)
# # Configure CORS properly to allow requests from your React frontend
# CORS(app, resources={r"/*": {"origins": "*"}})

# CORS(app, resources={r"/*": {"origins": [
#     "http://localhost:3000",
#     "https://mlytsummarize-kdso.vercel.app"
# ]}})

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

# # def extractive_summarization(transcript):
# #     """
# #     Summarizes the input transcript using the Extractive Summarization technique.
# #     """
# #     try:
# #         # If the transcript is empty, return a default message
# #         if not transcript or len(transcript.strip()) == 0:
# #             return "The transcript appears to be empty. Please try a different video."
        
# #         # Use the NLTK punkt tokenizer to split into sentences
# #         sentences = sent_tokenize(transcript)
        
# #         print(f"Number of sentences detected: {len(sentences)}")
        
# #         if len(sentences) <= 5:  # If transcript is very short, return it as is
# #             return transcript
        
# #         # Vectorize sentences
# #         vectorizer = CountVectorizer(stop_words='english')
        
# #         try:
# #             X = vectorizer.fit_transform(sentences)
            
# #             # Check if we can perform SVD
# #             if X.shape[0] < 2 or X.shape[1] < 2:
# #                 return transcript
                
# #             # Perform Truncated SVD for dimensionality reduction
# #             n_components = min(1, X.shape[1] - 1)  # Ensure we don't exceed matrix dimensions
# #             svd = TruncatedSVD(n_components=n_components, random_state=42)
# #             svd.fit(X)
# #             components = svd.transform(X)
            
# #             # Rank sentences based on the first singular vector
# #             ranked_sentences = [item[0] for item in sorted(enumerate(components), key=lambda item: -item[1])]
            
# #             # Select top sentences for summary (at least 3 sentences or 30% of the original)
# #             num_sentences = max(3, int(0.3 * len(sentences)))
# #             selected_sentences = sorted(ranked_sentences[:min(num_sentences, len(ranked_sentences))])
            
# #             # Compile the final summary
# #             summary = " ".join([sentences[idx] for idx in selected_sentences])
# #             return summary
# #         except Exception as e:
# #             print(f"Error during summarization process: {str(e)}")
# #             # Return a shortened version of transcript if summarization fails
# #             return " ".join(sentences[:min(10, len(sentences))])
# #     except Exception as e:
# #         print(f"Error in sentence tokenization: {str(e)}")
# #         # If tokenization fails, return a truncated version of the transcript
# #         if transcript and len(transcript) > 0:
# #             return transcript[:1000] + "... (Summary unavailable due to processing error, showing partial transcript)"
            
# #         else:
# #             return "Unable to generate summary. The video may not contain sufficient spoken content."





# def extractive_summarization(transcript):
#     """
#     Summarizes the input transcript using the Extractive Summarization technique.
#     Ensures summaries end with complete sentences (ending with full stops).
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
#             # Return first few complete sentences if summarization fails
#             # Instead of truncating with "...", return complete sentences
#             return " ".join(sentences[:min(10, len(sentences))])
#     except Exception as e:
#         print(f"Error in sentence tokenization: {str(e)}")
#         # If tokenization fails, return complete sentences from the beginning
#         if transcript and len(transcript) > 0:
#             # Get the first 1000 characters but ensure we end with a complete sentence
#             partial_text = transcript[:1000]
#             # Find the last period in the partial text
#             last_period_index = partial_text.rfind('.')
#             if last_period_index > 0:
#                 return transcript[:last_period_index + 1]  # Include the period
#             else:
#                 # If no period is found, return as is without ellipsis
#                 return partial_text
#         else:
#             return "Unable to generate summary. The video may not contain sufficient spoken content."

# def get_transcript_with_retry(transcriber, audio_file, max_retries=3):
#     """Attempts to get a transcript with retries"""
#     for attempt in range(max_retries):
#         try:
#             transcript = transcriber.transcribe(audio_file)
            
#             # Check if transcript has text
#             if hasattr(transcript, 'text') and transcript.text:
#                 text_content = transcript.text
#                 if len(text_content) > 0:
#                     print(f"Successful transcription on attempt {attempt+1} with {len(text_content)} characters")
#                     return text_content
#                 else:
#                     print(f"Empty transcript.text on attempt {attempt+1}")
#             else:
#                 print(f"No text attribute found on attempt {attempt+1}")
                
#                 # Try to find the text in other attributes
#                 for attr in ['transcript', 'result', 'content', 'output']:
#                     if hasattr(transcript, attr):
#                         content = getattr(transcript, attr)
#                         if isinstance(content, str) and len(content) > 0:
#                             return content
                
#                 # If all else fails, check if there's a summary attribute
#                 if hasattr(transcript, 'summary') and transcript.summary:
#                     return f"Summary from AssemblyAI: {transcript.summary}"
                    
#             # If we get here, we need to try again
#             if attempt < max_retries - 1:
#                 print(f"Retry transcription in 2 seconds...")
#                 time.sleep(2)
                
#         except Exception as e:
#             print(f"Transcription error on attempt {attempt+1}: {str(e)}")
#             if attempt < max_retries - 1:
#                 print(f"Retry transcription in 2 seconds...")
#                 time.sleep(2)
    
#     # If we get here, all retries failed
#     raise Exception("Failed to get transcript after multiple attempts")

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

#         # Print AssemblyAI version to help with debugging
#         try:
#             print(f"AssemblyAI version: {aai.__version__}")
#         except AttributeError:
#             print("Could not determine AssemblyAI version")

#         transcriber = aai.Transcriber()
        
#         # Try to get transcript with retries
#         text = get_transcript_with_retry(transcriber, audio_file)
        
#         if not text:
#             # If text is still empty, return a helpful error message
#             return jsonify({"summary": "The video does not appear to contain any recognizable speech. Please try another video."}), 200
        
#         print(f"Successfully transcribed {len(text)} characters of text")
        
#         # Generate summary from transcript
#         try:
#             summary = extractive_summarization(text)
#             print(f"Generated summary of {len(summary)} characters")
#         except Exception as e:
#             print(f"Error in summarization: {str(e)}")
#             summary = text[:1000]
        
#         # Return an empty response if summary is empty
#         if not summary or len(summary.strip()) == 0:
#             summary = "Unable to generate summary for this video. The content may not contain sufficient speech."
        
#         # Add better debug output
#         print(f"Returning summary response with {len(summary)} characters")
#         return jsonify({"summary": summary})
#     except Exception as e:
#         # Log the full traceback
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

# # @app.route("/getHindiTranscript", methods=["GET"])
# # def get_hindi_transcript():
# #     videolink = request.args.get('videolink')
# #     if not videolink:
# #         return jsonify({"error": "No video link provided"}), 400

# #     print(f"Downloading and transcribing (Hindi): {videolink}")

# #     try:
# #         audio_file = download_audio(videolink)
        
# #         # Add error checking for audio download
# #         if not os.path.exists(audio_file):
# #             raise Exception(f"Audio file {audio_file} was not created successfully")
        
# #         # Check audio file size
# #         audio_size = os.path.getsize(audio_file)
# #         print(f"Audio file size: {audio_size} bytes")
        
# #         if audio_size < 1000:  # Less than 1KB, likely an empty file
# #             raise Exception("Audio file is too small, possibly download failed")
            
# #         config = aai.TranscriptionConfig(language_code="hi")
# #         transcriber = aai.Transcriber(config=config)
        
# #         # Try to get transcript with retries
# #         try:
# #             text = get_transcript_with_retry(transcriber, audio_file)
# #         except Exception as e:
# #             print(f"Error getting Hindi transcript: {str(e)}")
# #             return jsonify({"summary": "Failed to transcribe Hindi audio. The video may not contain Hindi speech."}), 200
        
# #         if not text:
# #             # If text is still empty, return a helpful error message
# #             return jsonify({"summary": "The video does not appear to contain any recognizable Hindi speech. Please try another video."}), 200
        
# #         print(f"Successfully transcribed {len(text)} characters of Hindi text")
        
# #         # For Hindi, we'll use a simpler approach since our extractive summarization
# #         # might not work well with Hindi. We'll return a portion of the transcript.
# #         try:
# #             # Try to tokenize Hindi text into sentences
# #             sentences = sent_tokenize(text)
# #             summary = " ".join(sentences[:min(10, len(sentences))])
# #         except Exception as e:
# #             print(f"Error in Hindi summarization: {str(e)}")
# #             summary = text[:1000]
        
# #         # Add better debug output
# #         print(f"Returning Hindi summary response with {len(summary)} characters")
# #         return jsonify({"summary": summary})
# #     except Exception as e:
# #         # Log the full traceback
# #         traceback.print_exc()
        
# #         print(f"Error in Hindi transcript: {str(e)}")
# #         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
# #     finally:
# #         try:
# #             if os.path.exists("sameAudio.mp3"):
# #                 os.remove("sameAudio.mp3")
# #                 print("Audio file deleted successfully")
# #         except Exception as e:
# #             print(f"Error deleting audio file: {str(e)}")


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
#             # Look for both Hindi purna viram (।) and Latin period (.) as sentence delimiters
#             hindi_sentences = []
#             current_sentence = ""
            
#             for char in text:
#                 current_sentence += char
#                 if char in ['।', '.']:  # Both Hindi purna viram and Latin period
#                     hindi_sentences.append(current_sentence.strip())
#                     current_sentence = ""
            
#             # Add any remaining text as a sentence if it exists
#             if current_sentence.strip():
#                 hindi_sentences.append(current_sentence.strip())
            
#             # If tokenization worked, use the first few sentences
#             if hindi_sentences:
#                 summary = " ".join(hindi_sentences[:min(25, len(hindi_sentences))])
#             else:
#                 # Fallback to NLTK's sent_tokenize with additional handling
#                 sentences = sent_tokenize(text)
#                 summary = " ".join(sentences[:min(25, len(sentences))])
#         except Exception as e:
#             print(f"Error in Hindi summarization: {str(e)}")
#             # If all else fails, truncate the text but ensure it ends with a Hindi purna viram or period
#             partial_text = text[:1000]
#             # Find the last purna viram in the partial text
#             last_purna_viram = partial_text.rfind('।')
#             last_period = partial_text.rfind('.')
            
#             # Use the later of the two ending marks
#             last_mark = max(last_purna_viram, last_period)
            
#             if last_mark > 0:
#                 summary = text[:last_mark + 1]  # Include the punctuation mark
#             else:
#                 summary = partial_text  # If no sentence ending found, use as is
        
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

# @app.route("/ping", methods=["GET"])
# def ping():
#     """Health check endpoint"""
#     return jsonify({"status": "ok"}), 200



# # for development
# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0")


# # for production
# # if __name__ == "__main__":
# #     app.run(debug=False, host="0.0.0.0")










# ---------------  Cloud friendly code using yt caption

# from flask import Flask, request, jsonify
# import yt_dlp
# import assemblyai as aai
# import os
# import nltk
# from nltk.tokenize import sent_tokenize
# from sklearn.feature_extraction import text
# from sklearn.decomposition import TruncatedSVD
# from flask_cors import CORS
# import time
# import traceback
# import re
# from youtube_transcript_api import YouTubeTranscriptApi  

# # Download ONLY the punkt package that's needed (not punkt_tab)
# nltk.download('punkt')

# app = Flask(__name__)
# # Configure CORS properly to allow requests from your React frontend
# # CORS(app, resources={r"/*": {"origins": "*"}})
# CORS(app, resources={r"/*": {"origins": [
#     "http://localhost:5173",
#     "https://mlytsummarize-kdso.vercel.app",
#     "https://mlytsummarize-omjx.vercel.app"
# ]}})

# aai.settings.api_key = "85dfd1af7ea047f1abf886314afcbd7d"

# # Enhanced yt_dlp options for cloud environments
# ydl_opts = {
#     'format': 'worstaudio[ext=m4a]/worstaudio/worst',  # Use worst quality for faster processing
#     'outtmpl': '/tmp/sameAudio.%(ext)s',  # Use /tmp directory
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#     }],
#     'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
#     'referer': 'https://www.youtube.com/',
#     'sleep_interval': 1,
#     'max_sleep_interval': 3,
#     'retries': 2,
#     'fragment_retries': 2,
#     'socket_timeout': 30,
#     'nocheckcertificate': True,
# }

# def download_audio(video_url):
#     """Enhanced download_audio with better error handling"""
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             error_code = ydl.download([video_url])
#             if error_code != 0:
#                 raise Exception(f"yt_dlp failed with exit code: {error_code}")
        
#         # Check both possible locations
#         possible_files = ["/tmp/sameAudio.mp3", "sameAudio.mp3"]
#         for file_path in possible_files:
#             if os.path.exists(file_path):
#                 return file_path
        
#         raise Exception("Audio file not found after download")
#     except Exception as e:
#         raise Exception(f"Audio download failed: {str(e)}")

# def get_transcript_from_youtube_api(videolink, language='en'):
#     """Get transcript directly from YouTube's captions (no audio download needed)"""
#     try:
#         # Extract video ID from various YouTube URL formats
#         video_id_patterns = [
#             r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
#             r'youtu\.be\/([0-9A-Za-z_-]{11})',
#             r'youtube\.com\/embed\/([0-9A-Za-z_-]{11})'
#         ]
        
#         video_id = None
#         for pattern in video_id_patterns:
#             match = re.search(pattern, videolink)
#             if match:
#                 video_id = match.group(1)
#                 break
        
#         if not video_id:
#             raise Exception("Could not extract video ID from URL")
        
#         print(f"Extracted video ID: {video_id}")
        
#         # Try different language codes based on the requested language
#         if language == 'hi':
#             lang_codes = ['hi', 'hi-IN']
#         else:
#             lang_codes = ['en', 'en-US', 'en-GB', 'en-CA']
        
#         transcript_list = None
#         used_lang = None
        
#         # Try specific language codes first
#         for lang_code in lang_codes:
#             try:
#                 transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
#                 used_lang = lang_code
#                 print(f"Successfully got transcript using language code: {lang_code}")
#                 break
#             except Exception as e:
#                 print(f"Failed to get transcript for {lang_code}: {str(e)}")
#                 continue
        
#         # If specific languages fail, try auto-generated
#         if not transcript_list:
#             try:
#                 transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
#                 used_lang = "auto"
#                 print("Successfully got auto-generated transcript")
#             except Exception as e:
#                 raise Exception(f"Could not retrieve any transcript: {str(e)}")
        
#         # Convert transcript list to text
#         transcript_text = ' '.join([item['text'] for item in transcript_list])
        
#         if not transcript_text or len(transcript_text.strip()) == 0:
#             raise Exception("Transcript is empty")
        
#         print(f"Retrieved transcript with {len(transcript_text)} characters using {used_lang}")
#         return transcript_text
        
#     except Exception as e:
#         raise Exception(f"YouTube transcript API failed: {str(e)}")

# def extractive_summarization(transcript):
#     """
#     Summarizes the input transcript using the Extractive Summarization technique.
#     Ensures summaries end with complete sentences (ending with full stops).
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
#         vectorizer = text.CountVectorizer(stop_words='english')
        
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
#             # Return first few complete sentences if summarization fails
#             return " ".join(sentences[:min(10, len(sentences))])
#     except Exception as e:
#         print(f"Error in sentence tokenization: {str(e)}")
#         # If tokenization fails, return complete sentences from the beginning
#         if transcript and len(transcript) > 0:
#             # Get the first 1000 characters but ensure we end with a complete sentence
#             partial_text = transcript[:1000]
#             # Find the last period in the partial text
#             last_period_index = partial_text.rfind('.')
#             if last_period_index > 0:
#                 return transcript[:last_period_index + 1]  # Include the period
#             else:
#                 # If no period is found, return as is without ellipsis
#                 return partial_text
#         else:
#             return "Unable to generate summary. The video may not contain sufficient spoken content."

# def get_transcript_with_retry(transcriber, audio_file, max_retries=3):
#     """Attempts to get a transcript with retries"""
#     for attempt in range(max_retries):
#         try:
#             transcript = transcriber.transcribe(audio_file)
            
#             # Check if transcript has text
#             if hasattr(transcript, 'text') and transcript.text:
#                 text_content = transcript.text
#                 if len(text_content) > 0:
#                     print(f"Successful transcription on attempt {attempt+1} with {len(text_content)} characters")
#                     return text_content
#                 else:
#                     print(f"Empty transcript.text on attempt {attempt+1}")
#             else:
#                 print(f"No text attribute found on attempt {attempt+1}")
                
#                 # Try to find the text in other attributes
#                 for attr in ['transcript', 'result', 'content', 'output']:
#                     if hasattr(transcript, attr):
#                         content = getattr(transcript, attr)
#                         if isinstance(content, str) and len(content) > 0:
#                             return content
                
#                 # If all else fails, check if there's a summary attribute
#                 if hasattr(transcript, 'summary') and transcript.summary:
#                     return f"Summary from AssemblyAI: {transcript.summary}"
                    
#             # If we get here, we need to try again
#             if attempt < max_retries - 1:
#                 print(f"Retry transcription in 2 seconds...")
#                 time.sleep(2)
                
#         except Exception as e:
#             print(f"Transcription error on attempt {attempt+1}: {str(e)}")
#             if attempt < max_retries - 1:
#                 print(f"Retry transcription in 2 seconds...")
#                 time.sleep(2)
    
#     # If we get here, all retries failed
#     raise Exception("Failed to get transcript after multiple attempts")

# @app.route("/getEnglishTranscript", methods=["GET"])
# def get_english_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing: {videolink}")
#     audio_file = None

#     try:
#         # Method 1: Try YouTube transcript API first (faster and more reliable in cloud)
#         try:
#             print("Attempting to get transcript using YouTube transcript API...")
#             text = get_transcript_from_youtube_api(videolink, language='en')
#             print(f"Successfully got transcript from YouTube API: {len(text)} characters")
            
#             # Generate summary from transcript
#             summary = extractive_summarization(text)
#             print(f"Generated summary of {len(summary)} characters")
            
#             return jsonify({"summary": summary, "method": "youtube-api"})
            
#         except Exception as api_error:
#             print(f"YouTube transcript API failed: {api_error}")
#             print("Falling back to yt_dlp + AssemblyAI method...")
        
#         # Method 2: Fallback to your existing yt_dlp + AssemblyAI method
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
        
#         # Try to get transcript with retries
#         text = get_transcript_with_retry(transcriber, audio_file)
        
#         if not text:
#             # If text is still empty, return a helpful error message
#             return jsonify({"summary": "The video does not appear to contain any recognizable speech. Please try another video."}), 200
        
#         print(f"Successfully transcribed {len(text)} characters of text")
        
#         # Generate summary from transcript
#         try:
#             summary = extractive_summarization(text)
#             print(f"Generated summary of {len(summary)} characters")
#         except Exception as e:
#             print(f"Error in summarization: {str(e)}")
#             summary = text[:1000]
        
#         # Return an empty response if summary is empty
#         if not summary or len(summary.strip()) == 0:
#             summary = "Unable to generate summary for this video. The content may not contain sufficient speech."
        
#         # Add better debug output
#         print(f"Returning summary response with {len(summary)} characters")
#         return jsonify({"summary": summary, "method": "assemblyai"})
        
#     except Exception as e:
#         # Log the full traceback
#         traceback.print_exc()
        
#         print(f"Comprehensive error in English transcript: {str(e)}")
#         return jsonify({
#             "error": f"Failed to process video: {str(e)}", 
#             "suggestion": "This might be due to video restrictions or server limitations. Please try another video."
#         }), 500
#     finally:
#         # Clean up audio files from both possible locations
#         cleanup_files = ["/tmp/sameAudio.mp3", "sameAudio.mp3"]
#         if audio_file:
#             cleanup_files.append(audio_file)
        
#         for file_path in cleanup_files:
#             try:
#                 if os.path.exists(file_path):
#                     os.remove(file_path)
#                     print(f"Audio file {file_path} deleted successfully")
#             except Exception as e:
#                 print(f"Error deleting audio file {file_path}: {str(e)}")

# @app.route("/getHindiTranscript", methods=["GET"])
# def get_hindi_transcript():
#     videolink = request.args.get('videolink')
#     if not videolink:
#         return jsonify({"error": "No video link provided"}), 400

#     print(f"Downloading and transcribing (Hindi): {videolink}")
#     audio_file = None

#     try:
#         # Method 1: Try YouTube transcript API first for Hindi
#         try:
#             print("Attempting to get Hindi transcript using YouTube transcript API...")
#             text = get_transcript_from_youtube_api(videolink, language='hi')
#             print(f"Successfully got Hindi transcript from YouTube API: {len(text)} characters")
            
#             # Process Hindi text
#             try:
#                 # Hindi sentence processing
#                 hindi_sentences = []
#                 current_sentence = ""
                
#                 for char in text:
#                     current_sentence += char
#                     if char in ['।', '.']:  # Both Hindi purna viram and Latin period
#                         hindi_sentences.append(current_sentence.strip())
#                         current_sentence = ""
                
#                 # Add any remaining text as a sentence if it exists
#                 if current_sentence.strip():
#                     hindi_sentences.append(current_sentence.strip())
                
#                 # If tokenization worked, use the first few sentences
#                 if hindi_sentences:
#                     summary = " ".join(hindi_sentences[:min(20, len(hindi_sentences))])
#                 else:
#                     # Fallback to NLTK's sent_tokenize
#                     sentences = sent_tokenize(text)
#                     summary = " ".join(sentences[:min(20, len(sentences))])
#             except Exception as e:
#                 print(f"Error in Hindi summarization: {str(e)}")
#                 # Fallback to truncated text
#                 partial_text = text[:1000]
#                 last_purna_viram = partial_text.rfind('।')
#                 last_period = partial_text.rfind('.')
#                 last_mark = max(last_purna_viram, last_period)
                
#                 if last_mark > 0:
#                     summary = text[:last_mark + 1]
#                 else:
#                     summary = partial_text
            
#             return jsonify({"summary": summary, "method": "youtube-api"})
            
#         except Exception as api_error:
#             print(f"YouTube Hindi transcript API failed: {api_error}")
#             print("Falling back to yt_dlp + AssemblyAI method for Hindi...")
        
#         # Method 2: Fallback to yt_dlp + AssemblyAI
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
#             return jsonify({"summary": "The video does not appear to contain any recognizable Hindi speech. Please try another video."}), 200
        
#         print(f"Successfully transcribed {len(text)} characters of Hindi text")
        
#         # Process Hindi text (same logic as above)
#         try:
#             hindi_sentences = []
#             current_sentence = ""
            
#             for char in text:
#                 current_sentence += char
#                 if char in ['।', '.']:
#                     hindi_sentences.append(current_sentence.strip())
#                     current_sentence = ""
            
#             if current_sentence.strip():
#                 hindi_sentences.append(current_sentence.strip())
            
#             if hindi_sentences:
#                 summary = " ".join(hindi_sentences[:min(25, len(hindi_sentences))])
#             else:
#                 sentences = sent_tokenize(text)
#                 summary = " ".join(sentences[:min(25, len(sentences))])
#         except Exception as e:
#             print(f"Error in Hindi summarization: {str(e)}")
#             partial_text = text[:1000]
#             last_purna_viram = partial_text.rfind('।')
#             last_period = partial_text.rfind('.')
#             last_mark = max(last_purna_viram, last_period)
            
#             if last_mark > 0:
#                 summary = text[:last_mark + 1]
#             else:
#                 summary = partial_text
        
#         print(f"Returning Hindi summary response with {len(summary)} characters")
#         return jsonify({"summary": summary, "method": "assemblyai"})
        
#     except Exception as e:
#         # Log the full traceback
#         traceback.print_exc()
        
#         print(f"Error in Hindi transcript: {str(e)}")
#         return jsonify({
#             "error": f"Failed to process Hindi video: {str(e)}", 
#             "suggestion": "This might be due to video restrictions or server limitations. Please try another video."
#         }), 500
#     finally:
#         # Clean up audio files
#         cleanup_files = ["/tmp/sameAudio.mp3", "sameAudio.mp3"]
#         if audio_file:
#             cleanup_files.append(audio_file)
        
#         for file_path in cleanup_files:
#             try:
#                 if os.path.exists(file_path):
#                     os.remove(file_path)
#                     print(f"Audio file {file_path} deleted successfully")
#             except Exception as e:
#                 print(f"Error deleting audio file {file_path}: {str(e)}")

# @app.route("/ping", methods=["GET"])
# def ping():
#     """Health check endpoint"""
#     return jsonify({"status": "ok"}), 200

# # for development
# # if __name__ == "__main__":
# #     app.run(debug=True, host="0.0.0.0")

# # for production
# if __name__ == "__main__":
#     app.run(debug=False, host="0.0.0.0")















# groq api implemented
# groq api implemented
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

# Download ONLY the punkt package that's needed (not punkt_tab)
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
groq_client = Groq(
    api_key="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"
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

# Test endpoint to check Groq connectivity
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

# New endpoint for direct text summarization
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