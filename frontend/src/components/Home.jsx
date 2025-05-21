// import { useState } from 'react';
// import { Search, ClipboardCopy, Youtube, Globe, Loader, AlertCircle } from 'lucide-react';
// import BackgroundBeamsWithCollision from './BeamGround';

// export default function YouTubeTranscriptSummarizer() {
//   const [youtubeUrl, setYoutubeUrl] = useState('');
//   const [summary, setSummary] = useState('');
//   const [language, setLanguage] = useState('english');
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');
//   const [copied, setCopied] = useState(false);

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     setLoading(true);
//     setError('');
//     setSummary('');
    
//     try {
//       // const response = await fetch('https://your-backend-api.com/summarize', {
//       //   method: 'POST',
//       //   headers: {
//       //     'Content-Type': 'application/json',
//       //   },
//       //   body: JSON.stringify({ url: youtubeUrl, language }),
//       // });


//       const endpoint = language === 'english' 
//         ? 'http://localhost:5000/getEnglishTranscript' 
//         : 'http://localhost:5000/getHindiTranscript';

//       const response = await fetch(`${endpoint}?videolink=${youtubeUrl}`);
      
//       if (!response.ok) {
//         throw new Error('Failed to fetch summary');
//       }
//       console.log("data",response)
//       const data = await response.json();
//       setSummary(data.summary);
//     } catch (err) {
//       setError('Failed to get summary. Please check your URL and try again.');
//     } finally {
//       setLoading(false);
//     }
//   };

//   const copyToClipboard = () => {
//     navigator.clipboard.writeText(summary);
//     setCopied(true);
//     setTimeout(() => setCopied(false), 2000);
//   };

//   const extractVideoId = (url) => {
//     const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
//     const match = url.match(regExp);
//     return match && match[2].length === 11 ? match[2] : null;
//   };

//   const videoId = extractVideoId(youtubeUrl);
//   const thumbnailUrl = videoId ? `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg` : '/api/placeholder/640/360';

//   return (
//     <BackgroundBeamsWithCollision className="h-screen w-full flex flex-col items-center justify-center bg-black">
//       <div className="z-10 w-full max-w-4xl mx-auto px-4 py-8">
//         <div className="flex flex-col items-center text-center mb-8">
//           <div className="flex items-center gap-3 mb-2">
//             <Youtube className="text-red-600" size={36} />
//             <h1 className="text-3xl font-bold text-white">YouTube Transcript Summarizer</h1>
//           </div>
//           <p className="text-gray-300">Get quick summaries of YouTube videos in English or Hindi</p>
//         </div>
        
//         <div className=" bg-opacity-10  rounded-xl shadow-2xl p-8  border-opacity-20 
//         h-full w-full bg-gray-0  bg-clip-padding backdrop-filter backdrop-blur-none bg-opacity-20 ">
//           <form onSubmit={handleSubmit} className="mb-6">
//             <div className="mb-6">
//               <label htmlFor="youtubeUrl" className="block text-sm font-medium text-gray-200 mb-2">
//                 YouTube Video URL
//               </label>
//               <div className="relative">
//                 <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
//                   <Search className="h-5 w-5 text-gray-400" />
//                 </div>
//                 <input
//                   type="text"
//                   id="youtubeUrl"
//                   value={youtubeUrl}
//                   onChange={(e) => setYoutubeUrl(e.target.value)}
//                   placeholder="https://www.youtube.com/watch?v=..."
//                   className="block w-full pl-10 pr-3 py-3 bg-black bg-opacity-30 border border-gray-600 rounded-lg shadow-sm focus:ring-2 focus:ring-red-500 focus:border-red-500 text-white"
//                   required
//                 />
//               </div>
//             </div>
            
//             <div className="mb-8">
//               <p className="text-sm font-medium text-gray-200 mb-3">Select Language</p>
//               <div className="flex justify-center gap-8">
//                 <div className="flex items-center">
//                   <input
//                     type="radio"
//                     id="english"
//                     name="language"
//                     value="english"
//                     checked={language === 'english'}
//                     onChange={() => setLanguage('english')}
//                     className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300"
//                   />
//                   <label htmlFor="english" className="ml-2 block text-sm text-gray-200">
//                     English
//                   </label>
//                 </div>
//                 <div className="flex items-center">
//                   <input
//                     type="radio"
//                     id="hindi"
//                     name="language"
//                     value="hindi"
//                     checked={language === 'hindi'}
//                     onChange={() => setLanguage('hindi')}
//                     className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300"
//                   />
//                   <label htmlFor="hindi" className="ml-2 block text-sm text-gray-200">
//                     Hindi
//                   </label>
//                 </div>
//               </div>
//             </div>
            
//             <div className="flex justify-center">
//               <button
//                 type="submit"
//                 disabled={loading}
//                 className="flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 shadow-lg transition-colors w-full sm:w-auto"
//               >
//                 {loading ? (
//                   <>
//                     <Loader className="animate-spin -ml-1 mr-2 h-5 w-5" />
//                     Processing...
//                   </>
//                 ) : (
//                   <>
//                     <Globe className="mr-2 h-5 w-5" />
//                     Get {language === 'hindi' ? 'Hindi' : 'English'} Summary
//                   </>
//                 )}
//               </button>
//             </div>
//           </form>

//           {videoId && (
//             <div className="mb-6 overflow-hidden rounded-lg">
//               <div className="relative pb-9/16 overflow-hidden bg-gray-900" style={{ paddingBottom: '56.25%' }}>
//                 <img 
//                   src={thumbnailUrl} 
//                   alt="Video thumbnail" 
//                   className="absolute inset-0 w-full h-full object-cover" 
//                 />
//                 <div className="absolute inset-0 bg-black bg-opacity-30 flex items-center justify-center">
//                   <div className="bg-red-600 rounded-full p-3 opacity-90">
//                     <Youtube className="h-8 w-8 text-white" />
//                   </div>
//                 </div>
//               </div>
//             </div>
//           )}

//           {error && (
//             <div className="mb-6 p-4 rounded-md bg-red-900 bg-opacity-30 border border-red-500">
//               <div className="flex">
//                 <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
//                 <p className="text-sm text-red-300">{error}</p>
//               </div>
//             </div>
//           )}

//           {summary && (
//             <div className="relative mt-6">
//               <div className="absolute top-2 right-2">
//                 <button
//                   onClick={copyToClipboard}
//                   className="p-2 text-gray-300 hover:text-white focus:outline-none focus:ring-2 focus:ring-red-500 rounded-md"
//                   aria-label="Copy to clipboard"
//                 >
//                   <ClipboardCopy className="h-5 w-5" />
//                 </button>
//               </div>
//               <div className="p-6 bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700">
//                 <h3 className="text-lg font-medium text-gray-100 mb-2">Summary ({language === 'hindi' ? 'Hindi' : 'English'})</h3>
//                 <div className="prose max-w-none text-gray-300 whitespace-pre-wrap">{summary}</div>
//               </div>
//               {copied && (
//                 <div className="absolute bottom-2 right-2 bg-gray-800 text-white py-1 px-3 rounded text-sm">
//                   Copied!
//                 </div>
//               )}
//             </div>
//           )}
//         </div>
//       </div>
//     </BackgroundBeamsWithCollision>
//   );
// }















import { useState } from 'react';
import { Search, ClipboardCopy, Youtube, Globe, Loader, AlertCircle } from 'lucide-react';
import BackgroundBeamsWithCollision from './BeamGround';


// import { useState } from 'react';
// import { Search, ClipboardCopy, Youtube, Globe, Loader, AlertCircle } from 'lucide-react';
// import BackgroundBeamsWithCollision from './BeamGround';

export default function YouTubeTranscriptSummarizer() {
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [summary, setSummary] = useState('');
  const [language, setLanguage] = useState('english');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSummary('');
    
    try {
      const endpoint = language === 'english' 
        ? 'https://mlytsummarize.onrender.com/getEnglishTranscript' 
        // ? 'http://localhost:5000/getEnglishTranscript' 
        : 'https://mlytsummarize.onrender.com/getHindiTranscript';
        // : 'http://localhost:5000/getHindiTranscript';

      console.log(`Fetching summary from: ${endpoint} for URL: ${youtubeUrl}`);
      
      // const response = await fetch(`${endpoint}`, {
      const response = await fetch(`${endpoint}?videolink=${encodeURIComponent(youtubeUrl)}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });
      
      console.log(`Response status: ${response.status}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to fetch summary (Status: ${response.status})`);
      }
      
      const data = await response.json();
      console.log(`Received data:`, data);
      
      if (data && data.summary) {
        setSummary(data.summary);
        console.log(`Set summary of length: ${data.summary.length}`);
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'Failed to get summary. Please check your URL and try again.');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const extractVideoId = (url) => {
    if (!url) return null;
    
    // Handle multiple YouTube URL formats and remove additional parameters
    const cleanUrl = url.split('?')[0];  // Remove query parameters
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = cleanUrl.match(regExp);
    return match && match[2].length === 11 ? match[2] : null;
  };

  const videoId = extractVideoId(youtubeUrl);
  // Updated thumbnail URL with higher resolution and fallback
  const thumbnailUrl = videoId 
    ? `https://img.youtube.com/vi/${videoId}/hqdefault.jpg` 
    : '/api/placeholder/640/360';

  // Improve URL validation
  const isValidYouTubeUrl = (url) => {
    return extractVideoId(url) !== null;
  };

  return (
    <BackgroundBeamsWithCollision className="w-full flex flex-col items-center justify-center bg-black">
      <div className="z-10 w-full max-w-4xl mx-auto px-4 py-8">
        <div className="flex flex-col items-center text-center mb-8">
          <div className="flex items-center gap-3 mb-2">
            <Youtube className="text-red-600" size={36} />
            <h1 className="text-3xl font-bold text-white">YouTube Transcript Summarizer</h1>
          </div>
          <p className="text-gray-300">Get quick summaries of YouTube videos in English or Hindi</p>
        </div>
        
        <div className="bg-gray-900 bg-opacity-50 rounded-xl shadow-2xl p-8 border border-gray-700 
          w-full backdrop-filter backdrop-blur-sm">
          <form onSubmit={handleSubmit} className="mb-6">
            <div className="mb-6">
              <label htmlFor="youtubeUrl" className="block text-sm font-medium text-gray-200 mb-2">
                YouTube Video URL
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  id="youtubeUrl"
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="block w-full pl-10 pr-3 py-3 bg-black bg-opacity-30 border border-gray-600 rounded-lg shadow-sm focus:ring-2 focus:ring-red-500 focus:border-red-500 text-white"
                  required
                />
              </div>
              {youtubeUrl && !isValidYouTubeUrl(youtubeUrl) && (
                <p className="mt-1 text-sm text-red-400">Please enter a valid YouTube URL</p>
              )}
            </div>
            
            <div className="mb-8">
              <p className="text-sm font-medium text-gray-200 mb-3">Select Language</p>
              <div className="flex justify-center gap-8">
                <div className="flex items-center">
                  <input
                    type="radio"
                    id="english"
                    name="language"
                    value="english"
                    checked={language === 'english'}
                    onChange={() => setLanguage('english')}
                    className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300"
                  />
                  <label htmlFor="english" className="ml-2 block text-sm text-gray-200">
                    English
                  </label>
                </div>
                <div className="flex items-center">
                  <input
                    type="radio"
                    id="hindi"
                    name="language"
                    value="hindi"
                    checked={language === 'hindi'}
                    onChange={() => setLanguage('hindi')}
                    className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300"
                  />
                  <label htmlFor="hindi" className="ml-2 block text-sm text-gray-200">
                    Hindi
                  </label>
                </div>
              </div>
            </div>
            
            <div className="flex justify-center">
              <button
                type="submit"
                disabled={loading || (youtubeUrl && !isValidYouTubeUrl(youtubeUrl))}
                className={`flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white shadow-lg transition-colors w-full sm:w-auto ${
                  loading || (youtubeUrl && !isValidYouTubeUrl(youtubeUrl))
                    ? 'bg-gray-600'
                    : 'bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500'
                }`}
              >
                {loading ? (
                  <>
                    <Loader className="animate-spin -ml-1 mr-2 h-5 w-5" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Globe className="mr-2 h-5 w-5" />
                    Get {language === 'hindi' ? 'Hindi' : 'English'} Summary
                  </>
                )}
              </button>
            </div>
          </form>

          {/* {videoId && (
            <div className="mb-6 overflow-hidden rounded-lg">
              <div className="relative overflow-hidden bg-gray-900" style={{ paddingBottom: '56.25%' }}>
                <img 
                  src={thumbnailUrl} 
                  alt="Video thumbnail" 
                  className="absolute inset-0 w-full h-full object-cover" 
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = '/api/placeholder/640/360';
                  }}
                />
                <div className="absolute inset-0 bg-black bg-opacity-30 flex items-center justify-center">
                  <div className="bg-red-600 rounded-full p-3 opacity-90">
                    <Youtube className="h-8 w-8 text-white" />
                  </div>
                </div>
              </div>
            </div>
          )} */}

          {error && (
            <div className="mb-6 p-4 rounded-md bg-red-900 bg-opacity-30 border border-red-500">
              <div className="flex">
                <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
                <p className="text-sm text-red-300">{error}</p>
              </div>
            </div>
          )}

          {summary && (
            <div className="relative mt-6">
              <div className="absolute top-2 right-2">
                <button
                  onClick={copyToClipboard}
                  className="p-2 text-gray-300 hover:text-white focus:outline-none focus:ring-2 focus:ring-red-500 rounded-md"
                  aria-label="Copy to clipboard"
                >
                  <ClipboardCopy className="h-5 w-5" />
                </button>
                {copied && (
                <div className="absolute  bg-gray-800 text-white py-1 px-3 rounded text-sm">
                  Copied!
                </div>
              )}
              </div>
              <div className="p-6 bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700">
                <h3 className="text-lg font-medium text-gray-100 mb-2">Summary ({language === 'hindi' ? 'Hindi' : 'English'})</h3>
                <div className="prose max-w-none text-gray-300 whitespace-pre-wrap">{summary}</div>
              </div>
              
            </div>
          )}
        </div>
      </div>
    </BackgroundBeamsWithCollision>
  );
}