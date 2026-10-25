import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Mic, Square, Play, CheckCircle, AlertCircle, Sparkles, BarChart3, HelpCircle, Activity, Flame, History, AlertTriangle, LogIn, LogOut, UserPlus, X, Search, RefreshCw, LayoutDashboard, Compass, Volume2, Video as VideoIcon, TrendingUp, TrendingDown, Award, Target, ShieldCheck, Layers, Gauge, Trash2, Calendar, ChevronDown, ChevronUp } from 'lucide-react';
import confetti from 'canvas-confetti';
import { LineChart, Line, AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceArea, ReferenceLine } from 'recharts';
import './App.css';
import './modules/video/video.css';
import { VideoAnalysisModule } from './modules/video/VideoAnalysisModule';
import { ScoreOverview } from './modules/video/components/ScoreOverview';
import { SpeechDetailedMetrics } from './modules/video/components/SpeechDetailedMetrics';
import { FusionTimeline } from './modules/video/components/FusionTimeline';
import { EmotionBreakdown } from './modules/video/components/EmotionBreakdown';
import { FeedbackList } from './modules/video/components/FeedbackList';
import { PronunciationHighlight } from './modules/video/components/PronunciationHighlight';
import { deleteVideoSession } from './modules/video/videoApi';
import { apiUrl } from './api';
import type { ScoresBreakdown, SpeechFeatures } from './modules/video/types';

// Type definitions
interface AnalyticsData {
  has_data: boolean;
  total_sessions: number;
  streak_count: number;
  fluency: {
    current: number;
    baseline: number;
    pct_change: number;
    slope: number;
    status: string;
  };
  accuracy: {
    current: number;
    baseline: number;
    pct_change: number;
  };
  fillers: {
    current_fpm: number;
    baseline_fpm: number;
    pct_reduction: number;
  };
  pacing: {
    current_wpm: number;
    status: string;
  };
  mastery_index: {
    score: number;
    tier: string;
    label: string;
  };
  radar_data: Array<{
    dimension: string;
    baseline: number;
    current: number;
    fullMark: number;
  }>;
  timeline: Array<{
    name: string;
    session_id: string;
    type: string;
    fluency: number;
    ema_trend: number;
    accuracy: number;
    wpm: number;
    fillers: number;
    stammers: number;
    pauses: number;
  }>;
  weak_words: Array<{
    word: string;
    count: number;
    spoken_as: string[];
    status: 'mastered' | 'needs_drill';
  }>;
}

interface MismatchedWord {
  expected: string;
  spoken: string;
  index: number;
}


interface PauseDetail {
  start: number;
  end: number;
  duration: number;
}

interface FillerDetail {
  word: string;
  start: number;
  end: number;
  duration: number;
}

interface StammerDetail {
  text: string;
  type: string;
  start: number;
  end: number;
  duration: number;
}

interface AnalysisResult {
  transcript: string;
  duration_sec: number;
  word_count: number;
  wpm: number;
  filler_count: number;
  filler_words_found: string[];
  filler_details?: FillerDetail[];
  stammer_events: number;
  stammer_details?: StammerDetail[];
  long_pauses: number;
  pause_details?: PauseDetail[];
  sub_scores: {
    filler_score: number;
    stammer_score: number;
    pause_score: number;
    rate_score: number;
    clarity_score: number;
  };
  final_score: number;
  feedback: string[];
  recommended_exercises: string[];
  target_text?: string;
  word_error_rate?: number;
  mismatched_words?: MismatchedWord[];
  ai_pathologist_feedback?: string;
  streak_count?: number;
}

interface Exercise {
  _id: string;
  title: string;
  description: string;
  difficulty: string;
  trigger_condition: string;
}

interface SessionRecord {
  session_id: string;
  type: 'practice' | 'analysis' | 'exercise' | 'video';
  session_category?: 'practice' | 'analysis' | 'exercise' | 'video';
  created_at: string;
  spoken_text?: string;
  target_text?: string;
  transcript?: string;
  exercise_id?: string;
  exercise_title?: string;
  pronunciation_score?: number;
  final_score?: number;
  wpm?: number;
  filler_count?: number;
  filler_words_found?: string[];
  filler_details?: FillerDetail[];
  stammer_events?: number;
  stammer_details?: StammerDetail[];
  long_pauses?: number;
  duration_sec?: number;
  pause_details?: PauseDetail[];
  mismatched_words?: MismatchedWord[];
  word_error_rate?: number;
  sub_scores?: {
    filler_score: number;
    stammer_score: number;
    pause_score: number;
    rate_score: number;
    clarity_score: number;
  };
  feedback?: string[];
  recommended_exercises?: string[];
  ai_pathologist_feedback?: string;
}

interface User {
  id: string;
  name: string;
  email: string;
}


const TOPIC_PROMPTS = [
  "Explain what you had for breakfast today, or talk about a recent movie you watched.",
  "Describe your favorite hobby and why you enjoy spending time on it.",
  "Talk about a memorable vacation trip you took with your friends or family.",
  "Discuss your favorite book or film and what values it taught you.",
  "Explain the steps to make a simple dish or coffee.",
  "Describe what your dream job is and the skills needed to achieve it."
];

const EXERCISE_INSTRUCTION_MAP: Record<string, { objective: string; howToPerform: string[]; advantages: string[] }> = {
  silent_pause_drill: {
    objective: "Train your brain to replace involuntary vocal fillers (like 'um', 'uh', 'like', 'you know') with a clean, confident silent pause.",
    howToPerform: [
      "Scan the target text before speaking to get familiar with the sentence flow.",
      "As you start reading aloud, pay attention to moments where you hesitate or feel the urge to say a filler word.",
      "Instead of letting 'um' or 'uh' slip out, swallow, close your mouth, and pause silently for 1 second.",
      "Maintain comfortable posture, take a relaxed breath, and proceed to the next word."
    ],
    advantages: [
      "Significantly reduces auditory speech clutter, instantly sounding more professional.",
      "Gives your listener natural gaps to digest your message.",
      "Provides your brain with a cognitive breathing window to plan the next sentence."
    ]
  },
  slow_rate_reading: {
    objective: "Help prevent phoneme blocks and syllable repetitions by introducing prolonged vowel transitions and structured pacing.",
    howToPerform: [
      "Read the provided passage at roughly 50% of your normal speed.",
      "Slightly elongate the initial consonant or vowel of each word (e.g. 'S-s-peaking...').",
      "Connect words in a smooth, sliding tone rather than voicing them in sharp, staccato intervals.",
      "Keep your abdominal breathing steady and continuous throughout the sentence."
    ],
    advantages: [
      "Lowers tension in the vocal cords and articulatory muscles.",
      "Prevents physical sound blocks and repetitive stammers.",
      "Promotes a calm, meditative state during verbal output."
    ]
  },
  sentence_chunking: {
    objective: "Break complex sentences into digestible, rhythmic word groups to maintain vocal stamina and flow.",
    howToPerform: [
      "Look at the text and identify natural separators (commas, periods, conjunctions).",
      "Chunk your speech into brief, logical phrase groups of 3 to 5 words.",
      "Read each chunk with continuous phrasing, then insert a distinct 1.5-second silent pause.",
      "Inhale gently during each pause to keep your lungs filled with air."
    ],
    advantages: [
      "Prevents running out of breath before finishing long sentences.",
      "Improves vocal projection and listener comprehension.",
      "Establishes a structured, engaging cadence of delivery."
    ]
  },
  metronome_paced_reading: {
    objective: "Tackle rapid speaking speeds and speech cluttering by synchronizing syllables with a steady rhythmic pulse.",
    howToPerform: [
      "Focus on the visual tempo guide or count beats in your head.",
      "Articulate precisely one word or syllable per rhythmic interval.",
      "Do not rush ahead; force yourself to wait for the next beat before vocalizing.",
      "Slow down and over-emphasize word endings (like -ed, -s, -ing)."
    ],
    advantages: [
      "Overcomes conversational cluttering (running words together).",
      "Enforces a safe upper-limit speed threshold (ideally 120-140 WPM).",
      "Improves overall clarity and clarity scores."
    ]
  },
  timed_reading_challenge: {
    objective: "Help overcome over-cautious, hyper-hesitant speech styles by pacing your delivery against a target time window.",
    howToPerform: [
      "Review the text to ensure you know the pronunciation of all words.",
      "Start reading in a steady, conversational voice without back-tracking or self-correcting.",
      "Avoid extra pauses; read forward smoothly to finish within the target duration.",
      "Let mistakes go immediately rather than stopping to fix them."
    ],
    advantages: [
      "Combats excessive self-correction and hesitation blocks.",
      "Builds a natural, fluent, conversational rhythm.",
      "Boosts confidence in real-time, forward-moving speech."
    ]
  },
  articulation_drill: {
    objective: "Strengthen physical speech muscles and eliminate mumbling by exaggerating difficult consonant sounds.",
    howToPerform: [
      "Review the target tongue twister carefully.",
      "Exaggerate your lip, jaw, and tongue movements slightly as you speak.",
      "Deliver Plosive consonants (P, T, K, B, D, G) with crisp, firm air releases.",
      "Practice at a slower pace first to ensure every syllable is cleanly separated, then increase speed."
    ],
    advantages: [
      "Exercises and builds agility in speech muscles (articulators).",
      "Prevents mumbling and running syllables together.",
      "Ensures clear communication in noisy or professional settings."
    ]
  },
  advanced_impromptu_speaking: {
    objective: "Synthesize all positive speech habits (pacing, clear articulation, silent pauses) during spontaneous conversation.",
    howToPerform: [
      "Read the randomly selected prompt.",
      "Take 5-10 seconds to form a brief mental outline (Intro, Main Point, Summary).",
      "Start speaking freely without a script.",
      "Focus on slow pacing, breathing between thoughts, and absolute avoidance of filler words."
    ],
    advantages: [
      "Simulates real-world, high-pressure conversational speaking.",
      "Improves cognitive load management during speech.",
      "Consolidates and validates your progression toward fluent speech."
    ]
  }
};
const EXERCISE_LEVEL_PRESETS: Record<string, { beginner: string[]; intermediate: string[]; advanced: string[] }> = {
  silent_pause_drill: {
    beginner: [
      "Practice makes perfect.",
      "Speak slowly and clearly.",
      "Take a breath now."
    ],
    intermediate: [
      "Practice makes a man perfect, especially in language skills.",
      "A calm mind helps you coordinate words and speak without panic.",
      "Taking a brief silent pause between phrases lets you control your cadence."
    ],
    advanced: [
      "Deliberate pacing is not just about slow speed, but rather about incorporating natural, structured silences to allow your breathing to align with your thoughts.",
      "While speaking in public, pauses serve as a cognitive bridge, allowing both the speaker to plan the next sentence and the audience to digest the message.",
      "Establishing a steady verbal cadence requires continuous self-awareness, deep diaphragmatic inhalations, and the elimination of rapid speech spurts."
    ]
  },
  slow_rate_reading: {
    beginner: [
      "Keep a steady pace.",
      "Breathe in and out.",
      "Rushing causes blocks."
    ],
    intermediate: [
      "A steady breathing pattern helps reduce stammering and speech blocks during conversation.",
      "Slow, deliberate speech increases clarity and makes you sound professional.",
      "Slowing your word rate gives you time to anticipate and prevent sound repetitions."
    ],
    advanced: [
      "Vocal cord tension can be significantly minimized by consciously slowing down your speech rate to roughly one hundred and twenty words per minute.",
      "Speaking at a reduced velocity provides your brain with a safe buffer zone, allowing you to bypass physical blocks and speech repetitions.",
      "A controlled verbal delivery, combined with a relaxed physical posture, represents the foundation of speech therapy pacing techniques."
    ]
  },
  sentence_chunking: {
    beginner: [
      "Read short phrase groups.",
      "Pause after each chunk.",
      "Inhale during the pause."
    ],
    intermediate: [
      "To speak clearly and confidently, one must divide long sentences into chunks.",
      "Chunking complex sentences helps you maintain breath support throughout your speech.",
      "Insert a brief pause after logical word groups to establish a clean rhythm."
    ],
    advanced: [
      "Breaking your speech flow into logical phrase units of three to five words prevents vocal exhaustion and builds conversational stamina.",
      "By deliberately stopping at commas, conjunctions, and logical transition points, you create a rhythmic delivery that enhances listener comprehension.",
      "Diaphragmatic breathing during these structural pauses ensures your lungs are always filled, avoiding the gasping blocks typical of cluttered speech."
    ]
  },
  metronome_paced_reading: {
    beginner: [
      "One beat per word.",
      "Speak on the pulse.",
      "Wait for the beat."
    ],
    intermediate: [
      "Rhythm and timing are crucial when delivering a public speech to a large audience.",
      "Synchronize your syllables with the steady beat to regulate conversational speed.",
      "Force yourself to pause and wait for the next metronome pulse before speaking."
    ],
    advanced: [
      "Pacing your verbal output against an external rhythmic metronome pulse prevents the rapid syllable acceleration common in cluttering disorders.",
      "By matching the articulation of each word to a visual or auditory tempo guide, you enforce a safe speed threshold of under one hundred and thirty WPM.",
      "Exaggerating word endings and giving every vowel its full duration on the beat trains your articulators for consistent daily clarity."
    ]
  },
  timed_reading_challenge: {
    beginner: [
      "Read forward smoothly.",
      "Do not look back.",
      "Finish on time."
    ],
    intermediate: [
      "Clear communication is essential for professional success across global systems.",
      "Maintaining forward momentum is the key to conquering hyper-hesitant speech styles.",
      "Read the passage from start to finish without pausing to self-correct slips."
    ],
    advanced: [
      "To overcome excessive self-correction and speech block hesitations, you must train yourself to move forward continuously, regardless of minor errors.",
      "Pacing your reading against a strict target time window teaches you to maintain a steady vocal energy and bypass conversational blockades.",
      "Focusing on the final word of the paragraph rather than individual syllables helps build a smooth, uninterrupted conversational flow."
    ]
  },
  articulation_drill: {
    beginner: [
      "Red leather, yellow leather.",
      "Crisp consonants sound clear.",
      "Focus on word endings."
    ],
    intermediate: [
      "Peter Piper picked a peck of pickled peppers; did Peter Piper pick a peck of pickled peppers?",
      "She sells seashells by the seashore; the shells she sells are surely seashells.",
      "A big black bug bit a big black bear and made the big black bear bleed blood."
    ],
    advanced: [
      "Exaggerating your jaw, tongue, and lip placements when pronouncing hard plosive consonants ensures maximum intelligibility in professional settings.",
      "Crisp articulation of word endings, especially hard 't', 'd', and 'k' sounds, prevents mumbling and makes your speech sound polished and authoritative.",
      "Practice shifting your pitch and volume while maintaining crisp consonant separation to build control over your vocal articulators."
    ]
  },
  advanced_impromptu_speaking: {
    beginner: [
      "Talk about your favorite food.",
      "Describe your bedroom layout.",
      "What did you do yesterday?"
    ],
    intermediate: [
      "Explain what you had for breakfast today and how you prepared it.",
      "Describe a recent movie you watched and why you liked or disliked it.",
      "Talk about a memorable vacation trip you took and what you enjoyed most."
    ],
    advanced: [
      "Discuss your views on the future of artificial intelligence in modern healthcare and global education systems.",
      "Explain the key factors that contribute to career satisfaction and how you balance professional growth with personal life.",
      "Analyze the impact of social media platforms on adolescent mental health and suggest potential mitigation strategies."
    ]
  }
};

const getPresetsForExercise = (exerciseId: string, difficulty: 'beginner' | 'intermediate' | 'advanced'): string[] => {
  const levelPresets = EXERCISE_LEVEL_PRESETS[exerciseId];
  if (levelPresets) {
    return levelPresets[difficulty];
  }
  return [
    "Practice makes a man perfect, especially in language skills.",
    "A steady breathing pattern helps reduce stammering and speech blocks.",
    "Rhythm and timing are crucial when delivering a public speech."
  ];
};

const renderHighlightedTranscript = (transcript?: string, fillerWords: string[] = []) => {
  if (!transcript) return null;
  const fillerSet = new Set(fillerWords.map(w => w.toLowerCase()));
  const words = transcript.split(/\s+/);

  return words.map((w, idx) => {
    const cleanWord = w.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"]/g, '');
    const isFiller = fillerSet.has(cleanWord) ||
      /^(u+m+h*|u+h+m*|e+r+m*|a+h+|h+m+)$/.test(cleanWord) ||
      ['like', 'actually', 'basically', 'so', 'well', 'literally', 'honestly'].includes(cleanWord);

    if (isFiller) {
      return (
        <span
          key={idx}
          style={{
            backgroundColor: 'rgba(245, 158, 11, 0.22)',
            color: '#f59e0b',
            padding: '2px 6px',
            borderRadius: '4px',
            fontWeight: 700,
            border: '1px solid rgba(245, 158, 11, 0.4)',
            margin: '0 2px',
            display: 'inline-block'
          }}
          title="Detected filler word"
        >
          {w}
        </span>
      );
    }
    return <span key={idx}> {w}</span>;
  });
};

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'fluency' | 'video' | 'exercises'>(() => {
    const searchParams = new URLSearchParams(window.location.search);
    const path = window.location.pathname.toLowerCase();
    const hash = window.location.hash.toLowerCase();
    
    if (searchParams.get('tab') === 'video' || path.includes('video') || hash.includes('video')) {
      return 'video';
    }
    const tab = searchParams.get('tab');
    if (tab === 'fluency' || tab === 'exercises' || tab === 'dashboard') {
      return tab;
    }
    return 'dashboard';
  });
  const [status, setStatus] = useState<'idle' | 'recording' | 'processing' | 'success' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState('Ready to record');
  
  // Audio state
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [isAudioLoading, setIsAudioLoading] = useState(false);
  const [currentAudioType, setCurrentAudioType] = useState<'reference' | 'recording' | null>(null);
  const audioBlobRef = useRef<Blob | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const speechUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const ttsAbortControllerRef = useRef<AbortController | null>(null);
  const activeAudioRequestIdRef = useRef<number>(0);
  
  // Module 1 specific (Fluency Tracker)
  const [promptIndex, setPromptIndex] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [fluencyTargetText, setFluencyTargetText] = useState('');
  const [configTabM1, setConfigTabM1] = useState<'prompt' | 'groq'>('prompt');
  
  // Persistence / DB state
  const [streakCount, setStreakCount] = useState(0);
  const [sessionHistory, setSessionHistory] = useState<SessionRecord[]>([]);
  const [exercisesList, setExercisesList] = useState<Exercise[]>([]);
  const [exerciseSearch, setExerciseSearch] = useState('');
  
  // Swappable history tab & detailed expander state
  const [historyTab, setHistoryTab] = useState<'analysis' | 'exercise' | 'video'>('analysis');
  const [expandedSessionId, setExpandedSessionId] = useState<string | null>(null);

  // Analytics & Progress Tracking state
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [chartMetricMode, setChartMetricMode] = useState<'fluency' | 'disfluency' | 'pacing'>('fluency');

  // Groq AI practice text generator states
  const [aiTopic, setAiTopic] = useState('General Communication');
  const [aiLength, setAiLength] = useState('paragraph');
  const [aiFocusExercise] = useState('none');
  const [aiEnglishLevel, setAiEnglishLevel] = useState<'easy' | 'medium' | 'difficult'>('medium');
  const [isGenerating, setIsGenerating] = useState(false);

  // Exercise detail modal/instructions view state
  const [activeExerciseDetail, setActiveExerciseDetail] = useState<Exercise | null>(null);
  const [exerciseTargetText, setExerciseTargetText] = useState('');
  const [configTabExercise, setConfigTabExercise] = useState<'presets' | 'custom' | 'groq'>('presets');
  const [exerciseDifficulty, setExerciseDifficulty] = useState<'beginner' | 'intermediate' | 'advanced'>('beginner');

  const normalizeTtsText = (text: string) => {
    if (!text) return text;

    let normalized = text.trim();
    normalized = normalized.replace(/\s+/g, ' ');
    normalized = normalized.replace(/([a-z])([A-Z])/g, '$1 $2');
    normalized = normalized.replace(/([a-z])([0-9])/gi, '$1 $2');
    normalized = normalized.replace(/([0-9])([A-Za-z])/g, '$1 $2');
    normalized = normalized.replace(/([,.!?;:])(?=[A-Za-z])/g, '$1 ');
    normalized = normalized.replace(/\s+/g, ' ').trim();
    return normalized;
  };

  const startExerciseDrill = (ex: Exercise) => {
    setActiveExerciseDetail(ex);
    
    // Set default difficulty based on exercise metadata
    const diff = (ex.difficulty === 'beginner' || ex.difficulty === 'intermediate' || ex.difficulty === 'advanced') ? ex.difficulty : 'beginner';
    setExerciseDifficulty(diff);
    
    const defaultPresets = getPresetsForExercise(ex._id, diff);
    setExerciseTargetText(defaultPresets[0] || "Read this target passage out loud to practice and track your speech parameters.");

    // Reset settings config tab
    setConfigTabExercise('presets');

    // Reset past results/playback states to prevent leak
    setAnalysisResult(null);
    setAudioUrl(null);
    setStatus('idle');
    setStatusMessage('Ready to record');
    setLiveTranscript('');
  };

  // Auth state
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [authName, setAuthName] = useState('');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);

  // Live Browser STT state
  const [liveTranscript, setLiveTranscript] = useState('');
  const recognitionRef = useRef<any>(null);

  // Initialize data on mount
  useEffect(() => {
    // Check for cached user session
    const cachedUser = localStorage.getItem('speechai_user');
    const cachedToken = localStorage.getItem('speechai_token');
    
    if (cachedUser && cachedToken) {
      try {
        setCurrentUser(JSON.parse(cachedUser));
      } catch (e) {
        localStorage.removeItem('speechai_user');
        localStorage.removeItem('speechai_token');
      }
    }

    fetchExercises();

    // Setup speech recognition
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const rec = new SpeechRecognition();
      rec.continuous = true;
      rec.interimResults = true;
      rec.lang = 'en-IN';
      
      rec.onresult = (event: any) => {
        let finalTrans = '';
        let interimTrans = '';
        for (let i = 0; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTrans += event.results[i][0].transcript + ' ';
          } else {
            interimTrans += event.results[i][0].transcript;
          }
        }
        setLiveTranscript((finalTrans + interimTrans).trim());
      };
      recognitionRef.current = rec;
    }

    return () => {
      stopPlayback();
    };
  }, []);

  // Fetch reports when user shifts (logged in vs anonymous)
  useEffect(() => {
    fetchDashboardData();
  }, [currentUser]);

  const fetchDashboardData = async () => {
    const userId = currentUser ? currentUser.id : 'user_anonymous';
    const token = localStorage.getItem('speechai_token');
    const headers: HeadersInit = {};
    
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    try {
      const res = await fetch(apiUrl(`/reports/${userId}`), { headers });
      if (res.ok) {
        const data = await res.json();
        setStreakCount(data.streak_count);
        if (data.analytics) {
          setAnalyticsData(data.analytics);
        }
        const normalizedSessions = data.sessions.map((session: any) => {
          const hasExerciseMeta = session.session_category === 'exercise' || (session.exercise_id && session.exercise_id !== 'none') || !!session.exercise_title;
          const inferredType = session.type
            || (session.session_category === 'video' ? 'video'
              : session.session_category === 'exercise' ? 'exercise'
              : session.session_category === 'analysis' ? 'analysis'
              : hasExerciseMeta ? 'exercise'
              : 'analysis');
          return {
            ...session,
            type: inferredType,
          };
        });
        setSessionHistory(normalizedSessions);

        const availableTypes = new Set(normalizedSessions.map((session: any) => session.type));
        if (!availableTypes.has(historyTab)) {
          if (availableTypes.has('analysis')) setHistoryTab('analysis');
          else if (availableTypes.has('exercise')) setHistoryTab('exercise');
          else if (availableTypes.has('video')) setHistoryTab('video');
        }
      }
    } catch (e) {
      console.error("Error fetching reports:", e);
    }
  };

  const fetchExercises = async () => {
    try {
      const res = await fetch(apiUrl('/exercises'));
      if (res.ok) {
        const data = await res.json();
        setExercisesList(data);
      }
    } catch (e) {
      console.error("Error fetching exercises:", e);
    }
  };

  const handleGenerateText = async () => {
    setIsGenerating(true);
    try {
      const url = apiUrl(`/practice/generate?topic=${encodeURIComponent(aiTopic)}&length=${aiLength}&level=${aiEnglishLevel}&exercise_id=${aiFocusExercise}`);
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setFluencyTargetText(data.text);
        setAnalysisResult(null);
        alert(`Target passage (${aiEnglishLevel.toUpperCase()} English level) generated successfully using ${data.source}!`);
      } else {
        throw new Error("Failed to generate text");
      }
    } catch (e) {
      console.error(e);
      alert("Error generating text. Falling back to default sentence.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleGenerateExerciseText = async (exerciseId: string) => {
    setIsGenerating(true);
    try {
      const url = apiUrl(`/practice/generate?topic=${encodeURIComponent(aiTopic)}&length=${aiLength}&exercise_id=${exerciseId}`);
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setExerciseTargetText(data.text);
        setAnalysisResult(null);
        alert(`Practice text generated successfully using ${data.source}!`);
      } else {
        throw new Error("Failed to generate text");
      }
    } catch (e) {
      console.error(e);
      alert("Error generating text. Falling back to default sentence.");
    } finally {
      setIsGenerating(false);
    }
  };

  // Auth Submit Handlers
  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);

    const url = apiUrl(authMode === 'login' ? '/auth/login' : '/auth/register');
      
    const payload = authMode === 'login'
      ? { email: authEmail, password: authPassword }
      : { name: authName, email: authEmail, password: authPassword };

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Authentication failed');
      }

      if (authMode === 'login') {
        localStorage.setItem('speechai_token', data.access_token);
        localStorage.setItem('speechai_user', JSON.stringify(data.user));
        setCurrentUser(data.user);
        setShowAuthModal(false);
        setAuthEmail('');
        setAuthPassword('');
        
        confetti({
          particleCount: 80,
          spread: 60,
          colors: ['#6366f1', '#a855f7']
        });
      } else {
        alert("Registration successful! Please sign in using your credentials.");
        setAuthMode('login');
        setAuthName('');
      }
    } catch (err: any) {
      setAuthError(err.message || 'Network error');
    }
  };

  const handleSignOut = () => {
    localStorage.removeItem('speechai_token');
    localStorage.removeItem('speechai_user');
    setCurrentUser(null);
    setStreakCount(0);
    setSessionHistory([]);
  };

  const startRecording = async () => {
    stopPlayback();
    try {
      setLiveTranscript('');
      setAudioUrl(null);
      setAnalysisResult(null);
      audioBlobRef.current = null;
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const options = { mimeType: 'audio/webm' };
      
      let mediaRecorder;
      try {
        mediaRecorder = new MediaRecorder(stream, options);
      } catch (e) {
        mediaRecorder = new MediaRecorder(stream);
      }
      
      mediaRecorderRef.current = mediaRecorder;
      const chunks: Blob[] = [];
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunks.push(e.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        audioBlobRef.current = audioBlob;
        setAudioUrl(URL.createObjectURL(audioBlob));
        submitAudio(audioBlob);
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      setStatus('recording');
      setStatusMessage('Listening...');
      
      if (recognitionRef.current) {
        try {
          recognitionRef.current.start();
        } catch (e) {
          console.error("SpeechRecognition error:", e);
        }
      }
    } catch (err) {
      console.error("Failed to start recording:", err);
      setStatus('error');
      setStatusMessage('Microphone access denied or not found.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
      
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (e) {
          console.error("SpeechRecognition stop error:", e);
        }
      }
    }
  };

  const submitAudio = async (blob: Blob) => {
    setStatus('processing');
    setStatusMessage('Transcribing & analyzing speech...');
    
    const formData = new FormData();
    formData.append('audio', blob, 'recording.webm');
    
    const url = apiUrl('/analyze/speech');
    const text = activeTab === 'exercises' ? exerciseTargetText : fluencyTargetText;
    if (text) {
      formData.append('target_sentence', text);
    }
    if (activeTab === 'exercises' && activeExerciseDetail) {
      formData.append('exercise_id', activeExerciseDetail._id);
      formData.append('exercise_title', activeExerciseDetail.title);
    }

    const token = localStorage.getItem('speechai_token');
    const headers: HeadersInit = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }
      
      const data = await response.json();
      setAnalysisResult(data);
      if (data.final_score >= 8.0) {
        confetti({
          particleCount: 120,
          spread: 80,
          colors: ['#6366f1', '#a855f7', '#10b981'],
          origin: { y: 0.6 }
        });
      }
      
      if (data.streak_count !== undefined) {
        setStreakCount(data.streak_count);
      }
      fetchDashboardData();
      
      setStatus('success');
      setStatusMessage('Analysis complete!');
    } catch (err: any) {
      console.error("Network or Backend error:", err);
      setStatus('error');
      setStatusMessage(`Error: ${err.message || 'Could not connect to backend server.'}`);
    }
  };

  const getLiveWordColorClass = (_targetWord: string, index: number): string => {
    if (!liveTranscript) {
      return index === 0 ? 'practice-word current' : 'practice-word default';
    }

    const currentTarget = activeTab === 'exercises' ? exerciseTargetText : fluencyTargetText;
    const targetWords = currentTarget
      .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, "")
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean);
    const liveWords = liveTranscript
      .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, "")
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean);

    if (liveWords.length === 0) {
      return index === 0 ? 'practice-word current' : 'practice-word default';
    }

    let matchedTargetIdx = -1;
    let liveIdx = 0;

    for (let tIdx = 0; tIdx < targetWords.length; tIdx++) {
      if (liveIdx >= liveWords.length) break;
      const tWord = targetWords[tIdx];
      const lWord = liveWords[liveIdx];

      if (tWord === lWord || tWord.includes(lWord) || lWord.includes(tWord)) {
        matchedTargetIdx = tIdx;
        liveIdx++;
      } else {
        const nextTWord = tIdx + 1 < targetWords.length ? targetWords[tIdx + 1] : '';
        if (nextTWord && (nextTWord === lWord || nextTWord.includes(lWord) || lWord.includes(nextTWord))) {
          matchedTargetIdx = tIdx;
        } else {
          matchedTargetIdx = tIdx;
          liveIdx++;
        }
      }
    }

    if (index < matchedTargetIdx) {
      // Word has already been spoken by the user -> Blue
      return 'practice-word spoken';
    } else if (index === matchedTargetIdx) {
      // Current active word the user said / is saying -> Blue (current)
      return 'practice-word current';
    } else {
      // Not yet spoken -> Default neutral color
      return 'practice-word default';
    }
  };

  // ---------- Edge-like TTS helpers (browser SpeechSynthesis, prefers Microsoft/Edge voices when available) ----------
  const stopPlayback = () => {
    // Invalidate any active or in-flight audio requests
    activeAudioRequestIdRef.current += 1;

    // Abort in-flight network requests
    if (ttsAbortControllerRef.current) {
      try {
        ttsAbortControllerRef.current.abort();
      } catch (e) {
        console.warn('Error aborting TTS fetch:', e);
      }
      ttsAbortControllerRef.current = null;
    }

    // Stop and cleanup active HTMLAudioElement
    if (audioRef.current) {
      try {
        audioRef.current.onended = null;
        audioRef.current.onerror = null;
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        audioRef.current.src = '';
      } catch (e) {
        console.warn('Error stopping audio playback:', e);
      }
      audioRef.current = null;
    }

    // Cancel browser SpeechSynthesis
    if (typeof window !== 'undefined' && (window as any).speechSynthesis) {
      try {
        (window as any).speechSynthesis.cancel();
      } catch (e) {
        console.warn('Error cancelling speech synthesis:', e);
      }
    }

    if (speechUtteranceRef.current) {
      speechUtteranceRef.current.onend = null;
      speechUtteranceRef.current.onerror = null;
      speechUtteranceRef.current = null;
    }

    setIsAudioPlaying(false);
    setIsAudioLoading(false);
    setCurrentAudioType(null);
  };

  const speakText = async (text: string, rate = 0.95) => {
    if (!text || typeof window === 'undefined') return;

    const normalizedText = normalizeTtsText(text);
    if (!normalizedText) return;

    stopPlayback();

    const thisRequestId = ++activeAudioRequestIdRef.current;
    const abortController = new AbortController();
    ttsAbortControllerRef.current = abortController;

    setIsAudioLoading(true);
    setCurrentAudioType('reference');

    try {
      if ((window as any).speechSynthesis) {
        const utter = new SpeechSynthesisUtterance(normalizedText);
        utter.rate = rate;
        const voices = (window as any).speechSynthesis.getVoices() || [];
        const preferred = voices.find((v: any) => /Microsoft|Zira|Aria|Davis|Guy|Hazel|Eva|Gwyneth/i.test(v.name));
        if (preferred) utter.voice = preferred;

        utter.onend = () => {
          if (activeAudioRequestIdRef.current === thisRequestId) {
            setIsAudioPlaying(false);
            setIsAudioLoading(false);
            setCurrentAudioType(null);
            speechUtteranceRef.current = null;
          }
        };

        utter.onerror = () => {
          if (activeAudioRequestIdRef.current === thisRequestId) {
            setIsAudioPlaying(false);
            setIsAudioLoading(false);
            setCurrentAudioType(null);
            speechUtteranceRef.current = null;
          }
        };

        speechUtteranceRef.current = utter;
        (window as any).speechSynthesis.cancel();
        (window as any).speechSynthesis.speak(utter);
        setIsAudioLoading(false);
        setIsAudioPlaying(true);
        setCurrentAudioType('reference');
        return;
      }
    } catch (e) {
      console.warn('Browser TTS unavailable or failed; falling back to server audio:', e);
    }

    if (thisRequestId !== activeAudioRequestIdRef.current) return;

    try {
      const url = apiUrl(`/tts/generate?text=${encodeURIComponent(normalizedText)}`);
      const res = await fetch(url, { signal: abortController.signal });

      if (thisRequestId !== activeAudioRequestIdRef.current) {
        return;
      }

      if (!res.ok) {
        throw new Error(`TTS request failed: ${res.status}`);
      }

      const buf = await res.arrayBuffer();
      if (thisRequestId !== activeAudioRequestIdRef.current) {
        return;
      }

      const blob = new Blob([buf], { type: 'audio/mpeg' });
      const objectUrl = URL.createObjectURL(blob);
      const a = new Audio(objectUrl);
      a.preload = 'auto';
      a.load();

      if (thisRequestId !== activeAudioRequestIdRef.current) {
        URL.revokeObjectURL(objectUrl);
        return;
      }

      if (audioRef.current) {
        try {
          audioRef.current.onended = null;
          audioRef.current.onerror = null;
          audioRef.current.pause();
          audioRef.current.src = '';
        } catch (_) {}
      }

      audioRef.current = a;
      setIsAudioLoading(false);
      setIsAudioPlaying(true);
      setCurrentAudioType('reference');

      a.onended = () => {
        URL.revokeObjectURL(objectUrl);
        if (activeAudioRequestIdRef.current === thisRequestId) {
          setIsAudioPlaying(false);
          setIsAudioLoading(false);
          setCurrentAudioType(null);
          audioRef.current = null;
        }
      };

      a.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        if (activeAudioRequestIdRef.current === thisRequestId) {
          setIsAudioPlaying(false);
          setIsAudioLoading(false);
          setCurrentAudioType(null);
          audioRef.current = null;
        }
      };

      await a.play();
    } catch (e: any) {
      if (e.name === 'AbortError' || thisRequestId !== activeAudioRequestIdRef.current) {
        return;
      }
      console.warn('Server TTS failed:', e);
      setIsAudioLoading(false);
      setIsAudioPlaying(false);
      setCurrentAudioType(null);
    }
  };

  const playUserRecording = () => {
    if (!audioUrl) return;
    if ((isAudioPlaying || isAudioLoading) && currentAudioType === 'recording') {
      stopPlayback();
      return;
    }
    stopPlayback();
    const thisRequestId = ++activeAudioRequestIdRef.current;
    try {
      const a = new Audio(audioUrl);
      audioRef.current = a;
      setIsAudioPlaying(true);
      setCurrentAudioType('recording');
      a.onended = () => {
        if (activeAudioRequestIdRef.current === thisRequestId) {
          setIsAudioPlaying(false);
          setCurrentAudioType(null);
          audioRef.current = null;
        }
      };
      a.onerror = () => {
        if (activeAudioRequestIdRef.current === thisRequestId) {
          setIsAudioPlaying(false);
          setCurrentAudioType(null);
          audioRef.current = null;
        }
      };
      a.play().catch((e) => console.error('Playback failed:', e));
    } catch (e) {
      console.error('Error playing user recording:', e);
      setIsAudioPlaying(false);
      setCurrentAudioType(null);
    }
  };

  // Effective Analytics calculation (uses backend analyticsData if available, otherwise local derivation)
  const effectiveAnalytics: AnalyticsData = useMemo(() => {
    if (analyticsData && analyticsData.has_data) {
      return analyticsData;
    }
    const n = sessionHistory.length;
    if (n === 0) {
      return {
        has_data: false,
        total_sessions: 0,
        streak_count: streakCount,
        fluency: { current: 0.0, baseline: 0.0, pct_change: 0.0, slope: 0.0, status: 'No practice logs yet' },
        accuracy: { current: 0.0, baseline: 0.0, pct_change: 0.0 },
        fillers: { current_fpm: 0.0, baseline_fpm: 0.0, pct_reduction: 0.0 },
        pacing: { current_wpm: 0.0, status: 'No data' },
        mastery_index: { score: 15, tier: 'Foundational', label: 'Begin Practice' },
        radar_data: [
          { dimension: 'Articulation', baseline: 5, current: 5, fullMark: 10 },
          { dimension: 'Pacing & Rhythm', baseline: 5, current: 5, fullMark: 10 },
          { dimension: 'Pause Control', baseline: 5, current: 5, fullMark: 10 },
          { dimension: 'Phonetic Clarity', baseline: 5, current: 5, fullMark: 10 },
          { dimension: 'Filler Resistance', baseline: 5, current: 5, fullMark: 10 },
        ],
        timeline: [],
        weak_words: [],
      };
    }

    const chrono = [...sessionHistory].reverse();
    const k = Math.min(3, n);
    const scores = chrono.map(s => Number(s.final_score ?? s.pronunciation_score ?? 0));
    const base_fl = Number((scores.slice(0, k).reduce((a, b) => a + b, 0) / k).toFixed(1));
    const recent_fl = Number((scores.slice(-k).reduce((a, b) => a + b, 0) / k).toFixed(1));
    const fl_pct = base_fl > 0 ? Number((((recent_fl - base_fl) / base_fl) * 100).toFixed(1)) : 0;

    let slope = 0;
    if (n >= 2) {
      const x_bar = (n - 1) / 2.0;
      const y_bar = scores.reduce((a, b) => a + b, 0) / n;
      let num = 0;
      let den = 0;
      for (let i = 0; i < n; i++) {
        num += (i - x_bar) * (scores[i] - y_bar);
        den += Math.pow(i - x_bar, 2);
      }
      slope = den !== 0 ? Number((num / den).toFixed(3)) : 0;
    }

    const fl_status = slope > 0.12 ? 'Accelerating Progress' : slope < -0.12 ? 'Needs Targeted Drills' : 'Steady Cadence';

    const accuracies = chrono.map(s => {
      if (s.word_error_rate !== undefined) return Math.max(0, Math.min(100, (1 - s.word_error_rate) * 100));
      if (s.pronunciation_score !== undefined) return Math.min(100, s.pronunciation_score * 10);
      return Math.min(100, (s.final_score ?? 7) * 10);
    });
    const base_acc = Number((accuracies.slice(0, k).reduce((a, b) => a + b, 0) / k).toFixed(1));
    const recent_acc = Number((accuracies.slice(-k).reduce((a, b) => a + b, 0) / k).toFixed(1));
    const acc_pct = Number((((recent_acc - base_acc) / Math.max(1, base_acc)) * 100).toFixed(1));

    const fpms = chrono.map(s => {
      const fc = s.filler_count || 0;
      const dur = Math.max(0.15, (s.duration_sec || 60) / 60);
      return fc / dur;
    });
    const base_fpm = Number((fpms.slice(0, k).reduce((a, b) => a + b, 0) / k).toFixed(1));
    const recent_fpm = Number((fpms.slice(-k).reduce((a, b) => a + b, 0) / k).toFixed(1));
    const fpm_red_pct = base_fpm > 0 ? Number((((base_fpm - recent_fpm) / base_fpm) * 100).toFixed(1)) : 0;

    const wpms = chrono.map(s => s.wpm || 130);
    const recent_wpm = Number((wpms.slice(-k).reduce((a, b) => a + b, 0) / k).toFixed(1));
    const pacing_status = recent_wpm >= 120 && recent_wpm <= 150
      ? 'Optimal Target Cadence (120-150 WPM)'
      : recent_wpm < 120
      ? 'Deliberate / Slow Pace (<120 WPM)'
      : 'Brisk / Rapid Pace (>150 WPM)';

    const c_fl = (recent_fl / 10.0) * 30.0;
    const c_acc = (recent_acc / 100.0) * 25.0;
    const lastSub = chrono[chrono.length - 1]?.sub_scores;
    const c_pause = Math.min(20, ((lastSub?.pause_score ?? 7) + (lastSub?.stammer_score ?? 7)) / 20 * 20);
    const c_filler = Math.min(15, ((lastSub?.filler_score ?? 7) / 10) * 15);
    const c_streak = Math.min(10, (streakCount / 7.0) * 10);
    const mastery_score = Math.min(100, Math.max(15, Math.round(c_fl + c_acc + c_pause + c_filler + c_streak)));

    const tier = mastery_score >= 85 ? 'Master Speaker' : mastery_score >= 70 ? 'Proficient' : mastery_score >= 50 ? 'Developing' : 'Foundational';
    const label = mastery_score >= 85 ? 'Articulate & Highly Resilient' : mastery_score >= 70 ? 'Confident & Fluid Cadence' : mastery_score >= 50 ? 'Consistent Progress & Clarity' : 'Building Rhythm & Breath Control';

    const firstSub = chrono[0]?.sub_scores;
    const radar_data = [
      { dimension: 'Articulation', baseline: Number((base_acc / 10).toFixed(1)), current: Number((recent_acc / 10).toFixed(1)), fullMark: 10 },
      { dimension: 'Pacing & Rhythm', baseline: Math.max(3, Number((10 - Math.abs((wpms[0] || 125) - 135) / 8).toFixed(1))), current: Math.max(3, Number((10 - Math.abs(recent_wpm - 135) / 8).toFixed(1))), fullMark: 10 },
      { dimension: 'Pause Control', baseline: Number(firstSub?.pause_score ?? 6), current: Number(lastSub?.pause_score ?? 8), fullMark: 10 },
      { dimension: 'Phonetic Clarity', baseline: Number(firstSub?.clarity_score ?? 6.5), current: Number(lastSub?.clarity_score ?? 8.5), fullMark: 10 },
      { dimension: 'Filler Resistance', baseline: Number(firstSub?.filler_score ?? 5.5), current: Number(lastSub?.filler_score ?? 8), fullMark: 10 },
    ];

    let ema = scores[0] || 0;
    const alpha = 0.35;
    const timeline = chrono.map((s, idx) => {
      const sc = scores[idx];
      ema = alpha * sc + (1 - alpha) * ema;
      const date = new Date(s.created_at);
      const name = isNaN(date.getTime()) ? `Sess #${idx + 1}` : date.toLocaleDateString([], { month: 'short', day: 'numeric' });
      return {
        name,
        session_id: s.session_id,
        type: s.type,
        fluency: sc,
        ema_trend: Number(ema.toFixed(2)),
        accuracy: Number(accuracies[idx].toFixed(1)),
        wpm: wpms[idx],
        fillers: s.filler_count || 0,
        stammers: s.stammer_events || 0,
        pauses: s.long_pauses || 0,
      };
    });

    const word_freq: Record<string, { count: number; spoken: Set<string> }> = {};
    chrono.forEach(s => {
      (s.mismatched_words || []).forEach(m => {
        const w = (m.expected || '').trim().toLowerCase();
        if (w.length > 2) {
          if (!word_freq[w]) word_freq[w] = { count: 0, spoken: new Set() };
          word_freq[w].count += 1;
          if (m.spoken) word_freq[w].spoken.add(m.spoken);
        }
      });
    });
    const lastMismatches = new Set((chrono[chrono.length - 1]?.mismatched_words || []).map(m => (m.expected || '').toLowerCase()));
    const weak_words = Object.entries(word_freq)
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 8)
      .map(([w, data]) => ({
        word: w,
        count: data.count,
        spoken_as: Array.from(data.spoken).slice(0, 2),
        status: lastMismatches.has(w) ? ('needs_drill' as const) : ('mastered' as const),
      }));

    return {
      has_data: true,
      total_sessions: n,
      streak_count: streakCount,
      fluency: { current: recent_fl, baseline: base_fl, pct_change: fl_pct, slope, status: fl_status },
      accuracy: { current: recent_acc, baseline: base_acc, pct_change: acc_pct },
      fillers: { current_fpm: recent_fpm, baseline_fpm: base_fpm, pct_reduction: fpm_red_pct },
      pacing: { current_wpm: recent_wpm, status: pacing_status },
      mastery_index: { score: mastery_score, tier, label },
      radar_data,
      timeline,
      weak_words,
    };
  }, [analyticsData, sessionHistory, streakCount]);


  // Filtered exercises list
  const filteredExercises = exercisesList.filter(ex => 
    ex.title.toLowerCase().includes(exerciseSearch.toLowerCase()) || 
    ex.description.toLowerCase().includes(exerciseSearch.toLowerCase()) ||
    ex.difficulty.toLowerCase().includes(exerciseSearch.toLowerCase())
  );

  // Swappable session history timeline mapping
  const filteredHistory = sessionHistory.filter(s => {
    if (historyTab === 'exercise') {
      return s.session_category === 'exercise' || s.type === 'exercise';
    }
    if (historyTab === 'video') {
      return s.session_category === 'video' || s.type === 'video';
    }
    return s.session_category === 'analysis' || s.type === 'analysis';
  });

  const visibleHistory = filteredHistory;

  const handleDeleteHistorySession = async (sId: string, sType?: string) => {
    if (sType === 'video') {
      try {
        await deleteVideoSession(sId);
      } catch (e) {
        console.error('Failed to delete video session:', e);
      }
    }
    setSessionHistory(prev => prev.filter(s => {
      const id = s.session_id || (s as any)._id || (s as any).job_id || (s as any).id;
      return id !== sId;
    }));
    if (expandedSessionId === sId) {
      setExpandedSessionId(null);
    }
  };

  // Reset all ephemeral analysis, recording, and transcript activity data
  const resetActivityState = (targetTab?: 'dashboard' | 'fluency' | 'exercises' | 'video', keepExerciseDetail = false) => {
    // 1. Immediately stop any TTS or recorded audio playback
    stopPlayback();

    // 2. Stop microphone recording cleanly
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
        mediaRecorderRef.current.stream?.getTracks().forEach(track => track.stop());
      } catch (e) {
        console.error("Error stopping recorder on reset:", e);
      }
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {}
    }
    setIsRecording(false);

    // 3. Clear audio and transcript state
    setAudioUrl(null);
    audioBlobRef.current = null;
    setLiveTranscript('');

    // 4. Reset status
    setStatus('idle');
    setStatusMessage('Ready to record');

    // 5. Clear module evaluation results
    setAnalysisResult(null);

    // 6. Reset target passages and prompts to default fresh state
    setFluencyTargetText('');
    setPromptIndex(0);
    setConfigTabM1('prompt');

    // 7. Clear exercise drill if leaving exercises
    if (!keepExerciseDetail && targetTab !== 'exercises') {
      setActiveExerciseDetail(null);
      setExerciseTargetText('');
      setConfigTabExercise('presets');
    }
  };

  const prevTabRef = useRef(activeTab);

  const handleTabChange = (newTab: 'dashboard' | 'fluency' | 'exercises' | 'video', keepExerciseDetail = false) => {
    prevTabRef.current = newTab;
    resetActivityState(newTab, keepExerciseDetail);
    setActiveTab(newTab);
  };

  useEffect(() => {
    if (prevTabRef.current !== activeTab) {
      prevTabRef.current = activeTab;
      resetActivityState(activeTab, activeTab === 'exercises');
    }
  }, [activeTab]);

  return (
    <div className="container">
      <header className="app-header">
        <div className="logo-container" onClick={() => handleTabChange('dashboard')}>
          <Activity size={30} className="logo-icon" style={{ color: 'var(--primary)' }} />
          <span className="logo-text">SpeechAI</span>
        </div>

        {/* Desktop Navigation Tabs */}
        <div className="nav-tab-bar">
          <button className={`nav-tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => handleTabChange('dashboard')}>
            <LayoutDashboard size={16} />
            <span>Dashboard</span>
          </button>
          <button className={`nav-tab-btn ${activeTab === 'fluency' ? 'active' : ''}`} onClick={() => handleTabChange('fluency')}>
            <BarChart3 size={16} />
            <span><span className="nav-module-prefix">Module 1: </span>Fluency</span>
          </button>
          <button className={`nav-tab-btn ${activeTab === 'video' ? 'active' : ''}`} onClick={() => handleTabChange('video')}>
            <VideoIcon size={16} />
            <span><span className="nav-module-prefix">Module 2: </span>Video</span>
          </button>
          <button className={`nav-tab-btn ${activeTab === 'exercises' ? 'active' : ''}`} onClick={() => handleTabChange('exercises')}>
            <Compass size={16} />
            <span>Exercises</span>
          </button>
        </div>
        
        <div className="header-actions">
          <div className="streak-pill">
            <Flame size={17} className="streak-flame-icon" />
            <span className="streak-pill-text">{streakCount} Day Streak</span>
          </div>

          <nav className="nav-links">
            {currentUser ? (
              <div className="user-profile-info">
                <span className="user-welcome-text">Welcome, <strong>{currentUser.name}</strong></span>
                <button className="nav-link sign-out-btn" onClick={handleSignOut}>
                  <LogOut size={16} />
                  <span>Sign Out</span>
                </button>
              </div>
            ) : (
              <button className="nav-link sign-in-btn" onClick={() => { setShowAuthModal(true); setAuthMode('login'); setAuthError(null); }}>
                <LogIn size={16} />
                <span>Sign In</span>
              </button>
            )}
          </nav>
        </div>
      </header>

      <main className="main-content">

        {/* 1. DASHBOARD VIEW */}
        {activeTab === 'dashboard' && (
          <div className="view-fade">
            {/* Dashboard Hero Header */}
            <div className="dashboard-hero" style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
                <div>
                  <h1 className="hero-title" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Activity size={30} style={{ color: 'var(--primary)' }} />
                    Speech Fluency & Analytical Dashboard
                  </h1>
                  <p className="hero-subtitle">
                    Longitudinal progress tracking, exponential moving averages, 5-axis competency mapping, and disfluency reduction.
                  </p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <button 
                    onClick={fetchDashboardData}
                    className="chart-switcher-btn"
                    style={{ border: '1px solid var(--border-color)', padding: '8px 14px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                    title="Refresh analytics data"
                  >
                    <RefreshCw size={14} /> Refresh Analytics
                  </button>
                </div>
              </div>
            </div>

            {/* Executive KPI Cards Grid */}
            <div className="analytics-kpi-grid">
              {/* KPI 1: Compound Speech Mastery Index */}
              <div className="kpi-card">
                <div className="kpi-card-header">
                  <span className="kpi-card-title">Compound Mastery Index</span>
                  <div className="kpi-card-icon" style={{ background: 'rgba(99, 102, 241, 0.15)', color: 'var(--primary)' }}>
                    <Award size={18} />
                  </div>
                </div>
                <div className="kpi-card-value">
                  {effectiveAnalytics.mastery_index.score}
                  <span className="kpi-card-denom">/ 100</span>
                </div>
                <div className="kpi-card-footer">
                  <span className="badge-pct optimal">{effectiveAnalytics.mastery_index.tier}</span>
                  <span className="kpi-subtext" style={{ fontWeight: 600 }}>{effectiveAnalytics.mastery_index.label}</span>
                </div>
              </div>

              {/* KPI 2: Fluency Score Trajectory */}
              <div className="kpi-card">
                <div className="kpi-card-header">
                  <span className="kpi-card-title">Fluency Trajectory</span>
                  <div className="kpi-card-icon" style={{ background: 'rgba(34, 197, 94, 0.15)', color: 'var(--success)' }}>
                    <Activity size={18} />
                  </div>
                </div>
                <div className="kpi-card-value">
                  {effectiveAnalytics.fluency.current}
                  <span className="kpi-card-denom">/ 10</span>
                </div>
                <div className="kpi-card-footer">
                  <span className={`badge-pct ${effectiveAnalytics.fluency.pct_change >= 0 ? 'positive' : 'negative'}`}>
                    {effectiveAnalytics.fluency.pct_change >= 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                    {effectiveAnalytics.fluency.pct_change >= 0 ? '+' : ''}{effectiveAnalytics.fluency.pct_change}%
                  </span>
                  <span className="kpi-subtext">Baseline: {effectiveAnalytics.fluency.baseline}/10</span>
                </div>
              </div>

              {/* KPI 3: Pronunciation Accuracy */}
              <div className="kpi-card">
                <div className="kpi-card-header">
                  <span className="kpi-card-title">Pronunciation Accuracy</span>
                  <div className="kpi-card-icon" style={{ background: 'rgba(6, 182, 212, 0.15)', color: '#06b6d4' }}>
                    <ShieldCheck size={18} />
                  </div>
                </div>
                <div className="kpi-card-value">
                  {effectiveAnalytics.accuracy.current}%
                </div>
                <div className="kpi-card-footer">
                  <span className={`badge-pct ${effectiveAnalytics.accuracy.pct_change >= 0 ? 'positive' : 'negative'}`}>
                    {effectiveAnalytics.accuracy.pct_change >= 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                    {effectiveAnalytics.accuracy.pct_change >= 0 ? '+' : ''}{effectiveAnalytics.accuracy.pct_change}%
                  </span>
                  <span className="kpi-subtext">Baseline: {effectiveAnalytics.accuracy.baseline}%</span>
                </div>
              </div>

              {/* KPI 4: Vocal Filler Frequency */}
              <div className="kpi-card">
                <div className="kpi-card-header">
                  <span className="kpi-card-title">Vocal Filler Frequency</span>
                  <div className="kpi-card-icon" style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b' }}>
                    <Layers size={18} />
                  </div>
                </div>
                <div className="kpi-card-value">
                  {effectiveAnalytics.fillers.current_fpm}
                  <span className="kpi-card-denom">/ min</span>
                </div>
                <div className="kpi-card-footer">
                  <span className={`badge-pct ${effectiveAnalytics.fillers.pct_reduction >= 0 ? 'positive' : 'negative'}`}>
                    {effectiveAnalytics.fillers.pct_reduction >= 0 ? '-' : '+'}{Math.abs(effectiveAnalytics.fillers.pct_reduction)}%
                  </span>
                  <span className="kpi-subtext">Baseline: {effectiveAnalytics.fillers.baseline_fpm}/min</span>
                </div>
              </div>

              {/* KPI 5: Speaking Cadence */}
              <div className="kpi-card">
                <div className="kpi-card-header">
                  <span className="kpi-card-title">Speaking Cadence</span>
                  <div className="kpi-card-icon" style={{ background: 'rgba(139, 92, 246, 0.15)', color: 'var(--accent)' }}>
                    <Gauge size={18} />
                  </div>
                </div>
                <div className="kpi-card-value">
                  {effectiveAnalytics.pacing.current_wpm}
                  <span className="kpi-card-denom">WPM</span>
                </div>
                <div className="kpi-card-footer">
                  <span className={`badge-pct ${effectiveAnalytics.pacing.current_wpm >= 120 && effectiveAnalytics.pacing.current_wpm <= 150 ? 'optimal' : 'neutral'}`}>
                    {effectiveAnalytics.pacing.current_wpm >= 120 && effectiveAnalytics.pacing.current_wpm <= 150 ? 'Optimal' : effectiveAnalytics.pacing.current_wpm < 120 ? 'Slow' : 'Brisk'}
                  </span>
                  <span className="kpi-subtext">Target: 120-150 WPM</span>
                </div>
              </div>
            </div>

            {/* Deep Analytics: Longitudinal Chart & Radar */}
            <div className="analytics-chart-row">
              {/* Left Column: Longitudinal Trajectory Chart */}
              <div className="glass-card chart-container-card" style={{ display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
                  <h3 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Activity size={20} style={{ color: 'var(--primary)' }} />
                    Longitudinal Progress & Trajectory Analytics
                  </h3>
                  <div className="chart-switcher">
                    <button 
                      className={`chart-switcher-btn ${chartMetricMode === 'fluency' ? 'active' : ''}`}
                      onClick={() => setChartMetricMode('fluency')}
                    >
                      <Activity size={13} /> Fluency & EMA
                    </button>
                    <button 
                      className={`chart-switcher-btn ${chartMetricMode === 'disfluency' ? 'active' : ''}`}
                      onClick={() => setChartMetricMode('disfluency')}
                    >
                      <AlertTriangle size={13} /> Disfluency Stack
                    </button>
                    <button 
                      className={`chart-switcher-btn ${chartMetricMode === 'pacing' ? 'active' : ''}`}
                      onClick={() => setChartMetricMode('pacing')}
                    >
                      <Gauge size={13} /> Cadence Band
                    </button>
                  </div>
                </div>

                {effectiveAnalytics.timeline.length === 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '260px', color: 'var(--text-muted)' }}>
                    <HelpCircle size={32} style={{ marginBottom: '10px' }} />
                    <p style={{ fontSize: '13px' }}>No session recordings logged yet. Record a practice or fluency session to view your trajectory.</p>
                  </div>
                ) : (
                  <div style={{ width: '100%', height: 280 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      {chartMetricMode === 'fluency' ? (
                        <AreaChart data={effectiveAnalytics.timeline} margin={{ top: 15, right: 15, left: -20, bottom: 0 }}>
                          <defs>
                            <linearGradient id="fluencyGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.4} />
                              <stop offset="95%" stopColor="var(--primary)" stopOpacity={0.0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                          <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                          <YAxis domain={[0, 10]} ticks={[0, 2, 4, 6, 8, 10]} stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                          <Tooltip
                            contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)', borderRadius: 'var(--radius-sm)' }}
                            labelStyle={{ color: 'var(--text-primary)', fontWeight: 'bold' }}
                          />
                          <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />
                          {effectiveAnalytics.fluency.baseline > 0 && (
                            <ReferenceLine y={effectiveAnalytics.fluency.baseline} stroke="var(--text-muted)" strokeDasharray="4 4" label={{ value: `Baseline (${effectiveAnalytics.fluency.baseline})`, fill: 'var(--text-muted)', fontSize: 10, position: 'insideTopLeft' }} />
                          )}
                          <Area type="monotone" dataKey="fluency" name="Fluency Score (/10)" stroke="var(--primary)" strokeWidth={2.5} fillOpacity={1} fill="url(#fluencyGrad)" dot={{ r: 3 }} />
                          <Line type="monotone" dataKey="ema_trend" name="EMA Trend (α=0.35)" stroke="#06b6d4" strokeWidth={2} strokeDasharray="3 3" dot={false} />
                        </AreaChart>
                      ) : chartMetricMode === 'disfluency' ? (
                        <AreaChart data={effectiveAnalytics.timeline} margin={{ top: 15, right: 15, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                          <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                          <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                          <Tooltip
                            contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)', borderRadius: 'var(--radius-sm)' }}
                            labelStyle={{ color: 'var(--text-primary)', fontWeight: 'bold' }}
                          />
                          <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />
                          <Area type="monotone" dataKey="fillers" name="Fillers" stackId="1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.6} />
                          <Area type="monotone" dataKey="stammers" name="Stammers" stackId="1" stroke="#ec4899" fill="#ec4899" fillOpacity={0.6} />
                          <Area type="monotone" dataKey="pauses" name="Long Pauses" stackId="1" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                        </AreaChart>
                      ) : (
                        <LineChart data={effectiveAnalytics.timeline} margin={{ top: 15, right: 15, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                          <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                          <YAxis domain={[60, 220]} stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                          <Tooltip
                            contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)', borderRadius: 'var(--radius-sm)' }}
                            labelStyle={{ color: 'var(--text-primary)', fontWeight: 'bold' }}
                          />
                          <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />
                          <ReferenceArea y1={120} y2={150} fill="rgba(6, 182, 212, 0.12)" stroke="rgba(6, 182, 212, 0.3)" />
                          <ReferenceLine y={135} stroke="#06b6d4" strokeDasharray="3 3" label={{ value: '135 WPM Target', fill: '#06b6d4', fontSize: 10, position: 'insideBottomRight' }} />
                          <Line type="monotone" dataKey="wpm" name="Speaking Rate (WPM)" stroke="#06b6d4" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                        </LineChart>
                      )}
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              {/* Right Column: 5-Axis Competency Radar & Streak */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div className="glass-card" style={{ padding: '20px', borderRadius: 'var(--radius-lg)' }}>
                  <h3 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Target size={18} style={{ color: 'var(--primary)' }} />
                    5-Axis Competency Radar (0-10)
                  </h3>
                  <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: '4px 0 10px 0' }}>
                    Baseline vs. Current multi-axial performance comparison.
                  </p>
                  <div className="radar-wrapper" style={{ height: '220px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart cx="50%" cy="50%" outerRadius="70%" data={effectiveAnalytics.radar_data}>
                        <PolarGrid stroke="rgba(255,255,255,0.1)" />
                        <PolarAngleAxis dataKey="dimension" stroke="var(--text-secondary)" fontSize={10} />
                        <PolarRadiusAxis angle={30} domain={[0, 10]} stroke="rgba(255,255,255,0.2)" fontSize={9} />
                        <Radar name="Baseline" dataKey="baseline" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.25} />
                        <Radar name="Current" dataKey="current" stroke="var(--primary)" fill="var(--primary)" fillOpacity={0.5} />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="glass-card stat-card-highlight" style={{ padding: '18px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <Flame size={30} style={{ color: 'var(--secondary)', fill: 'var(--secondary)' }} />
                      <div>
                        <h3 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', margin: 0 }}>Daily Goal Streak</h3>
                        <p style={{ fontSize: '11px', color: 'var(--text-secondary)', margin: '2px 0 0 0' }}>Consecutive UTC days</p>
                      </div>
                    </div>
                    <div style={{ fontSize: '30px', fontWeight: '800', color: 'var(--text-primary)' }}>
                      {streakCount} <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Days</span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', paddingTop: '10px', borderTop: '1px solid var(--border-color)', fontSize: '12px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Total Logged Sessions:</span>
                    <strong style={{ color: 'var(--text-primary)' }}>{effectiveAnalytics.total_sessions}</strong>
                  </div>
                </div>
              </div>
            </div>


            {/* Swappable History Timelines Logs Panel */}
            <div className="glass-card" style={{ marginTop: '24px', textAlign: 'left', borderRadius: 'var(--radius-lg)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <History size={22} style={{ color: 'var(--primary)' }} />
                  <h3 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)', margin: 0 }}>
                    {historyTab === 'analysis' ? 'Fluency History (Module 1)' : historyTab === 'video' ? 'Video Practice History (Module 2)' : 'Exercise History'}
                  </h3>
                </div>

                {/* Sub-tab Swapper with 3 tabs */}
                <div style={{ display: 'flex', gap: '6px', backgroundColor: 'rgba(255,255,255,0.02)', padding: '2px', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                  <button 
                    className={`nav-tab-btn ${historyTab === 'analysis' ? 'active' : ''}`}
                    onClick={() => { setHistoryTab('analysis'); setExpandedSessionId(null); }}
                    style={{ fontSize: '12px', padding: '6px 12px' }}
                  >
                    Module 1: Fluency
                  </button>
                  <button 
                    className={`nav-tab-btn ${historyTab === 'exercise' ? 'active' : ''}`}
                    onClick={() => { setHistoryTab('exercise'); setExpandedSessionId(null); }}
                    style={{ fontSize: '12px', padding: '6px 12px' }}
                  >
                    Exercises
                  </button>
                  <button 
                    className={`nav-tab-btn ${historyTab === 'video' ? 'active' : ''}`}
                    onClick={() => { setHistoryTab('video'); setExpandedSessionId(null); }}
                    style={{ fontSize: '12px', padding: '6px 12px' }}
                  >
                    Module 2: Video
                  </button>
                </div>
              </div>

              {filteredHistory.length === 0 ? (
                <p style={{ color: 'var(--text-secondary)', fontSize: '14px', margin: 0 }}>No session logs found for this module. Record a session to populate this list!</p>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {visibleHistory.map((session, idx) => {
                    const date = new Date(session.created_at);
                    const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    const sessionId = session.session_id || (session as any)._id || (session as any).job_id || (session as any).id || String(idx);
                    const isExpanded = expandedSessionId === sessionId;
                    const isVideoSession = session.type === 'video' || session.session_category === 'video';
                    const isExercisePractice = session.session_category === 'practice' || (session.type === 'exercise' && session.pronunciation_score !== undefined);
                    const isExerciseAnalysis = session.session_category === 'analysis' || (session.type === 'exercise' && session.final_score !== undefined);
                    const badgeColor = isExercisePractice || session.type === 'practice'
                      ? 'var(--primary-light)'
                      : isVideoSession
                      ? 'rgba(6, 182, 212, 0.15)'
                      : 'var(--secondary-light)';
                    const badgeTextColor = isExercisePractice || session.type === 'practice'
                      ? 'var(--primary)'
                      : isVideoSession
                      ? '#06b6d4'
                      : 'var(--secondary)';
                    const summaryScoreText = isExercisePractice
                      ? `Practice score: ${session.pronunciation_score ?? 0}/10`
                      : isExerciseAnalysis
                        ? `Fluency score: ${session.final_score ?? 0}/10`
                        : isVideoSession
                          ? `Overall score: ${((session as any).scores?.overall_10 ?? session.final_score ?? session.pronunciation_score ?? 0).toFixed(1)}/10`
                          : session.type === 'analysis'
                            ? `Fluency score: ${session.final_score ?? 0}/10`
                            : `Exercise score: ${session.pronunciation_score ?? session.final_score ?? 0}/10`;

                    // Video Session specific detailed payload normalizations
                    const rawScores = (session as any).scores || {};
                    const normalizedScores: ScoresBreakdown = {
                      F: rawScores.F ?? null,
                      P: rawScores.P ?? null,
                      N: rawScores.N ?? null,
                      E: rawScores.E ?? null,
                      G: rawScores.G ?? null,
                      V: rawScores.V ?? null,
                      overall_100: rawScores.overall_100 ?? (rawScores.overall_10 !== undefined ? rawScores.overall_10 * 10 : (session.final_score !== undefined ? session.final_score * 10 : 0)),
                      overall_10: rawScores.overall_10 ?? session.final_score ?? 0,
                      weights_used: rawScores.weights_used ?? {},
                      unavailable: rawScores.unavailable ?? []
                    };

                    const rawSpeech = (session as any).speech || {};
                    const normalizedSpeech: SpeechFeatures = {
                      transcript: rawSpeech.transcript || session.transcript || (session as any).spoken_text || '',
                      word_count: rawSpeech.word_count || ((rawSpeech.transcript || session.transcript || '').split(/\s+/).filter(Boolean).length || 1),
                      wpm: rawSpeech.wpm ?? session.wpm ?? 0,
                      filler_count: rawSpeech.filler_count ?? session.filler_count ?? 0,
                      filler_types: rawSpeech.filler_types || [],
                      repetition_count: rawSpeech.repetition_count ?? session.stammer_events ?? 0,
                      long_pauses: rawSpeech.long_pauses ?? session.long_pauses ?? 0,
                      pause_ratio: rawSpeech.pause_ratio ?? 0,
                      clarity_raw: rawSpeech.clarity_raw ?? 0,
                      pause_events: rawSpeech.pause_events || session.pause_details || [],
                      filler_details: rawSpeech.filler_details || [],
                      stammer_details: rawSpeech.stammer_details || []
                    };

                    const durationSec = Math.max(1, (session as any).duration_sec || 1);
                    const normalizedTimeline = (session as any).timeline || [];
                    const cameraFacingPct = Math.round(((session as any).visual?.camera_facing_ratio ?? 0) * 100);

                    return (
                      <div key={idx} style={{ display: 'flex', flexDirection: 'column', padding: '16px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                        {isVideoSession ? (
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', flexWrap: 'wrap', gap: '12px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                              <div style={{
                                width: '46px',
                                height: '46px',
                                borderRadius: '50%',
                                background: 'var(--primary-light)',
                                border: '2px solid var(--primary)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontWeight: 800,
                                fontSize: '15px',
                                color: 'var(--text-primary)',
                                flexShrink: 0
                              }}>
                                {normalizedScores.overall_10.toFixed(1)}
                              </div>

                              <div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                                  <span style={{ 
                                    fontSize: '10px', 
                                    padding: '2px 8px', 
                                    borderRadius: '4px', 
                                    fontWeight: 'bold',
                                    textTransform: 'uppercase',
                                    backgroundColor: 'rgba(6, 182, 212, 0.15)',
                                    color: '#06b6d4'
                                  }}>
                                    Module 2: Video
                                  </span>
                                  <span style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'capitalize' }}>
                                    {((session as any).task_type || 'free_talk').replace('_', ' ')}
                                  </span>
                                  <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: 'var(--radius-full)', background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}>
                                    {Math.round((session as any).duration_sec || 0)}s
                                  </span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12px', color: 'var(--text-muted)' }}>
                                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                    <Calendar size={12} /> {dateStr}
                                  </span>
                                  <span>•</span>
                                  <span>{normalizedSpeech.wpm} WPM</span>
                                  <span>•</span>
                                  <span>{cameraFacingPct}% Gaze</span>
                                </div>
                                {normalizedSpeech.transcript && (
                                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px', fontStyle: 'italic', maxWidth: '650px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    "{normalizedSpeech.transcript}"
                                  </p>
                                )}
                              </div>
                            </div>

                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                              <button 
                                onClick={() => setExpandedSessionId(isExpanded ? null : sessionId)}
                                style={{ 
                                  fontSize: '12px', 
                                  padding: '6px 14px', 
                                  backgroundColor: isExpanded ? 'var(--bg-primary)' : 'var(--primary-light)', 
                                  border: '1px solid var(--primary)', 
                                  borderRadius: 'var(--radius-sm)', 
                                  color: 'var(--text-primary)',
                                  cursor: 'pointer',
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '6px'
                                }}
                              >
                                {isExpanded ? (
                                  <>
                                    <span>Hide Details</span>
                                    <ChevronUp size={14} />
                                  </>
                                ) : (
                                  <>
                                    <span>View Report</span>
                                    <ChevronDown size={14} />
                                  </>
                                )}
                              </button>
                              <button
                                onClick={() => handleDeleteHistorySession(sessionId, session.type)}
                                style={{
                                  background: 'none',
                                  border: 'none',
                                  color: 'var(--text-muted)',
                                  padding: '6px',
                                  cursor: 'pointer',
                                  borderRadius: '4px',
                                  display: 'flex',
                                  alignItems: 'center'
                                }}
                                title="Delete session"
                              >
                                <Trash2 size={16} />
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', flexWrap: 'wrap', gap: '10px' }}>
                            <div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <span style={{ 
                                  fontSize: '10px', 
                                  padding: '2px 8px', 
                                  borderRadius: '4px', 
                                  fontWeight: 'bold',
                                  textTransform: 'uppercase',
                                  backgroundColor: badgeColor,
                                  color: badgeTextColor
                                }}>
                                  {session.type}
                                </span>
                                <span style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--text-primary)' }}>
                                  {summaryScoreText}
                                </span>
                              </div>
                              <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '6px' }}>
                                {session.type === 'practice' ? `Target: "${session.target_text}"`
                                  : session.type === 'analysis' ? `Spoken: "${session.transcript || 'Speech'}"`
                                  : `Exercise: ${session.exercise_title || session.exercise_id || 'Practice Drill'}`}
                              </p>
                            </div>
                            
                            <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{dateStr}</span>
                              <button 
                                onClick={() => setExpandedSessionId(isExpanded ? null : sessionId)}
                                style={{ 
                                  fontSize: '12px', 
                                  padding: '6px 12px', 
                                  backgroundColor: isExpanded ? 'var(--bg-primary)' : 'var(--primary-light)', 
                                  border: '1px solid var(--primary)', 
                                  borderRadius: 'var(--radius-sm)', 
                                  color: 'var(--text-primary)',
                                  cursor: 'pointer'
                                }}
                              >
                                {isExpanded ? 'Hide Details' : 'View Report'}
                              </button>
                            </div>
                          </div>
                        )}

                        {/* Expandable detailed scorecard panel */}
                        {isExpanded && (
                          <div className="view-fade">
                            {isVideoSession ? (
                              <div style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '12px', borderBottom: '1px solid var(--border-color)', flexWrap: 'wrap', gap: '8px' }}>
                                  <h3 style={{ fontSize: '18px', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
                                    <VideoIcon size={20} style={{ color: '#06b6d4' }} />
                                    Multimodal Session Analysis Report
                                  </h3>
                                  <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                                    Recorded {dateStr}
                                  </span>
                                </div>

                                {/* 1. Score Overview & Radar */}
                                <ScoreOverview
                                  scores={normalizedScores}
                                  durationSec={durationSec}
                                />

                                {/* 1.5 Detailed Speech Analytics */}
                                {normalizedSpeech && (
                                  <>
                                    <SpeechDetailedMetrics
                                      speech={normalizedSpeech}
                                      durationSec={durationSec}
                                      taskType={(session as any).task_type || 'free_talk'}
                                    />
                                    {(((session as any).task_type === 'custom_topic') || (session as any).prompt_text || session.target_text) && (
                                      <PronunciationHighlight
                                        targetText={(session as any).prompt_text || session.target_text || ''}
                                        spokenText={normalizedSpeech.transcript}
                                      />
                                    )}
                                  </>
                                )}

                                {/* 2. 5-Second Window Fusion Timeline */}
                                {normalizedTimeline.length > 0 && (
                                  <FusionTimeline timeline={normalizedTimeline} />
                                )}

                                {/* 3. Emotion Breakdown & Feedback List */}
                                <div className="vid-results-grid">
                                  <EmotionBreakdown
                                    distribution={(session as any).visual?.emotion_distribution || {}}
                                    dominantEmotion={(session as any).visual?.dominant_emotion || 'neutral'}
                                  />
                                  <FeedbackList
                                    feedback={(session as any).feedback || []}
                                    qualityWarnings={(session as any).quality_warnings || (session as any).visual?.quality_warnings || []}
                                  />
                                </div>
                              </div>
                            ) : isExercisePractice ? (
                              <div style={{ marginTop: '16px', padding: '16px', backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                                <h4 style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '12px' }}>{session.type === 'exercise' ? 'Exercise Pronunciation Metrics' : 'Pronunciation Accuracy Metrics'}</h4>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '12px' }}>
                                  <span style={{ fontSize: '12px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', padding: '4px 10px', borderRadius: '4px' }}>
                                    Score: <strong>{session.pronunciation_score}/10</strong>
                                  </span>
                                  <span style={{ fontSize: '12px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', padding: '4px 10px', borderRadius: '4px' }}>
                                    Word Error Rate (WER): <strong>{(session.word_error_rate ? session.word_error_rate * 100 : 0).toFixed(0)}%</strong>
                                  </span>
                                </div>
                                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                                  <strong>Spoken text matched:</strong> <em>"{session.spoken_text}"</em>
                                </p>
                                {session.mismatched_words && session.mismatched_words.length > 0 && (
                                  <div style={{ marginTop: '12px' }}>
                                    <span style={{ fontSize: '12px', color: 'var(--error)', fontWeight: 'bold', display: 'block', marginBottom: '6px' }}>Mismatched words:</span>
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                      {session.mismatched_words.map((w, idx) => (
                                        <span key={idx} style={{ backgroundColor: 'var(--error-light)', color: 'var(--error)', padding: '2px 8px', borderRadius: '4px', fontSize: '12px' }}>
                                          Expected: "{w.expected}" → Said: "{w.spoken || '[not detected / skipped]'}"
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div style={{ marginTop: '16px', padding: '16px', backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                                <h4 style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '12px' }}>Detailed Fluency Metrics</h4>
                                
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: '10px', marginBottom: '15px' }}>
                                  <div style={{ padding: '10px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', textAlign: 'center' }}>
                                    <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>WPM Rate</span>
                                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: 'var(--primary)', marginTop: '4px' }}>{session.wpm} WPM</div>
                                  </div>
                                  <div style={{ padding: '10px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', textAlign: 'center' }}>
                                    <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Filler Words</span>
                                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: 'var(--primary)', marginTop: '4px' }}>{session.filler_count}</div>
                                  </div>
                                  <div style={{ padding: '10px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', textAlign: 'center' }}>
                                    <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Stammers</span>
                                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: 'var(--primary)', marginTop: '4px' }}>{session.stammer_events}</div>
                                  </div>
                                  <div style={{ padding: '10px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', textAlign: 'center' }}>
                                    <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Long Pauses</span>
                                    <div style={{ fontSize: '15px', fontWeight: 'bold', color: 'var(--primary)', marginTop: '4px' }}>{session.long_pauses}</div>
                                  </div>
                                </div>

                                {session.sub_scores && (
                                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(90px, 1fr))', gap: '8px', marginBottom: '15px' }}>
                                    {Object.entries(session.sub_scores).map(([name, score], idx) => (
                                      <div key={idx} style={{ padding: '8px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', textAlign: 'center', borderColor: score < 6 ? 'var(--error)' : 'var(--border-color)' }}>
                                        <span style={{ fontSize: '9px', color: 'var(--text-secondary)', textTransform: 'capitalize' }}>{name.replace('_score', '')}</span>
                                        <div style={{ fontSize: '13px', fontWeight: 'bold', color: score < 6 ? 'var(--error)' : 'var(--success)', marginTop: '2px' }}>{score}/10</div>
                                      </div>
                                    ))}
                                  </div>
                                )}

                                {session.pause_details && session.pause_details.length > 0 && (
                                   <div style={{ marginBottom: '12px' }}>
                                     <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Detected pauses:</span>
                                     
                                     {/* Interactive visual timeline inside logs details */}
                                     <div style={{ 
                                       position: 'relative', 
                                       height: '10px', 
                                       backgroundColor: 'var(--bg-primary)', 
                                       border: '1px solid var(--border-color)', 
                                       borderRadius: '5px', 
                                       overflow: 'hidden', 
                                       width: '100%', 
                                       marginBottom: '6px' 
                                     }}>
                                       {session.pause_details.map((pause, idx) => {
                                         const total = session.duration_sec || 1;
                                         const leftPct = (pause.start / total) * 100;
                                         const widthPct = (pause.duration / total) * 100;
                                         return (
                                           <div 
                                             key={idx} 
                                             style={{ 
                                               position: 'absolute', 
                                               left: `${leftPct}%`, 
                                               width: `${widthPct}%`, 
                                               height: '100%', 
                                               backgroundColor: 'var(--error)',
                                               opacity: 0.8
                                             }}
                                           />
                                         );
                                       })}
                                     </div>
                                     <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                                       <span>0.0s</span>
                                       <span>{(session.duration_sec || 0).toFixed(1)}s</span>
                                     </div>

                                     <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                       {session.pause_details.map((pause, idx) => (
                                         <span key={idx} style={{ fontSize: '10px', backgroundColor: 'var(--warning-light)', color: 'var(--warning)', padding: '2px 6px', borderRadius: '4px' }}>
                                           {pause.duration}s gap at {pause.start}s
                                         </span>
                                       ))}
                                     </div>
                                   </div>
                                 )}

                                 {session.filler_details && session.filler_details.length > 0 && (
                                   <div style={{ marginBottom: '12px' }}>
                                     <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Detected filler words:</span>
                                     
                                     <div style={{ 
                                       position: 'relative', 
                                       height: '10px', 
                                       backgroundColor: 'var(--bg-primary)', 
                                       border: '1px solid var(--border-color)', 
                                       borderRadius: '5px', 
                                       overflow: 'hidden', 
                                       width: '100%', 
                                       marginBottom: '6px' 
                                     }}>
                                       {session.filler_details.map((filler, idx) => {
                                         const total = session.duration_sec || 1;
                                         const leftPct = (filler.start / total) * 100;
                                         const widthPct = Math.max((filler.duration / total) * 100, 1.2);
                                         return (
                                           <div 
                                             key={idx} 
                                             style={{ 
                                               position: 'absolute', 
                                               left: `${leftPct}%`, 
                                               width: `${widthPct}%`, 
                                               height: '100%', 
                                               backgroundColor: '#f59e0b',
                                               opacity: 0.85
                                             }}
                                             title={`Filler: "${filler.word}" (${filler.duration}s from ${filler.start}s to ${filler.end}s)`}
                                           />
                                         );
                                       })}
                                     </div>
                                     <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                                       <span>0.0s</span>
                                       <span>{(session.duration_sec || 0).toFixed(1)}s</span>
                                     </div>

                                     <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                       {session.filler_details.map((filler, idx) => (
                                         <span key={idx} style={{ fontSize: '10px', backgroundColor: 'rgba(245, 158, 11, 0.12)', color: '#d97706', padding: '2px 6px', borderRadius: '4px', fontWeight: 500 }}>
                                           "{filler.word}" at {filler.start}s ({filler.duration}s)
                                         </span>
                                       ))}
                                     </div>
                                   </div>
                                 )}

                                 {session.stammer_details && session.stammer_details.length > 0 && (
                                   <div style={{ marginBottom: '12px' }}>
                                     <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Detected stammers & repetitions:</span>
                                     
                                     <div style={{ 
                                       position: 'relative', 
                                       height: '10px', 
                                       backgroundColor: 'var(--bg-primary)', 
                                       border: '1px solid var(--border-color)', 
                                       borderRadius: '5px', 
                                       overflow: 'hidden', 
                                       width: '100%', 
                                       marginBottom: '6px' 
                                     }}>
                                       {session.stammer_details.map((stammer, idx) => {
                                         const total = session.duration_sec || 1;
                                         const leftPct = (stammer.start / total) * 100;
                                         const widthPct = Math.max((stammer.duration / total) * 100, 1.2);
                                         return (
                                           <div 
                                             key={idx} 
                                             style={{ 
                                               position: 'absolute', 
                                               left: `${leftPct}%`, 
                                               width: `${widthPct}%`, 
                                               height: '100%', 
                                               backgroundColor: '#ec4899',
                                               opacity: 0.85
                                             }}
                                             title={`Stammer (${stammer.type}): "${stammer.text}" (${stammer.duration}s from ${stammer.start}s to ${stammer.end}s)`}
                                           />
                                         );
                                       })}
                                     </div>
                                     <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                                       <span>0.0s</span>
                                       <span>{(session.duration_sec || 0).toFixed(1)}s</span>
                                     </div>

                                     <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                       {session.stammer_details.map((stammer, idx) => (
                                         <span key={idx} style={{ fontSize: '10px', backgroundColor: 'rgba(236, 72, 153, 0.12)', color: '#db2777', padding: '2px 6px', borderRadius: '4px', fontWeight: 500 }}>
                                           "{stammer.text}" ({stammer.type}) at {stammer.start}s
                                         </span>
                                       ))}
                                     </div>
                                   </div>
                                 )}

                                 {session.ai_pathologist_feedback && (
                                   <div style={{ marginBottom: '15px', padding: '10px', borderLeft: '3px solid var(--accent)', backgroundColor: 'rgba(99,102,241,0.01)' }}>
                                     <span style={{ fontSize: '11px', color: 'var(--accent)', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '2px' }}>
                                       <Sparkles size={11} /> AI Speech Pathologist Assessment:
                                     </span>
                                     <p style={{ fontSize: '12px', color: 'var(--text-primary)', fontStyle: 'italic', margin: '0' }}>
                                       "{session.ai_pathologist_feedback}"
                                     </p>
                                   </div>
                                 )}

                                {session.feedback && session.feedback.length > 0 && (
                                  <div style={{ marginBottom: '15px' }}>
                                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Therapy Feedback:</span>
                                    <ul style={{ fontSize: '12px', color: 'var(--text-secondary)', paddingLeft: '16px' }}>
                                      {session.feedback.map((f, idx) => <li key={idx}>{f}</li>)}
                                    </ul>
                                  </div>
                                )}

                                {session.recommended_exercises && session.recommended_exercises.length > 0 && (
                                  <div>
                                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Recommended Drills:</span>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                      {session.recommended_exercises.map((exId, idx) => {
                                        const exDetail = exercisesList.find(item => item._id === exId);
                                        return (
                                          <div key={idx} style={{ padding: '8px 12px', backgroundColor: 'var(--bg-secondary)', borderRadius: '4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <div>
                                              <span style={{ fontSize: '12px', fontWeight: 'bold', color: 'var(--text-primary)' }}>{exDetail ? exDetail.title : exId.replace(/_/g, ' ')}</span>
                                              <p style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{exDetail ? exDetail.description : "Speech drill to improve fluency."}</p>
                                            </div>
                                            <button 
                                              onClick={() => {
                                                if (exDetail) {
                                                  handleTabChange('exercises', true);
                                                  startExerciseDrill(exDetail);
                                                } else {
                                                  const fallbackEx = {
                                                    _id: exId,
                                                    title: exId.replace(/_/g, ' '),
                                                    description: "Speech drill to improve fluency.",
                                                    difficulty: 'intermediate',
                                                    trigger_condition: 'automatic'
                                                  };
                                                  handleTabChange('exercises', true);
                                                  startExerciseDrill(fallbackEx);
                                                }
                                              }}
                                              style={{ fontSize: '11px', padding: '4px 8px', backgroundColor: 'var(--primary-light)', border: '1px solid var(--primary)', borderRadius: '3px', color: 'var(--text-primary)', cursor: 'pointer' }}
                                            >
                                              Launch
                                            </button>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {/* 2. MODULE 1: FLUENCY MODULE */}
        {activeTab === 'fluency' && (
          <div className="view-fade">
            <div className="dashboard-hero" style={{ marginBottom: '24px' }}>
              <h1 className="hero-title">Module 1: Fluency Tracker</h1>
              <p className="hero-subtitle">Practice impromptu topics or read dynamic target passages to track filler words, pacing, and silences.</p>
            </div>

            {/* Target Practice Text card (renders only if fluencyTargetText is set) */}
            {fluencyTargetText && (
              <div className="glass-card" style={{ marginBottom: '24px', textAlign: 'left' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <h3 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-secondary)', margin: '0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    📖 Target Reading Passage
                  </h3>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <button
                      onClick={() => (isAudioPlaying || isAudioLoading) && currentAudioType === 'reference' ? stopPlayback() : speakText(fluencyTargetText)}
                      style={{
                        fontSize: '11px',
                        backgroundColor: 'rgba(255,255,255,0.03)',
                        border: '1px solid var(--border-color)',
                        color: 'var(--text-primary)',
                        padding: '6px 10px',
                        borderRadius: 'var(--radius-sm)',
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '6px',
                        transition: 'var(--transition-fast)'
                      }}
                    >
                      {(isAudioPlaying || isAudioLoading) && currentAudioType === 'reference' ? (
                        <Square size={12} style={{ color: 'var(--error)' }} />
                      ) : (
                        <Volume2 size={12} style={{ color: 'var(--primary)' }} />
                      )}
                      {(isAudioPlaying || isAudioLoading) && currentAudioType === 'reference' ? 'Stop' : 'Listen Reference'}
                    </button>
                    <button 
                      onClick={() => { setFluencyTargetText(''); setAnalysisResult(null); }}
                      style={{ 
                        fontSize: '11px', 
                        backgroundColor: 'rgba(239, 68, 68, 0.1)', 
                        border: '1px solid rgba(239, 68, 68, 0.3)', 
                        color: 'var(--error)', 
                        padding: '4px 8px', 
                        borderRadius: 'var(--radius-sm)', 
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}
                    >
                      ✕ Clear passage to speak impromptu
                    </button>
                  </div>
                </div>
                <div className="practice-sentence-box" style={{ fontSize: '18px', lineHeight: '1.6', backgroundColor: 'var(--bg-primary)', border: 'none' }}>
                  {fluencyTargetText.trim().split(/\s+/).map((word, idx) => (
                    <span key={idx} className={getLiveWordColorClass(word, idx)}>{word} </span>
                  ))}
                </div>
              </div>
            )}

            <div className="workspace-layout">
              {/* Left Column: Speaking Topic Prompt OR Passage Generator */}
              <div className="workspace-panel config-panel">
                <div>
                  <h2 className="panel-title">1. Fluency Settings</h2>
                  <p className="panel-subtitle" style={{ marginBottom: '16px' }}>Practice impromptu topic prompts or generate dynamic reading paragraphs.</p>
                  
                  {/* Segmented Switcher */}
                  <div style={{ display: 'flex', gap: '4px', backgroundColor: 'rgba(255,255,255,0.02)', padding: '4px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', marginBottom: '20px' }}>
                    <button 
                      onClick={() => setConfigTabM1('prompt')}
                      style={{ 
                        flex: 1, 
                        padding: '8px', 
                        fontSize: '12px', 
                        fontWeight: '600',
                        backgroundColor: configTabM1 === 'prompt' ? 'var(--bg-secondary)' : 'transparent', 
                        border: configTabM1 === 'prompt' ? '1px solid var(--border-color)' : 'none', 
                        color: configTabM1 === 'prompt' ? 'var(--text-primary)' : 'var(--text-secondary)',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        transition: 'var(--transition-fast)'
                      }}
                    >
                      💡 Topic Prompts
                    </button>
                    <button 
                      onClick={() => setConfigTabM1('groq')}
                      style={{ 
                        flex: 1, 
                        padding: '8px', 
                        fontSize: '12px', 
                        fontWeight: '600',
                        backgroundColor: configTabM1 === 'groq' ? 'rgba(168, 85, 247, 0.1)' : 'transparent', 
                        border: configTabM1 === 'groq' ? '1px solid var(--secondary)' : 'none', 
                        color: configTabM1 === 'groq' ? 'var(--text-primary)' : 'var(--text-secondary)',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        transition: 'var(--transition-fast)'
                      }}
                    >
                      ✨ Groq AI
                    </button>
                  </div>

                  {/* Render content based on sub-tab selection */}
                  {configTabM1 === 'prompt' && (
                    <div className="practice-sentence-box" style={{ fontSize: '15px', fontStyle: 'italic', color: 'var(--text-secondary)', minHeight: '90px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', backgroundColor: 'rgba(255,255,255,0.01)', border: '1px dashed var(--border-color)', padding: '16px' }}>
                      <p style={{ margin: '0' }}>💡 Impromptu Prompt: "{TOPIC_PROMPTS[promptIndex]}"</p>
                      <button 
                        onClick={() => setPromptIndex((promptIndex + 1) % TOPIC_PROMPTS.length)}
                        style={{ 
                          alignSelf: 'flex-end', 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '6px', 
                          fontSize: '11px', 
                          backgroundColor: 'rgba(255,255,255,0.04)', 
                          padding: '4px 10px', 
                          borderRadius: 'var(--radius-sm)', 
                          border: '1px solid var(--border-color)',
                          color: 'var(--text-primary)',
                          cursor: 'pointer',
                          marginTop: '8px'
                        }}
                      >
                        <RefreshCw size={11} /> Next Prompt
                      </button>
                    </div>
                  )}

                  {configTabM1 === 'groq' && (
                    <div className="glass-card" style={{ padding: '16px', border: '1px solid var(--border-color)', backgroundColor: 'rgba(255,255,255,0.01)' }}>
                      <h3 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <Sparkles size={14} style={{ color: 'var(--accent)' }} /> Generate Custom Text with Groq AI
                      </h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div className="form-group" style={{ marginBottom: '4px' }}>
                          <label className="form-label" style={{ fontSize: '10px' }}>Topic Keyword</label>
                          <input 
                            type="text" 
                            className="form-input" 
                            style={{ padding: '6px 12px', fontSize: '13px' }}
                            value={aiTopic}
                            onChange={e => setAiTopic(e.target.value)}
                            placeholder="Space, Cooking, AI..."
                          />
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                          <div className="form-group" style={{ marginBottom: '4px' }}>
                            <label className="form-label" style={{ fontSize: '10px' }}>Length</label>
                            <select 
                              style={{ 
                                padding: '6px', 
                                backgroundColor: 'var(--bg-primary)', 
                                color: 'var(--text-primary)',
                                border: '1px solid var(--border-color)',
                                borderRadius: 'var(--radius-sm)',
                                fontSize: '13px'
                              }}
                              value={aiLength === 'sentence' ? 'paragraph' : aiLength}
                              onChange={e => setAiLength(e.target.value)}
                            >
                              <option value="paragraph">Paragraph</option>
                              <option value="long_paragraph">Long Paragraph</option>
                            </select>
                          </div>
                          <div className="form-group" style={{ marginBottom: '4px' }}>
                            <label className="form-label" style={{ fontSize: '10px' }}>Level of English</label>
                            <select 
                              style={{ 
                                padding: '6px', 
                                backgroundColor: 'var(--bg-primary)', 
                                color: 'var(--text-primary)',
                                border: '1px solid var(--border-color)',
                                borderRadius: 'var(--radius-sm)',
                                fontSize: '13px'
                              }}
                              value={aiEnglishLevel}
                              onChange={e => setAiEnglishLevel(e.target.value as 'easy' | 'medium' | 'difficult')}
                            >
                              <option value="easy">Easy</option>
                              <option value="medium">Medium</option>
                              <option value="difficult">Difficult</option>
                            </select>
                          </div>
                        </div>
                        <button 
                          onClick={handleGenerateText} 
                          className="submit-btn" 
                          disabled={isGenerating}
                          style={{ padding: '8px', fontSize: '13px', marginTop: '5px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                        >
                          {isGenerating ? (
                            <>
                              <RefreshCw size={14} className="spin" /> Generating...
                            </>
                          ) : (
                            <>
                              <Sparkles size={14} /> Generate with Groq AI
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column: Symmetrical Recording Workspace */}
              <div className="workspace-panel recording-panel">
                <div>
                  <h2 className="panel-title">2. Speak & Record</h2>
                  <p className="panel-subtitle" style={{ marginBottom: '20px' }}>
                    {!fluencyTargetText 
                      ? 'Speak freely on the prompt. Pacing, stammers, and silences will be tracked.' 
                      : 'Read the target passage aloud. Word accuracy and pacing will be tracked.'}
                  </p>
                </div>

                <div className={`recorder-container ${isRecording ? 'recording' : ''}`} style={{ width: '100%', minHeight: '230px', margin: '0', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <div className={`status-badge ${status}`}>
                    {status === 'recording' && <span className="pulse-dot" style={{ width: '8px', height: '8px', backgroundColor: 'var(--error)', borderRadius: '50%', display: 'inline-block', marginRight: '6px' }}></span>}
                    {statusMessage}
                  </div>

                  <div className="record-btn-wrapper">
                    {!isRecording ? (
                      <button className="record-btn" onClick={startRecording}>
                        <Mic size={36} />
                      </button>
                    ) : (
                      <button className="record-btn recording" onClick={stopRecording}>
                        <Square size={32} />
                      </button>
                    )}
                  </div>

                  {isRecording && (
                    <div className="visualizer-waves">
                      <div className="wave-bar"></div>
                      <div className="wave-bar"></div>
                      <div className="wave-bar"></div>
                      <div className="wave-bar"></div>
                      <div className="wave-bar"></div>
                      <div className="wave-bar"></div>
                      <div className="wave-bar"></div>
                      <div className="wave-bar"></div>
                    </div>
                  )}

                  {liveTranscript && (
                    <div style={{ marginTop: '20px', padding: '10px', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)', width: '100%', textAlign: 'left' }}>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Client Live Feed (Web Speech):</span>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>{liveTranscript}...</p>
                    </div>
                  )}

                  {audioUrl && !isRecording && (
                    <div style={{ marginTop: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Playback:</span>
                      <button
                        onClick={() => playUserRecording()}
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '6px',
                          fontSize: '12px',
                          fontWeight: '600',
                          padding: '6px 12px',
                          backgroundColor: 'rgba(255, 255, 255, 0.03)',
                          border: '1px solid var(--border-color)',
                          borderRadius: 'var(--radius-sm)',
                          color: 'var(--text-primary)',
                          cursor: 'pointer',
                          transition: 'var(--transition-fast)'
                        }}
                      >
                        {(isAudioPlaying || isAudioLoading) && currentAudioType === 'recording' ? (
                          <Square size={14} style={{ color: 'var(--error)' }} />
                        ) : (
                          <Play size={14} style={{ color: 'var(--success)' }} />
                        )}
                        {(isAudioPlaying || isAudioLoading) && currentAudioType === 'recording' ? 'Stop' : 'Play My Recording'}
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Pronunciation Accuracy Analysis card (renders if target passage was practiced) */}
            {analysisResult && (fluencyTargetText || analysisResult.target_text || analysisResult.word_error_rate !== undefined || (analysisResult.mismatched_words && analysisResult.mismatched_words.length > 0)) && (
              <div className="glass-card result-section" style={{ marginTop: '30px', textAlign: 'left' }}>
                <h3 className="feedback-title" style={{ textAlign: 'center', marginBottom: '20px' }}>Pronunciation Accuracy Analysis</h3>
                
                {/* Passage with words highlighted in Green (correct) and Red (wrong pronunciation) */}
                <div style={{ marginBottom: '24px', padding: '18px', backgroundColor: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '12px' }}>
                    📖 Pronunciation Passage Evaluation
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center', fontSize: '18px', lineHeight: '1.8' }}>
                    {(fluencyTargetText || analysisResult.target_text || '').trim().split(/\s+/).map((word: string, index: number) => {
                      const cleanWord = word.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, "");
                      const mismatchedIndices = new Set((analysisResult.mismatched_words || []).map((w: any) => w.index));
                      const isMismatched = mismatchedIndices.has(index);
                      return (
                        <span key={index} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                          {isMismatched && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                speakText(cleanWord);
                              }}
                              title={`Listen to correct pronunciation of "${cleanWord}"`}
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '3px',
                                padding: '2px 6px',
                                backgroundColor: 'rgba(99, 102, 241, 0.15)',
                                border: '1px solid var(--primary)',
                                borderRadius: '4px',
                                color: 'var(--primary)',
                                fontSize: '11px',
                                fontWeight: 700,
                                cursor: 'pointer',
                                transition: 'all 0.15s ease',
                                verticalAlign: 'middle',
                              }}
                            >
                              <Volume2 size={11} />
                              <span>Listen</span>
                            </button>
                          )}
                          <span 
                            className={`practice-word ${isMismatched ? 'incorrect' : 'correct'}`}
                          >
                            {word}{' '}
                          </span>
                        </span>
                      );
                    })}
                  </div>
                  {/* Pronunciation Legend */}
                  <div style={{ display: 'flex', gap: '20px', marginTop: '16px', paddingTop: '12px', borderTop: '1px solid var(--border-color)', fontSize: '13px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#22c55e' }}></span>
                      <span style={{ color: '#22c55e', fontWeight: 600 }}>Green: Correct Pronunciation</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#ef4444' }}></span>
                      <span style={{ color: '#ef4444', fontWeight: 600 }}>Red: Wrong Pronunciation / Missed</span>
                    </div>
                  </div>
                </div>

                <div className="result-grid">
                  <div className="score-panel">
                    <div className="score-circle">
                      <span className="score-val">
                        {analysisResult.word_error_rate !== undefined 
                          ? Math.max(0, Math.round((1 - analysisResult.word_error_rate) * 10 * 10) / 10) 
                          : analysisResult.final_score}
                      </span>
                      <span className="score-label">/ 10</span>
                    </div>
                    <h4 className="score-heading">Accuracy Score</h4>
                    <p className="score-desc">
                      Accuracy: {(((1 - (analysisResult.word_error_rate ?? 0))) * 100).toFixed(0)}% matching
                    </p>
                  </div>

                  <div className="feedback-panel">
                    <h4 className="feedback-title">Articulation & Word Alignment</h4>
                    <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '16px' }}>
                      Target words matched against your recorded audio. Words pronounced correctly are highlighted in green, and wrong/mispronounced words are highlighted in red.
                    </p>
                    {analysisResult.mismatched_words && analysisResult.mismatched_words.length > 0 ? (
                      <ul className="feedback-list">
                        {analysisResult.mismatched_words.map((w: any, idx: number) => (
                          <li className="feedback-item" key={idx} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                              <span className="feedback-bullet" style={{ color: '#ef4444' }}>▸</span>
                              <span>
                                At index {w.index + 1}: {w.spoken ? (
                                  <>Expected <strong style={{ color: '#22c55e' }}>"{w.expected}"</strong> but recorded <strong style={{ color: '#ef4444' }}>"{w.spoken}"</strong></>
                                ) : (
                                  <>Expected <strong style={{ color: '#22c55e' }}>"{w.expected}"</strong> <span style={{ color: '#ef4444', fontStyle: 'italic' }}>(not detected / skipped)</span></>
                                )}
                              </span>
                            </div>
                            <button
                              type="button"
                              onClick={() => speakText(w.expected)}
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '5px',
                                fontSize: '11px',
                                fontWeight: 600,
                                padding: '4px 10px',
                                backgroundColor: 'rgba(99, 102, 241, 0.12)',
                                border: '1px solid var(--primary)',
                                borderRadius: 'var(--radius-sm)',
                                color: 'var(--text-primary)',
                                cursor: 'pointer',
                                transition: 'var(--transition-fast)',
                                whiteSpace: 'nowrap'
                              }}
                              title={`Listen to correct pronunciation of "${w.expected}"`}
                            >
                              <Volume2 size={12} style={{ color: 'var(--primary)' }} />
                              Listen
                            </button>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', fontWeight: 'bold' }}>
                        <CheckCircle size={18} /> Excellent clarity! You didn't make any phonetic mistakes.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Fluency Score Report */}
            {analysisResult && (
              <div className="glass-card result-section" style={{ marginTop: '24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <h3 className="feedback-title" style={{ margin: 0 }}>Fluency & Clarity Analysis</h3>
                  <button
                    onClick={() => resetActivityState('fluency')}
                    style={{
                      fontSize: '12px',
                      fontWeight: '600',
                      padding: '6px 14px',
                      backgroundColor: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-sm)',
                      color: 'var(--text-primary)',
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      transition: 'var(--transition-fast)'
                    }}
                    title="Clear current report and start a new fluency prompt"
                  >
                    <RefreshCw size={13} />
                    Start New Practice
                  </button>
                </div>
                
                <div className="result-grid">
                  <div className="score-panel">
                    <div className="score-circle" style={{ borderColor: 'var(--secondary-light)', borderTopColor: 'var(--secondary)' }}>
                      <span className="score-val">{analysisResult.final_score}</span>
                      <span className="score-label">/ 10</span>
                    </div>
                    <h4 className="score-heading">Overall Fluency Score</h4>
                    <p className="score-desc">
                      Pacing: {analysisResult.wpm} WPM ({analysisResult.duration_sec.toFixed(1)}s)
                    </p>
                  </div>
                  
                  <div className="feedback-panel">
                    <h4 className="feedback-title">Disfluency Markers</h4>
                    <ul className="feedback-list">
                      <li className="feedback-item">
                        <span className="feedback-bullet">▸</span>
                        <span>Filler words: <strong>{analysisResult.filler_count}</strong> {analysisResult.filler_words_found.length > 0 && `(${analysisResult.filler_words_found.join(', ')})`}</span>
                      </li>
                      <li className="feedback-item">
                        <span className="feedback-bullet">▸</span>
                        <span>Stammer / repetitions: <strong>{analysisResult.stammer_events}</strong> instances</span>
                      </li>
                      <li className="feedback-item">
                        <span className="feedback-bullet">▸</span>
                        <span>Long pauses (&gt;1.5s): <strong>{analysisResult.long_pauses}</strong> detected</span>
                      </li>
                      <li className="feedback-item">
                        <span className="feedback-bullet">▸</span>
                        <span>Speech rate: <strong>{analysisResult.wpm} WPM</strong> (Ideal: 120-150)</span>
                      </li>
                    </ul>

                    {analysisResult.transcript && (
                      <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                          <h5 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', margin: 0 }}>
                            🎙️ Spoken Transcript & Tagged Fillers:
                          </h5>
                          {analysisResult.filler_count > 0 && (
                            <span style={{ fontSize: '11px', color: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.12)', padding: '2px 8px', borderRadius: '4px', fontWeight: 600 }}>
                              {analysisResult.filler_count} filler {analysisResult.filler_count === 1 ? 'word' : 'words'} detected
                            </span>
                          )}
                        </div>
                        <p style={{ fontSize: '13px', lineHeight: '1.8', color: 'var(--text-primary)', margin: 0 }}>
                          {renderHighlightedTranscript(analysisResult.transcript, analysisResult.filler_words_found)}
                        </p>
                      </div>
                    )}

                    {analysisResult.pause_details && analysisResult.pause_details.length > 0 && (
                      <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                        <h5 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '8px' }}>⏱️ Detected Pause Locations:</h5>
                        
                        {/* Interactive Pause timeline bar */}
                        <div style={{ 
                          position: 'relative', 
                          height: '12px', 
                          backgroundColor: 'var(--bg-primary)', 
                          border: '1px solid var(--border-color)', 
                          borderRadius: '6px', 
                          overflow: 'hidden', 
                          width: '100%', 
                          marginBottom: '8px' 
                        }}>
                          {analysisResult.pause_details.map((pause, idx) => {
                            const total = analysisResult.duration_sec || 1;
                            const leftPct = (pause.start / total) * 100;
                            const widthPct = (pause.duration / total) * 100;
                            return (
                              <div 
                                key={idx} 
                                style={{ 
                                  position: 'absolute', 
                                  left: `${leftPct}%`, 
                                  width: `${widthPct}%`, 
                                  height: '100%', 
                                  backgroundColor: 'var(--error)',
                                  opacity: 0.8
                                }}
                                title={`Pause: ${pause.duration}s (from ${pause.start}s to ${pause.end}s)`}
                              />
                            );
                          })}
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                          <span>0.0s</span>
                          <span>{(analysisResult.duration_sec || 0).toFixed(1)}s</span>
                        </div>

                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                          {analysisResult.pause_details.map((pause, idx) => (
                            <span key={idx} style={{ fontSize: '11px', backgroundColor: 'var(--warning-light)', color: 'var(--warning)', padding: '2px 8px', borderRadius: '4px' }}>
                              {pause.duration}s gap at {pause.start}s
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Detected Filler Words Timeline and Chips */}
                    {analysisResult.filler_details && analysisResult.filler_details.length > 0 && (
                      <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                        <h5 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          🗣️ Detected Filler Word Locations:
                        </h5>
                        
                        {/* Interactive Filler timeline bar */}
                        <div style={{ 
                          position: 'relative', 
                          height: '12px', 
                          backgroundColor: 'var(--bg-primary)', 
                          border: '1px solid var(--border-color)', 
                          borderRadius: '6px', 
                          overflow: 'hidden', 
                          width: '100%', 
                          marginBottom: '8px' 
                        }}>
                          {analysisResult.filler_details.map((filler, idx) => {
                            const total = analysisResult.duration_sec || 1;
                            const leftPct = (filler.start / total) * 100;
                            const widthPct = Math.max((filler.duration / total) * 100, 1.2);
                            return (
                              <div 
                                key={idx} 
                                style={{ 
                                  position: 'absolute', 
                                  left: `${leftPct}%`, 
                                  width: `${widthPct}%`, 
                                  height: '100%', 
                                  backgroundColor: '#f59e0b',
                                  opacity: 0.85
                                }}
                                title={`Filler: "${filler.word}" (${filler.duration}s from ${filler.start}s to ${filler.end}s)`}
                              />
                            );
                          })}
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                          <span>0.0s</span>
                          <span>{(analysisResult.duration_sec || 0).toFixed(1)}s</span>
                        </div>

                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                          {analysisResult.filler_details.map((filler, idx) => (
                            <span key={idx} style={{ fontSize: '11px', backgroundColor: 'rgba(245, 158, 11, 0.12)', color: '#d97706', padding: '2px 8px', borderRadius: '4px', fontWeight: 500 }}>
                              "{filler.word}" at {filler.start}s ({filler.duration}s)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Detected Stammers and Repetitions Timeline and Chips */}
                    {analysisResult.stammer_details && analysisResult.stammer_details.length > 0 && (
                      <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                        <h5 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          ⚡ Detected Stammers & Repetitions:
                        </h5>
                        
                        {/* Interactive Stammer timeline bar */}
                        <div style={{ 
                          position: 'relative', 
                          height: '12px', 
                          backgroundColor: 'var(--bg-primary)', 
                          border: '1px solid var(--border-color)', 
                          borderRadius: '6px', 
                          overflow: 'hidden', 
                          width: '100%', 
                          marginBottom: '8px' 
                        }}>
                          {analysisResult.stammer_details.map((stammer, idx) => {
                            const total = analysisResult.duration_sec || 1;
                            const leftPct = (stammer.start / total) * 100;
                            const widthPct = Math.max((stammer.duration / total) * 100, 1.2);
                            return (
                              <div 
                                key={idx} 
                                style={{ 
                                  position: 'absolute', 
                                  left: `${leftPct}%`, 
                                  width: `${widthPct}%`, 
                                  height: '100%', 
                                  backgroundColor: '#ec4899',
                                  opacity: 0.85
                                }}
                                title={`Stammer (${stammer.type}): "${stammer.text}" (${stammer.duration}s from ${stammer.start}s to ${stammer.end}s)`}
                              />
                            );
                          })}
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                          <span>0.0s</span>
                          <span>{(analysisResult.duration_sec || 0).toFixed(1)}s</span>
                        </div>

                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                          {analysisResult.stammer_details.map((stammer, idx) => (
                            <span key={idx} style={{ fontSize: '11px', backgroundColor: 'rgba(236, 72, 153, 0.12)', color: '#db2777', padding: '2px 8px', borderRadius: '4px', fontWeight: 500 }}>
                              "{stammer.text}" ({stammer.type}) at {stammer.start}s
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* AI pathologist assessment feedback */}
                    {analysisResult.ai_pathologist_feedback && (
                      <div className="feedback-panel" style={{ marginTop: '16px', borderLeft: '4px solid var(--accent)', backgroundColor: 'rgba(99, 102, 241, 0.02)' }}>
                        <h4 className="feedback-title" style={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}>
                          <Sparkles size={14} /> AI Speech Pathologist Assessment
                        </h4>
                        <p style={{ fontSize: '13px', lineHeight: '1.6', color: 'var(--text-primary)', fontStyle: 'italic', margin: '4px 0 0' }}>
                          "{analysisResult.ai_pathologist_feedback}"
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Sub-scores warning rings */}
                <div style={{ marginTop: '30px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '12px' }}>
                  <div style={{ 
                    padding: '16px', 
                    backgroundColor: 'var(--bg-secondary)', 
                    borderRadius: 'var(--radius-sm)', 
                    textAlign: 'center', 
                    border: '1px solid var(--border-color)',
                    borderColor: analysisResult.sub_scores.filler_score < 6 ? 'var(--error)' : 'var(--border-color)',
                    boxShadow: analysisResult.sub_scores.filler_score < 6 ? '0 0 10px rgba(244, 63, 94, 0.15)' : 'none'
                  }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}>
                      Filler Score
                      {analysisResult.sub_scores.filler_score < 6 && <AlertTriangle size={12} style={{ color: 'var(--error)' }} />}
                    </div>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: analysisResult.sub_scores.filler_score < 6 ? 'var(--error)' : 'var(--primary)', marginTop: '4px' }}>{analysisResult.sub_scores.filler_score}/10</div>
                  </div>

                  <div style={{ 
                    padding: '16px', 
                    backgroundColor: 'var(--bg-secondary)', 
                    borderRadius: 'var(--radius-sm)', 
                    textAlign: 'center', 
                    border: '1px solid var(--border-color)',
                    borderColor: analysisResult.sub_scores.stammer_score < 6 ? 'var(--error)' : 'var(--border-color)',
                    boxShadow: analysisResult.sub_scores.stammer_score < 6 ? '0 0 10px rgba(244, 63, 94, 0.15)' : 'none'
                  }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}>
                      Stammer Score
                      {analysisResult.sub_scores.stammer_score < 6 && <AlertTriangle size={12} style={{ color: 'var(--error)' }} />}
                    </div>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: analysisResult.sub_scores.stammer_score < 6 ? 'var(--error)' : 'var(--primary)', marginTop: '4px' }}>{analysisResult.sub_scores.stammer_score}/10</div>
                  </div>

                  <div style={{ 
                    padding: '16px', 
                    backgroundColor: 'var(--bg-secondary)', 
                    borderRadius: 'var(--radius-sm)', 
                    textAlign: 'center', 
                    border: '1px solid var(--border-color)',
                    borderColor: analysisResult.sub_scores.pause_score < 6 ? 'var(--error)' : 'var(--border-color)',
                    boxShadow: analysisResult.sub_scores.pause_score < 6 ? '0 0 10px rgba(244, 63, 94, 0.15)' : 'none'
                  }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}>
                      Pause Score
                      {analysisResult.sub_scores.pause_score < 6 && <AlertTriangle size={12} style={{ color: 'var(--error)' }} />}
                    </div>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: analysisResult.sub_scores.pause_score < 6 ? 'var(--error)' : 'var(--primary)', marginTop: '4px' }}>{analysisResult.sub_scores.pause_score}/10</div>
                  </div>

                  <div style={{ 
                    padding: '16px', 
                    backgroundColor: 'var(--bg-secondary)', 
                    borderRadius: 'var(--radius-sm)', 
                    textAlign: 'center', 
                    border: '1px solid var(--border-color)',
                    borderColor: analysisResult.sub_scores.rate_score < 6 ? 'var(--error)' : 'var(--border-color)',
                    boxShadow: analysisResult.sub_scores.rate_score < 6 ? '0 0 10px rgba(244, 63, 94, 0.15)' : 'none'
                  }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}>
                      Rate Score
                      {analysisResult.sub_scores.rate_score < 6 && <AlertTriangle size={12} style={{ color: 'var(--error)' }} />}
                    </div>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: analysisResult.sub_scores.rate_score < 6 ? 'var(--error)' : 'var(--primary)', marginTop: '4px' }}>{analysisResult.sub_scores.rate_score}/10</div>
                  </div>

                  <div style={{ 
                    padding: '16px', 
                    backgroundColor: 'var(--bg-secondary)', 
                    borderRadius: 'var(--radius-sm)', 
                    textAlign: 'center', 
                    border: '1px solid var(--border-color)',
                    borderColor: analysisResult.sub_scores.clarity_score < 6 ? 'var(--error)' : 'var(--border-color)',
                    boxShadow: analysisResult.sub_scores.clarity_score < 6 ? '0 0 10px rgba(244, 63, 94, 0.15)' : 'none'
                  }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}>
                      Clarity Score
                      {analysisResult.sub_scores.clarity_score < 6 && <AlertTriangle size={12} style={{ color: 'var(--error)' }} />}
                    </div>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: analysisResult.sub_scores.clarity_score < 6 ? 'var(--error)' : 'var(--primary)', marginTop: '4px' }}>{analysisResult.sub_scores.clarity_score}/10</div>
                  </div>
                </div>

                {/* Recommendations and feedback matching */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '20px', marginTop: '20px', textAlign: 'left' }}>
                  <div className="feedback-panel">
                    <h4 className="feedback-title" style={{ color: 'var(--accent)' }}>Therapy Feedback & Advice</h4>
                    <ul className="feedback-list">
                      {analysisResult.feedback.map((f, idx) => (
                        <li className="feedback-item" key={idx}>
                          <span className="feedback-bullet">✦</span>
                          <span>{f}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="feedback-panel">
                    <h4 className="feedback-title" style={{ color: 'var(--success)' }}>Recommended Database Exercises</h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      {analysisResult.recommended_exercises.map((exId, idx) => {
                        const exDetail = exercisesList.find(item => item._id === exId);
                        
                        return (
                          <div key={idx} className="recommended-exercise-item">
                            <div style={{ flex: 1 }}>
                              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                                <h5 style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--text-primary)' }}>
                                  {exDetail ? exDetail.title : exId.replace(/_/g, ' ')}
                                </h5>
                                {exDetail && (
                                  <span style={{ 
                                    fontSize: '9px', 
                                    padding: '1px 6px', 
                                    borderRadius: '3px',
                                    textTransform: 'uppercase',
                                    fontWeight: 'bold',
                                    backgroundColor: exDetail.difficulty === 'beginner' ? 'rgba(16, 185, 129, 0.1)' : exDetail.difficulty === 'intermediate' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(244, 63, 94, 0.1)',
                                    color: exDetail.difficulty === 'beginner' ? 'var(--success)' : exDetail.difficulty === 'intermediate' ? 'var(--warning)' : 'var(--error)'
                                  }}>
                                    {exDetail.difficulty}
                                  </span>
                                )}
                              </div>
                              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                                {exDetail ? exDetail.description : "Practice this drill to work on your matching speech fluency indicator."}
                              </p>
                            </div>
                            
                            <button 
                              onClick={() => {
                                if (exDetail) {
                                  handleTabChange('exercises', true);
                                  startExerciseDrill(exDetail);
                                } else {
                                  const fallbackEx = {
                                    _id: exId,
                                    title: exId.replace(/_/g, ' '),
                                    description: "Practice this drill to work on your matching speech fluency indicator.",
                                    difficulty: 'intermediate',
                                    trigger_condition: 'automatic'
                                  };
                                  handleTabChange('exercises', true);
                                  startExerciseDrill(fallbackEx);
                                }
                              }}
                              style={{ 
                                fontSize: '12px', 
                                padding: '6px 12px', 
                                backgroundColor: 'var(--primary-light)', 
                                border: '1px solid var(--primary)', 
                                borderRadius: 'var(--radius-sm)', 
                                color: 'var(--text-primary)',
                                cursor: 'pointer',
                                marginLeft: '15px'
                              }}
                            >
                              Start Drill
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* 4. EXERCISES LIBRARY VIEW */}
        {activeTab === 'exercises' && (
          <div className="view-fade">
            {activeExerciseDetail ? (
              <div style={{ width: '100%' }}>
                {/* Back button */}
                <button 
                  onClick={() => setActiveExerciseDetail(null)}
                  style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '6px', 
                    background: 'none', 
                    border: 'none', 
                    color: 'var(--text-secondary)', 
                    cursor: 'pointer', 
                    fontSize: '14px', 
                    marginBottom: '20px',
                    padding: '0',
                    transition: 'var(--transition-fast)'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-primary)'}
                  onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}
                >
                  ← Back to Drills Library
                </button>

                {/* Full-width Exercise Guidelines Card (Visible First) */}
                <div className="glass-card" style={{ padding: '24px', border: '1px solid var(--border-color)', marginBottom: '24px', textAlign: 'left' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid var(--border-color)', paddingBottom: '16px', marginBottom: '20px' }}>
                    <div>
                      <span style={{ 
                        fontSize: '10px', 
                        padding: '3px 8px', 
                        borderRadius: '4px',
                        textTransform: 'uppercase',
                        fontWeight: 'bold',
                        backgroundColor: activeExerciseDetail.difficulty === 'beginner' ? 'rgba(16, 185, 129, 0.1)' : activeExerciseDetail.difficulty === 'intermediate' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(244, 63, 94, 0.1)',
                        color: activeExerciseDetail.difficulty === 'beginner' ? 'var(--success)' : activeExerciseDetail.difficulty === 'intermediate' ? 'var(--warning)' : 'var(--error)',
                        display: 'inline-block',
                        marginBottom: '8px'
                      }}>
                        {activeExerciseDetail.difficulty} Level (Default)
                      </span>
                      <h1 style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--text-primary)', margin: '0 0 4px 0' }}>
                        {activeExerciseDetail.title}
                      </h1>
                      <p style={{ fontSize: '14px', color: 'var(--text-secondary)', margin: '0' }}>
                        {activeExerciseDetail.description}
                      </p>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px' }}>
                    <div>
                      <h4 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        🎯 Clinical Objective
                      </h4>
                      <p style={{ margin: '0', fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                        {EXERCISE_INSTRUCTION_MAP[activeExerciseDetail._id]?.objective || "This therapy exercise aims to help you master control over your pacing, articulation, and timing during speech delivery."}
                      </p>
                    </div>

                    <div>
                      <h4 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        📖 How to Perform
                      </h4>
                      <ul style={{ margin: '0', padding: '0 0 0 14px', fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '4px', lineHeight: '1.4' }}>
                        {(EXERCISE_INSTRUCTION_MAP[activeExerciseDetail._id]?.howToPerform || []).map((step, sIdx) => (
                          <li key={sIdx}>{step}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <h4 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        ✨ Therapeutic Benefits
                      </h4>
                      <ul style={{ margin: '0', padding: '0 0 0 14px', fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '4px', lineHeight: '1.4' }}>
                        {(EXERCISE_INSTRUCTION_MAP[activeExerciseDetail._id]?.advantages || []).map((adv, aIdx) => (
                          <li key={aIdx} style={{ listStyleType: 'square' }}>{adv}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>

                {/* Target Practice Text card (only renders if exerciseTargetText is set and it's not impromptu speaking) */}
                {activeExerciseDetail._id !== "advanced_impromptu_speaking" && exerciseTargetText && (
                  <div className="glass-card" style={{ marginBottom: '24px', textAlign: 'left' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <h3 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-secondary)', margin: '0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        📖 Target Practice Passage
                      </h3>
                      <button
                        onClick={() => (isAudioPlaying || isAudioLoading) && currentAudioType === 'reference' ? stopPlayback() : speakText(exerciseTargetText)}
                        style={{
                          fontSize: '11px',
                          backgroundColor: 'rgba(255,255,255,0.03)',
                          border: '1px solid var(--border-color)',
                          color: 'var(--text-primary)',
                          padding: '6px 10px',
                          borderRadius: 'var(--radius-sm)',
                          cursor: 'pointer',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '6px',
                          transition: 'var(--transition-fast)'
                        }}
                      >
                        {(isAudioPlaying || isAudioLoading) && currentAudioType === 'reference' ? (
                          <Square size={12} style={{ color: 'var(--error)' }} />
                        ) : (
                          <Volume2 size={12} style={{ color: 'var(--primary)' }} />
                        )}
                        {(isAudioPlaying || isAudioLoading) && currentAudioType === 'reference' ? 'Stop' : 'Listen Reference'}
                      </button>
                    </div>
                    <div className="practice-sentence-box" style={{ fontSize: '18px', lineHeight: '1.6', backgroundColor: 'var(--bg-primary)', border: 'none' }}>
                      {exerciseTargetText.trim().split(/\s+/).map((word, idx) => (
                        <span key={idx} className={getLiveWordColorClass(word, idx)}>{word} </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Impromptu Topic Prompt Card (for advanced impromptu speaking) */}
                {activeExerciseDetail._id === "advanced_impromptu_speaking" && (
                  <div className="glass-card" style={{ marginBottom: '24px', textAlign: 'left' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <h3 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-secondary)', margin: '0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        💡 Impromptu Speaking Topic
                      </h3>
                      <button 
                        onClick={() => setPromptIndex((promptIndex + 1) % TOPIC_PROMPTS.length)}
                        style={{ 
                          display: 'inline-flex', 
                          alignItems: 'center', 
                          gap: '6px', 
                          fontSize: '11px', 
                          backgroundColor: 'rgba(255,255,255,0.04)', 
                          padding: '4px 10px', 
                          borderRadius: 'var(--radius-sm)', 
                          border: '1px solid var(--border-color)',
                          color: 'var(--text-primary)',
                          cursor: 'pointer'
                        }}
                      >
                        <RefreshCw size={12} /> Next Topic Prompt
                      </button>
                    </div>
                    <div className="practice-sentence-box" style={{ fontSize: '18px', lineHeight: '1.6', backgroundColor: 'var(--bg-primary)', border: 'none', fontStyle: 'italic' }}>
                      "{TOPIC_PROMPTS[promptIndex]}"
                    </div>
                  </div>
                )}

                <div className="workspace-layout">
                  {/* Left Column: Practice Settings configuration matching Module 1 */}
                  <div className="workspace-panel config-panel" style={{ textAlign: 'left', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <div>
                      <h2 className="panel-title">1. Practice Settings</h2>
                      <p className="panel-subtitle" style={{ marginBottom: '16px' }}>Configure difficulty level and how to load target passages.</p>

                      {/* Difficulty Level Switcher */}
                      <div className="form-group" style={{ marginBottom: '16px' }}>
                        <label className="form-label" style={{ fontSize: '11px' }}>Difficulty Level</label>
                        <div style={{ display: 'flex', gap: '4px', backgroundColor: 'rgba(255,255,255,0.02)', padding: '4px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                          {(['beginner', 'intermediate', 'advanced'] as const).map((lvl) => (
                            <button
                              key={lvl}
                              onClick={() => {
                                setExerciseDifficulty(lvl);
                                const newPresets = getPresetsForExercise(activeExerciseDetail._id, lvl);
                                setExerciseTargetText(newPresets[0]);
                                setAnalysisResult(null);
                              }}
                              style={{
                                flex: 1,
                                padding: '6px',
                                fontSize: '11px',
                                fontWeight: '600',
                                textTransform: 'capitalize',
                                backgroundColor: exerciseDifficulty === lvl ? 'var(--bg-secondary)' : 'transparent',
                                border: exerciseDifficulty === lvl ? '1px solid var(--border-color)' : 'none',
                                color: exerciseDifficulty === lvl ? 'var(--text-primary)' : 'var(--text-secondary)',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                transition: 'var(--transition-fast)'
                              }}
                            >
                              {lvl}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Unified drill tab switcher */}
                      <div style={{ display: 'flex', gap: '4px', backgroundColor: 'rgba(255,255,255,0.02)', padding: '4px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', marginBottom: '16px' }}>
                        <button 
                          onClick={() => { setConfigTabExercise('presets'); setAnalysisResult(null); }}
                          style={{ 
                            flex: 1, 
                            padding: '8px', 
                            fontSize: '12px', 
                            fontWeight: '600',
                            backgroundColor: configTabExercise === 'presets' ? 'var(--bg-secondary)' : 'transparent', 
                            border: configTabExercise === 'presets' ? '1px solid var(--border-color)' : 'none', 
                            color: configTabExercise === 'presets' ? 'var(--text-primary)' : 'var(--text-secondary)',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            transition: 'var(--transition-fast)'
                          }}
                        >
                          📋 Presets
                        </button>
                        <button 
                          onClick={() => { setConfigTabExercise('custom'); setAnalysisResult(null); }}
                          style={{ 
                            flex: 1, 
                            padding: '8px', 
                            fontSize: '12px', 
                            fontWeight: '600',
                            backgroundColor: configTabExercise === 'custom' ? 'var(--bg-secondary)' : 'transparent', 
                            border: configTabExercise === 'custom' ? '1px solid var(--border-color)' : 'none', 
                            color: configTabExercise === 'custom' ? 'var(--text-primary)' : 'var(--text-secondary)',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            transition: 'var(--transition-fast)'
                          }}
                        >
                          ✏ Custom
                        </button>
                        <button 
                          onClick={() => { setConfigTabExercise('groq'); setAnalysisResult(null); }}
                          style={{ 
                            flex: 1, 
                            padding: '8px', 
                            fontSize: '12px', 
                            fontWeight: '600',
                            backgroundColor: configTabExercise === 'groq' ? 'rgba(168, 85, 247, 0.1)' : 'transparent', 
                            border: configTabExercise === 'groq' ? '1px solid var(--secondary)' : 'none', 
                            color: configTabExercise === 'groq' ? 'var(--text-primary)' : 'var(--text-secondary)',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            transition: 'var(--transition-fast)'
                          }}
                        >
                          ✨ Groq AI
                        </button>
                      </div>

                      {/* Sub-tab content */}
                      {configTabExercise === 'presets' && (
                        <div className="form-group" style={{ marginBottom: '0' }}>
                          <label className="form-label">Select Drill Preset</label>
                          <select 
                            style={{ 
                              padding: '12px', 
                              backgroundColor: 'var(--bg-primary)', 
                              color: 'var(--text-primary)',
                              border: '1px solid var(--border-color)',
                              borderRadius: 'var(--radius-sm)',
                              width: '100%',
                              fontSize: '14px'
                            }}
                            value={exerciseTargetText}
                            onChange={(e) => { setExerciseTargetText(e.target.value); setAnalysisResult(null); }}
                          >
                            {getPresetsForExercise(activeExerciseDetail._id, exerciseDifficulty).map((presetText, pIdx) => (
                              <option key={pIdx} value={presetText}>
                                {presetText.length > 50 ? `${presetText.substring(0, 50)}...` : presetText}
                              </option>
                            ))}
                          </select>
                        </div>
                      )}

                      {configTabExercise === 'custom' && (
                        <div className="form-group" style={{ marginBottom: '0' }}>
                          <label className="form-label">Write Custom Target Text</label>
                          <input 
                            type="text" 
                            className="form-input" 
                            style={{ padding: '12px', fontSize: '14px', width: '100%' }}
                            value={exerciseTargetText}
                            onChange={(e) => { setExerciseTargetText(e.target.value); setAnalysisResult(null); }}
                            placeholder="Type custom text to practice in this drill..."
                          />
                        </div>
                      )}

                      {configTabExercise === 'groq' && (
                        <div className="glass-card" style={{ padding: '16px', border: '1px solid var(--border-color)', backgroundColor: 'rgba(255,255,255,0.01)' }}>
                          <h3 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <Sparkles size={14} style={{ color: 'var(--accent)' }} /> Generate Custom Text with Groq AI
                          </h3>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            <div className="form-group" style={{ marginBottom: '4px' }}>
                              <label className="form-label" style={{ fontSize: '10px' }}>Topic Keyword</label>
                              <input 
                                type="text" 
                                className="form-input" 
                                style={{ padding: '6px 12px', fontSize: '13px', width: '100%' }}
                                value={aiTopic}
                                onChange={e => setAiTopic(e.target.value)}
                                placeholder="Space, Science, Speech..."
                              />
                            </div>
                            <button 
                              onClick={() => handleGenerateExerciseText(activeExerciseDetail._id)} 
                              className="submit-btn" 
                              disabled={isGenerating}
                              style={{ padding: '8px', fontSize: '13px', marginTop: '5px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                            >
                              {isGenerating ? (
                                <><RefreshCw size={14} className="spin" /> Generating...</>
                              ) : (
                                <><Sparkles size={14} /> Generate with Groq AI</>
                              )}
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right Column: Speak & Record workspace (Exact mirror of M1/M2 layout) */}
                  <div className="workspace-panel recording-panel" style={{ textAlign: 'center' }}>
                    <h2 className="panel-title">2. Speak & Record</h2>
                    <p className="panel-subtitle" style={{ marginBottom: '20px' }}>Activate the mic and read the target passage cleanly. The analysis evaluates speech targets.</p>
                    
                    <div className="recorder-container" style={{ minHeight: '260px' }}>
                      <div className={`status-badge ${status}`}>
                        {status === 'recording' && <span className="pulse-dot" style={{ width: '8px', height: '8px', backgroundColor: 'var(--error)', borderRadius: '50%', display: 'inline-block', marginRight: '6px' }}></span>}
                        {statusMessage}
                      </div>

                      <div className="record-btn-wrapper" style={{ margin: '20px 0' }}>
                        {isRecording && (
                          <div style={{
                            position: 'absolute',
                            top: '-8px',
                            left: '-8px',
                            right: '-8px',
                            bottom: '-8px',
                            borderRadius: '50%',
                            border: '2px solid var(--error)',
                            animation: 'ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite',
                            opacity: 0.7,
                            pointerEvents: 'none'
                          }} />
                        )}
                        <button 
                          onClick={isRecording ? stopRecording : startRecording} 
                          className={`record-btn ${isRecording ? 'recording' : ''}`}
                          disabled={status === 'processing'}
                          style={{ border: 'none' }}
                        >
                          {isRecording ? <Square size={26} /> : <Mic size={26} />}
                        </button>
                      </div>

                      {/* Visualizer wave bars showing client speaking */}
                      {isRecording && (
                        <div className="visualizer-waves">
                          <div className="wave-bar"></div>
                          <div className="wave-bar"></div>
                          <div className="wave-bar"></div>
                          <div className="wave-bar"></div>
                          <div className="wave-bar"></div>
                          <div className="wave-bar"></div>
                          <div className="wave-bar"></div>
                          <div className="wave-bar"></div>
                        </div>
                      )}

                      {liveTranscript && (
                        <div style={{ marginTop: '20px', padding: '10px', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)', width: '100%', textAlign: 'left' }}>
                          <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Client Live Feed (Web Speech):</span>
                          <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>{liveTranscript}...</p>
                        </div>
                      )}

                      {audioUrl && !isRecording && (
                        <div style={{ marginTop: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
                          <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Playback:</span>
                          <button
                            onClick={() => playUserRecording()}
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '6px',
                              fontSize: '12px',
                              fontWeight: '600',
                              padding: '6px 12px',
                              backgroundColor: 'rgba(255, 255, 255, 0.03)',
                              border: '1px solid var(--border-color)',
                              borderRadius: 'var(--radius-sm)',
                              color: 'var(--text-primary)',
                              cursor: 'pointer',
                              transition: 'var(--transition-fast)'
                            }}
                          >
                            {(isAudioPlaying || isAudioLoading) && currentAudioType === 'recording' ? (
                              <Square size={14} style={{ color: 'var(--error)' }} />
                            ) : (
                              <Play size={14} style={{ color: 'var(--success)' }} />
                            )}
                            {(isAudioPlaying || isAudioLoading) && currentAudioType === 'recording' ? 'Stop' : 'Play My Recording'}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Below: Results evaluation reports for Exercises */}
                
                {/* Pronunciation scoring card (renders if target passage accuracy was evaluated) */}
                {analysisResult && (analysisResult.word_error_rate !== undefined || (analysisResult.mismatched_words && analysisResult.mismatched_words.length > 0)) && (
                  <div className="glass-card result-section" style={{ marginTop: '30px', textAlign: 'left' }}>
                    <h3 className="feedback-title" style={{ textAlign: 'center', marginBottom: '20px' }}>Pronunciation Accuracy Analysis</h3>
                    
                    {/* Passage with words highlighted in Green (correct) and Red (wrong pronunciation) */}
                    <div style={{ marginBottom: '24px', padding: '18px', backgroundColor: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                      <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '12px' }}>
                        📖 Pronunciation Passage Evaluation
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center', fontSize: '18px', lineHeight: '1.8' }}>
                        {(exerciseTargetText || analysisResult.target_text || '').trim().split(/\s+/).map((word: string, index: number) => {
                          const cleanWord = word.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, "");
                          const mismatchedIndices = new Set((analysisResult.mismatched_words || []).map((w: any) => w.index));
                          const isMismatched = mismatchedIndices.has(index);
                          return (
                            <span key={index} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                              {isMismatched && (
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    speakText(cleanWord);
                                  }}
                                  title={`Listen to correct pronunciation of "${cleanWord}"`}
                                  style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '3px',
                                    padding: '2px 6px',
                                    backgroundColor: 'rgba(99, 102, 241, 0.15)',
                                    border: '1px solid var(--primary)',
                                    borderRadius: '4px',
                                    color: 'var(--primary)',
                                    fontSize: '11px',
                                    fontWeight: 700,
                                    cursor: 'pointer',
                                    transition: 'all 0.15s ease',
                                    verticalAlign: 'middle',
                                  }}
                                >
                                  <Volume2 size={11} />
                                  <span>Listen</span>
                                </button>
                              )}
                              <span 
                                className={`practice-word ${isMismatched ? 'incorrect' : 'correct'}`}
                              >
                                {word}{' '}
                              </span>
                            </span>
                          );
                        })}
                      </div>
                      {/* Pronunciation Legend */}
                      <div style={{ display: 'flex', gap: '20px', marginTop: '16px', paddingTop: '12px', borderTop: '1px solid var(--border-color)', fontSize: '13px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#22c55e' }}></span>
                          <span style={{ color: '#22c55e', fontWeight: 600 }}>Green: Correct Pronunciation</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#ef4444' }}></span>
                          <span style={{ color: '#ef4444', fontWeight: 600 }}>Red: Wrong Pronunciation / Missed</span>
                        </div>
                      </div>
                    </div>

                    <div className="result-grid">
                      <div className="score-panel">
                        <div className="score-circle">
                          <span className="score-val">
                            {analysisResult.word_error_rate !== undefined 
                              ? Math.max(0, Math.round((1 - analysisResult.word_error_rate) * 10 * 10) / 10) 
                              : analysisResult.final_score}
                          </span>
                          <span className="score-label">/ 10</span>
                        </div>
                        <h4 className="score-heading">Accuracy Score</h4>
                        <p className="score-desc">
                          Accuracy: {(((1 - (analysisResult.word_error_rate ?? 0))) * 100).toFixed(0)}% matching
                        </p>
                      </div>

                      <div className="feedback-panel">
                        <h4 className="feedback-title">Articulation & Word Alignment</h4>
                        <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '16px' }}>
                          Target words matched against your spoken audio. Words pronounced correctly are highlighted in green, and wrong/mispronounced words are highlighted in red.
                        </p>
                        {analysisResult.mismatched_words && analysisResult.mismatched_words.length > 0 ? (
                          <ul className="feedback-list">
                            {analysisResult.mismatched_words.map((w: any, idx: number) => (
                              <li className="feedback-item" key={idx} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                  <span className="feedback-bullet" style={{ color: '#ef4444' }}>▸</span>
                                  <span>
                                    At index {w.index + 1}: {w.spoken ? (
                                      <>Expected <strong style={{ color: '#22c55e' }}>"{w.expected}"</strong> but recorded <strong style={{ color: '#ef4444' }}>"{w.spoken}"</strong></>
                                    ) : (
                                      <>Expected <strong style={{ color: '#22c55e' }}>"{w.expected}"</strong> <span style={{ color: '#ef4444', fontStyle: 'italic' }}>(not detected / skipped)</span></>
                                    )}
                                  </span>
                                </div>
                                <button
                                  type="button"
                                  onClick={() => speakText(w.expected)}
                                  style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '5px',
                                    fontSize: '11px',
                                    fontWeight: 600,
                                    padding: '4px 10px',
                                    backgroundColor: 'rgba(99, 102, 241, 0.12)',
                                    border: '1px solid var(--primary)',
                                    borderRadius: 'var(--radius-sm)',
                                    color: 'var(--text-primary)',
                                    cursor: 'pointer',
                                    transition: 'var(--transition-fast)',
                                    whiteSpace: 'nowrap'
                                  }}
                                  title={`Listen to correct pronunciation of "${w.expected}"`}
                                >
                                  <Volume2 size={12} style={{ color: 'var(--primary)' }} />
                                  Listen
                                </button>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <div style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', fontWeight: 'bold' }}>
                            <CheckCircle size={18} /> Excellent clarity! You didn't make any phonetic mistakes.
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Fluency & Clarity scoring card */}
                {analysisResult && (
                  <div className="glass-card result-section" style={{ marginTop: '30px', textAlign: 'left' }}>
                    <h3 className="feedback-title" style={{ textAlign: 'center' }}>Fluency & Clarity Analysis</h3>
                    
                    <div className="result-grid">
                      <div className="score-panel">
                        <div className="score-circle" style={{ borderColor: 'var(--secondary-light)', borderTopColor: 'var(--secondary)' }}>
                          <span className="score-val">{analysisResult.final_score}</span>
                          <span className="score-label">/ 10</span>
                        </div>
                        <h4 className="score-heading">Overall Fluency Score</h4>
                        <p className="score-desc">
                          Pacing: {analysisResult.wpm} WPM ({analysisResult.duration_sec.toFixed(1)}s)
                        </p>
                      </div>

                      <div className="feedback-panel">
                        <h4 className="feedback-title">Disfluency Markers</h4>
                        <ul className="feedback-list">
                          <li className="feedback-item">
                            <span className="feedback-bullet">▸</span>
                            <span>Filler words: <strong>{analysisResult.filler_count}</strong> {analysisResult.filler_words_found.length > 0 && `(${analysisResult.filler_words_found.join(', ')})`}</span>
                          </li>
                          <li className="feedback-item">
                            <span className="feedback-bullet">▸</span>
                            <span>Stammer / repetitions: <strong>{analysisResult.stammer_events}</strong> instances</span>
                          </li>
                          <li className="feedback-item">
                            <span className="feedback-bullet">▸</span>
                            <span>Long pauses (&gt;1.5s): <strong>{analysisResult.long_pauses}</strong> detected</span>
                          </li>
                          <li className="feedback-item">
                            <span className="feedback-bullet">▸</span>
                            <span>Speech rate: <strong>{analysisResult.wpm} WPM</strong> (Ideal: 120-150)</span>
                          </li>
                        </ul>

                        {analysisResult.transcript && (
                          <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                              <h5 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', margin: 0 }}>
                                🎙️ Spoken Transcript & Tagged Fillers:
                              </h5>
                              {analysisResult.filler_count > 0 && (
                                <span style={{ fontSize: '11px', color: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.12)', padding: '2px 8px', borderRadius: '4px', fontWeight: 600 }}>
                                  {analysisResult.filler_count} filler {analysisResult.filler_count === 1 ? 'word' : 'words'} detected
                                </span>
                              )}
                            </div>
                            <p style={{ fontSize: '13px', lineHeight: '1.8', color: 'var(--text-primary)', margin: 0 }}>
                              {renderHighlightedTranscript(analysisResult.transcript, analysisResult.filler_words_found)}
                            </p>
                          </div>
                        )}

                        {analysisResult.pause_details && analysisResult.pause_details.length > 0 && (
                          <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                            <h5 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '8px' }}>⏱️ Detected Pause Locations:</h5>
                            
                            {/* Interactive Pause timeline bar */}
                            <div style={{ 
                              position: 'relative', 
                              height: '12px', 
                              backgroundColor: 'var(--bg-primary)', 
                              border: '1px solid var(--border-color)', 
                              borderRadius: '6px', 
                              overflow: 'hidden', 
                              width: '100%', 
                              marginBottom: '8px' 
                            }}>
                              {analysisResult.pause_details.map((pause, idx) => {
                                const total = analysisResult.duration_sec || 1;
                                const leftPct = (pause.start / total) * 100;
                                const widthPct = (pause.duration / total) * 100;
                                return (
                                  <div 
                                    key={idx} 
                                    style={{ 
                                      position: 'absolute', 
                                      left: `${leftPct}%`, 
                                      width: `${widthPct}%`, 
                                      height: '100%', 
                                      backgroundColor: 'var(--error)',
                                      opacity: 0.8
                                    }}
                                    title={`Pause: ${pause.duration}s (from ${pause.start}s to ${pause.end}s)`}
                                  />
                                );
                              })}
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                              <span>0.0s</span>
                              <span>{(analysisResult.duration_sec || 0).toFixed(1)}s</span>
                            </div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                              {analysisResult.pause_details.map((pause, idx) => (
                                <span key={idx} style={{ fontSize: '11px', backgroundColor: 'var(--warning-light)', color: 'var(--warning)', padding: '2px 8px', borderRadius: '4px' }}>
                                  {pause.duration}s gap at {pause.start}s
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Detected Filler Words Timeline and Chips */}
                        {analysisResult.filler_details && analysisResult.filler_details.length > 0 && (
                          <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                            <h5 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                              🗣️ Detected Filler Word Locations:
                            </h5>
                            
                            {/* Interactive Filler timeline bar */}
                            <div style={{ 
                              position: 'relative', 
                              height: '12px', 
                              backgroundColor: 'var(--bg-primary)', 
                              border: '1px solid var(--border-color)', 
                              borderRadius: '6px', 
                              overflow: 'hidden', 
                              width: '100%', 
                              marginBottom: '8px' 
                            }}>
                              {analysisResult.filler_details.map((filler, idx) => {
                                const total = analysisResult.duration_sec || 1;
                                const leftPct = (filler.start / total) * 100;
                                const widthPct = Math.max((filler.duration / total) * 100, 1.2);
                                return (
                                  <div 
                                    key={idx} 
                                    style={{ 
                                      position: 'absolute', 
                                      left: `${leftPct}%`, 
                                      width: `${widthPct}%`, 
                                      height: '100%', 
                                      backgroundColor: '#f59e0b',
                                      opacity: 0.85
                                    }}
                                    title={`Filler: "${filler.word}" (${filler.duration}s from ${filler.start}s to ${filler.end}s)`}
                                  />
                                );
                              })}
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                              <span>0.0s</span>
                              <span>{(analysisResult.duration_sec || 0).toFixed(1)}s</span>
                            </div>

                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                              {analysisResult.filler_details.map((filler, idx) => (
                                <span key={idx} style={{ fontSize: '11px', backgroundColor: 'rgba(245, 158, 11, 0.12)', color: '#d97706', padding: '2px 8px', borderRadius: '4px', fontWeight: 500 }}>
                                  "{filler.word}" at {filler.start}s ({filler.duration}s)
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Detected Stammers and Repetitions Timeline and Chips */}
                        {analysisResult.stammer_details && analysisResult.stammer_details.length > 0 && (
                          <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
                            <h5 style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                              ⚡ Detected Stammers & Repetitions:
                            </h5>
                            
                            {/* Interactive Stammer timeline bar */}
                            <div style={{ 
                              position: 'relative', 
                              height: '12px', 
                              backgroundColor: 'var(--bg-primary)', 
                              border: '1px solid var(--border-color)', 
                              borderRadius: '6px', 
                              overflow: 'hidden', 
                              width: '100%', 
                              marginBottom: '8px' 
                            }}>
                              {analysisResult.stammer_details.map((stammer, idx) => {
                                const total = analysisResult.duration_sec || 1;
                                const leftPct = (stammer.start / total) * 100;
                                const widthPct = Math.max((stammer.duration / total) * 100, 1.2);
                                return (
                                  <div 
                                    key={idx} 
                                    style={{ 
                                      position: 'absolute', 
                                      left: `${leftPct}%`, 
                                      width: `${widthPct}%`, 
                                      height: '100%', 
                                      backgroundColor: '#ec4899',
                                      opacity: 0.85
                                    }}
                                    title={`Stammer (${stammer.type}): "${stammer.text}" (${stammer.duration}s from ${stammer.start}s to ${stammer.end}s)`}
                                  />
                                );
                              })}
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                              <span>0.0s</span>
                              <span>{(analysisResult.duration_sec || 0).toFixed(1)}s</span>
                            </div>

                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                              {analysisResult.stammer_details.map((stammer, idx) => (
                                <span key={idx} style={{ fontSize: '11px', backgroundColor: 'rgba(236, 72, 153, 0.12)', color: '#db2777', padding: '2px 8px', borderRadius: '4px', fontWeight: 500 }}>
                                  "{stammer.text}" ({stammer.type}) at {stammer.start}s
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* AI pathologist assessment feedback */}
                        {analysisResult.ai_pathologist_feedback && (
                          <div className="feedback-panel" style={{ marginTop: '16px', borderLeft: '4px solid var(--accent)', backgroundColor: 'rgba(99, 102, 241, 0.02)', padding: '12px 16px' }}>
                            <h4 className="feedback-title" style={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', margin: '0' }}>
                              <Sparkles size={14} /> AI Speech Pathologist Assessment
                            </h4>
                            <p style={{ fontSize: '13px', lineHeight: '1.6', color: 'var(--text-primary)', fontStyle: 'italic', margin: '6px 0 0' }}>
                              "{analysisResult.ai_pathologist_feedback}"
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <>
                <div className="dashboard-hero" style={{ marginBottom: '24px' }}>
                  <h1 className="hero-title">Speech Therapy Drills & Exercises Library</h1>
                  <p className="hero-subtitle">Search exercises loaded directly from the database and practice specific pronunciation or speech rhythms.</p>
                </div>

                {/* Filter Search Bar */}
                <div style={{ display: 'flex', gap: '10px', maxWidth: '500px', margin: '0 auto 30px' }}>
                  <div style={{ position: 'relative', flex: 1 }}>
                    <Search size={18} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                    <input 
                      type="text" 
                      className="form-input" 
                      style={{ paddingLeft: '40px', width: '100%' }}
                      value={exerciseSearch}
                      onChange={e => setExerciseSearch(e.target.value)}
                      placeholder="Search by drill title, difficulty, or tags..."
                    />
                  </div>
                </div>

                {filteredExercises.length === 0 ? (
                  <p style={{ color: 'var(--text-secondary)', textAlign: 'center' }}>No speech drills match your search criteria. Try a different query!</p>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '20px', textAlign: 'left' }}>
                    {filteredExercises.map((ex, idx) => (
                      <div key={idx} className="glass-card exercise-library-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', borderLeft: `4px solid ${ex.difficulty === 'beginner' ? 'var(--success)' : ex.difficulty === 'intermediate' ? 'var(--warning)' : 'var(--error)'}` }}>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                            <h4 style={{ fontSize: '15px', fontWeight: 'bold', color: 'var(--text-primary)' }}>{ex.title}</h4>
                            <span style={{ 
                              fontSize: '10px', 
                              padding: '2px 8px', 
                              borderRadius: '4px',
                              textTransform: 'uppercase',
                              fontWeight: 'bold',
                              backgroundColor: ex.difficulty === 'beginner' ? 'rgba(16, 185, 129, 0.1)' : ex.difficulty === 'intermediate' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(244, 63, 94, 0.1)',
                              color: ex.difficulty === 'beginner' ? 'var(--success)' : ex.difficulty === 'intermediate' ? 'var(--warning)' : 'var(--error)'
                            }}>
                              {ex.difficulty}
                            </span>
                          </div>
                          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5', marginBottom: '12px' }}>{ex.description}</p>
                          
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '15px' }}>
                            <code>Triggers on: {ex.trigger_condition}</code>
                          </div>
                        </div>

                        <button 
                          onClick={() => startExerciseDrill(ex)}
                          className="submit-btn" 
                          style={{ fontSize: '13px', padding: '8px', fontWeight: 'bold', marginTop: '0', boxShadow: 'none' }}
                        >
                          Start Drill
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* 3. MODULE 2: VIDEO & COMMUNICATION ANALYSIS */}
        {activeTab === 'video' && (
          <div className="view-fade">
            <VideoAnalysisModule currentUser={currentUser} />
          </div>
        )}
      </main>

      {/* User Registration/Login Overlay Modal */}
      {showAuthModal && (
        <div className="modal-overlay">
          <div className="modal-content glass">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                {authMode === 'login' ? <LogIn size={20} /> : <UserPlus size={20} />}
                {authMode === 'login' ? 'Sign In to SpeechAI' : 'Create SpeechAI Account'}
              </h3>
              <button onClick={() => setShowAuthModal(false)} style={{ color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer' }}>
                <X size={20} />
              </button>
            </div>

            {authError && (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', backgroundColor: 'var(--error-light)', border: '1px solid var(--error)', padding: '10px', borderRadius: 'var(--radius-sm)', color: 'var(--error)', fontSize: '13px', marginBottom: '18px', textAlign: 'left' }}>
                <AlertCircle size={16} style={{ flexShrink: 0 }} />
                <span>{authError}</span>
              </div>
            )}

            <form onSubmit={handleAuthSubmit}>
              {authMode === 'register' && (
                <div className="form-group">
                  <label className="form-label">Full Name</label>
                  <input 
                    type="text" 
                    className="form-input" 
                    placeholder="Enter your name" 
                    value={authName} 
                    onChange={e => setAuthName(e.target.value)}
                    required
                  />
                </div>
              )}

              <div className="form-group">
                <label className="form-label">Email Address</label>
                <input 
                  type="email" 
                  className="form-input" 
                  placeholder="name@example.com" 
                  value={authEmail} 
                  onChange={e => setAuthEmail(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">Password</label>
                <input 
                  type="password" 
                  className="form-input" 
                  placeholder="••••••••" 
                  value={authPassword} 
                  onChange={e => setAuthPassword(e.target.value)}
                  required
                />
              </div>

              <button type="submit" className="submit-btn">
                {authMode === 'login' ? 'Sign In' : 'Register Account'}
              </button>
            </form>

            <span className="auth-toggle-link" onClick={() => { setAuthMode(authMode === 'login' ? 'register' : 'login'); setAuthError(null); }}>
              {authMode === 'login' ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
