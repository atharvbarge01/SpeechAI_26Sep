import React, { useState } from 'react';
import { MessageSquare, Clock, Sparkles, RefreshCw, ArrowRight } from 'lucide-react';
import { apiUrl } from '../../../api';

interface ModuleConfigPanelProps {
  onStartSession: (config: {
    taskType: 'free_talk' | 'interview' | 'custom_topic';
    customParagraph?: string;
  }) => void;
}

const QUICK_TOPICS = [
  "Artificial Intelligence",
  "Space Exploration",
  "Leadership & Teamwork",
  "Climate Action",
  "Healthy Daily Habits",
  "Future of Remote Work"
];

export const ModuleConfigPanel: React.FC<ModuleConfigPanelProps> = ({ onStartSession }) => {
  const [taskType, setTaskType] = useState<'free_talk' | 'interview' | 'custom_topic'>('free_talk');
  
  // Custom Topic states
  const [aiTopic, setAiTopic] = useState('Artificial Intelligence');
  const [aiLength, setAiLength] = useState<'sentence' | 'paragraph' | 'long_paragraph'>('paragraph');
  const [aiEnglishLevel, setAiEnglishLevel] = useState<'easy' | 'medium' | 'difficult'>('medium');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedText, setGeneratedText] = useState<string | null>(null);

  const handleGenerate = async (topicParam: string = aiTopic) => {
    setIsGenerating(true);
    setGeneratedText(null);
    try {
      const url = apiUrl(`/practice/generate?topic=${encodeURIComponent(topicParam)}&length=${aiLength}&level=${aiEnglishLevel}&exercise_id=none`);
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setGeneratedText(data.text);
      } else {
        throw new Error('Failed to generate paragraph');
      }
    } catch (e) {
      console.error(e);
      const fallbackText = "Speaking in front of an audience can be intimidating at first, but with steady pacing and deliberate breaths, anyone can deliver a powerful message.";
      setGeneratedText(fallbackText);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleStart = () => {
    onStartSession({
      taskType,
      customParagraph: taskType === 'custom_topic' ? (generatedText || "Please generate a topic first.") : undefined
    });
  };

  return (
    <div className="workspace-panel" style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 className="panel-title">Select Practice Mode</h2>
        <p className="panel-subtitle">Choose the speaking context and difficulty for your multimodal communication session.</p>
      </div>
      
      <div className="vid-task-selector">
        <div
          className={`vid-task-option ${taskType === 'free_talk' ? 'active' : ''}`}
          onClick={() => setTaskType('free_talk')}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <MessageSquare size={16} style={{ color: 'var(--primary)' }} />
            <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>Free Talk</span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Conversational prompts (120-150 WPM)</p>
        </div>

        <div
          className={`vid-task-option ${taskType === 'interview' ? 'active' : ''}`}
          onClick={() => setTaskType('interview')}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <Clock size={16} style={{ color: 'var(--secondary)' }} />
            <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>Interview Answer</span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>AI Interviewer speaks questions</p>
        </div>

        <div
          className={`vid-task-option ${taskType === 'custom_topic' ? 'active' : ''}`}
          onClick={() => setTaskType('custom_topic')}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <Sparkles size={16} style={{ color: 'var(--accent)' }} />
            <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>Custom Topic AI</span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Generate reading passages on any topic</p>
        </div>
      </div>

      {taskType === 'custom_topic' && (
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>Generator Settings</h3>
          
          <div className="vid-quick-chips">
            {QUICK_TOPICS.map(topic => (
              <button
                key={topic}
                type="button"
                className={`vid-chip-btn ${aiTopic === topic ? 'active' : ''}`}
                onClick={() => {
                  setAiTopic(topic);
                  handleGenerate(topic);
                }}
              >
                {topic}
              </button>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr auto', gap: '10px', alignItems: 'flex-end' }}>
            <div className="form-group" style={{ marginBottom: '0' }}>
              <label className="form-label" style={{ fontSize: '10px' }}>Topic Keyword</label>
              <input
                type="text"
                className="form-input"
                style={{ padding: '7px 12px', fontSize: '13px', width: '100%' }}
                value={aiTopic}
                onChange={(e) => setAiTopic(e.target.value)}
              />
            </div>
            <div className="form-group" style={{ marginBottom: '0' }}>
              <label className="form-label" style={{ fontSize: '10px' }}>Length</label>
              <select
                className="form-input"
                style={{ width: '100%', padding: '7px 10px', fontSize: '13px' }}
                value={aiLength}
                onChange={(e) => setAiLength(e.target.value as any)}
              >
                <option value="sentence">Sentence</option>
                <option value="paragraph">Paragraph</option>
                <option value="long_paragraph">Long Paragraph</option>
              </select>
            </div>
            <div className="form-group" style={{ marginBottom: '0' }}>
              <label className="form-label" style={{ fontSize: '10px' }}>Level of English</label>
              <select
                className="form-input"
                style={{ width: '100%', padding: '7px 10px', fontSize: '13px' }}
                value={aiEnglishLevel}
                onChange={(e) => setAiEnglishLevel(e.target.value as 'easy' | 'medium' | 'difficult')}
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="difficult">Difficult</option>
              </select>
            </div>
            <button
              className="vid-btn-secondary"
              onClick={() => handleGenerate()}
              disabled={isGenerating}
              style={{ padding: '8px 18px', fontSize: '13px' }}
            >
              {isGenerating ? <RefreshCw size={14} className="spin" /> : <Sparkles size={14} />} Generate
            </button>
          </div>

          {generatedText && (
            <div style={{ background: 'rgba(255,255,255,0.05)', padding: '12px', borderRadius: '8px', fontSize: '14px', lineHeight: 1.5 }}>
              {generatedText}
            </div>
          )}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '16px' }}>
        <button
          className="vid-btn-primary"
          onClick={handleStart}
          disabled={taskType === 'custom_topic' && !generatedText}
          style={{ padding: '12px 24px', fontSize: '16px' }}
        >
          <span>Continue to Recording</span>
          <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
};
