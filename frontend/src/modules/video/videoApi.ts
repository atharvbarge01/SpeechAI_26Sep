import type { VideoJobResponse, VideoSessionDetail } from './types';
import { apiUrl } from '../../api';

const BASE_URL = (import.meta.env.VITE_VIDEO_API_URL || apiUrl('/video')).replace(/\/$/, '');

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem('speechai_token');
  const headers: HeadersInit = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

export async function checkVideoHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) return false;
    const data = await res.json();
    return data.status === 'healthy';
  } catch {
    return false;
  }
}

export async function submitVideoAnalysis(
  videoBlob: Blob,
  taskType: string = 'free_talk',
  promptText?: string
): Promise<VideoJobResponse> {
  const formData = new FormData();
  formData.append('video', videoBlob, 'recording.webm');
  formData.append('task_type', taskType);
  if (promptText) {
    formData.append('prompt_text', promptText);
  }

  const headers = getAuthHeaders();
  const res = await fetch(`${BASE_URL}/analyze`, {
    method: 'POST',
    headers,
    body: formData
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Upload failed (${res.status}): ${errText}`);
  }

  return await res.json();
}

export async function pollSessionStatus(jobId: string): Promise<VideoSessionDetail> {
  const headers = getAuthHeaders();
  const res = await fetch(`${BASE_URL}/session/${jobId}`, {
    headers
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch session status (${res.status})`);
  }

  return await res.json();
}

export async function fetchUserVideoReports(userId: string): Promise<VideoSessionDetail[]> {
  const headers = getAuthHeaders();
  const res = await fetch(`${BASE_URL}/reports/${userId}`, {
    headers
  });

  if (!res.ok) return [];
  return await res.json();
}

export async function deleteVideoSession(jobId: string): Promise<boolean> {
  const headers = getAuthHeaders();
  const res = await fetch(`${BASE_URL}/session/${jobId}`, {
    method: 'DELETE',
    headers
  });
  return res.ok;
}
